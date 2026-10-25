"""Choose the matrix's request-rate ladder from a measured calibration sweep.

The matrix asks for "4 rate points spanning underload to overload". That is only
meaningful relative to a *measured* saturation point of this fleet, with the
router in front of it, rather than one guessed from a single replica. (An
earlier version of this docstring also cited P1's measured co-location cost of
-34% throughput. That finding was withdrawn on 2026-10-24, because its four
replicas shared one port; measuring the knee on the real fleet still stands.)

Saturation is located from the SERVER's behaviour, by the first rate where

  * median TTFT rises above `latency_factor` x its lowest-rate baseline
    (queueing: under open-loop overload the backlog grows and every request
    waits longer), or
  * the error rate rises above `err_thresh` (admission control shedding).

Median, not p95. TTFT is measured from each request's *scheduled* arrival, the
correct open-loop method, so a client-side stall inflates TTFT directly. Stalls
hit a few requests, so they pollute the tail and leave the median clean: in
calibration run 5215612, p95 was highest at the LOWEST rates (0.84 s and 1.50 s
at 2 and 4 req/s) while p50 sat flat at 0.07-0.10 s. A p95 criterion would have
located the knee exactly where the fleet was idlest.

An earlier version used "achieved throughput < keep_up_frac x offered". That
ratio was computed over total elapsed time including the drain tail, and on
short phases it fell steadily with offered rate on a fleet that never saturated
(0.92 -> 0.66 across 2..32 req/s, zero errors). It fired at 4 req/s, called that
the knee, and produced an all-underload ladder that would have compared four
routing policies on a fleet that never contends. The docstring also promised
two independent signals so one noisy phase could not move the knee, while the
code broke on the first signal that fired. Both are corrected here.

Client keep-up is a SEPARATE check. If the generator cannot dispatch at the
offered rate, the server has not saturated; the measurement at that rate is
simply invalid. Such rates are reported and excluded, never promoted to a knee.

If no rate saturates, that is reported rather than papered over: the ladder then
spans the measured range and the caller is told the top of it was never driven
into overload.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_calibration(raw_dir: Path, policy: str = "round_robin") -> list[dict[str, Any]]:
    rows = []
    for p in sorted(raw_dir.glob("*.json")):
        try:
            o = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        cfg = o.get("full_config_dict", {}).get("phase")
        if not cfg or cfg.get("policy") != policy:
            continue
        if o.get("kind") == "failure_mode":
            continue
        s = o.get("summary") or {}
        if s.get("achieved_request_rate_per_s") is None:
            continue
        rows.append(
            {
                "offered": float(cfg["request_rate"]),
                "achieved": float(s["achieved_request_rate_per_s"]),
                "error_rate": s.get("error_rate"),
                "ttft_p50": (s.get("ttft_s") or {}).get("p50"),
                "ttft_p95": (s.get("ttft_s") or {}).get("p95"),
                "status": o.get("status"),
                "artifact": p.name,
            }
        )
    rows.sort(key=lambda r: r["offered"])
    return rows


def pick(
    rows: list[dict[str, Any]],
    *,
    keep_up_frac: float = 0.90,
    err_thresh: float = 0.01,
    latency_factor: float = 3.0,
    n: int = 4,
) -> dict[str, Any]:
    """Return {'rates': [...], 'knee': float|None, 'reason': str, 'table': rows}."""
    if not rows:
        return {"rates": [], "knee": None, "reason": "no calibration artifacts found", "table": []}

    # Client keep-up: a rate the generator could not actually offer is not a
    # measurement of the server at all. Mark it and drop it before knee search.
    client_lagged = [
        r["offered"] for r in rows
        if r.get("achieved") is not None and r["achieved"] < keep_up_frac * r["offered"]
    ]
    usable = [r for r in rows if r["offered"] not in client_lagged]
    if not usable:
        return {
            "rates": [], "knee": None, "table": rows,
            "reason": "the load generator fell behind at every calibration rate; "
                      "no server measurement is valid",
        }

    base = next((r["ttft_p50"] for r in usable if r.get("ttft_p50")), None)
    knee = None
    knee_signal = None
    for r in usable:
        blowup = bool(base and r.get("ttft_p50") and r["ttft_p50"] > latency_factor * base)
        shedding = (r.get("error_rate") or 0.0) > err_thresh
        if blowup or shedding:
            knee = r["offered"]
            knee_signal = (
                f"median TTFT {r['ttft_p50']:.3f}s > {latency_factor:g}x baseline {base:.3f}s"
                if blowup else f"error rate {r['error_rate']:.3f} > {err_thresh:g}"
            )
            break

    rows = usable
    offered = [r["offered"] for r in rows]
    if knee is None:
        # Never driven into overload. Say so; do not invent a higher rate than
        # was measured, and do not label the top of the ladder "overload".
        hi = offered[-1]
        rates = _spread(offered, n)
        return {
            "rates": rates,
            "knee": None,
            "reason": (
                f"no rate in the calibration ladder saturated the fleet "
                f"(median TTFT never exceeded {latency_factor:g}x baseline and no "
                f"shedding up to {hi:g} req/s); the ladder spans the measured "
                f"range but its top point is NOT verified overload"
                + (f"; client fell behind at {client_lagged}" if client_lagged else "")
            ),
            "table": rows,
        }

    # One clearly under, one approaching, one at the knee, one past it.
    below = [o for o in offered if o < knee]
    above = [o for o in offered if o > knee]
    picks: list[float] = []
    if below:
        picks.append(below[0])                      # clearly underloaded
        if len(below) > 1:
            picks.append(below[len(below) // 2])    # approaching
        if len(below) > 2 and below[-1] not in picks:
            picks.append(below[-1])                 # just under
    picks.append(knee)                              # at the knee
    if above:
        picks.append(above[min(len(above) - 1, 1)])  # clearly past it

    picks = sorted(set(picks))
    # Trim toward n while keeping the extremes, which define the span.
    while len(picks) > n:
        picks.pop(len(picks) // 2)
    return {
        "rates": picks,
        "knee": knee,
        "reason": f"knee located at {knee:g} req/s ({knee_signal})"
                  + (f"; client fell behind at {client_lagged}, excluded" if client_lagged else ""),
        "table": rows,
    }


def _spread(offered: list[float], n: int) -> list[float]:
    """Pick up to n rates evenly by index from the MEASURED ladder, extremes kept.

    Only calibrated rates are returned, never interpolated ones: a rate the
    fleet was not calibrated at has no evidence behind its load label.
    """
    if len(offered) <= n:
        return list(offered)
    idx = sorted({round(i * (len(offered) - 1) / (n - 1)) for i in range(n)})
    return [offered[i] for i in idx]


def format_table(rows: list[dict[str, Any]]) -> str:
    out = [
        f"{'offered':>8} {'achieved':>9} {'ttft p50':>9} {'ttft p95':>9} "
        f"{'err':>7}  status"
    ]
    for r in rows:
        def f(v: Any, nd: int = 3) -> str:
            return "      -" if v is None else f"{v:.{nd}f}"

        out.append(
            f"{r['offered']:>8.1f} {r['achieved']:>9.2f} {f(r['ttft_p50']):>9} "
            f"{f(r['ttft_p95']):>9} {f(r['error_rate']):>7}  {r['status']}"
        )
    return "\n".join(out)
