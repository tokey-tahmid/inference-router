"""Choose the matrix's request-rate ladder from a measured calibration sweep.

The matrix asks for "4 rate points spanning underload to overload". That is only
meaningful relative to a *measured* saturation point, and guessing it would be
wrong here in a knowable direction: P1 measured that co-locating four
single-GPU replicas costs -34% throughput versus isolation, so a ladder derived
from isolated-replica throughput would place every "overload" point inside the
stable region and the matrix would compare four policies that never contend.

Saturation is located by the first rate where the fleet stops keeping up, using
two independent signals so a single noisy phase cannot move the knee:

  * achieved throughput falls below `keep_up_frac` of offered, and/or
  * the error rate rises above `err_thresh` (admission control shedding).

Both are read from committed artifacts. If no rate saturates, that is reported
rather than papered over: the ladder then spans the measured range and the
caller is told the top of it was never driven into overload.
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
    n: int = 4,
) -> dict[str, Any]:
    """Return {'rates': [...], 'knee': float|None, 'reason': str, 'table': rows}."""
    if not rows:
        return {"rates": [], "knee": None, "reason": "no calibration artifacts found", "table": []}

    knee = None
    for r in rows:
        saturated = r["achieved"] < keep_up_frac * r["offered"]
        shedding = (r["error_rate"] or 0.0) > err_thresh
        if saturated or shedding:
            knee = r["offered"]
            break

    offered = [r["offered"] for r in rows]
    if knee is None:
        # Never driven into overload. Say so; do not invent a higher rate than
        # was measured, and do not label the top of the ladder "overload".
        lo, hi = offered[0], offered[-1]
        rates = _spread(lo, hi, n)
        return {
            "rates": rates,
            "knee": None,
            "reason": (
                f"no rate in the calibration ladder saturated the fleet "
                f"(max offered {hi:g} req/s still kept up); the ladder spans the "
                f"measured range but its top point is NOT verified overload"
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
        "reason": f"knee located at {knee:g} req/s (first rate where achieved < "
                  f"{keep_up_frac:g}x offered or error rate > {err_thresh:g})",
        "table": rows,
    }


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
