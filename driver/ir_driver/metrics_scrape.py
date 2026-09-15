"""Prometheus scraping and per-phase differencing.

The differencing is the whole point of this module.

Both vLLM's and the router's counters are **cumulative for the life of the
process**, and the experiment deliberately runs many phases against one
long-lived fleet so that the four policies are compared on identically-warmed
replicas. Reading a counter at the end of a phase therefore gives that phase's
value *plus every phase before it*.

P1 hit exactly this and it produced a result that looked like data: a 0.9-ratio
phase following a 0.0-ratio phase read low, so the prefix-caching comparison
became sensitive to phase ordering. The fix there and here is the same. Scrape
before and after, difference, and treat a negative delta as evidence the server
restarted rather than as a measurement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>[^\s]+)\s*$"
)


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse an exposition into `name{labels}` -> value.

    The full label set is kept in the key so per-replica series stay distinct.
    Values that are NaN or +Inf are dropped: they are not measurements.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if not m:
            continue
        try:
            val = float(m.group("value"))
        except ValueError:
            continue
        if val != val or val in (float("inf"), float("-inf")):
            continue
        labels = m.group("labels")
        key = f"{m.group('name')}{{{labels}}}" if labels else m.group("name")
        out[key] = val
    return out


async def scrape(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    """Scrape one /metrics endpoint. Errors are recorded, not raised."""
    try:
        resp = await client.get(f"{base_url}/metrics", timeout=20.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("metrics_scrape_failed", url=base_url, error=str(exc))
        return {"scrape_error": str(exc)}
    return {"samples": parse_prometheus(resp.text), "raw": resp.text}


def _total(samples: dict[str, float], *prefixes: str) -> float | None:
    """Sum every series whose name matches one of `prefixes`.

    Returns None when no series matched. **None, not 0.0** — deliberately.
    P1's prefix-cache hit rate read as a clean 0.000 at every sharing ratio for
    a while, and the only reason that was diagnosable as a metric-name change
    (vLLM 0.27.1 dropped the `gpu_` infix) rather than as "prefix caching does
    not work here" was that the adapter reported the metric absent instead of
    zero. An absent measurement and a measured zero are different facts.
    """
    hits = [v for k, v in samples.items() if any(k.startswith(p) for p in prefixes)]
    return sum(hits) if hits else None


@dataclass
class PrefixCacheDelta:
    """Per-phase prefix cache counters for one replica."""

    replica: str
    queries: float | None
    hits: float | None
    hit_rate: float | None
    status: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def prefix_cache_delta(
    replica: str, before: dict[str, Any], after: dict[str, Any]
) -> PrefixCacheDelta:
    """Difference two vLLM scrapes into a per-phase prefix cache hit rate.

    Counter names: vLLM 0.27.1 dropped the `gpu_` infix, so both spellings are
    accepted (old first, for an older pin). Copied from P1's verified adapter.
    """
    if "samples" not in before or "samples" not in after:
        return PrefixCacheDelta(replica, None, None, None, "scrape_failed")

    b, a = before["samples"], after["samples"]
    q_before = _total(b, "vllm:gpu_prefix_cache_queries_total", "vllm:prefix_cache_queries_total")
    q_after = _total(a, "vllm:gpu_prefix_cache_queries_total", "vllm:prefix_cache_queries_total")
    h_before = _total(b, "vllm:gpu_prefix_cache_hits_total", "vllm:prefix_cache_hits_total")
    h_after = _total(a, "vllm:gpu_prefix_cache_hits_total", "vllm:prefix_cache_hits_total")

    if q_after is None or h_after is None:
        return PrefixCacheDelta(replica, None, None, None, "counters_absent")

    # A missing "before" means this is the first scrape of this server. That is a
    # genuine unknown, not a zero.
    if q_before is None or h_before is None:
        return PrefixCacheDelta(replica, None, None, None, "no_baseline_scrape")

    dq = q_after - q_before
    dh = h_after - h_before
    if dq < 0 or dh < 0:
        # Counters only go up within one process lifetime, so a negative delta
        # means the server restarted mid-experiment. Discard rather than report.
        return PrefixCacheDelta(replica, dq, dh, None, "counter_went_backwards")
    if dq == 0:
        return PrefixCacheDelta(replica, 0.0, dh, None, "no_queries_in_phase")
    return PrefixCacheDelta(replica, dq, dh, dh / dq, "ok")


def router_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Difference the router's own counters over a phase.

    Only monotone counters are differenced; gauges (`_inflight`, `_healthy`,
    `last_transition_unix_us`) are taken from the `after` scrape as-is, because
    a difference of a gauge is meaningless.
    """
    if "samples" not in before or "samples" not in after:
        return {"status": "scrape_failed"}

    b, a = before["samples"], after["samples"]
    gauge_markers = ("_inflight", "_queued", "_healthy", "last_transition_unix_us", "active_policy")

    deltas: dict[str, float] = {}
    gauges: dict[str, float] = {}
    for key, av in a.items():
        if any(m in key for m in gauge_markers):
            gauges[key] = av
            continue
        bv = b.get(key)
        if bv is None:
            continue
        d = av - bv
        if d < 0:
            log.warning("router_counter_went_backwards", metric=key, before=bv, after=av)
            continue
        deltas[key] = d
    return {"status": "ok", "counter_deltas": deltas, "gauges": gauges}


async def reset_prefix_caches(
    client: httpx.AsyncClient, replica_urls: dict[str, str]
) -> dict[str, Any]:
    """Clear every replica's prefix cache between phases.

    Requires `VLLM_SERVER_DEV_MODE=1` on the server: in vLLM 0.27.1 the endpoint
    lives in `vllm/entrypoints/serve/dev/cache/api_router.py` and is registered
    only inside `if envs.VLLM_SERVER_DEV_MODE:`. Without it the call 404s and
    every phase inherits a warm cache from the one before it, which for *this*
    project would be fatal: a prefix-affinity arm following another
    prefix-affinity arm would post a flattering TTFT that had nothing to do with
    its own routing.

    The HTTP status is recorded rather than assumed. That recording is precisely
    what caught the 404 in P1.
    """
    out: dict[str, Any] = {}
    for rid, url in replica_urls.items():
        try:
            r = await client.post(f"{url}/reset_prefix_cache", timeout=60.0)
            out[rid] = {"ok": r.status_code < 300, "status": r.status_code}
        except httpx.HTTPError as exc:
            out[rid] = {"ok": False, "error": str(exc)}
    all_ok = all(v.get("ok") for v in out.values())
    if not all_ok:
        log.warning("prefix_cache_reset_incomplete", detail=out)
    return {"all_ok": all_ok, "per_replica": out}
