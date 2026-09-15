"""Summary statistics, including the fairness metric.

Kept separate from both the load generator and the analysis so that the *same*
function computes a value written into a raw artifact and a value recomputed
during plotting. Two implementations of "Jain's index" that disagree by a
denominator would be very hard to notice and would invalidate the headline
fairness comparison.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def percentiles(values: Sequence[float], ps: Sequence[float] = (50, 95, 99)) -> dict[str, float]:
    """Percentiles by linear interpolation, keyed `p50`, `p95`, ...

    Returns an empty dict for an empty input rather than NaN-filled keys: a
    phase in which nothing completed has no latency distribution, and emitting
    NaNs would let them propagate into a plot as though they were data.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {}
    return {f"p{int(p)}": float(np.percentile(arr, p)) for p in ps}


def summarize(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }
    out.update(percentiles(arr))
    return out


def jains_index(counts: Sequence[float]) -> float | None:
    """Jain's fairness index over per-replica load.

        J = (sum x_i)^2 / (n * sum x_i^2)

    1.0 is a perfectly even split; 1/n is everything on one replica. Scale-free,
    so it compares across request rates without normalisation, which is why it
    is preferred here over a raw standard deviation.

    Returns None for an empty fleet or an all-zero split: "no requests were
    routed" has no fairness, and reporting 1.0 (which the formula gives as 0/0
    resolved naively) would say the opposite of the truth.
    """
    arr = np.asarray(list(counts), dtype=float)
    if arr.size == 0:
        return None
    s = arr.sum()
    sq = (arr**2).sum()
    if s <= 0 or sq <= 0:
        return None
    return float(s**2 / (arr.size * sq))


def coefficient_of_variation(counts: Sequence[float]) -> float | None:
    """CV of per-replica load: std/mean. 0 is perfectly even.

    Reported alongside Jain's index because the two emphasise different things:
    Jain's compresses near 1.0 for mild imbalance, while CV keeps resolving it.
    A policy that is 15% imbalanced reads as J=0.98 (easy to call "fair") but
    CV=0.15, which is the more legible number for a small fleet.
    """
    arr = np.asarray(list(counts), dtype=float)
    if arr.size == 0:
        return None
    m = arr.mean()
    if m <= 0:
        return None
    return float(arr.std(ddof=0) / m)


def max_over_mean(counts: Sequence[float]) -> float | None:
    """Peak-to-mean load ratio.

    The most operationally direct of the three: it is the factor by which the
    hottest replica is overloaded relative to an even split, which is what
    actually determines the tail latency when affinity concentrates a hot
    prefix.
    """
    arr = np.asarray(list(counts), dtype=float)
    if arr.size == 0:
        return None
    m = arr.mean()
    if m <= 0:
        return None
    return float(arr.max() / m)


def fairness(counts: Sequence[float]) -> dict[str, Any]:
    return {
        "per_replica_counts": [float(c) for c in counts],
        "jains_index": jains_index(counts),
        "coefficient_of_variation": coefficient_of_variation(counts),
        "max_over_mean": max_over_mean(counts),
        "n_replicas": len(list(counts)),
    }


def median_with_spread(values: Sequence[float]) -> dict[str, Any]:
    """Median plus min/max, the reporting form the workspace conventions require.

    A bare mean of one run is explicitly disallowed; every figure shows spread.
    """
    arr = np.asarray([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def streaming_evidence(records: Sequence[Any]) -> dict[str, Any]:
    """Client-side evidence that responses were streamed, not buffered.

    This is the under-load half of the non-buffering verification; the unit test
    in `tests/streaming.rs` is the deterministic half.

    The discriminator is the ratio of TTFT to end-to-end latency. If the router
    buffered, the client would see nothing until generation finished and the
    ratio would sit at ~1.0 for every request. A genuinely streaming path gives
    a ratio well below 1 whenever more than a couple of tokens are generated.
    `chunks_per_request` is the corroborating quantity: a buffered response
    arrives as very few chunks regardless of how many tokens it contains.
    """
    ratios = []
    chunks = []
    for r in records:
        ttft = getattr(r, "ttft", None)
        e2e = getattr(r, "e2e", None)
        if ttft is None or e2e is None or e2e <= 0:
            continue
        ratios.append(ttft / e2e)
        chunks.append(r.chunks)
    if not ratios:
        return {"n": 0, "status": "no_completed_streaming_requests"}
    return {
        "n": len(ratios),
        "ttft_over_e2e": summarize(ratios),
        "chunks_per_request": summarize(chunks),
    }
