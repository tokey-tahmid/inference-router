"""Load raw artifacts into tidy records.

Reads `results/raw/` only, never writes there. Every figure and table is
regenerated from this module rather than transcribed, so a number in the README
can always be traced back to the artifact it came from.

Two exclusions are applied here, once, so every downstream consumer gets them:

* `warmup=true` phases. The first run of any configuration pays CUDA graph
  capture, allocator growth and cache population. Recorded, excluded, never
  silently dropped.
* `status != "ok"` phases. A phase whose prefix-cache reset failed, or whose
  load generator fell behind its own schedule, is not the configuration it
  claims to be. It stays in `results/raw/` and is reported in the inventory as
  a failed or degraded configuration, because a failed config is data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results" / "raw"


@dataclass
class Phase:
    """One measured (policy, ratio, rate, repetition) point."""

    path: Path
    raw: dict[str, Any]

    @property
    def policy(self) -> str:
        return self.raw["full_config_dict"]["phase"]["policy"]

    @property
    def ratio(self) -> float:
        return float(self.raw["full_config_dict"]["phase"]["shared_prefix_ratio"])

    @property
    def rate(self) -> float:
        return float(self.raw["full_config_dict"]["phase"]["request_rate"])

    @property
    def repetition(self) -> int:
        return int(self.raw.get("repetition_index", 0))

    @property
    def warmup(self) -> bool:
        return bool(self.raw.get("warmup", False))

    @property
    def status(self) -> str:
        return self.raw.get("status", "unknown")

    @property
    def summary(self) -> dict[str, Any]:
        return self.raw.get("summary", {})

    def ttft(self, p: str) -> float | None:
        return self.summary.get("ttft_s", {}).get(p)

    def e2e(self, p: str) -> float | None:
        return self.summary.get("e2e_s", {}).get(p)

    @property
    def throughput_req_s(self) -> float | None:
        return self.summary.get("achieved_request_rate_per_s")

    @property
    def output_tok_s(self) -> float | None:
        return self.summary.get("output_tokens_per_s")

    @property
    def error_rate(self) -> float | None:
        return self.summary.get("error_rate")

    @property
    def jains_index(self) -> float | None:
        return self.summary.get("fairness", {}).get("jains_index")

    @property
    def cv(self) -> float | None:
        return self.summary.get("fairness", {}).get("coefficient_of_variation")

    @property
    def max_over_mean(self) -> float | None:
        return self.summary.get("fairness", {}).get("max_over_mean")

    @property
    def fleet_prefix_hit_rate(self) -> float | None:
        """Fleet-wide prefix cache hit rate: total hits / total queries.

        Summed across replicas rather than averaged over per-replica rates.
        Averaging rates would weight a replica that served three requests the
        same as one that served three hundred, and under prefix affinity the
        load is deliberately uneven, so that weighting would systematically
        flatter exactly the policy under test.
        """
        per = self.summary.get("prefix_cache_per_replica", {})
        q = sum(v["queries"] for v in per.values() if v.get("queries") is not None)
        h = sum(v["hits"] for v in per.values() if v.get("hits") is not None)
        if not q:
            return None
        return h / q

    @property
    def per_replica_hit_rates(self) -> dict[str, float]:
        per = self.summary.get("prefix_cache_per_replica", {})
        return {k: v["hit_rate"] for k, v in per.items() if v.get("hit_rate") is not None}

    @property
    def theoretical_bound(self) -> float | None:
        return self.summary.get("theoretical_prefix_hit_rate_bound")

    @property
    def spill_fraction(self) -> float | None:
        """Fraction of routing decisions that spilled off the affinity primary.

        The mechanism number. A latency difference between the bounded and
        unbounded affinity policies is only interpretable next to how often the
        bound actually fired.
        """
        rr = self.summary.get("route_reason_counts", {})
        total = sum(rr.values())
        if not total:
            return None
        spilled = rr.get("affinity_spill", 0) + rr.get("affinity_spill_exhausted", 0)
        return spilled / total

    @property
    def streaming_ttft_over_e2e(self) -> float | None:
        ev = self.summary.get("streaming_evidence", {})
        return ev.get("ttft_over_e2e", {}).get("p50")

    @property
    def chunks_per_request(self) -> float | None:
        ev = self.summary.get("streaming_evidence", {})
        return ev.get("chunks_per_request", {}).get("p50")

    def key(self) -> tuple[str, float, float]:
        return (self.policy, self.ratio, self.rate)


def iter_artifacts(raw_dir: Path = RAW) -> Iterator[Phase]:
    for p in sorted(raw_dir.glob("*.json")):
        try:
            with p.open() as fh:
                obj = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            # A corrupt artifact is reported, never skipped silently.
            print(f"WARNING: cannot read {p.name}: {exc}")
            continue
        if "full_config_dict" not in obj or "phase" not in obj.get("full_config_dict", {}):
            continue  # verification / density artifacts have a different shape
        yield Phase(path=p, raw=obj)


def load_measurements(raw_dir: Path = RAW) -> tuple[list[Phase], dict[str, Any]]:
    """Return (usable phases, inventory).

    The inventory is printed alongside every figure run so the exclusions are
    visible rather than implicit.
    """
    every = list(iter_artifacts(raw_dir))
    usable = [p for p in every if not p.warmup and p.status == "ok"]
    inventory = {
        "artifacts_total": len(every),
        "warmup_excluded": sum(1 for p in every if p.warmup),
        "degraded_excluded": sum(1 for p in every if not p.warmup and p.status == "degraded"),
        "failed_excluded": sum(1 for p in every if not p.warmup and p.status == "failed"),
        "usable": len(usable),
        "policies": sorted({p.policy for p in usable}),
        "ratios": sorted({p.ratio for p in usable}),
        "rates": sorted({p.rate for p in usable}),
        "repetitions_per_config": {},
        "degraded_detail": [
            {"phase": p.raw.get("phase_id"), "problems": p.raw.get("problems")}
            for p in every
            if not p.warmup and p.status != "ok"
        ],
    }
    counts: dict[str, int] = {}
    for p in usable:
        k = f"{p.policy}|{p.ratio}|{p.rate}"
        counts[k] = counts.get(k, 0) + 1
    inventory["repetitions_per_config"] = counts
    if counts:
        inventory["min_repetitions"] = min(counts.values())
        inventory["max_repetitions"] = max(counts.values())
    return usable, inventory


def group(phases: list[Phase]) -> dict[tuple[str, float, float], list[Phase]]:
    out: dict[tuple[str, float, float], list[Phase]] = {}
    for p in phases:
        out.setdefault(p.key(), []).append(p)
    return out


def hardware_caption(phases: list[Phase]) -> str:
    """Build the figure caption from what the artifacts actually recorded.

    Never hand-written: a caption that says "4x A100" while the artifacts say
    something else is exactly the kind of quiet mismatch that makes a portfolio
    figure indefensible.
    """
    if not phases:
        return "no measurements"
    r = phases[0].raw
    fleet = r["full_config_dict"].get("fleet", {})
    parts = []
    gpu = r.get("gpu_model")
    if gpu:
        parts.append(f"{r.get('gpu_count', '?')}x {gpu}")
    if r.get("driver_version"):
        parts.append(f"driver {r['driver_version']}")
    nrep = len(fleet.get("replica_ids", []))
    if nrep:
        rpg = fleet.get("replicas_per_gpu")
        parts.append(f"{nrep} replicas" + (f" ({rpg}/GPU)" if rpg else ""))
    if r.get("backend_name"):
        parts.append(f"{r['backend_name']} {r.get('backend_version', '?')}")
    if r.get("model_id"):
        parts.append(str(r["model_id"]))
    return ", ".join(parts)
