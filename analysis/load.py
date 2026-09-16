"""Load raw artifacts into tidy records.

Reads `results/raw/` only, never writes there. Every figure and table is
regenerated from this module rather than transcribed, so a number in the README
can always be traced back to the artifact it came from.

Selection and exclusions are applied here, once, so every downstream consumer
gets them:

* **One matrix run.** Only the most recent matrix run (by `run_id`) is used,
  or the one named by `IR_RUN_ID`. Runs from different jobs used different
  nodes and rate ladders; pooling their repetitions would mix fleets. Earlier
  runs are listed in the inventory as superseded, not dropped silently.
* `warmup=true` phases. The first run of any configuration pays CUDA graph
  capture, allocator growth and cache population. Recorded, excluded, never
  silently dropped.
* Phases with a **server-side** problem (prefix-cache reset failed, fleet not
  drained, no successful request) are not the configuration they claim to be
  and are excluded.
* Phases whose only problem is `load_generator_fell_behind_schedule` are
  **kept and flagged**. In job 5216059 the client missed its 0.25 s dispatch
  tolerance on more than 2% of requests even at 4 req/s, so excluding them
  would leave most configurations below N=3. The lag is a client artefact and
  is handled in the metric instead: TTFT is reported both from scheduled
  arrival (open-loop, includes client lag) and from actual dispatch (client
  lag removed), and the dispatch-lag p95 is reported next to them.
"""

from __future__ import annotations

import json
import os
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

    def ttft_from_dispatch(self, p: str) -> float | None:
        """TTFT measured from when the request was actually sent.

        Computed from the per-request records stored in the artifact, not
        estimated. Removes client dispatch lag; a stall on the receive side
        would still be included, so this is a lower bound on the client's
        contribution, not a pure server number.
        """
        vals = sorted(
            r["first_token_at"] - r["sent_at"]
            for r in self.raw.get("per_request", [])
            if r.get("ok") and r.get("first_token_at") is not None and r.get("sent_at") is not None
        )
        return _pct(vals, p)

    def send_delay(self, p: str) -> float | None:
        return self.summary.get("send_delay_s", {}).get(p)

    @property
    def client_lagged(self) -> bool:
        return "load_generator_fell_behind_schedule" in (self.raw.get("problems") or [])

    @property
    def run_id(self) -> str:
        return str(self.raw.get("run_id"))

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

    @property
    def num_prefix_groups(self) -> int:
        return int(self.raw["full_config_dict"]["phase"].get("num_prefix_groups", 1))

    def key(self) -> tuple[str, float, float]:
        return (self.policy, self.ratio, self.rate)


def _pct(sorted_vals: list[float], p: str) -> float | None:
    """Nearest-rank percentile on pre-sorted values; `p` like "p50"."""
    if not sorted_vals:
        return None
    q = float(p.lstrip("p")) / 100.0
    k = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


CLIENT_ONLY_PROBLEMS = {"load_generator_fell_behind_schedule"}


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
        if obj.get("kind") == "failure_mode":
            continue  # separate experiment, loaded by load_failure_trials
        yield Phase(path=p, raw=obj)


def load_failure_trials(raw_dir: Path = RAW) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Failure-mode artifacts from the most recent failure job (or IR_FAILURE_JOB).

    Every trial is its own driver process and so its own run_id; the Slurm job
    is what groups a set of trials run against the same code and node.
    """
    trials = []
    for p in sorted(raw_dir.glob("*.json")):
        try:
            with p.open() as fh:
                obj = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: cannot read {p.name}: {exc}")
            continue
        if obj.get("kind") == "failure_mode":
            obj["_path"] = p.name
            trials.append(obj)
    jobs = sorted({str(t.get("slurm_job_id")) for t in trials})
    chosen = os.environ.get("IR_FAILURE_JOB") or (
        max(jobs, key=lambda j: max(str(t.get("utc_timestamp")) for t in trials
                                    if str(t.get("slurm_job_id")) == j))
        if jobs else None
    )
    sel = [t for t in trials if str(t.get("slurm_job_id")) == chosen]
    return sel, {"slurm_job_id": chosen, "trials": len(sel),
                 "superseded_jobs": [j for j in jobs if j != chosen]}


def prefix_group_counts(raw_dir: Path = RAW) -> list[int]:
    """Distinct workload shapes present. Each is analysed as its own experiment."""
    return sorted({p.num_prefix_groups for p in iter_artifacts(raw_dir)})


def load_measurements(
    raw_dir: Path = RAW, num_prefix_groups: int = 1
) -> tuple[list[Phase], dict[str, Any]]:
    """Return (usable phases, inventory).

    The inventory is printed alongside every figure run so the exclusions are
    visible rather than implicit.
    """
    allruns = [p for p in iter_artifacts(raw_dir) if p.num_prefix_groups == num_prefix_groups]
    first_seen: dict[str, str] = {}
    for p in allruns:
        ts = str(p.raw.get("utc_timestamp", ""))
        first_seen[p.run_id] = min(first_seen.get(p.run_id, ts), ts)
    pinned = os.environ.get("IR_RUN_ID")
    chosen = pinned if pinned in first_seen else (
        max(first_seen, key=lambda k: first_seen[k]) if first_seen else None
    )
    every = [p for p in allruns if p.run_id == chosen]

    def server_ok(p: Phase) -> bool:
        return set(p.raw.get("problems") or []) <= CLIENT_ONLY_PROBLEMS

    usable = [p for p in every if not p.warmup and server_ok(p)]
    inventory = {
        "num_prefix_groups": num_prefix_groups,
        "run_id": chosen,
        "slurm_job_id": every[0].raw.get("slurm_job_id") if every else None,
        "superseded_runs": {
            rid: {
                "artifacts": sum(1 for p in allruns if p.run_id == rid),
                "slurm_job_id": next(
                    (p.raw.get("slurm_job_id") for p in allruns if p.run_id == rid), None
                ),
            }
            for rid in sorted(first_seen)
            if rid != chosen
        },
        "artifacts_total": len(every),
        "warmup_excluded": sum(1 for p in every if p.warmup),
        "server_problem_excluded": sum(1 for p in every if not p.warmup and not server_ok(p)),
        "client_lag_flagged_kept": sum(1 for p in usable if p.client_lagged),
        "usable": len(usable),
        "policies": sorted({p.policy for p in usable}),
        "ratios": sorted({p.ratio for p in usable}),
        "rates": sorted({p.rate for p in usable}),
        "repetitions_per_config": {},
        "degraded_detail": [
            {"phase": p.raw.get("phase_id"), "problems": p.raw.get("problems")}
            for p in every
            if not p.warmup and not server_ok(p)
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
