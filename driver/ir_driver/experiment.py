"""Phase orchestration and raw-artifact emission.

One *phase* is one (policy, shared-prefix ratio, request rate, repetition) point.
Phases run against a single long-lived fleet and a single long-lived router; the
policy is switched via the router's `/admin/policy` endpoint between phases.

That structure is the central experimental-design decision of this project and
it is worth stating why. Four vLLM replicas take minutes each to stream 15 GB of
weights off Lustre. Restarting them per policy would cost more node-hours than
the measurement and would make the four policies a comparison of four
separately-warmed fleets. Holding the fleet fixed and swapping only the routing
rule makes the policy the single independent variable.

The hazard that structure creates is cross-phase contamination: a phase would
otherwise inherit the previous phase's warm prefix caches, and for *this*
project that is fatal rather than cosmetic, because a warm cache is exactly what
prefix affinity is claimed to produce. Every phase therefore begins with an
explicit prefix-cache reset on every replica, a drain wait, and a recorded check
that both actually happened. A phase whose reset failed is marked and excluded
in analysis rather than silently reported.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import httpx
import structlog

from . import metrics_scrape as ms
from . import stats
from .loadgen import PhaseResult, run_phase
from .provenance import RunProvenance, backend_provenance, load_lockfile

log = structlog.get_logger(__name__)


@dataclass
class PhaseSpec:
    policy: str
    shared_prefix_ratio: float
    request_rate: float
    repetition_index: int
    warmup: bool = False

    num_requests: int = 240
    input_tokens: int = 1024
    output_tokens: int = 128
    shared_prefix_tokens: int = 512
    num_prefix_groups: int = 1
    seed: int = 1234

    # Router tuning held fixed across the matrix unless a phase overrides it.
    prefix_hash_tokens: int = 512
    bounded_overload_factor: float = 1.25
    bounded_min_capacity: int = 4

    def phase_id(self) -> str:
        w = "warmup" if self.warmup else f"rep{self.repetition_index}"
        return (
            f"{self.policy}__ratio{self.shared_prefix_ratio:g}"
            f"__rate{self.request_rate:g}__{w}"
        )


@dataclass
class FleetSpec:
    """The replica topology, recorded verbatim in every artifact.

    A routing result is meaningless without it: 'prefix affinity improved TTFT
    12%' is a different claim on four isolated GPUs than on eight replicas
    packed two per card.
    """

    router_url: str
    replicas: dict[str, str]  # id -> base url
    placements: dict[str, str] = field(default_factory=dict)
    replicas_per_gpu: int | None = None
    gpus_used: int | None = None
    nodes_used: int | None = None
    topology_note: str = ""


async def _wait_drained(
    client: httpx.AsyncClient, router_url: str, timeout_s: float = 120.0
) -> dict[str, Any]:
    """Wait until the router reports nothing in flight and every replica healthy.

    Recorded, not assumed: a phase that starts while the previous phase is still
    draining measures a fleet under residual load and is not the configuration
    it claims to be.
    """
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            r = await client.get(f"{router_url}/admin/fleet", timeout=15.0)
            r.raise_for_status()
            last = r.json()
        except httpx.HTTPError as exc:
            last = {"error": str(exc)}
            await asyncio.sleep(1.0)
            continue
        if last.get("admitted_inflight", 1) == 0 and last.get("admission_waiting", 1) == 0:
            return {"drained": True, "waited_s": None, "fleet": last}
        await asyncio.sleep(0.5)
    return {"drained": False, "timeout_s": timeout_s, "fleet": last}


async def _set_policy(client: httpx.AsyncClient, router_url: str, spec: PhaseSpec) -> dict[str, Any]:
    body = {
        "policy": spec.policy,
        "prefix_hash_tokens": spec.prefix_hash_tokens,
        "bounded_overload_factor": spec.bounded_overload_factor,
        "bounded_min_capacity": spec.bounded_min_capacity,
    }
    r = await client.post(f"{router_url}/admin/policy", json=body, timeout=30.0)
    r.raise_for_status()
    got = r.json()
    active = got.get("active", {}).get("policy")
    if active != spec.policy:
        # A silently ignored policy switch would mean a whole arm ran under the
        # wrong rule while its artifacts claimed otherwise.
        raise RuntimeError(
            f"router reports active policy {active!r} after requesting {spec.policy!r}"
        )
    return got


def build_workload(spec: PhaseSpec, tokenizer: Any) -> tuple[list[Any], Any]:
    """Build the request set using P1's generator, imported not copied."""
    from llm_inference_bench.workload import (  # noqa: PLC0415
        LengthSpec,
        PromptFormat,
        WorkloadGenerator,
        WorkloadSpec,
    )

    ws = WorkloadSpec(
        num_requests=spec.num_requests,
        input_len=LengthSpec(mean=spec.input_tokens),
        output_len=LengthSpec(mean=spec.output_tokens),
        shared_prefix_ratio=spec.shared_prefix_ratio,
        shared_prefix_tokens=spec.shared_prefix_tokens,
        num_prefix_groups=spec.num_prefix_groups,
        # Per-repetition seed: repetitions must be independent samples, not the
        # same workload replayed, or N=3 measures only run-to-run server noise
        # and reports it as workload variance.
        seed=spec.seed + 1000 * spec.repetition_index,
        prompt_format=PromptFormat.TOKEN_IDS,
    )
    gen = WorkloadGenerator(tokenizer, ws)
    return gen.generate(), ws


async def run_one_phase(
    spec: PhaseSpec,
    fleet: FleetSpec,
    tokenizer: Any,
    model_served_name: str,
    out_dir: Path,
    provenance: RunProvenance,
    backend_prov: dict[str, Any],
    *,
    settle_s: float = 3.0,
) -> Path:
    """Run one phase end to end and write its raw artifact. Returns the path."""
    requests, workload_spec = build_workload(spec, tokenizer)

    from llm_inference_bench.workload import theoretical_prefix_hit_rate  # noqa: PLC0415

    bound = theoretical_prefix_hit_rate(requests, spec.shared_prefix_tokens)

    async with httpx.AsyncClient() as admin:
        policy_ack = await _set_policy(admin, fleet.router_url, spec)
        drain = await _wait_drained(admin, fleet.router_url)
        reset = await ms.reset_prefix_caches(admin, fleet.replicas)
        # Let the engines settle after a cache wipe before the clock starts.
        await asyncio.sleep(settle_s)

        before_replicas = {
            rid: await ms.scrape(admin, url) for rid, url in fleet.replicas.items()
        }
        before_router = await ms.scrape(admin, fleet.router_url)

    result: PhaseResult = await run_phase(
        fleet.router_url,
        requests,
        spec.request_rate,
        model_served_name,
        seed=spec.seed + 7 * spec.repetition_index,
    )

    async with httpx.AsyncClient() as admin:
        after_replicas = {
            rid: await ms.scrape(admin, url) for rid, url in fleet.replicas.items()
        }
        after_router = await ms.scrape(admin, fleet.router_url)
        fleet_after = await _wait_drained(admin, fleet.router_url, timeout_s=300.0)

    artifact = _assemble(
        spec=spec,
        fleet=fleet,
        result=result,
        workload_spec=workload_spec,
        theoretical_bound=bound,
        policy_ack=policy_ack,
        drain_before=drain,
        drain_after=fleet_after,
        cache_reset=reset,
        before_replicas=before_replicas,
        after_replicas=after_replicas,
        before_router=before_router,
        after_router=after_router,
        provenance=provenance,
        backend_prov=backend_prov,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{provenance.run_id}__{spec.phase_id()}.json"
    # Append-only: never overwrite an existing artifact. If a phase id repeats,
    # that is a bug in the matrix, and silently clobbering the earlier result
    # would hide it.
    if path.exists():
        raise FileExistsError(f"artifact already exists, refusing to overwrite: {path}")
    with path.open("w") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True, default=str)
    log.info(
        "artifact_written",
        path=str(path),
        phase=spec.phase_id(),
        status=artifact["status"],
        ok=artifact["summary"]["requests_ok"],
        failed=artifact["summary"]["requests_failed"],
    )
    return path


def _assemble(**kw: Any) -> dict[str, Any]:
    spec: PhaseSpec = kw["spec"]
    fleet: FleetSpec = kw["fleet"]
    result: PhaseResult = kw["result"]

    records = result.records
    ok = [r for r in records if r.ok]
    failed = [r for r in records if not r.ok]

    # Per-replica dispatch counts, attributed client-side from the response
    # header. Cross-checked against the router's own counter deltas below; a
    # disagreement means the header and the routing decision diverged, which
    # would corrupt the fairness metric.
    counts: dict[str, int] = {rid: 0 for rid in fleet.replicas}
    for r in records:
        if r.replica in counts:
            counts[r.replica] += 1

    route_reasons: dict[str, int] = {}
    for r in records:
        if r.route_reason:
            route_reasons[r.route_reason] = route_reasons.get(r.route_reason, 0) + 1

    ttfts = [r.ttft for r in ok if r.ttft is not None]
    e2es = [r.e2e for r in ok if r.e2e is not None]

    per_replica_ttft = {}
    for rid in fleet.replicas:
        vals = [r.ttft for r in ok if r.replica == rid and r.ttft is not None]
        if vals:
            per_replica_ttft[rid] = stats.summarize(vals)

    cache = {
        rid: ms.prefix_cache_delta(
            rid, kw["before_replicas"].get(rid, {}), kw["after_replicas"].get(rid, {})
        ).as_dict()
        for rid in fleet.replicas
    }

    router_d = ms.router_delta(kw["before_router"], kw["after_router"])

    # Status is derived from what happened, never asserted. A phase where the
    # cache reset failed, or where the generator could not keep to schedule, is
    # recorded as degraded so analysis can exclude it rather than average it in.
    late = [r.send_delay for r in records if r.send_delay > 0.25]
    problems = []
    if not kw["cache_reset"].get("all_ok"):
        problems.append("prefix_cache_reset_failed")
    if not kw["drain_before"].get("drained"):
        problems.append("fleet_not_drained_before_phase")
    if len(late) > 0.02 * max(len(records), 1):
        problems.append("load_generator_fell_behind_schedule")
    if not ok:
        problems.append("no_successful_requests")
    status = "ok" if not problems else "degraded"

    completed_output_tokens = sum(
        r.completion_tokens for r in ok if r.completion_tokens is not None
    )

    # Achieved rate over the DISPATCH window, not over total elapsed time.
    #
    # `len(ok) / elapsed_s` divides by a span that runs to the last *completion*,
    # so it always includes the drain tail. On a short phase that tail is a large
    # share of elapsed, and the ratio achieved/offered then falls steadily with
    # offered rate on a fleet that is nowhere near saturated (calibration run
    # 5215612: 0.92 -> 0.66 across 2..32 req/s while TTFT p50 stayed flat at
    # 0.07-0.10 s with zero errors). That artifact was read as a saturation knee.
    #
    # The dispatch window (first send to last send) is the interval over which
    # load was actually offered, so rate over it measures whether the generator
    # kept up, free of the tail. `elapsed_s` and the old ratio are kept, renamed,
    # so older artifacts stay comparable.
    sends = [r.sent_at for r in records if r.sent_at is not None]
    dispatch_window_s = (max(sends) - min(sends)) if len(sends) > 1 else None

    summary: dict[str, Any] = {
        "requests_attempted": len(records),
        "requests_ok": len(ok),
        "requests_failed": len(failed),
        "error_rate": len(failed) / len(records) if records else None,
        "elapsed_s": result.elapsed_s,
        "dispatch_window_s": dispatch_window_s,
        # Requests actually dispatched per second over the offered window. This
        # is the keep-up signal; compare it with offered_request_rate_per_s.
        "achieved_request_rate_per_s": (
            (len(records) - 1) / dispatch_window_s if dispatch_window_s else None
        ),
        # The tail-contaminated form, retained for comparability, never used to
        # locate a knee.
        "completion_rate_incl_tail_per_s": (
            len(ok) / result.elapsed_s if result.elapsed_s > 0 else None
        ),
        "offered_request_rate_per_s": spec.request_rate,
        "output_tokens_total": completed_output_tokens,
        "output_tokens_per_s": (
            completed_output_tokens / result.elapsed_s if result.elapsed_s > 0 else None
        ),
        "ttft_s": stats.summarize(ttfts),
        "e2e_s": stats.summarize(e2es),
        "send_delay_s": stats.summarize([r.send_delay for r in records]),
        "late_dispatches": len(late),
        "fairness": stats.fairness([counts[rid] for rid in sorted(counts)]),
        "fairness_replica_order": sorted(counts),
        "per_replica_dispatch_counts": counts,
        "per_replica_ttft_s": per_replica_ttft,
        "route_reason_counts": route_reasons,
        "prefix_cache_per_replica": cache,
        "theoretical_prefix_hit_rate_bound": kw["theoretical_bound"],
        "streaming_evidence": stats.streaming_evidence(ok),
        "status_code_counts": _count(r.status for r in records),
        "error_kinds": _count(
            (r.error or "")[:80] for r in failed if r.error
        ),
    }

    prov = kw["provenance"].as_dict()
    prov.update(kw["backend_prov"])

    return {
        **prov,
        "artifact_schema_version": 1,
        "status": status,
        "problems": problems,
        "warmup": spec.warmup,
        "repetition_index": spec.repetition_index,
        "phase_id": spec.phase_id(),
        "full_config_dict": {
            "phase": asdict(spec),
            "workload_spec_fingerprint": kw["workload_spec"].fingerprint(),
            "workload_spec": _workload_spec_dict(kw["workload_spec"]),
            "fleet": {
                "router_url": fleet.router_url,
                "replica_ids": sorted(fleet.replicas),
                "replica_urls": fleet.replicas,
                "placements": fleet.placements,
                "replicas_per_gpu": fleet.replicas_per_gpu,
                "gpus_used": fleet.gpus_used,
                "nodes_used": fleet.nodes_used,
                "topology_note": fleet.topology_note,
            },
        },
        "policy_ack": kw["policy_ack"],
        "drain_before_phase": kw["drain_before"],
        "drain_after_phase": kw["drain_after"],
        "cache_reset": kw["cache_reset"],
        "summary": summary,
        "router_metrics_delta": router_d,
        # Full exposition kept verbatim. P1 recovered two separate metric bugs
        # after the fact purely because raw telemetry was stored next to the
        # normalised keys, at a cost of tens of KB per artifact. Cheap insurance.
        "raw_prometheus": {
            "router_after": kw["after_router"].get("raw"),
            "replicas_after": {
                rid: v.get("raw") for rid, v in kw["after_replicas"].items()
            },
        },
        "per_request": [r.as_dict() for r in records],
        "phase_wall_start_unix": result.wall_start_unix,
        "phase_wall_end_unix": result.wall_end_unix,
    }


def _workload_spec_dict(ws: Any) -> dict[str, Any]:
    d = asdict(ws)
    return {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
            for k, v in d.items()}


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        k = str(i)
        out[k] = out.get(k, 0) + 1
    return out


def build_matrix(
    policies: Sequence[str],
    ratios: Sequence[float],
    rates: Sequence[float],
    repetitions: int,
    *,
    warmup: bool = True,
    min_duration_s: float = 0.0,
    **phase_kwargs: Any,
) -> list[PhaseSpec]:
    """Enumerate the experiment matrix.

    Ordering is policy-outermost so a policy's whole ratio x rate block runs
    contiguously. Within a block, ratio varies slower than rate. This matters
    because it puts the per-policy cost of the fleet's thermal and scheduler
    state in the same place for every policy rather than interleaving it.

    A warmup repetition is prepended to every configuration and marked
    `warmup=true`: the first run of any config pays CUDA graph capture,
    allocator growth and cache population. It is recorded and excluded in
    analysis, never silently dropped.

    `min_duration_s` scales the request count with rate so every phase offers
    load for at least that long. A fixed count shrinks the arrival window as
    rate rises (200 requests at 64 req/s is ~3 s), which is too short for an
    overloaded fleet to build a backlog, so overload would read as underload.
    """
    out: list[PhaseSpec] = []
    base_n = phase_kwargs.pop("num_requests", PhaseSpec.num_requests)
    for policy in policies:
        for ratio in ratios:
            for rate in rates:
                phase_kwargs["num_requests"] = max(base_n, math.ceil(rate * min_duration_s))
                if warmup:
                    out.append(
                        PhaseSpec(
                            policy=policy,
                            shared_prefix_ratio=ratio,
                            request_rate=rate,
                            repetition_index=0,
                            warmup=True,
                            **phase_kwargs,
                        )
                    )
                for rep in range(1, repetitions + 1):
                    out.append(
                        PhaseSpec(
                            policy=policy,
                            shared_prefix_ratio=ratio,
                            request_rate=rate,
                            repetition_index=rep,
                            warmup=False,
                            **phase_kwargs,
                        )
                    )
    return out


def load_backend_provenance(lockfile: Path | None = None) -> dict[str, Any]:
    lock = load_lockfile(lockfile) if lockfile else load_lockfile()
    return backend_provenance(lock, "vllm", "Qwen_Qwen2_5-7B-Instruct")
