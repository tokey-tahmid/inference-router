"""Driver entry point.

    python -m ir_driver.cli matrix   --router http://127.0.0.1:8080 --replicas r0=... ...
    python -m ir_driver.cli verify   --router ... --replicas ...
    python -m ir_driver.cli failure  --router ... --replicas ... --kill r2

`verify` is the gate that must pass before any matrix runs: it drives one live
replica through the router and checks the things that would invalidate every
subsequent number if they were false — that token-ID prompts round-trip exactly,
that `ignore_eos` is honoured, and that responses actually stream rather than
arriving in one piece.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import structlog

from . import metrics_scrape as ms
from . import stats
from .experiment import (
    FleetSpec,
    PhaseSpec,
    build_matrix,
    build_workload,
    load_backend_provenance,
    run_one_phase,
)
from .loadgen import run_phase
from .provenance import RunProvenance

REPO = Path(__file__).resolve().parents[2]

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger("ir_driver")


def parse_replicas(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--replicas expects id=url, got {it!r}")
        rid, url = it.split("=", 1)
        out[rid] = url.rstrip("/")
    return out


def load_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer  # noqa: PLC0415

    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


# ---------------------------------------------------------------- verify -----


async def cmd_verify(args: argparse.Namespace) -> int:
    """Single-replica end-to-end verification through the router.

    Everything checked here is a precondition of the whole experiment, and each
    one is checked by *recording an outcome* rather than by assuming it. P1's
    notes are emphatic on this point: every defect on the road to its first real
    measurement was caught by a check that recorded a result, and none by
    reading code.
    """
    replicas = parse_replicas(args.replicas)
    tok = load_tokenizer(args.model_path)
    prov = RunProvenance.capture(REPO, sys.argv)
    backend = load_backend_provenance(Path(args.lockfile) if args.lockfile else None)

    checks: dict[str, Any] = {}

    async with httpx.AsyncClient(timeout=120.0) as c:
        fleet = (await c.get(f"{args.router}/admin/fleet")).json()
        checks["fleet"] = fleet

        # 1. Token-ID prompts must round-trip exactly, or the shared-prefix knob
        #    is measuring nothing. Checked through the ROUTER, not directly
        #    against the replica: the router re-serialises the body, and a
        #    JSON round trip that turned ints into floats would break exactness
        #    in a way invisible from the replica side.
        probe = list(range(1000, 1064))  # 64 token ids
        r = await c.post(
            f"{args.router}/v1/completions",
            json={
                "model": args.served_name,
                "prompt": probe,
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
            },
        )
        body = r.json()
        reported = body.get("usage", {}).get("prompt_tokens")
        checks["token_id_prompt_fidelity"] = {
            "sent_prompt_tokens": len(probe),
            "reported_prompt_tokens": reported,
            "ok": reported == len(probe),
            "status": r.status_code,
            "routed_to": r.headers.get("x-router-replica"),
        }

        # 2. ignore_eos must be honoured or the output-length axis is not
        #    controlled and cross-phase comparisons are illegitimate.
        want = 64
        r = await c.post(
            f"{args.router}/v1/completions",
            json={
                "model": args.served_name,
                "prompt": probe,
                "max_tokens": want,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": False,
            },
        )
        got = r.json().get("usage", {}).get("completion_tokens")
        checks["output_length_exact"] = {
            "requested": want,
            "reported": got,
            "ok": got == want,
        }

        # 3. Streaming must not be buffered. Measured the only way that proves
        #    it: record when each chunk arrives at the client and check the first
        #    one lands well before the last.
        arrivals: list[float] = []
        t0 = time.perf_counter()
        headers: dict[str, str] = {}
        async with c.stream(
            "POST",
            f"{args.router}/v1/completions",
            json={
                "model": args.served_name,
                "prompt": probe,
                "max_tokens": 128,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as resp:
            headers = dict(resp.headers)
            async for chunk in resp.aiter_bytes():
                if chunk:
                    arrivals.append(time.perf_counter() - t0)
        if arrivals:
            ttft, e2e = arrivals[0], arrivals[-1]
            checks["streaming_not_buffered"] = {
                "chunks": len(arrivals),
                "ttft_s": ttft,
                "e2e_s": e2e,
                "ttft_over_e2e": ttft / e2e if e2e > 0 else None,
                # A buffering proxy gives ratio ~1.0 and chunks ~1. Requiring
                # both keeps a single slow chunk from passing the check.
                "ok": len(arrivals) > 10 and (ttft / e2e) < 0.5 if e2e > 0 else False,
                "router_replica": headers.get("x-router-replica"),
                "router_policy": headers.get("x-router-policy"),
            }
        else:
            checks["streaming_not_buffered"] = {"ok": False, "error": "no chunks received"}

        # 4. The prefix cache endpoint and counters must actually exist, or the
        #    per-replica cache hit rate is unmeasurable.
        checks["cache_reset"] = await ms.reset_prefix_caches(c, replicas)
        scr = {rid: await ms.scrape(c, url) for rid, url in replicas.items()}
        checks["prefix_cache_counters_present"] = {
            rid: any(
                k.startswith(("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"))
                for k in v.get("samples", {})
            )
            for rid, v in scr.items()
        }

    # 5. A small real load through the router, to confirm the whole path works
    #    before anything expensive is committed.
    spec = PhaseSpec(
        policy="round_robin",
        shared_prefix_ratio=0.6,
        request_rate=args.rate,
        repetition_index=1,
        num_requests=args.num_requests,
    )
    reqs, _ws = build_workload(spec, tok)
    result = await run_phase(args.router, reqs, args.rate, args.served_name, seed=1)
    ok = [r for r in result.records if r.ok]
    checks["smoke_load"] = {
        "attempted": len(result.records),
        "ok": len(ok),
        "failed": len(result.records) - len(ok),
        "elapsed_s": result.elapsed_s,
        "ttft_s": stats.summarize([r.ttft for r in ok if r.ttft is not None]),
        "e2e_s": stats.summarize([r.e2e for r in ok if r.e2e is not None]),
        "streaming_evidence": stats.streaming_evidence(ok),
        "per_replica": {
            rid: sum(1 for r in result.records if r.replica == rid) for rid in replicas
        },
    }

    gate_keys = [
        "token_id_prompt_fidelity",
        "output_length_exact",
        "streaming_not_buffered",
    ]
    passed = all(checks[k].get("ok") for k in gate_keys)
    artifact = {
        **prov.as_dict(),
        **backend,
        "artifact_schema_version": 1,
        "kind": "single_replica_verification",
        "status": "ok" if passed else "failed",
        "gate_keys": gate_keys,
        "checks": checks,
        "full_config_dict": {"args": vars(args)},
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"verify__{prov.run_id}.json"
    with path.open("w") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True, default=str)
    log.info("verification_written", path=str(path), passed=passed)
    print(json.dumps({k: checks[k] for k in gate_keys}, indent=2, default=str))
    return 0 if passed else 1


# ---------------------------------------------------------------- matrix -----


async def cmd_matrix(args: argparse.Namespace) -> int:
    replicas = parse_replicas(args.replicas)
    placements = parse_replicas(args.placements) if args.placements else {}
    tok = load_tokenizer(args.model_path)
    prov = RunProvenance.capture(REPO, sys.argv)
    backend = load_backend_provenance(Path(args.lockfile) if args.lockfile else None)

    fleet = FleetSpec(
        router_url=args.router,
        replicas=replicas,
        placements=placements,
        replicas_per_gpu=args.replicas_per_gpu,
        gpus_used=args.gpus_used,
        nodes_used=args.nodes_used,
        topology_note=args.topology_note,
    )

    phases = build_matrix(
        policies=args.policies,
        ratios=args.ratios,
        rates=args.rates,
        repetitions=args.repetitions,
        warmup=not args.no_warmup,
        num_requests=args.num_requests,
        min_duration_s=args.min_duration_s,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        shared_prefix_tokens=args.shared_prefix_tokens,
        num_prefix_groups=args.num_prefix_groups,
        prefix_hash_tokens=args.shared_prefix_tokens,
    )
    log.info("matrix_planned", phases=len(phases), run_id=prov.run_id)

    out_dir = Path(args.out_dir)
    deadline = time.monotonic() + args.budget_s if args.budget_s else None

    done = 0
    for i, spec in enumerate(phases):
        if deadline and time.monotonic() > deadline:
            # Stop cleanly rather than being killed by Slurm mid-write. A
            # truncated artifact is worse than a missing one.
            log.warning(
                "budget_exhausted_stopping_early",
                completed=done,
                planned=len(phases),
                remaining=len(phases) - i,
            )
            break
        try:
            await run_one_phase(
                spec,
                fleet,
                tok,
                args.served_name,
                out_dir,
                prov,
                backend,
            )
            done += 1
        except Exception as exc:  # noqa: BLE001
            # A failed configuration is data. Record it with its error rather
            # than dropping it from the sweep.
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{prov.run_id}__{spec.phase_id()}__FAILED.json"
            with path.open("w") as fh:
                json.dump(
                    {
                        **prov.as_dict(),
                        **backend,
                        "artifact_schema_version": 1,
                        "status": "failed",
                        "phase_id": spec.phase_id(),
                        "warmup": spec.warmup,
                        "repetition_index": spec.repetition_index,
                        "full_config_dict": {"phase": spec.__dict__},
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    fh,
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            log.error("phase_failed", phase=spec.phase_id(), error=str(exc), artifact=str(path))
        log.info("matrix_progress", completed=i + 1, planned=len(phases))

    log.info("matrix_complete", completed=done, planned=len(phases), run_id=prov.run_id)
    return 0


# --------------------------------------------------------------- failure -----


async def cmd_failure(args: argparse.Namespace) -> int:
    """Kill a replica mid-load and measure detection, error rate and recovery.

    Detection time is measured against the *externally recorded* kill timestamp,
    not against the router's belief about when the replica died. The router
    reports `router_replica_last_transition_unix_us` in wall-clock microseconds
    precisely so those two clocks are comparable.
    """
    replicas = parse_replicas(args.replicas)
    tok = load_tokenizer(args.model_path)
    prov = RunProvenance.capture(REPO, sys.argv)
    backend = load_backend_provenance(Path(args.lockfile) if args.lockfile else None)

    if args.kill not in replicas:
        raise SystemExit(f"--kill {args.kill!r} is not one of {sorted(replicas)}")

    spec = PhaseSpec(
        policy=args.policy,
        shared_prefix_ratio=args.ratio,
        request_rate=args.rate,
        repetition_index=args.repetition_index,
        num_requests=args.num_requests,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
    )
    reqs, ws = build_workload(spec, tok)

    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(
            f"{args.router}/admin/policy",
            json={"policy": args.policy, "prefix_hash_tokens": args.shared_prefix_tokens},
        )
        r.raise_for_status()
        await ms.reset_prefix_caches(c, replicas)
        before_router = await ms.scrape(c, args.router)

    kill_event: dict[str, Any] = {}
    restart_event: dict[str, Any] = {}
    timeline: list[dict[str, Any]] = []
    load_done = asyncio.Event()

    async def run_cmd(cmd: str, ev: dict[str, Any], prefix: str) -> None:
        ev[f"{prefix}_requested_unix"] = time.time()
        proc = await asyncio.create_subprocess_exec(
            *cmd.split(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        ev[f"{prefix}_completed_unix"] = time.time()
        ev[f"{prefix}_returncode"] = proc.returncode
        ev[f"{prefix}_output"] = (out or b"").decode("utf-8", "replace")[:4000]

    async def poller() -> None:
        """Record the killed replica's health state throughout the trial.

        Reading `last_transition_unix_us` once at the end only yields the
        *latest* transition; with a restart that is the recovery, and the
        ejection would be lost. Polling keeps both, and the transition times
        themselves still come from the router's own microsecond clock.
        """
        async with httpx.AsyncClient(timeout=5.0) as pc:
            while not load_done.is_set():
                t = time.time()
                try:
                    fl = (await pc.get(f"{args.router}/admin/fleet")).json()
                    for rep in fl.get("replicas", []):
                        if rep["id"] == args.kill:
                            timeline.append({
                                "poll_unix": t,
                                "healthy": rep.get("healthy"),
                                "ejections_total": rep.get("ejections_total"),
                                "recoveries_total": rep.get("recoveries_total"),
                                "last_transition_unix_us": rep.get("last_transition_unix_us"),
                                "fleet_healthy": fl.get("healthy_replicas"),
                                "capacity": fl.get("capacity"),
                            })
                except (httpx.HTTPError, ValueError) as exc:
                    timeline.append({"poll_unix": t, "error": str(exc)[:200]})
                try:
                    await asyncio.wait_for(load_done.wait(), args.poll_interval_ms / 1000.0)
                except asyncio.TimeoutError:
                    pass

    async def restarter() -> None:
        if not args.restart_command:
            return
        while "kill_completed_unix" not in kill_event:
            await asyncio.sleep(0.2)
        await asyncio.sleep(args.restart_after_s)
        await run_cmd(args.restart_command, restart_event, "restart")
        log.warning("replica_restart_launched", replica=args.kill, detail=restart_event)

    async def killer() -> None:
        await asyncio.sleep(args.kill_after_s)
        kill_event["kill_requested_unix"] = time.time()
        # SIGKILL, not SIGTERM: the point is an abrupt failure, and a graceful
        # shutdown would let vLLM drain in-flight requests, which is a different
        # (and much kinder) experiment than the one intended.
        proc = await asyncio.create_subprocess_exec(
            *args.kill_command.split(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        kill_event["kill_completed_unix"] = time.time()
        kill_event["kill_returncode"] = proc.returncode
        kill_event["kill_output"] = (out or b"").decode("utf-8", "replace")[:2000]
        log.warning("replica_killed", replica=args.kill, detail=kill_event)

    async def load() -> Any:
        try:
            return await run_phase(args.router, reqs, args.rate, args.served_name, seed=99)
        finally:
            load_done.set()

    result, _, _, _ = await asyncio.gather(load(), killer(), poller(), restarter())

    async with httpx.AsyncClient(timeout=60.0) as c:
        after_router = await ms.scrape(c, args.router)
        fleet_after = (await c.get(f"{args.router}/admin/fleet")).json()

    # Transitions from the polled timeline: the first poll where each counter
    # moved carries the router's own timestamp of that transition.
    def first_transition_us(counter: str) -> int | None:
        base = next((p.get(counter) for p in timeline if p.get(counter) is not None), None)
        for p in timeline:
            v = p.get(counter)
            if v is not None and base is not None and v > base:
                return p.get("last_transition_unix_us")
        return None

    transition_us = first_transition_us("ejections_total")
    recovery_us = first_transition_us("recoveries_total")
    detection_s = None
    if transition_us and kill_event.get("kill_requested_unix"):
        detection_s = (transition_us / 1e6) - kill_event["kill_requested_unix"]
    recovery_s = None
    if recovery_us and restart_event.get("restart_requested_unix"):
        recovery_s = (recovery_us / 1e6) - restart_event["restart_requested_unix"]
    ejected_s = None
    if recovery_us and transition_us:
        ejected_s = (recovery_us - transition_us) / 1e6

    records = result.records
    ok = [r for r in records if r.ok]
    # Requests are bucketed relative to the kill so the error burst can be
    # located in time rather than smeared across the whole phase.
    kill_offset = (
        kill_event["kill_requested_unix"] - result.wall_start_unix
        if kill_event.get("kill_requested_unix")
        else None
    )
    recovery_offset = (
        recovery_us / 1e6 - result.wall_start_unix if recovery_us else None
    )
    buckets: dict[str, dict[str, int]] = {}
    if kill_offset is not None:
        for r in records:
            if r.scheduled_at < kill_offset:
                b = "before_kill"
            elif detection_s is None or r.scheduled_at < kill_offset + max(detection_s, 0):
                b = "during_detection_window"
            elif recovery_offset is None or r.scheduled_at < recovery_offset:
                b = "after_ejection"
            else:
                b = "after_recovery"
            d = buckets.setdefault(b, {"total": 0, "ok": 0, "failed": 0})
            d["total"] += 1
            d["ok" if r.ok else "failed"] += 1

    artifact = {
        **prov.as_dict(),
        **backend,
        "artifact_schema_version": 1,
        "kind": "failure_mode",
        "status": "ok" if ok else "degraded",
        "repetition_index": args.repetition_index,
        "warmup": False,
        "phase_id": f"failure__{args.policy}__ratio{args.ratio:g}__rate{args.rate:g}__rep{args.repetition_index}",
        "full_config_dict": {
            "phase": spec.__dict__,
            "killed_replica": args.kill,
            "kill_after_s": args.kill_after_s,
            "kill_command": args.kill_command,
            "workload_spec_fingerprint": ws.fingerprint(),
            "fleet": {"replicas": replicas, "router_url": args.router},
        },
        "kill_event": kill_event,
        "restart_event": restart_event,
        "health_timeline": timeline,
        "recovery": {
            "restart_after_kill_s": args.restart_after_s if args.restart_command else None,
            "router_recovery_unix_us": recovery_us,
            "restart_to_readmission_s": recovery_s,
            "ejected_duration_s": ejected_s,
            "recovery_offset_from_phase_start_s": recovery_offset,
        },
        "detection": {
            "router_transition_unix_us": transition_us,
            "kill_requested_unix": kill_event.get("kill_requested_unix"),
            "detection_latency_s": detection_s,
            "health_interval_ms": args.health_interval_ms,
            "unhealthy_threshold": args.unhealthy_threshold,
            "expected_lower_bound_s": (
                args.health_interval_ms * args.unhealthy_threshold / 1000.0
            ),
        },
        "summary": {
            "requests_attempted": len(records),
            "requests_ok": len(ok),
            "requests_failed": len(records) - len(ok),
            "error_rate": (len(records) - len(ok)) / len(records) if records else None,
            "buckets_relative_to_kill": buckets,
            "kill_offset_from_phase_start_s": kill_offset,
            "ttft_s": stats.summarize([r.ttft for r in ok if r.ttft is not None]),
            "e2e_s": stats.summarize([r.e2e for r in ok if r.e2e is not None]),
            "per_replica_dispatch_counts": {
                rid: sum(1 for r in records if r.replica == rid) for rid in replicas
            },
            "send_delay_s": stats.summarize([r.send_delay for r in records]),
            "status_code_counts": {
                str(s): sum(1 for r in records if r.status == s)
                for s in {r.status for r in records}
            },
        },
        "fleet_after": fleet_after,
        "router_metrics_delta": ms.router_delta(before_router, after_router),
        "raw_prometheus": {"router_after": after_router.get("raw")},
        "per_request": [r.as_dict() for r in records],
        "phase_wall_start_unix": result.wall_start_unix,
        "phase_wall_end_unix": result.wall_end_unix,
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{prov.run_id}__{artifact['phase_id']}.json"
    with path.open("w") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True, default=str)
    log.info("failure_artifact_written", path=str(path), detection_s=detection_s)
    return 0


# ------------------------------------------------------------------ main -----


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ir_driver")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--router", required=True)
        sp.add_argument("--replicas", nargs="+", required=True, help="id=url ...")
        sp.add_argument("--model-path", required=True)
        sp.add_argument("--served-name", default="qwen2.5-7b-instruct")
        sp.add_argument("--out-dir", required=True)
        sp.add_argument("--lockfile", default=None)

    v = sub.add_parser("verify", help="single-replica end-to-end gate")
    common(v)
    v.add_argument("--rate", type=float, default=2.0)
    v.add_argument("--num-requests", type=int, default=32)

    m = sub.add_parser("matrix", help="run the full experiment matrix")
    common(m)
    m.add_argument("--policies", nargs="+", default=[
        "round_robin", "least_loaded", "prefix_affinity", "prefix_affinity_bounded"
    ])
    m.add_argument("--ratios", nargs="+", type=float, default=[0.0, 0.3, 0.6, 0.9])
    m.add_argument("--rates", nargs="+", type=float, required=True)
    m.add_argument("--repetitions", type=int, default=3)
    m.add_argument("--no-warmup", action="store_true")
    m.add_argument("--num-requests", type=int, default=240)
    m.add_argument(
        "--min-duration-s",
        type=float,
        default=0.0,
        help="raise the per-phase request count so arrivals span at least this long",
    )
    m.add_argument("--input-tokens", type=int, default=1024)
    m.add_argument("--output-tokens", type=int, default=128)
    m.add_argument("--shared-prefix-tokens", type=int, default=512)
    m.add_argument("--num-prefix-groups", type=int, default=1)
    m.add_argument("--placements", nargs="*", default=None, help="id=placement ...")
    m.add_argument("--replicas-per-gpu", type=int, default=None)
    m.add_argument("--gpus-used", type=int, default=None)
    m.add_argument("--nodes-used", type=int, default=None)
    m.add_argument("--topology-note", default="")
    m.add_argument(
        "--budget-s",
        type=float,
        default=None,
        help="stop cleanly after this many seconds rather than being killed mid-write",
    )

    f = sub.add_parser("failure", help="kill a replica mid-load")
    common(f)
    f.add_argument("--kill", required=True, help="replica id to kill")
    f.add_argument("--kill-command", required=True, help="shell-less command that kills it")
    f.add_argument("--kill-after-s", type=float, default=30.0)
    f.add_argument("--policy", default="round_robin")
    f.add_argument("--ratio", type=float, default=0.6)
    f.add_argument("--rate", type=float, required=True)
    f.add_argument("--repetition-index", type=int, default=1)
    f.add_argument("--num-requests", type=int, default=400)
    f.add_argument("--input-tokens", type=int, default=1024)
    f.add_argument("--output-tokens", type=int, default=128)
    f.add_argument("--shared-prefix-tokens", type=int, default=512)
    f.add_argument("--health-interval-ms", type=int, default=500)
    f.add_argument("--unhealthy-threshold", type=int, default=2)
    f.add_argument("--restart-command", default=None,
                   help="shell-less command that relaunches the killed replica")
    f.add_argument("--restart-after-s", type=float, default=30.0,
                   help="delay after the kill completes before relaunching")
    f.add_argument("--poll-interval-ms", type=int, default=250)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fn = {"verify": cmd_verify, "matrix": cmd_matrix, "failure": cmd_failure}[args.cmd]
    return asyncio.run(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
