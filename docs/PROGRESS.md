# PROGRESS: P3 `inference-router`

Last updated: 2026-09-17.

This file is the honest status of the project. Where something has not been
measured it says **not measured**, and nothing here carries a number that does
not trace to a committed artifact under `results/raw/` or
`results/calibration/`.

---

## Definition-of-done audit

Checked against the PROJECT 3 "Definition of done" in the workspace spec.

| requirement | status | evidence |
|---|---|---|
| All four policies measured against a live >=4-replica deployment with N>=3 | **MET** | Job 5216059: 4 policies x 4 ratios x 4 calibrated rates x (3 reps + 1 warmup) = 256 phases, 4 vLLM replicas, run `4c21b52cf3d1432c`. Job 5216974: 4 policies x 2 ratios x 3 rates x 4 = 96 phases with 216 prefix groups, run `3623b8dbfa6443b1`. Every configuration has N=3 usable repetitions (`results/RESULTS.md` inventory). |
| SSE streaming verified non-buffering under load | **MET** | Deterministic: `tests/streaming.rs`. Live gate, job 5216059 stage 1 (`results/raw/verify__ea06b12cf8ae4e58.json`): 32/32 requests, 130 chunks each, TTFT/e2e p50 0.048. Under matrix load: 130 chunks per request in every phase at every rate (figure `streaming_non_buffering.png`). |
| Prefix-affinity load-imbalance failure mode demonstrated with real data | **MET** | Single-prefix matrix: plain affinity at ratio 0.9 puts 3.66-3.74x the mean load on one replica, doubles median TTFT at 16 req/s, and sheds 42% of requests at 48 req/s against round robin's 10%. The bounded policy holds peak/mean to 1.21-1.26 at 16-48 req/s. README "Results". |
| README states hardware, replica count, and topology | **MET** | README "Hardware": node type, GPU, driver as recorded, NVLink mesh, 4 replicas one per GPU on one node, NUMA pinning, admission cap, backend digest and model revision. |
| Router binary, config file, clippy-clean, unit + integration tests | **MET** | Build job 5215489 (rustc 1.98.1 via rustup): `cargo build --release`, `clippy -D warnings`, rustfmt check, all unit and integration tests pass. |
| Four routing policies behind a common trait | **MET** | `src/policy/mod.rs` `RoutingPolicy`; selectable at runtime via `POST /admin/policy`. |
| Deployment: Slurm, replicas fanned out, topology documented | **MET** | `slurm/lib_replicas.sh`; topology recorded in every artifact's `full_config_dict.fleet`. Single node; see "not done" below for multi-node. |
| Kubernetes manifests | **DELIBERATELY ABSENT** | No Kubernetes cluster is available, and the spec says not to commit untested manifests. |
| Failure-mode experiment (budget permitting) | **MET** | Job 5216528: 12 trials (4 policies x 3), replica SIGKILLed and relaunched under load; detection, readmission, error bursts and post-recovery share per policy. |

Replica density (spec: "measure, don't assume"): **measured**. Two 7B replicas
fit on one 40 GB card at util 0.45 each, with a 24,608-token KV cache each
(6.01x concurrency at 4,096 tokens) against 357,520 tokens (87.29x) for one
replica at 0.90 (jobs 5216059 stage 2 and 5213921 arm A).

---

## Findings in one paragraph each

**Single hot prefix.** Round robin already reaches the no-eviction hit-rate
bound, so prefix affinity gains nothing on cache and loses on balance. This
contradicts the expected headline for this workload, and it is reported as such.

**Tenant mix (216 prefixes).** Affinity reaches the bound (0.338 against 0.144
for round robin at ratio 0.9) with even load, and cuts median TTFT by 25% at
8 req/s. At 32 req/s the latency advantage is gone (1.35x round robin).

**Failure.** Detection 0.70-1.52 s with 500 ms health checks and a threshold of
two; no failures after ejection or after readmission in any trial; readmission
109.8-119.4 s after the relaunch, dominated by vLLM cold start.

---

## Caveats that apply to the numbers

- Client dispatch lag: most phases exceed the generator's own 0.25 s tolerance
  on more than 2% of requests. They are kept and flagged, and TTFT from actual
  dispatch is reported alongside (`analysis/load.py` docstring). The median
  TTFTs quoted barely change between the two clocks.
- Past the saturation point, latency describes admitted requests only.
- Absolute latencies are co-located-fleet numbers (P1 job 5162712).
- `-dirty` stamps: every job-5216059 artifact except the stage-1 verify
  artifact (265 files), caused only by the job's own untracked output files
  (since fixed in `provenance.git_sha`); 2 of the 12 failure trials in 5216528,
  from uncommitted analysis-only edits. The driver code that ran is the
  recorded SHA in both cases. The 216-prefix matrix (5216974) and the other 10
  failure trials are clean. The superseded job 5215612 is dirty throughout.
  See `docs/notes.md`.

---

## Things deliberately not done, and why

- **Multi-node (8 replicas across 2 nodes).** Would make network latency
  visible in routing, at double the node cost per phase. A single-node
  4-replica fleet satisfies ">= 4 replicas". Not measured, not estimated.
- **A matrix on the 2-per-card topology.** Its KV envelope is about 14.5x
  smaller per replica, so it serves a different workload; pooling it with the
  1-per-card matrix would confound policy with capacity.
- **Root-causing the client dispatch lag.** The generator and router run
  unpinned on the same node as four pinned replicas. Pinning them to free cores
  is the obvious next step; it was not needed to reach the conclusions above.
- **SGLang behind the router.** The router speaks the OpenAI API but has only
  been exercised against vLLM 0.27.1.
