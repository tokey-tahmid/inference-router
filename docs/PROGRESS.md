# PROGRESS — P3 `inference-router`

Last updated: 2026-09-15.

This file is the honest status of the project. Where something has not been
measured it says **not measured**, and nothing here carries a number that does
not trace to a committed artifact under `results/raw/`.

---

## Definition-of-done audit

Checked against the PROJECT 3 "Definition of done" in the workspace `CLAUDE.md`.

| requirement | status | evidence |
|---|---|---|
| All four policies measured against a live >=4-replica deployment with N>=3 | **NOT MET — not yet measured** | Matrix job staged as `slurm/run_full_experiment.sh` stage 5. `results/raw/` contains no matrix artifacts yet. |
| SSE streaming verified non-buffering **under load** | **PARTIAL** | Verified deterministically against a mock upstream in `tests/streaming.rs` (5 tests). Against a **live vLLM replica**: not yet measured — staged as stage 1 of the full-experiment job. |
| Prefix-affinity load-imbalance failure mode demonstrated with real data | **NOT MET — not yet measured** | Demonstrated in a unit test (`unbounded_affinity_ignores_load_completely`) but a unit test is not "real data". Requires the matrix. |
| README states hardware, replica count, and topology | **MET** | `README.md` "Hardware" section names the node, GPU model, driver, NVLink topology, and the exact backend digest and model revision copied from P1's lockfile. |
| Router binary, config file, clippy-clean, unit + integration tests | **MET (build), pending re-confirm** | Build job `5213920`: `cargo build --release --all-targets` and `cargo clippy --all-targets -- -D warnings` both clean; 64 of 66 unit tests passed. The 2 failures were wrong test expectations (documented in `docs/notes.md`), fixed, rebuild in flight as job `5214089`. |
| Four routing policies behind a common trait | **MET** | `src/policy/mod.rs` defines `RoutingPolicy`; `round_robin`, `least_loaded`, `prefix_affinity`, `prefix_affinity_bounded` all implement it and are selectable at runtime via `POST /admin/policy`. |
| Deployment: Slurm, replicas fanned out, topology documented | **MET** | `slurm/lib_replicas.sh` launches NUMA-pinned replicas; topology recorded into every artifact's `full_config_dict.fleet`. |
| Kubernetes manifests | **DELIBERATELY ABSENT** | No Kubernetes cluster is available here, and the workspace spec says not to commit untested manifests. |
| Failure-mode experiment (budget permitting) | **NOT MET — not yet measured** | `slurm/run_failure_mode.sh` is written and ready; not yet run. |

**Summary: the router is built and tested; the measurements are not done.**
That is the gap that matters, and it is stated plainly rather than dressed up.

---

## What has actually been measured

### Single-replica KV cache envelope (job `5213921`, arm A)

One Qwen2.5-7B-Instruct replica, bf16, TP=1, `gpu_memory_utilization=0.90` on an
A100-SXM4-40GB:

| quantity | measured |
|---|---|
| available KV cache memory | 19.09 GiB |
| GPU KV cache size | 357,520 tokens |
| max concurrency at 4,096 tokens/request | 87.29x |
| GPU memory used | 36,126 MiB of 40 GB |
| cold start (weights from Lustre to healthy) | 199 s |

### Two replicas per card: first attempt failed, and the attempt was flawed

Job `5213921` arm B launched two replicas on GPU0 at
`gpu_memory_utilization=0.47` each, 30 s apart. One came up; the other died:

```
ValueError: No available memory for the cache blocks.
```

That result is **not** usable as "two replicas do not fit", because the launches
were 30 s apart against a ~200 s weight load, so both profiled free memory while
the other was still allocating. The probe has been rewritten to launch
sequentially (first fully healthy before the second starts) and to give both
replicas the same half-card budget. **Re-measurement pending** as stage 2 of the
full-experiment job.

The job also timed out: it was submitted to `qos=test`, whose ceiling is 30
minutes, and never reached its third arm.

### Build and toolchain (jobs `5211435`-`5213920`)

- MeluXina exposes exactly one Rust: `Rust/1.85.1-GCCcore-14.2.0`. `reqwest`'s
  `url` -> `idna` -> `icu_*` chain now requires 1.88, so the build needs
  `resolver = "3"` plus `rust-version = "1.85"` **and** a freshly generated
  `Cargo.lock` (a stale lock silently defeats the MSRV-aware resolver).
- `cargo clippy --all-targets --all-features -- -D warnings` is clean.

Full detail and the reasoning behind each fix is in `docs/notes.md`.

---

## What is staged and ready to run

`slurm/run_full_experiment.sh` (job `5214090`, queued) runs five stages in one
6-hour allocation:

1. **SSE non-buffering gate** against one live vLLM replica. Hard gate: the job
   exits if it fails, because a buffering proxy produces a complete-looking but
   meaningless dataset.
2. **Replica density**, sequential, both replicas at half the card.
3. **Four replicas**, one per GPU, NUMA-pinned.
4. **Rate calibration** against that fleet, with the knee located from two
   independent signals and reported honestly if no rate saturates.
5. **The matrix**: 4 policies x 4 sharing ratios x the calibrated rates x N=3,
   plus a warmup repetition per configuration that is recorded and excluded.

Then `sbatch slurm/run_analysis.sh` regenerates every figure and table, and
`slurm/run_failure_mode.sh` runs the kill-a-replica experiment.

### If time runs short

The driver takes a `--budget-s` and stops cleanly between phases rather than
being SIGKILLed mid-write, so a partial matrix is a set of complete artifacts
rather than a truncated one. The matrix is ordered policy-outermost, so an early
stop loses whole policies rather than corrupting every cell. If the budget will
not cover the full matrix, the **rate axis** is what should be cut: the
policy-by-sharing-ratio interaction is the headline, and all four policies at
all four ratios matter more than four rate points.

---

## Things deliberately not done, and why

- **Multi-node (8 replicas across 2 nodes).** The more interesting topology for
  routing, because network latency becomes visible in the results. It doubles
  the node-hour cost of every matrix cell, and a single-node 4-replica fleet
  already satisfies the ">= 4 replicas" requirement. Recorded as an unmeasured
  absence; not estimated.
- **Kubernetes manifests with an HPA.** No cluster to run them on.
- **SGLang as a second backend.** Out of scope for P3; the router is
  backend-agnostic in that it speaks the OpenAI API, but this has only been
  exercised against vLLM 0.27.1.

## Open questions for a human

- **Is a 2-replicas-per-card topology worth measuring as a second arm?** If the
  sequential probe shows it works, each replica would have roughly 3 GB of KV
  cache against the 19 GB a solo replica gets. That is a small enough envelope
  that it changes the *workload* the replicas can serve, not just the topology,
  so the two configurations should probably not be pooled in one figure. My
  inclination is to report it as a capacity finding and keep the matrix on the
  1-per-card fleet, but it is a judgement call about what the portfolio should
  show.
- **Absolute latencies here are fleet numbers, not isolated-replica numbers.**
  P1 job `5162712` measured co-location alone costing +150.7% TTFT p50 and
  -34.1% throughput. The policy comparison is unaffected (every policy runs
  against the same co-located fleet in the same job), but any absolute figure
  quoted from this project needs that caveat attached, and the README carries
  it.
