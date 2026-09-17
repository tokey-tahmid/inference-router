# Node-hours consumed: P3 `inference-router`

Account `p201466`, partition `gpu`. **A `gpu` job bills the whole 4-GPU node**
regardless of how many GPUs it uses, so a build job that touches no GPU still
costs a node-hour at the same rate as the matrix.

Allocation is monthly and resets on the 1st. `myquota` on 2026-09-15 reported
`p201466` at **0 of 130 GPU node-hours used** for September, so the full
month's allocation was available at the start of this project.

Elapsed times below are from `sacct`; node-hours are `elapsed x AllocNodes`.

| job | what | elapsed | node-hours | outcome |
|---|---|---|---|---|
| 5211435 | ir-build | 00:00:29 | 0.008 | FAILED, `reqwest` has no `http1` feature |
| 5211438 | ir-build | 00:00:51 | 0.014 | FAILED, `icu_*` MSRV > 1.85.1 |
| 5211452 | ir-build | 00:00:31 | 0.009 | FAILED, stale `Cargo.lock` defeated the MSRV resolver |
| 5211464 | ir-build | 00:06:57 | 0.116 | FAILED, one clippy lint |
| 5213920 | ir-build | 00:12:16 | 0.204 | FAILED, 2 wrong unit-test expectations |
| 5213921 | ir-density | 00:30:36 | 0.510 | TIMEOUT in `qos=test`; density arm A measured, arm C not reached |
| 5214089 | ir-build | 00:09:16 | 0.154 | FAILED, 1 integration test asserted the opposite of `least_loaded` |
| 5214119 | ir-build | 00:01:57 | 0.033 | FAILED, `yoke-derive` needs newer Rust than the module; moved to rustup stable |
| 5215430 | ir-build | 00:02:07 | 0.035 | FAILED, clippy `sliced_string_as_bytes` on the new toolchain |
| 5215489 | ir-build | 00:03:56 | 0.066 | build and all tests pass (rustc 1.98.1) |
| 5215612 | ir-full | 00:29:27 | 0.491 | stages 1-4 ran; knee criterion wrong and budget floored to 600 s; 8 matrix phases (superseded) |
| 5215887 | ir-full | 00:00:31 | 0.009 | stopped by the new logic gate before stage 1 (`_spread` undefined) |
| 5216059 | ir-full | 03:43:47 | 3.730 | full experiment: SSE gate, density, calibration, 256-phase matrix |
| 5216182 | ir-analysis | 00:00:36 | 0.010 | analysis smoke test |
| 5216294 | ir-failure | 00:16:31 | 0.275 | cancelled: structlog `event=` crash after each kill |
| 5216528 | ir-failure | 02:11:36 | 2.193 | failure mode: 12 kill/relaunch trials |
| 5216635 | ir-analysis | 00:00:46 | 0.013 | analysis smoke test |
| 5216974 | ir-tenant | 01:51:25 | 1.857 | 216-prefix (tenant mix) matrix, 96 phases |
| 5217319 | ir-analysis | 00:01:40 | 0.028 | analysis |
| 5217400 | ir-analysis | 00:01:44 | 0.029 | analysis (final) |

Cancelled-while-pending jobs 5213922, 5214090 and 5214120 never started and cost nothing.

**P3 total: 9.78 node-hours** (from `sacct`, 2026-09-17) of the ~30 budgeted. `myquota` on
2026-09-17 reported `p201466` at 18 of 130 GPU node-hours used for September
(all projects).

Note on job states: `5213921` shows `TIMEOUT` rather than `COMPLETED`. Its
arm-A measurements are still sound because vLLM emits them at engine startup,
long before the 30-minute `qos=test` ceiling was hit; what the timeout cost was
arm C, which never ran. See `docs/notes.md` for the full caveat.

## Where the budget is going

The matrix is the expensive item and its cost is dominated by holding four
replicas live, not by the load itself:

- Fleet cold start: ~200 s per replica from Lustre, staggered, so ~5 minutes to
  bring up four.
- Each matrix phase is a cache reset, a drain wait, a settle, and 200 requests
  at the phase's offered rate. At the low end of the rate ladder a phase is
  dominated by its own arrival schedule.

The single-allocation design of `slurm/run_full_experiment.sh` exists partly for
this reason: five separate submissions would each pay the fleet cold start, and
four of the five stages would pay it for work that takes minutes.

## Notes on avoidable spend

The five failed build jobs cost ~0.35 node-hours between them, and every one of
those failures would have been caught in seconds by a local `cargo check`. There
is no cargo on the login node, so on this cluster that class of iteration is
charged to the GPU allocation. Batching lint, build and test into one job (which
`slurm/build.sh` does) is what keeps it to one queue slot per attempt rather
than three.
