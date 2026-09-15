# Node-hours consumed — P3 `inference-router`

Account `p201466`, partition `gpu`. **A `gpu` job bills the whole 4-GPU node**
regardless of how many GPUs it uses, so a build job that touches no GPU still
costs a node-hour at the same rate as the matrix.

Allocation is monthly and resets on the 1st. `myquota` on 2026-09-15 reported
`p201466` at **0 of 130 GPU node-hours used** for September, so the full
month's allocation was available at the start of this project.

Elapsed times below are from `sacct`; node-hours are `elapsed x AllocNodes`.

| job | what | elapsed | node-hours | outcome |
|---|---|---|---|---|
| 5211435 | ir-build | 00:00:29 | 0.008 | FAILED — `reqwest` has no `http1` feature |
| 5211438 | ir-build | 00:00:51 | 0.014 | FAILED — `icu_*` MSRV > 1.85.1 |
| 5211452 | ir-build | 00:00:31 | 0.009 | FAILED — stale `Cargo.lock` defeated the MSRV resolver |
| 5211464 | ir-build | 00:06:57 | 0.116 | FAILED — one clippy lint (`needless_question_mark`) |
| 5213920 | ir-build | 00:12:16 | 0.204 | FAILED — 64/66 unit tests pass, 2 wrong test expectations |
| 5213921 | ir-density | 00:30:36 | 0.510 | TIMEOUT in `qos=test` — arms A and B measured, C not reached |
| 5213922 | ir-verify | — | 0.000 | cancelled, dependency never satisfied |
| 5214089 | ir-build | 00:09:16 | 0.154 | FAILED — 66/66 unit tests pass; 1 integration test asserted the opposite of `least_loaded`'s purpose |
| 5214090 | ir-full | — | 0.000 | cancelled, dependency never satisfied |
| 5214119 | ir-build | in flight | — | — |
| 5214120 | ir-full | in flight | — | — |

**Consumed so far: ~1.02 node-hours** of the ~30 budgeted for P3.

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
