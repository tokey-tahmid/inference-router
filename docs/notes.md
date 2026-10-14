# Notes: findings, surprises, and things that were measured rather than assumed

Running log. Entries are dated and cite the job that observed them, so anything
here can be re-checked rather than trusted.

---

## 2026-09-15 — the cluster has exactly one Rust, and it is older than the crate ecosystem

Build jobs `5211435`, `5211438`, `5211452`, `5211464`.

`module -t avail Rust` on a `gpu` node lists exactly one toolchain:

```
Rust/1.85.1-GCCcore-14.2.0
```

That is a problem `reqwest` inherits three levels down. Its `url` -> `idna` ->
`icu_*` chain has since raised its MSRV to 1.88, so the default resolve picks
crates that cannot compile here:

```
error: rustc 1.85.1 is not supported by the following packages:
  icu_collections@2.3.0 requires rustc 1.88
  ...
  idna_adapter@1.2.2 requires rustc 1.86
```

The fix is `resolver = "3"` plus `rust-version = "1.85"` in `Cargo.toml`:
cargo's MSRV-aware resolver then picks the newest versions that still support
the pinned toolchain (`icu_collections` 2.1.1, `idna_adapter` 1.2.1). Far more
robust than `cargo update --precise` on a dozen `icu_*` crates, which would need
redoing every time the chain moves.

**The part worth remembering:** setting the resolver had *no effect* on the next
run (`5211452` failed identically). A `Cargo.lock` generated before the
`rust-version` pin keeps its MSRV-incompatible selections, and the MSRV-aware
resolver only runs when there is no lock to reuse. The build script now runs
`cargo generate-lockfile` explicitly and prints the selected versions of the
MSRV-sensitive chain, so a future failure of the same shape is diagnosable from
the log instead of by bisection.

### rustfmt is applied in the job, not checked

The login node has no cargo, so there is no pre-submission hook that could have
formatted the tree, and nothing can be syntax-checked before submission at all
(login Python is 3.6). A check-only `cargo fmt --check` gate would fail every
job until someone hand-formatted Rust they cannot run a formatter against. The
build job applies `cargo fmt --all` and then verifies the apply converged.

---

## 2026-09-15 — both unit-test failures were wrong tests, not wrong code

Build job `5213920`: 64 passed, 2 failed. Both failures were mistakes in the
*expectations*, and in both cases the implementation was correct.

**1. The FNV vector was FNV-1, not FNV-1a.** `fnv1a_matches_published_vectors`
asserted `fnv1a64(b"foobar") == 0x85267b256108 9d4d`. The implementation
produced `0x85944171f73967e8`.

**How that was established, which matters more than the fix.** The tempting move
when a test fails against an implementation you just wrote is to paste the
observed value into the expectation, and doing that here would have made the
test permanently worthless: it would then assert only "this function is
deterministic", which is true of any hash including a wrong one. Instead the
digest was recomputed from the FNV-1a definition in a throwaway Python
implementation written from the spec, not from the Rust:

```python
h = 0xcbf29ce484222325
for c in b"foobar":
    h ^= c
    h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF   # -> 0x85944171f73967e8
```

That independently reproduces the Rust output and the two other vectors
(`""`, `"a"`) that were already passing. `0x8526...` is the FNV-**1** digest;
the two algorithms differ only in whether the xor comes before or after the
multiply, which is exactly why the wrong constant looks plausible.

The general point: a test vector beats a round-trip property here. A test
asserting only "the same input hashes the same way twice" would have passed
against either algorithm, and the hash choice is load-bearing precisely because
it has to be stable across toolchains and machines for two runs of the same
experiment to be comparable.

**2. The bounded-affinity exhaustion test could not exhaust.** The test loaded
replicas to 30/30/25 with `factor = 1.0` and expected
`AffinitySpillExhausted`. But the capacity bound is `ceil(factor x mean)` =
`ceil(28.33)` = 29, and 25 < 29, so r2 was under cap and the policy correctly
spilled to it rather than falling through.

The general point: with `factor = 1.0`, "every replica at or over cap" requires
every replica to be *equal*, because any replica below the mean is necessarily
below `ceil(mean)`. The test now levels all three at 30 and asserts the reason
rather than a specific index, since the tiebreak among equals rotates by design.

---

## 2026-09-15 — two 7B replicas on one 40 GB card: the first attempt failed, and the failure mode is instructive

Density probe job `5213921`, on `mel2036`.

### Provenance caveat: this job's `sacct` state is TIMEOUT, and the numbers are still sound

`sacct -j 5213921` reports `TIMEOUT` at `00:30:36`, not `COMPLETED`. A reader
checking the job state needs to know why the arm-A figures below are quoted
anyway, so, plainly:

- The probe was submitted to `qos=test`, whose ceiling is 30 minutes. It has
  three arms and never reached the third.
- **The numbers quoted below are emitted by vLLM during engine startup**, before
  any load is applied. Arm A's replica was healthy at 199 s and printed its KV
  cache sizing then; that line is in `ir_density_5213921.out` and is complete.
- What the timeout truncated is **arm C** (four replicas, one per GPU), which
  never ran, and the tail of arm B. It did not truncate, interrupt, or
  invalidate the arm-A measurement.

So the correct reading is: arm A measured, arm B attempted and inconclusive (see
below, and the attempt was flawed on its own terms), arm C **not measured**.

**Arm A — one replica, `gpu_memory_utilization=0.90`: healthy in 199 s.**

```
Available KV cache memory: 19.09 GiB
GPU KV cache size: 357,520 tokens
Maximum concurrency for 4,096 tokens per request: 87.29x
GPU0 memory used: 36,126 MiB
```

So a single 7B replica on an A100-40GB has ~19 GB of KV cache and can hold ~87
concurrent 4k-token sequences. That is a comfortable replica.

**Arm B — two replicas on GPU0 at `gpu_memory_utilization=0.47` each: one died.**

The replica on port 8111 came up (400 s). The one on 8110 died at init:

```
File ".../vllm/v1/core/kv_cache_utils.py", line 758, in _check_enough_kv_cache_memory
    raise ValueError(
ValueError: No available memory for the cache blocks. Try increasing
`gpu_memory_utilization` when initializing the engine
```

**Why this is not simply "it does not fit".** `gpu_memory_utilization` is a
fraction of the card's *total* memory, but vLLM sizes the KV cache from memory
that is *free at init time*. The two replicas were launched 30 s apart, which is
nowhere near enough: weight loading takes ~200 s. So both were profiling free
memory while the other was still allocating its 15.2 GB of weights, and the
loser saw a card that was already nearly full.

The arithmetic is tight even without the race. 0.47 x 40 GB = 18.8 GB budget,
minus ~15.2 GB of bf16 weights, minus a ~0.5 GB CUDA context, minus the
activation peak during profiling, leaves very little for KV. Whether it leaves
*any* is the question, and a replica that loads but holds three sequences is not
a usable replica.

Those are two different findings and the first attempt could not tell them
apart, so the probe was rewritten to launch **sequentially**: the first replica
must be fully healthy before the second starts. That removes the race and makes
the answer attributable.

### The rewrite was wrong too, in a different way

Worth recording because it is the more embarrassing of the two and the easier
one to have shipped. The sequential rewrite kept the stage-1 replica alive (it
was already up at `gpu_memory_utilization=0.90`) and started the "second"
replica alongside it at 0.45.

That is not a two-replicas-per-card test at all. Arm A measured the 0.90 replica
occupying **36,126 MiB of the card's 40 GB**. There is no room for a second
15.2 GB weight load in the ~4 GB left, so the experiment could only ever have
returned "no", and that "no" would have been a fact about 0.90 + 0.45 > 1.0
rather than about whether the topology is viable.

It would also have looked like a *confirmation* of the first attempt's failure,
which is what makes it dangerous: two independent-looking runs agreeing on a
wrong answer. The fixed probe gives both replicas the same half-card budget
(0.45 each, falling back to 0.42) and tears the stage-1 replica down first.

Re-run is staged as step 2 of `slurm/run_full_experiment.sh`, and the finding is
written to `results/raw/` as an artifact rather than left in a log, because the
topology decision depends on it.

### What the answer will constrain, either way

This is a capacity finding about the **workload**, not only about the topology,
and that is the non-obvious part. A solo replica has ~19 GB of KV cache
(357,520 tokens, 87.29x concurrency at 4k). Two half-card replicas would each
have roughly 3 GB after weights, an order of magnitude less. A fleet built that
way cannot admit the same concurrency or the same context lengths, so the two
topologies are not two ways of serving one workload: they are two different
serving envelopes. Pooling them in a routing figure would compare policies
across configurations that differ in what they can physically admit, which is
why the matrix stays on the 1-per-card fleet and density is reported separately.

**Operational note:** the probe was submitted to `qos=test`, whose limit is 30
minutes, and it timed out before its third arm. At ~200 s per replica cold start
from Lustre, any multi-arm replica experiment needs `qos=short`. Also worth
knowing: `qos=test` has a max-jobs-per-user limit, so submitting three probes at
once serialises them rather than running them in parallel.

---

## 2026-09-15 — why the whole experiment became one Slurm job

Originally staged as five separate submissions (build, density, verify,
calibrate, matrix). Two things pushed it into one:

1. **Queue wait dominates the work.** Several of these stages are minutes of
   compute behind an unpredictable wait for a 4-GPU node.
2. **The stages are not independent measurements.** The matrix's rate ladder has
   to span underload to overload *for the fleet it runs against*, and the
   saturation point of four co-located replicas is a measured quantity. P1 job
   `5162712` measured co-location costing -34.1% throughput versus isolation, so
   a ladder derived from isolated-replica throughput would put every "overload"
   point inside the stable region and the matrix would compare four policies
   that never contend.

So `run_full_experiment.sh` measures the knee in stage 4 and feeds it to stage 5
in the same allocation, on the same node, minutes apart. `pick_rates.py` locates
the knee from server behaviour (median TTFT above 3x its low-rate baseline, or
error rate above 1%) and **reports when no rate saturated** rather than
labelling the top of the ladder "overload" on faith. The first version of that
criterion was wrong; see the next section.

Stage 1 is a hard gate: if SSE streaming is not verified non-buffering against a
live replica, the job exits before spending node-hours on a matrix whose
headline metric would be meaningless.

---

## 2026-09-16: the first knee was a measurement artefact, and so was the first "-dirty"

**False knee.** Job 5215612 calibrated at 2..32 req/s and put the knee at
4 req/s, from "achieved throughput < 0.9 x offered". Achieved was computed as
completions over total elapsed time, drain tail included, so on short phases it
falls with offered rate whether or not anything saturates. The same artifacts
show median TTFT flat at 0.07-0.10 s and zero errors across the whole ladder:
the fleet was never loaded. The matrix it would have fed compared four routing
policies on an idle fleet. Two more problems in the same job: the job end time
came from `date -d "+5:59:30"`, which `date` cannot parse, and an `|| echo 0`
fallback turned that into a silent 600 s budget floor, so only 8 phases ran.

Fixes: the knee is now median TTFT against the lowest-rate baseline, plus
error rate. Median, because TTFT is measured from scheduled arrival (the
correct open-loop method) and a client stall therefore lands in TTFT directly,
but only for the few requests it hits: p95 was *highest at the lowest rates*
in 5215612. Achieved rate is now measured over the dispatch window only, and a
rate the generator could not offer is reported and excluded, never promoted to
a knee. The ladder was widened and phase length now scales with rate so a
backlog has time to form.

Job 5216059 calibration (artifacts in `results/calibration/5216059/`):

| offered | achieved | TTFT p50 (s) | TTFT p95 (s) | error rate |
|---|---|---|---|---|
| 4 | 3.78 | 0.071 | 0.417 | 0.000 |
| 8 | 8.18 | 0.068 | 0.178 | 0.000 |
| 16 | 16.01 | 0.092 | 0.739 | 0.000 |
| 32 | 31.33 | 0.099 | 2.623 | 0.000 |
| 48 | 47.44 | 0.340 | 9.912 | 0.190 |
| 64 | 61.43 | 0.195 | 16.057 | 0.242 |
| 96 | 75.01 | 23.410 | 58.375 | 0.323 |
| 128 | 123.73 | 38.635 | 85.760 | 0.348 |

Knee at 48 req/s: median TTFT 0.340 s against a 0.071 s baseline, and
admission control shedding 19% of requests. Selected matrix ladder: 4, 16, 48,
128. At 96 req/s the generator dispatched only 75 req/s and that row was
excluded from the knee search; it did not affect the ladder. Every row is
flagged `degraded` (more than 2% of requests dispatched over 0.25 s late),
including the idle 4 req/s row, so client-side scheduling lag is present at
all rates and is part of why TTFT p95 is noisy even when unloaded. Past the
knee, latency percentiles describe *admitted* requests only; the error rate is
the other half of the overload result and must be read with them.

Known weakness: the keep-up test compares a realised Poisson dispatch rate to
the nominal rate at 0.9. With ~120 requests per low-rate phase the realised
rate alone varies by roughly 1/sqrt(120) ~ 9%, so a low rate can be excluded
by chance. It did not happen here.

**Spurious "-dirty".** The matrix artifacts from 5216059 carry
`f0289c3a0b3e89d5ab0b0460088449f8928fc5c5-dirty` (pre-rewrite; see the SHA
map below), while that job's stage-1
verify artifact carries the same SHA clean. No tracked file changed during the
job: `git status --porcelain` counted the job's own untracked result files.
The artifacts stay as written (append-only); the code at that SHA is exactly
what ran. `provenance.git_sha` now passes `--untracked-files=no`.

Two later failure-mode trials (job 5216528, pre-rewrite SHA `451d2a88`) are also stamped
dirty, from analysis-only edits that were uncommitted while those trials
started. The failure driver itself was committed at that SHA. The remaining
10 trials and the 216-prefix matrix (5216974) are clean.

---

## Design decisions recorded here because they are not obvious from the code

### The in-flight guard lives inside the response stream

A request is in flight for as long as its response body is still streaming,
which is long after the axum handler returns. `InflightGuard` is therefore moved
into the relay stream's state and dropped with it, so the decrement happens on
normal completion, upstream error, and client disconnect alike.

Decrementing in the handler would report a fleet that looks idle while every
replica is saturated, and `least_loaded` would then route on a number that is
always zero. That failure produces no error and no crash: it produces a
plausible dataset in which least-loaded and round-robin are inexplicably
identical.

### `least_loaded` rotates its tiebreak

On an idle fleet every replica reads zero load. A lowest-index tiebreak would
send *all* traffic to replica 0, which reads as a catastrophic fairness result
and is entirely an artefact of the tiebreak. This has a regression test
(`an_idle_fleet_is_spread_evenly_rather_than_piled_on_replica_zero`), because
the fairness metric is exactly the instrument that would otherwise have
presented the bug as a finding about load-based routing.

### Rendezvous hashing, not modulo or a ring

Affinity placement ranks replicas by highest-random-weight score over their
**stable indices**. When a replica is ejected, only the keys that mapped to it
move. A `key % n_healthy` scheme would remap essentially every key the instant
the healthy count changed, flushing every replica's prefix cache at once during
the failure experiment and turning a recovery measurement into a measurement of
mass cache invalidation. `hash.rs` asserts this property directly rather than
asserting it in a comment.

### The `min_capacity` floor on the bounded policy

Without it, an idle fleet has `total_load` near zero, so `ceil(factor x mean)`
rounds to 1 and the bounded policy spills on the *second* concurrent request to
any key. It would become round-robin exactly in the underload regime where
affinity is cheapest and where the cache benefit should be largest, and the
bounded-vs-unbounded comparison would be measuring the floor rather than the
bound.

### Open loop, and latency measured from the scheduled arrival

A closed-loop generator cannot express overload: as the fleet slows, offered
load falls with it and the system self-throttles into a stable point, so every
"overload" arm silently becomes a different, lower load. Arrivals are a Poisson
process on wall clock, and latency is measured from the **scheduled** arrival
rather than from when the client got around to sending, so generator lag counts
as the real client-visible latency it is. Each phase records its own
`send_delay` distribution and is marked `degraded` if the driver itself fell
behind, because in that case the phase measured the driver, not the fleet.

### Cumulative counters are differenced per phase

vLLM's `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` (the
`gpu_` infix was dropped in 0.27.1) and the router's own counters are cumulative
for the life of the process, and the matrix deliberately runs many phases
against one long-lived fleet. Reading a counter at the end of a phase gives that
phase plus every phase before it.

P1 hit this and it produced a result that looked like data: a 0.9-ratio phase
following a 0.0-ratio phase read low, so the comparison became sensitive to
phase ordering. Here, scrapes are taken before and after every phase and
differenced, a negative delta is treated as evidence the server restarted rather
than as a measurement, and a missing counter is reported **absent rather than
zero** — an absent measurement and a measured zero are different facts, and P1's
prefix-cache bug was diagnosable only because of that distinction.

---

## Open questions

- Whether two replicas per card is viable at all once the startup race is
  removed. Being re-measured sequentially; the arithmetic suggests the KV cache
  left over would be small enough to change the workload rather than just the
  topology, in which case a 2-per-card fleet is not comparable to a 1-per-card
  one and should not be pooled with it.
- Multi-node (8 replicas across 2 nodes) is not attempted. It is the more
  interesting topology for routing because network latency becomes visible, but
  it doubles the node-hour cost of every matrix cell and the single-node
  4-replica fleet already satisfies the ">= 4 replicas" requirement. Recorded as
  an unmeasured absence, not estimated.

---

## 2026-09-17: history rewritten to remove two incorrect co-author trailers

Two early commits (`WIP: initial inference-router scaffold` and `WIP:
inference-router progress before rate-limit interruption`) carried a
co-author trailer added in error. GitHub credits co-authors as
contributors, so the published repository listed one.

Removing a trailer rewrites the commit, and with it every descendant, so 29 of
30 commits changed SHA. No content changed: `git diff` between the pre-rewrite
and post-rewrite trees is empty, the commit count, subjects, author and
committer identities and dates are unchanged, and all 30 commits are authored
and committed by the repository owner.

**This breaks a provenance link, and the mapping below is the repair.** Every
artifact under `results/raw/` and `results/calibration/` records the `git_sha`
it was produced at, and those are pre-rewrite SHAs. The artifacts are
append-only and were *not* edited. To resolve one, map it here:

| `git_sha` recorded in artifacts | commit after the rewrite | artifacts |
|---|---|---|
| `f0289c3a0b3e89d5ab0b0460088449f8928fc5c5-dirty` | `2525671bcee6d097dc6fde4d092fb49218f28b55` (plus uncommitted changes) | 265 |
| `efe9130005e346179369992cbf01bbf8bd5c14a3` | `8881bc8485de65e959fa1d715dcd51355183e8a2` | 100 |
| `c921ce89f398b320919bef280dfb969e353ca73b-dirty` | `e972a28e079c8a11f7b7e73a588f1d80977999c2` (plus uncommitted changes) | 10 |
| `7782bbf5b9544461d3dea2ef9e60a65025aea809` | `75a541939d60991fdb3a34cbe51f32daeefc5cf4` | 6 |
| `451d2a880735197e4c72259184ceee6b456204da-dirty` | `cd564e5432d6b9035322361bd2bba19d737a1efd` (plus uncommitted changes) | 2 |
| `f0289c3a0b3e89d5ab0b0460088449f8928fc5c5` | `2525671bcee6d097dc6fde4d092fb49218f28b55` | 1 |

The complete 30-commit mapping is in `docs/sha_map_2026-09-17.tsv`. A `-dirty`
suffix means the tree carried uncommitted changes when the artifact was
written; see the provenance note above for what those were in each case.

No archive branch of the old history is published: leaving the original commits
reachable on the remote would keep the co-author credited, which is the thing
the rewrite exists to remove.
