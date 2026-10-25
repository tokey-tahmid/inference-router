# Notes: findings, surprises, and things measured rather than assumed

This is a running log, in date order. Each entry names the job that observed what it describes, so anything here can be checked again rather than taken on trust.

---

## 2026-09-15: The cluster has exactly one Rust, and it's older than the crate ecosystem

The first build jobs (5211435, 5211438, 5211452 and 5211464) ran into the toolchain before they reached any of my code. `module -t avail Rust` on a GPU node lists exactly one toolchain, `Rust/1.85.1-GCCcore-14.2.0`. That's a problem `reqwest` inherits three levels down, because its chain of `url`, `idna` and `icu_*` crates had since raised its minimum supported Rust version to 1.88, so the default dependency resolution picked crates that couldn't compile there.

```
error: rustc 1.85.1 is not supported by the following packages:
  icu_collections@2.3.0 requires rustc 1.88
  ...
  idna_adapter@1.2.2 requires rustc 1.86
```

My first fix was `resolver = "3"` together with `rust-version = "1.85"` in `Cargo.toml`. With those, cargo's version-aware resolver picks the newest versions that still support the pinned toolchain (`icu_collections` 2.1.1 and `idna_adapter` 1.2.1), which is far more robust than pinning a dozen `icu_*` crates by hand.

The part worth remembering is that setting the resolver had no effect on the next run, and job 5211452 failed in exactly the same way. A `Cargo.lock` created before the `rust-version` pin keeps its incompatible selections, because the version-aware resolver only runs when there's no lock file to reuse. So the build script now runs `cargo generate-lockfile` explicitly and prints the versions it selected for that sensitive chain, which makes a future failure of the same shape diagnosable from the log rather than by bisection.

That fix later stopped being enough, when a transitive crate needed a newer compiler without declaring it (build 5214119). So the build now installs a current stable toolchain with rustup into project storage, records the exact version it used, and no longer loads the cluster's Rust module. `slurm/build.sh` explains the reasoning in full.

### rustfmt is applied in the job, not checked

The login node has no cargo, so there's no hook before submission that could format the tree, and nothing can be checked before submission at all, because the login node's Python is 3.6. A check-only `cargo fmt --check` gate would therefore fail every job until someone hand-formatted Rust they couldn't run a formatter on. Instead, the build job applies `cargo fmt --all` and then verifies that the result is stable.

---

## 2026-09-15: Both unit-test failures were wrong tests, not wrong code

Build job 5213920 ran 66 tests, and two failed. In both cases the mistake was in the expectation, and the implementation was correct.

### The FNV test vector wasn't a real FNV digest

`fnv1a_matches_published_vectors` asserted that `fnv1a64(b"foobar")` equals `0x85267b2561089d4d`, while the implementation produced `0x85944171f73967e8`.

How I established which one was right matters more than the fix. The tempting move, when a test fails against code you've just written, is to paste the observed value into the expectation. Doing that here would have made the test permanently worthless, because it would only assert that the function is deterministic, which is true of any hash, including a wrong one. Instead, I recomputed the digest from the FNV-1a definition in a throwaway Python implementation, written from the specification rather than from the Rust.

```python
h = 0xcbf29ce484222325
for c in b"foobar":
    h ^= c
    h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF   # -> 0x85944171f73967e8
```

That independently reproduces the Rust output and the two other test vectors (the empty string and "a") that were already passing. The value in the test wasn't an FNV digest of "foobar" at all, since FNV-1 gives `0x340d8765a4dda9c2`. An earlier version of this entry, and a comment in `src/policy/hash.rs`, called it the FNV-1 digest, which I found was also wrong when I recomputed both on 2026-10-24. The most likely explanation is that the constant was written from memory rather than copied from a reference, which is exactly the failure this check exists to catch.

The general point is that a published test vector beats a round-trip property here. A test that only checks that the same input hashes the same way twice would have passed with either algorithm, and the choice of hash carries real weight, because it has to be stable across toolchains and machines for two runs of the same experiment to be comparable.

### The bounded-affinity exhaustion test could never exhaust

The test loaded three replicas to 30, 30 and 25 requests with a bound factor of 1.0, and expected the policy to report that every replica was over its bound. However, the bound is the ceiling of the factor times the mean load, which is the ceiling of 28.33, or 29. Since 25 is below 29, the third replica was under its bound, and the policy correctly moved the request there instead of giving up.

The general point is that with a factor of 1.0, "every replica at or over its bound" requires every replica to have the same load, because any replica below the mean is necessarily below the ceiling of the mean. The test now loads all three replicas to 30 and checks the reason for the decision rather than a particular replica, since the tiebreak among equal replicas rotates by design.

---

## 2026-09-15: Two 7B replicas on one 40 GB card: the first attempt failed, and how it failed is instructive

This was the density probe, job 5213921.

### Why this job's TIMEOUT state doesn't invalidate its numbers

`sacct -j 5213921` reports the job as `TIMEOUT` at 30 minutes and 36 seconds, not `COMPLETED`, so a reader checking the job state deserves to know why I still quote its first arm. The probe went to the test queue, which allows 30 minutes, and it has three arms, so it never reached the third. However, the numbers I quote below are printed by vLLM during engine startup, before any load is applied. Arm A's replica was healthy after 199 seconds and printed its KV cache sizing then, and that line is complete in the job's log. What the timeout cut off was arm C (four replicas, one per GPU), which never ran, and the tail of arm B. So arm A is measured, arm B was attempted but inconclusive (and flawed on its own terms, as described below), and arm C is not measured.

### Arm A: one replica at a utilisation of 0.90, healthy in 199 seconds

```
Available KV cache memory: 19.09 GiB
GPU KV cache size: 357,520 tokens
Maximum concurrency for 4,096 tokens per request: 87.29x
GPU0 memory used: 36,126 MiB
```

So a single 7B replica on an A100-40GB has about 19 GB of KV cache and can hold about 87 concurrent sequences of 4,096 tokens, which makes it a comfortable replica.

### Arm B: two replicas on GPU 0 at a utilisation of 0.47 each, and one died

The replica on port 8111 came up after 400 seconds, while the one on port 8110 died during initialisation.

```
File ".../vllm/v1/core/kv_cache_utils.py", line 758, in _check_enough_kv_cache_memory
    raise ValueError(
ValueError: No available memory for the cache blocks. Try increasing
`gpu_memory_utilization` when initializing the engine
```

That doesn't simply mean "it doesn't fit". `gpu_memory_utilization` is a fraction of the card's total memory, but vLLM sizes the KV cache from the memory that's free when it initialises. The two replicas started 30 seconds apart, which is nowhere near enough when loading the weights takes about 200 seconds. So both were measuring free memory while the other was still loading its 15.2 GB of weights, and the loser saw a card that was already nearly full.

The arithmetic is tight even without that race. A utilisation of 0.47 on 40 GB is an 18.8 GB budget, and after about 15.2 GB of bf16 weights, about 0.5 GB of CUDA context and the activation peak during profiling, very little is left for the KV cache. Whether anything is left at all is the real question, because a replica that loads but holds only three sequences isn't a usable replica. Those are two different findings, and the first attempt couldn't tell them apart. So I rewrote the probe to launch the replicas one after the other, with the first fully healthy before the second starts, which removes the race and makes the answer attributable.

### The rewrite was wrong too, in a different way

This one is worth recording, because it's the more embarrassing of the two and the easier one to have shipped. The sequential rewrite kept the first stage's replica running, which was already up at a utilisation of 0.90, and started the "second" replica alongside it at 0.45.

That isn't a test of two replicas per card at all. Arm A had measured the 0.90 replica occupying 36,126 MiB of the card's 40 GB, so there was no room for another 15.2 GB of weights in the 4 GB left. The experiment could only ever have answered "no", and that "no" would have been a fact about 0.90 plus 0.45 being more than 1.0, not about whether the topology works. Worse, it would have looked like a confirmation of the first attempt's failure, which is what makes it dangerous, because two independent-looking runs would have agreed on a wrong answer. The fixed probe gives both replicas the same half-card budget, 0.45 each with a fallback to 0.42, and shuts the first stage's replica down before it starts.

The re-run became step 2 of `slurm/run_full_experiment.sh`, and its finding is written to `results/raw/` as an artifact rather than left in a log, because the choice of topology depends on it. (It found that two replicas do fit at 0.45 each, but with about 14.5 times less KV cache per replica, as the README describes.)

### What the answer constrains

This is a capacity finding about the workload, not only about the topology, and that's the part that isn't obvious. A replica on its own has about 19 GB of KV cache, while two half-card replicas have roughly 3 GB each after the weights, an order of magnitude less. A fleet built that way can't admit the same concurrency or the same context lengths, so the two topologies aren't two ways of serving one workload but two different serving envelopes. Pooling them in a routing comparison would compare policies across configurations that differ in what they can physically admit. That's why the matrix stays on the one-per-card fleet, and density is reported separately.

There's also an operational lesson. The test queue's 30-minute limit is too short for this kind of probe, because at about 200 seconds per cold replica start, any experiment with several arms of replicas needs the short queue instead. In addition, the test queue limits how many jobs one user can run at once, so submitting three probes together runs them one after another rather than in parallel.

---

## 2026-09-15: Why the whole experiment became one Slurm job

I originally staged the experiment as five separate submissions: build, density, verification, calibration and the matrix. Two things pushed them into one job.

Firstly, the wait in the queue dominated the work, because several stages are only minutes of compute behind an unpredictable wait for a node with four GPUs. Secondly, the stages aren't independent measurements. The matrix's rate ladder has to span underload to overload for the fleet it runs against, and the saturation point of this fleet, with the router in front of it, is a quantity to measure rather than derive from one replica. At the time, I also argued that co-locating four replicas costs a third of their throughput, citing llm-inference-bench's job 5162712. That measurement was withdrawn on 2026-10-24, because its four replicas shared one port and so measured one server. The decision still stands on the first reason, and measuring the knee on the real fleet remains the right method.

So `run_full_experiment.sh` measures the knee in stage 4 and feeds it to stage 5 in the same allocation, on the same node, minutes apart. `pick_rates.py` locates the knee from the server's behaviour, as the first rate where the median time to the first token rises above three times its low-rate baseline or the error rate rises above 1%. It also reports when no rate saturated, rather than labelling the top of the ladder "overload" on faith. The first version of that criterion was wrong, as the next entry explains.

Stage 1 is a hard gate. If streaming isn't verified as non-buffering against a live replica, the job exits before spending node-hours on a matrix whose headline metric would be meaningless.

---

## 2026-09-16: The first knee was a measurement artefact, and so was the first "-dirty"

### A false knee

Job 5215612 calibrated at 2 to 32 requests per second and put the knee at 4, using the rule "achieved throughput below 0.9 times the offered rate". However, it computed the achieved rate as completions over the total elapsed time, including the time to drain the last requests, so on short phases that rate falls as the offered rate rises, whether or not anything saturates. The same artifacts show the median time to the first token flat at 0.07 to 0.10 seconds, with no errors across the whole ladder, so the fleet was never loaded at all. The matrix it would have fed would have compared four routing policies on an idle fleet.

The same job had two more problems. It computed its end time with `date -d "+5:59:30"`, which `date` can't parse, and an `|| echo 0` fallback quietly turned that into a 600-second budget, so only eight phases ran.

I fixed all three. The knee now compares the median time to the first token with the lowest-rate baseline, together with the error rate. I chose the median because the first-token time is measured from the scheduled arrival, which is the correct open-loop method, so a client stall lands directly in the first-token time, but only for the few requests it hits. In fact, the 95th percentile was highest at the lowest rates in job 5215612. In addition, the achieved rate is now measured over the dispatch window only, and a rate that the generator couldn't actually offer is reported and excluded rather than promoted to a knee. Lastly, the ladder is wider, and each phase's length now grows with the rate, so that a backlog has time to form.

Job 5216059 then produced this calibration (artifacts in `results/calibration/5216059/`).

| offered (req/s) | achieved (req/s) | median first token (s) | 95th percentile (s) | error rate |
|---|---|---|---|---|
| 4 | 3.78 | 0.071 | 0.417 | 0.000 |
| 8 | 8.18 | 0.068 | 0.178 | 0.000 |
| 16 | 16.01 | 0.092 | 0.739 | 0.000 |
| 32 | 31.33 | 0.099 | 2.623 | 0.000 |
| 48 | 47.44 | 0.340 | 9.912 | 0.190 |
| 64 | 61.43 | 0.195 | 16.057 | 0.242 |
| 96 | 75.01 | 23.410 | 58.375 | 0.323 |
| 128 | 123.73 | 38.635 | 85.760 | 0.348 |

The knee is at 48 requests per second, where the median first token reaches 0.34 seconds against a 0.07 second baseline and admission control sheds 19% of requests. So the matrix ladder became 4, 16, 48 and 128. At 96 requests per second the generator only managed 75, so that row was excluded from the knee search, and it didn't affect the ladder. Every row is flagged `degraded`, because more than 2% of requests were sent over 0.25 seconds late, even in the idle 4 requests per second row. So client-side scheduling lag is present at all rates, and it's part of why the 95th percentile is noisy even when the fleet is idle. Past the knee, the latency percentiles describe only the admitted requests, and the error rate is the other half of the overload result.

One weakness remains. The keep-up test compares a realised Poisson dispatch rate with 0.9 times the nominal rate, and with about 120 requests in a low-rate phase, the realised rate alone varies by roughly 9%, so a low rate could be excluded by chance. It didn't happen here.

### A spurious "-dirty"

The matrix artifacts from job 5216059 record their commit with a `-dirty` suffix, while that job's stage-1 verification artifact records the same commit as clean. No tracked file changed during the job, because `git status --porcelain` was counting the job's own new result files. The artifacts stay as written, since they're append-only, and the code at that commit is exactly what ran. `provenance.git_sha` now passes `--untracked-files=no`.

Two later failure trials (job 5216528) are also marked dirty, because of uncommitted edits to the analysis scripts while those trials started. The failure driver itself was committed at that point. The other ten trials and the 216-prefix matrix (job 5216974) are clean.

---

## Design decisions that aren't obvious from the code

### The in-flight counter lives inside the response stream

A request is in flight for as long as its response is still streaming, which is long after the axum handler returns. So `InflightGuard` moves into the state of the relay stream and is dropped with it, and the count goes down on normal completion, on an upstream error and on a client disconnect alike. If the handler decremented it instead, the router would report an idle-looking fleet while every replica was saturated, and `least_loaded` would route on a number that was always zero. That failure produces no error and no crash, only a plausible dataset in which least loaded and round robin are inexplicably identical.

### `least_loaded` rotates its tiebreak

On an idle fleet every replica reads zero load, and a tiebreak by lowest index would send all traffic to replica 0. That would read as a catastrophic fairness result while being entirely an artefact of the tiebreak. It has a regression test (`an_idle_fleet_is_spread_evenly_rather_than_piled_on_replica_zero`), because the fairness metric is exactly the instrument that would otherwise have presented this bug as a finding about load-based routing.

### Rendezvous hashing, not modulo or a ring

Affinity placement ranks the replicas by their highest-random-weight score over stable indices, so when a replica is taken out, only the keys that mapped to it move. A scheme based on the key modulo the number of healthy replicas would remap almost every key the moment that number changed. During the failure experiment, that would flush every replica's prefix cache at once and turn a recovery measurement into a measurement of mass cache invalidation. `hash.rs` tests this property directly, rather than asserting it in a comment.

### The minimum capacity on the bounded policy

Without a floor, an idle fleet has a total load near zero, so the bound rounds up to 1 and the bounded policy would spill on the second concurrent request for any key. It would then behave like round robin exactly when the fleet is underloaded, which is where affinity is cheapest and the cache benefit should be largest, and the comparison between the bounded and unbounded policies would measure the floor rather than the bound. `min_capacity` prevents that.

### Open loop, with latency measured from the scheduled arrival

A closed-loop generator can't express overload, because as the fleet slows down, the offered load falls with it and the system settles at a stable point, so every "overload" arm silently becomes a different, lower load. Here, requests arrive as a Poisson process on the wall clock, and latency is measured from each request's scheduled arrival, rather than from when the client got round to sending it, so generator lag counts as the real client-visible latency that it is. Each phase records its own distribution of send delays and is marked `degraded` if the driver fell behind, because in that case the phase measured the driver, not the fleet.

### Cumulative counters are differenced per phase

vLLM's `vllm:prefix_cache_hits_total` and `vllm:prefix_cache_queries_total` (version 0.27.1 dropped the `gpu_` part of their names) and the router's own counters keep counting for the life of the process, while the matrix deliberately runs many phases against one long-lived fleet. So reading a counter at the end of a phase gives that phase plus every phase before it. llm-inference-bench ran into this, and it produced a result that looked like data, because a phase with a high sharing ratio that followed one with none read low, which made the comparison depend on the order of the phases.

Here, the harness reads the counters before and after every phase and takes the difference. A negative difference is treated as evidence that the server restarted rather than as a measurement, and a missing counter is reported as absent rather than zero. An absent measurement and a measured zero are different facts, and llm-inference-bench's prefix-cache bug was only diagnosable because of that difference.

---

## 2026-10-24: A withdrawn citation, site details removed, and the recorded commits

### The co-location caveat

Several places in this repository cited llm-inference-bench's measurement that packing four single-GPU replicas onto one node costs 150% on the median first token and 34% of throughput: the README's hardware caveat, the entry above on why the experiment became one job, and comments in `pick_rates.py`, `calibrate_rates.sh` and `lib_replicas.sh`. The matrix and full-experiment scripts also wrote that claim into each artifact's topology note. That measurement was withdrawn on 2026-10-24, because its packing script gave all four replicas the same port, so all four load generators measured one server on one GPU. Measured again with separate ports and the same vLLM 0.27.1 image, four copies on one node performed the same as one copy alone, within half a percent.

This repository was never affected by that bug, because every replica here runs on its own port (8100 plus its index) and writes its own log. The routing comparison never depended on the caveat either, because every policy ran against the same fleet in the same job. So no result here changes. I corrected the README, the entry above and the comments, and the scripts no longer write the claim into new artifacts. The existing artifacts keep their original topology notes, because they're a record of what was written at the time.

### Site details removed

Before publishing, I replaced site-specific strings in the raw and calibration artifacts, such as host names, file paths and account names, with neutral placeholders like `${IR_DATA}`, `${HOME}` and `node2061`, using `scripts/scrub_private.py`. Host names also appear inside some metric names, as a replica's placement label, and those changed in the same way. A check that compares every value by position confirmed that every number, every flag and the structure of all 391 files are identical to what was written. In addition, the scripts no longer name an account or absolute paths, because those now come from a git-ignored `site.env`.

### The recorded commits

Every artifact records the git commit it was produced at. This repository's history was rewritten in September 2026 to correct commit metadata, so those commits now have new identifiers, while the project's files in each commit are unchanged. The `-dirty` suffix means the tree had uncommitted changes when the artifact was written, as described above.

| commit recorded in artifacts | same tree in this history |
|---|---|
| `f0289c3a0b3e89d5ab0b0460088449f8928fc5c5` (with and without `-dirty`) | `2525671` |
| `efe9130005e346179369992cbf01bbf8bd5c14a3` | `8881bc8` |
| `c921ce89f398b320919bef280dfb969e353ca73b-dirty` | `e972a28` |
| `7782bbf5b9544461d3dea2ef9e60a65025aea809` | `75a5419` |
| `451d2a880735197e4c72259184ceee6b456204da-dirty` | `cd564e5` |
