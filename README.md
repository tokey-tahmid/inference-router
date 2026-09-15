# inference-router

A prefix-aware request router, in Rust, in front of a fleet of LLM inference
replicas — and a measurement harness that compares four routing policies under
controlled load on real A100 hardware.

> **Status of the numbers.** Every measured value in this repository traces to a
> committed artifact under `results/raw/`. Anything not yet measured says
> "not measured" rather than carrying an estimate. See `docs/PROGRESS.md` for
> exactly what has and has not been run.

---

## Why prefix-aware routing matters at fleet scale

A serving engine's prefix cache is a **per-replica** structure. vLLM will reuse
the KV blocks of a system prompt it has already prefilled — but only on the
replica that prefilled it. Nothing in the engine knows that another replica,
one hop away, is holding the identical 512-token prefix in HBM.

That makes cache locality a property of the **routing decision**, not of the
engine. It is the part of the system the engine cannot optimise for you, and it
is invisible from inside any single replica.

The arithmetic is blunt. Take `N` requests that share one long system prompt and
a fleet of `R` replicas:

| routing | cold prefills of the shared prefix | best-case hit rate on those tokens |
|---|---|---|
| round robin | `R` | `1 - R/N` |
| perfect prefix affinity | `1` | `1 - 1/N` |

With `N=240` and `R=4` that is the difference between 236/240 and 239/240 — not
much. But the interesting regime is not one prefix. It is a *tenant mix*: many
distinct prefixes, each used by a handful of requests, arriving interleaved.
There, round robin shatters every prefix across all `R` replicas and each one
pays `R` cold prefills instead of one, while the KV blocks it evicts to do so
are the ones the next request wanted. Prefix affinity converts a fleet-wide
scatter into `R` independent, coherent working sets.

Prefill is where this is spendable. P1 measured TTFT scaling essentially
linearly with input length past ~1024 tokens on this hardware (0.0714 s at 1024
tokens, 0.5725 s at 8192, vLLM 0.27.1 at TP=1 — see
`../llm-inference-bench/docs/notes.md`). A cached prefix removes that work
outright, so the effect lands squarely on time-to-first-token.

### And why it is not free

Affinity and balance are in direct tension, and this is the part a router
demo usually skips.

If one prefix is hotter than the others — one popular system prompt, one large
tenant — pure affinity sends all of its traffic to one replica while the rest
idle. The cache hit rate goes up and the tail latency goes up *with* it, because
a saturated replica queues. The policy optimises the metric it was given and
degrades the metric anyone actually cares about.

So this repository implements both the naive policy and a bounded one, and the
experiment is designed to find the crossover rather than to assert it:

- **`prefix_affinity`** — deliberately blind to load. Its failure mode is the
  point, not an oversight.
- **`prefix_affinity_bounded`** — consistent hashing with bounded loads. A
  request takes its affinity primary only while that replica is under a capacity
  bound derived from the fleet's mean load; otherwise it spills down the
  rendezvous ranking.

The router reports *why* it made each decision (`x-router-route-reason`, and a
`router_route_decisions_total` metric), so a latency difference between the two
can be attributed to how often the bound actually fired rather than guessed at.

---

## Design notes worth reading

### The streaming passthrough is the hard part

The obvious proxy implementation buffers:

```rust
let body = upstream.text().await?;   // wrong
Ok((StatusCode::OK, body))
```

It *works*. Every request succeeds, the JSON is byte-identical, the tests pass.
What it destroys is time-to-first-token: the client receives nothing until the
replica has generated the **last** token, so TTFT collapses onto end-to-end
latency. At P1's measured 12.3 ms/token, a 128-token generation turns a ~70 ms
first token into a ~1.6 s one.

Since prefix affinity's entire claimed benefit is a *prefill* effect, a
buffering proxy would erase exactly the signal this project exists to measure —
while still producing a complete, clean, plausible dataset. That is the most
dangerous class of bug in the whole repository.

`src/proxy.rs` consumes the upstream body as a `Stream<Item = Bytes>` and hands
it to `axum::body::Body::from_stream` without ever collecting it. Three details
that a naive passthrough gets wrong are handled explicitly and documented in
that file: no total-request timeout (only an inter-chunk read timeout), the
in-flight guard living *inside the response stream* so accounting survives
client disconnects, and hop-by-hop plus `Content-Length` headers being dropped
rather than forwarded onto a re-framed body.

It is verified two ways, because one is not enough:

- **Deterministically**, in `tests/streaming.rs`: a mock upstream emits one
  chunk then stalls, and the test fails unless the first chunk reached the
  client before the stall ended. Also covers chunk-boundary preservation,
  in-flight accounting across the stream's lifetime, client disconnect, and
  concurrency.
- **Under real load**, in every experiment artifact: each phase records the
  distribution of TTFT/e2e per request. A buffering proxy pins that ratio at
  1.0. `results/figures/streaming_non_buffering.png` plots it.

### Rendezvous hashing, not a hash ring

Affinity placement uses highest-random-weight hashing over **stable replica
indices**. When a replica is ejected, only the keys that mapped to *it* move,
and they redistribute evenly over the survivors.

A `key % n_healthy` scheme would remap essentially every key the moment the
healthy count changed — flushing every replica's prefix cache at once during the
failure experiment, and turning a recovery measurement into a measurement of
mass cache invalidation. `src/policy/hash.rs` asserts the property directly.

The hash is FNV-1a rather than `DefaultHasher`, which is explicitly not stable
across Rust releases; an affinity key that meant something different after a
toolchain bump would silently break comparability between two runs of the same
experiment.

### One fleet, four policies, switched at runtime

Each vLLM replica takes minutes to stream 15 GB of weights off Lustre.
Relaunching the fleet per policy would cost more node-hours than the measurement
and, worse, would make the comparison one of four *separately warmed fleets*
rather than of four routing rules.

So the router exposes `POST /admin/policy` and the driver switches the rule
under one long-lived fleet. The hazard that creates — a phase inheriting the
previous phase's warm prefix caches — is fatal here rather than cosmetic,
because a warm cache is precisely what prefix affinity is claimed to produce.
Every phase therefore resets every replica's prefix cache, waits for the fleet
to drain, and **records whether both actually happened**. A phase whose reset
failed is marked `degraded` and excluded in analysis rather than quietly
reported.

(`/reset_prefix_cache` needs `VLLM_SERVER_DEV_MODE=1` on vLLM 0.27.1 — it is
registered only behind that flag. P1 found this the honest way, by recording the
HTTP status instead of assuming success, and getting a 404.)

### Open-loop load, not closed-loop

The experiment axis is *request rate*, so arrivals are a Poisson process
dispatched on wall clock regardless of whether earlier requests finished.

A closed-loop generator cannot express overload at all: as the fleet slows, the
offered load falls with it and the system self-throttles into a stable point.
Every "overload" arm would silently become a different, lower load. Latency is
measured from the **scheduled** arrival, not from when a worker got around to
sending, so generator lag counts as the real client-visible latency it is. Each
phase records its own `send_delay` distribution and is marked degraded if the
driver itself fell behind.

---

## Architecture

```
driver (Python, open-loop Poisson)         analysis/
  imports P1's workload generator            reads results/raw/ only
        |                                    writes figures + RESULTS.md
        v
   inference-router  (Rust, tokio + axum)
        |  /v1/completions, /v1/chat/completions   SSE passthrough
        |  /metrics (Prometheus), /admin/policy, /admin/fleet
        |  health checking, ejection + recovery, bounded admission
        v
   r0        r1        r2        r3        vLLM 0.27.1, TP=1 each
   GPU0      GPU1      GPU2      GPU3      Qwen2.5-7B-Instruct bf16
   NUMA3     NUMA1     NUMA7     NUMA5     NUMA-pinned, full NVLink NV4 mesh
```

| component | file |
|---|---|
| routing trait + policy registry | `src/policy/mod.rs` |
| rendezvous hashing, affinity keys | `src/policy/hash.rs` |
| the four policies | `src/policy/{round_robin,least_loaded,prefix_affinity}.rs` |
| SSE passthrough proxy | `src/proxy.rs` |
| admission control / backpressure | `src/admission.rs` |
| health checking, ejection, recovery | `src/health.rs` |
| replica state and in-flight accounting | `src/replica.rs` |
| Prometheus exposition | `src/metrics.rs` |
| load generation (open loop) | `driver/ir_driver/loadgen.py` |
| phase orchestration + artifacts | `driver/ir_driver/experiment.py` |
| fairness metrics | `driver/ir_driver/stats.py` |

### Routing policies

| policy | rule | fallback |
|---|---|---|
| `round_robin` | rotate over the healthy set | — |
| `least_loaded` | fewest in-flight + queued; **rotating** tiebreak | — |
| `prefix_affinity` | highest rendezvous score for the prompt-prefix hash | unhealthy primary → next-ranked healthy; no prompt → least loaded |
| `prefix_affinity_bounded` | as above, while the primary is under `max(min_capacity, ceil(factor x mean_load))` | over bound → spill down the ranking; all over → least loaded |

The rotating tiebreak in `least_loaded` is not a detail. On an idle fleet every
replica reads zero load, and a lowest-index tiebreak would send *all* traffic to
replica 0 — a catastrophic-looking fairness result that is entirely an artefact
of the tiebreak. `src/policy/least_loaded.rs` has a regression test for it.

---

## Running it

Everything runs as a Slurm batch job on the `gpu` partition under account
`p201466`. The login node has no `module`, no `cargo` and only Python 3.6, so a
batch job is the first opportunity to compile or run anything.

```bash
sbatch slurm/build.sh                   # build, rustfmt, clippy -D warnings, tests
sbatch slurm/probe_replica_density.sh   # how many 7B replicas fit on a 40 GB card
sbatch slurm/verify_single_replica.sh   # gate: one live replica, end to end
sbatch slurm/calibrate_rates.sh         # measure the fleet's saturation point
sbatch slurm/run_matrix.sh --rates <measured ladder>
sbatch slurm/run_failure_mode.sh        # kill a replica mid-load
sbatch slurm/run_analysis.sh            # regenerate every figure and table
```

`slurm/verify_single_replica.sh` is a gate, not a formality. It checks the four
things that would invalidate every subsequent number if they were silently
false: token-ID prompts round-tripping exactly **through the router**,
`ignore_eos` being honoured, responses genuinely streaming, and the
prefix-cache endpoint and counters existing under the names vLLM 0.27.1 uses.

`slurm/run_matrix.sh` refuses to run without a measured rate ladder. The rate
axis has to span underload to overload *for this fleet*, and that is a measured
quantity, not a guess.

---

## Hardware

MeluXina, partition `gpu`. Per node: 4x NVIDIA A100-SXM4-40GB, 128 logical cores
(4 sockets x 16 c x 2 t), 480 GB RAM, diskless. Full NVLink `NV4` mesh between
every GPU pair — there is no intra-node PCIe path between GPUs, so no
NVLink-vs-PCIe contrast is observable within a node.

Backend and model identifiers are copied verbatim from P1's
`provision.lock.json` into every artifact and are never retyped:

| | |
|---|---|
| backend | vLLM 0.27.1 (`sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`) |
| model | `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28` |
| precision | bf16 |

### Replica density constrains the *workload*, not just the topology

The spec asks whether two 7B replicas fit per 40 GB card. The answer matters for
more than how many boxes to draw in the diagram.

Measured (job `5213921`, arm A): one replica at `gpu_memory_utilization=0.90`
gets **19.09 GiB of KV cache, 357,520 tokens, 87.29x max concurrency at 4,096
tokens per request**, occupying 36,126 MiB of the card. Two replicas each asking
for half the card would have roughly 3 GB apiece once 15.2 GB of bf16 weights
are subtracted: an order of magnitude less cache each.

That is not a smaller version of the same fleet. A 2-per-card fleet cannot admit
the same concurrency or the same context lengths, so the two topologies are two
different **serving envelopes**, not two ways of serving one workload. Pooling
them into one routing figure would compare policies across configurations that
differ in what they can physically admit, and part of any measured difference
would be a difference in what the replicas were allowed to accept in the first
place.

So the experiment matrix runs on the **1-replica-per-card** fleet, and replica
density is reported separately as a capacity finding. (The two-per-card
measurement itself is still pending; see `docs/PROGRESS.md`.)

### A caveat that applies to every absolute number here

P1 measured that **co-locating four single-GPU replicas on one node is not
free**: +150.7% TTFT p50 and -34.1% output throughput versus the same replica
running alone on a node (job `5162712`). The binding resource is host-side
bandwidth on the prefill path, not the NVLink mesh.

This fleet is co-located by design — that *is* the topology under test. So the
absolute latencies here are fleet numbers, not isolated-replica numbers. The
policy comparison is unaffected, because every policy runs against the same
co-located fleet in the same job, and that is the comparison this project makes.

---

## Results

See `results/RESULTS.md` (generated) and `results/figures/`. Status of each
experiment, including what has not been run, is in `docs/PROGRESS.md`.

## Reproducing

Every figure and table is regenerated by one command — `sbatch
slurm/run_analysis.sh` — which reads `results/raw/` and writes
`results/figures/` plus `results/RESULTS.md`. Nothing in either is transcribed
by hand.
