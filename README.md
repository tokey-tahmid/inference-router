# inference-router

A prefix-aware request router, in Rust, in front of a fleet of LLM inference
replicas, and a measurement harness that compares four routing policies under
controlled load on real A100 hardware at ICHEC.

> **Status: measured.** Four routing policies against a live 4-replica vLLM
> fleet, N=3, in two workload shapes (256 + 96 phases), plus 12 replica
> kill-and-restart trials. Every number below traces to a committed artifact
> under `results/raw/`; the tables are generated into `results/RESULTS.md`.
> `docs/PROGRESS.md` has the definition-of-done audit.

---

## Why prefix-aware routing matters at fleet scale

A serving engine's prefix cache is a **per-replica** structure. vLLM will reuse
the KV blocks of a system prompt it has already prefilled, but only on the
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

With `N=240` and `R=4` that is the difference between 236/240 and 239/240, not
much. But the interesting regime is not one prefix. It is a *tenant mix*: many
distinct prefixes, each used by a handful of requests, arriving interleaved.
There, round robin shatters every prefix across all `R` replicas and each one
pays `R` cold prefills instead of one, while the KV blocks it evicts to do so
are the ones the next request wanted. Prefix affinity converts a fleet-wide
scatter into `R` independent, coherent working sets.

Prefill is where this is spendable. P1 measured TTFT scaling essentially
linearly with input length past ~1024 tokens on this hardware (0.0714 s at 1024
tokens, 0.5725 s at 8192, vLLM 0.27.1 at TP=1, see
`../llm-inference-bench/docs/notes.md`). A cached prefix removes that work
outright, so the effect lands squarely on time-to-first-token.

### And why it is not free

Affinity and balance are in direct tension, and this is the part a router
demo usually skips.

If one prefix is hotter than the others (one popular system prompt, one large
tenant), pure affinity sends all of its traffic to one replica while the rest
idle. The cache hit rate goes up and the tail latency goes up *with* it, because
a saturated replica queues. The policy optimises the metric it was given and
degrades the metric anyone actually cares about.

So this repository implements both the naive policy and a bounded one, and the
experiment is designed to find the crossover rather than to assert it:

- **`prefix_affinity`**: deliberately blind to load. Its failure mode is the
  point, not an oversight.
- **`prefix_affinity_bounded`**: consistent hashing with bounded loads. A
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
buffering proxy would erase exactly the signal this project exists to measure , 
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
healthy count changed, flushing every replica's prefix cache at once during the
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
under one long-lived fleet. The hazard that creates (a phase inheriting the
previous phase's warm prefix caches) is fatal here rather than cosmetic,
because a warm cache is precisely what prefix affinity is claimed to produce.
Every phase therefore resets every replica's prefix cache, waits for the fleet
to drain, and **records whether both actually happened**. A phase whose reset
failed is marked `degraded` and excluded in analysis rather than quietly
reported.

(`/reset_prefix_cache` needs `VLLM_SERVER_DEV_MODE=1` on vLLM 0.27.1, it is
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
| `round_robin` | rotate over the healthy set | none |
| `least_loaded` | fewest in-flight + queued; **rotating** tiebreak | none |
| `prefix_affinity` | highest rendezvous score for the prompt-prefix hash | unhealthy primary → next-ranked healthy; no prompt → least loaded |
| `prefix_affinity_bounded` | as above, while the primary is under `max(min_capacity, ceil(factor x mean_load))` | over bound → spill down the ranking; all over → least loaded |

The rotating tiebreak in `least_loaded` is not a detail. On an idle fleet every
replica reads zero load, and a lowest-index tiebreak would send *all* traffic to
replica 0, a catastrophic-looking fairness result that is entirely an artefact
of the tiebreak. `src/policy/least_loaded.rs` has a regression test for it.

---

## Running it

Everything runs as a Slurm batch job on the `gpu` partition under account
`p201466`. The login node has no `module`, no `cargo` and only Python 3.6, so a
batch job is the first opportunity to compile or run anything.

```bash
sbatch slurm/build.sh                   # build, rustfmt, clippy -D warnings, tests
sbatch slurm/run_full_experiment.sh     # SSE gate, density, fleet, calibration, matrix
sbatch --time=03:30:00 slurm/run_matrix.sh --rates 8 32 48 \
    --ratios 0.6 0.9 --num-requests 960 --num-prefix-groups 216   # tenant mix
sbatch slurm/run_failure_mode.sh 16     # kill and relaunch a replica, per policy
sbatch slurm/run_analysis.sh            # regenerate every figure and table
```

`slurm/verify_single_replica.sh` is a gate, not a formality. It checks the four
things that would invalidate every subsequent number if they were silently
false: token-ID prompts round-tripping exactly **through the router**,
`ignore_eos` being honoured, responses genuinely streaming, and the
prefix-cache endpoint and counters existing under the names vLLM 0.27.1 uses.

`run_full_experiment.sh` runs those gates and the density probe in the same
allocation as the matrix, then calibrates the rate ladder against the fleet it
is about to measure. `slurm/run_matrix.sh` refuses to run without a measured
rate ladder. The rate
axis has to span underload to overload *for this fleet*, and that is a measured
quantity, not a guess.

---

## Hardware

MeluXina, partition `gpu`. Per node: 4x NVIDIA A100-SXM4-40GB, 128 logical cores
(4 sockets x 16 c x 2 t), 480 GB RAM, diskless. The experiment nodes reported
driver 595.91.07 (recorded in every artifact).

**Fleet under test:** 4 replicas, one TP=1 vLLM replica per GPU at
`gpu_memory_utilization=0.90`, each pinned to its GPU's NUMA-local cores, all
on one node, with the router and load generator on the same node over
loopback. Router admission: at most 64 in-flight requests per healthy replica,
excess rejected with 429. Full NVLink `NV4` mesh between
every GPU pair, there is no intra-node PCIe path between GPUs, so no
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
tokens per request**, occupying 36,126 MiB of the card.

Measured (job `5216059`, stage 2, `results/raw/density_probe_5216059.json`):
**two replicas do fit** on one card at `gpu_memory_utilization=0.45` each,
launched sequentially, but each gets a **24,608-token KV cache, 6.01x max
concurrency at 4,096 tokens**. That is about 14.5x less cache per replica than
the solo configuration, and the pair together holds about 7% of what one solo
replica holds.

That is not a smaller version of the same fleet. A 2-per-card fleet cannot admit
the same concurrency or the same context lengths, so the two topologies are two
different **serving envelopes**, not two ways of serving one workload. Pooling
them into one routing figure would compare policies across configurations that
differ in what they can physically admit, and part of any measured difference
would be a difference in what the replicas were allowed to accept in the first
place.

So the experiment matrix runs on the **1-replica-per-card** fleet, and replica
density is reported separately as a capacity finding.

### A caveat that applies to every absolute number here

P1 measured that **co-locating four single-GPU replicas on one node is not
free**: +150.7% TTFT p50 and -34.1% output throughput versus the same replica
running alone on a node (job `5162712`). The binding resource is host-side
bandwidth on the prefill path, not the NVLink mesh.

This fleet is co-located by design, that *is* the topology under test. So the
absolute latencies here are fleet numbers, not isolated-replica numbers. The
policy comparison is unaffected, because every policy runs against the same
co-located fleet in the same job, and that is the comparison this project makes.

---

## Results

Full tables: [`results/RESULTS.md`](results/RESULTS.md). Figures:
`results/figures/`. Medians over N=3 repetitions with (min-max). TTFT is
measured from each request's *scheduled* open-loop arrival.

### Calibration: where this fleet saturates

Rate calibration in the same allocation (`results/calibration/5216059/`):
median TTFT stays at 0.07-0.10 s with zero errors up to 32 req/s. At 48 req/s
it reaches 0.340 s and the router rejects 19% of requests, so the matrix ran at
**4, 16, 48 and 128 req/s** (underloaded, approaching saturation, at the
saturation point, far past it). Past saturation, latency percentiles describe
*admitted* requests only and must be read with the error rate: at 128 req/s,
plain prefix affinity shows a "faster" TTFT than round robin only because it
rejects more (error rate 0.69 vs 0.34 at ratio 0.9).

### One hot prefix: affinity buys nothing and concentrates load

Workload: 1,024-token prompts, 128-token outputs, one shared 512-token system
prompt on a fraction of requests (the shared-prefix ratio). At ratio 0.9:

| policy | fleet hit rate (4 / 16 req/s) | peak / mean replica load | TTFT p50 at 16 req/s | error rate at 48 req/s |
|---|---|---|---|---|
| round robin | 0.440 / 0.444 | 1.00 | 0.067 s | 0.102 |
| least loaded | 0.440 / 0.444 | 1.00-1.01 | 0.067 s | 0.131 |
| prefix affinity | 0.448 / 0.448 | 3.69-3.72 | 0.151 s | 0.416 |
| prefix affinity (bounded) | 0.440 / 0.444 | 1.26-1.90 | 0.070 s | 0.101 |

With a single prefix, round robin already reaches the no-eviction bound: it pays
four cold prefills instead of one, out of thousands of requests. Plain prefix
affinity therefore gains nothing on cache hits (0.448 vs 0.444), and what it
does instead is send almost all traffic to one replica. The hottest replica
carries 3.7x the mean load, median TTFT more than doubles at 16 req/s (a load
the fleet handles easily), and at the saturation point it sheds 42% of
requests against round robin's 10%. **This is the load-imbalance failure mode,
measured.** The bounded policy caps it: peak/mean 1.21-1.26 at 16-48 req/s, median TTFT
at 16 req/s back to 0.070 s, and an error rate at 48 req/s matching round
robin's (0.101 vs 0.102), although its median TTFT there is still higher
(0.218 s vs 0.123 s).

This contradicts the expected headline that affinity helps at high sharing
ratios, for this workload. The next experiment shows why.

### Many prefixes (tenant mix): affinity wins on cache, with limits

Workload: 216 distinct 512-token prefixes, 960 requests per phase, so at ratio
0.9 each prefix is reused about four times. The no-eviction hit-rate bound is
0.3375 (0.1875 at ratio 0.6). At ratio 0.9:

| policy | fleet hit rate (8 req/s) | TTFT p50 at 8 req/s | TTFT p50 at 32 req/s | error rate at 48 req/s |
|---|---|---|---|---|
| round robin | 0.144 | 0.092 s | 0.110 s | 0.209 |
| least loaded | 0.140 | 0.093 s | 0.360 s | 0.222 |
| prefix affinity | 0.338 | 0.069 s | 0.148 s | 0.149 |
| prefix affinity (bounded) | 0.263 | 0.090 s | 0.276 s | 0.184 |

Here round robin scatters each prefix's handful of reuses across four replicas,
and most of them land on a replica that has never seen that prefix. Prefix
affinity reaches the bound exactly (0.338) with load still even (peak/mean
1.05), and at 8 req/s cuts median TTFT by 25% (0.069 s vs 0.092 s). **Cache
locality is a fleet-level property: the same engines with the same caches
served 2.3x more prefix tokens from cache purely because of where requests
were sent.**

The latency win does not survive load. At 32 req/s affinity's median TTFT is
1.35x round robin's, and the bounded policy (which spills to less-loaded
replicas and so loses some locality, hit rate 0.263-0.309) is slower still.
At 48 req/s the medians swing widely between repetitions (round robin
0.374-2.872 s), so the TTFT ratios in `RESULTS.md` at that rate are not a
reliable ranking. Affinity did reject fewer requests there (0.149 vs 0.209).

### Replica failure and recovery

Twelve trials (`results/RESULTS.md`, job 5216528): 16 req/s, ratio 0.6, one of
four replicas SIGKILLed 30 s in (whole process tree) and relaunched 30 s later,
a fresh fleet per trial. Health checks every 500 ms, ejection after two
consecutive failures.

| policy | detection (s) | relaunch to readmission (s) | error rate | killed replica's share after readmission (median) |
|---|---|---|---|---|
| round robin | 1.23 (0.99-1.52) | 109.8 (109.8-116.3) | 0.0018 | 0.250 |
| least loaded | 1.04 (0.99-1.12) | 113.9 (111.3-114.5) | 0.0034 | 0.251 |
| prefix affinity | 1.14 (1.04-1.18) | 112.8 (110.9-115.1) | 0.0006 | 0.099 |
| prefix affinity (bounded) | 0.74 (0.70-0.75) | 113.2 (111.9-119.4) | 0.0025 | 0.292 |

Every failed request is attributable to the killed replica: either its stream
broke mid-response, or the router returned a 502 naming it for a request
dispatched between the kill and the ejection. **No request failed after
ejection or after readmission, in any trial.** Readmission is dominated by
vLLM cold start (~110 s), and every policy puts the replica back to its
pre-kill share. Those shares differ by policy, and that explains the error
rates. Under plain prefix affinity the killed replica is not the shared
prefix's primary, so it only ever carries its hash share of the 40% of requests
without the shared prefix (0.106-0.108 before the kill, 0.099-0.104 after).
It had little in flight to lose, hence the lowest error rate. The bounded
policy spills shared-prefix traffic onto it, so it carries about 0.29.

### Streaming under load

Every matrix phase delivered a median of 130 SSE chunks per 128-token request
at every rate. At 4-16 req/s the median TTFT/e2e ratio is 0.035-0.056. At
128 req/s it rises to 0.30-0.88 because queue wait dominates TTFT, not because
the proxy buffers: the chunk count is unchanged. The deterministic half of the
check is `tests/streaming.rs`.

### Measurement caveats

- **Client dispatch lag.** The Python load generator (unpinned, sharing the
  node with four replicas) missed its 0.25 s dispatch tolerance on more than
  2% of requests in most phases, including at 4 req/s. Those phases are kept
  and flagged. `RESULTS.md` also reports TTFT measured from actual dispatch and
  the lag p95, and the median TTFTs above barely move between the two clocks.
  At 128 req/s the generator lags by seconds, so that rate is "at least
  heavily overloaded" rather than exactly 128 req/s.
- **Admission control shapes overload.** The 429s are the router's own
  64-per-replica in-flight cap, a configuration choice, not an engine limit.
- **Co-location.** See the Hardware caveat; absolute latencies are fleet numbers.
- **Superseded runs.** Job 5215612's 8 matrix phases used an invalid rate
  ladder (see `docs/notes.md`) and are listed but not used.

## Reproducing

Every figure and table is regenerated by one command, `sbatch
slurm/run_analysis.sh`, which reads `results/raw/` and writes
`results/figures/` plus `results/RESULTS.md`. Nothing in either is transcribed
by hand.
