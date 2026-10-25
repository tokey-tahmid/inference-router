# inference-router

This is a prefix-aware request router, written in Rust, that sits in front of a fleet of LLM inference replicas, together with a harness that compares four routing policies under controlled load on real A100 hardware.

I measured all four policies against a live fleet of four vLLM replicas, three times each, in two workload shapes (256 phases with one shared prefix and 96 phases with many), plus twelve trials where I killed a replica and restarted it under load. Every number below traces to a committed artifact under `results/raw/`, and the full tables are generated into `results/RESULTS.md`.

> **Correction, 2026-10-24.** An earlier version of this README warned that every absolute latency here was inflated by running four replicas on one node, citing a measurement from llm-inference-bench. That measurement was withdrawn, because its four replicas shared one port and so measured one server. [Co-location](#co-location) explains what changed. The routing results themselves are unaffected.

## Why prefix-aware routing matters at fleet scale

A serving engine's prefix cache belongs to one replica. vLLM will reuse the KV blocks of a system prompt it has already processed, but only on the replica that processed it, and nothing in the engine knows that another replica one hop away holds the identical 512-token prefix in its memory. So cache locality is a property of the routing decision rather than of the engine. It's the part of the system the engine can't optimise for you, and it's invisible from inside any single replica.

The arithmetic is simple. Take N requests that share one long system prompt, and a fleet of R replicas.

| routing | cold prefills of the shared prefix | best possible hit rate on those tokens |
|---|---|---|
| round robin | R | 1 - R/N |
| perfect prefix affinity | 1 | 1 - 1/N |

With 240 requests and four replicas, that's the difference between 236 out of 240 and 239 out of 240, which is not much. However, the interesting case isn't one prefix. It's a mix of tenants, with many distinct prefixes that are each used by a handful of requests arriving interleaved. There, round robin scatters every prefix across all four replicas, so each prefix pays four cold prefills instead of one, and the blocks it evicts to make room are the ones the next request wanted. Prefix affinity turns that fleet-wide scatter into four separate, coherent working sets.

Prefill is where the saving shows. llm-inference-bench measured the time to the first token growing almost linearly with input length beyond about 1,024 tokens on this hardware, from 0.07 seconds at 1,024 tokens to 0.57 seconds at 8,192, with vLLM 0.27.1 on one GPU. A cached prefix removes that work entirely, so the effect lands squarely on the time to the first token.

### And why it isn't free

Affinity and balance pull in opposite directions, and this is the part a router demo usually skips. If one prefix is hotter than the others, such as one popular system prompt or one large tenant, pure affinity sends all of its traffic to one replica while the others sit idle. The cache hit rate goes up, but the tail latency goes up with it, because a saturated replica queues. In other words, the policy optimises the metric it was given and degrades the one people actually care about.

So this repository implements both the naive policy and a bounded one, and the experiment is designed to find the crossover rather than assume it. `prefix_affinity` is deliberately blind to load, and its failure mode is the point rather than an oversight. `prefix_affinity_bounded` uses consistent hashing with bounded loads, so a request goes to its preferred replica only while that replica is under a capacity bound derived from the fleet's mean load, and otherwise it moves down the ranking. The router also reports why it made each decision, in the `x-router-route-reason` header and the `router_route_decisions_total` metric, so a latency difference between the two can be traced to how often the bound actually fired rather than guessed at.

## Design notes worth reading

### The streaming passthrough is the hard part

The obvious proxy implementation buffers the whole response.

```rust
let body = upstream.text().await?;   // wrong
Ok((StatusCode::OK, body))
```

It works, in the sense that every request succeeds, the JSON is byte-identical and the tests pass. What it destroys is the time to the first token, because the client receives nothing until the replica has generated the last token, so the first-token time collapses onto the end-to-end latency. At about 12 ms per token, a 128-token answer would turn a 70 ms first token into a 1.6 second one. Since prefix affinity's whole claimed benefit is a prefill effect, a buffering proxy would erase exactly the signal this project exists to measure, while still producing a complete, clean and plausible dataset. That makes it the most dangerous class of bug in the repository.

`src/proxy.rs` therefore consumes the upstream body as a stream of byte chunks and hands it straight to `axum::body::Body::from_stream`, without ever collecting it. It also handles three details that a naive passthrough gets wrong, each documented in that file. There is no timeout on the whole request, only on the gap between chunks. The in-flight counter lives inside the response stream, so the accounting survives a client disconnecting. Lastly, hop-by-hop headers and `Content-Length` are dropped rather than forwarded onto a body that has been re-framed.

I verify the streaming in two ways, because one isn't enough. Firstly, `tests/streaming.rs` checks it deterministically, using a mock upstream that sends one chunk and then stalls, and the test fails unless the first chunk reached the client before the stall ended. The same file covers chunk boundaries, in-flight accounting over the stream's lifetime, client disconnects and concurrency. Secondly, every experiment artifact records, for each request, the ratio of the time to the first token to the end-to-end latency. A buffering proxy would pin that ratio at 1.0, and `results/figures/streaming_non_buffering.png` plots it under real load.

### Rendezvous hashing, not a hash ring

Affinity placement uses highest-random-weight (rendezvous) hashing over stable replica indices. When a replica is taken out of service, only the keys that mapped to it move, and they spread evenly over the survivors. By contrast, a scheme based on the key modulo the number of healthy replicas would remap almost every key the moment that number changed. That would flush every replica's prefix cache at once during the failure experiment, turning a recovery measurement into a measurement of mass cache invalidation. `src/policy/hash.rs` tests the property directly.

The hash is FNV-1a rather than Rust's `DefaultHasher`, which is explicitly not stable across Rust releases. An affinity key that meant something different after a toolchain update would silently break the comparison between two runs of the same experiment.

### One fleet, four policies, switched at runtime

Each vLLM replica takes minutes to stream 15 GB of weights from the parallel filesystem. Relaunching the fleet for each policy would cost more node-hours than the measurement itself. Worse, it would compare four separately warmed fleets rather than four routing rules. So the router exposes `POST /admin/policy`, and the driver switches the rule on one long-lived fleet.

That creates its own hazard, because a phase could inherit the previous phase's warm prefix caches, and a warm cache is exactly what prefix affinity claims to produce. Therefore, every phase resets every replica's prefix cache, waits for the fleet to drain, and records whether both actually happened. A phase whose reset failed is marked `degraded` and excluded from the analysis rather than quietly reported. (On vLLM 0.27.1, the reset endpoint only exists when `VLLM_SERVER_DEV_MODE=1` is set. llm-inference-bench found this by recording the HTTP status instead of assuming success, and getting a 404.)

### Open-loop load, not closed-loop

The experiment's axis is the request rate, so requests arrive as a Poisson process on the wall clock, whether or not earlier requests have finished. A closed-loop generator can't express overload at all, because as the fleet slows down, the offered load falls with it and the system settles at a stable point. Every "overload" arm would silently become a different, lower load. In addition, latency is measured from each request's scheduled arrival, not from when a worker got round to sending it, so generator lag counts as the real client-visible latency that it is. Each phase records its own distribution of send delays and is marked degraded if the driver itself fell behind.

## Architecture

```
driver (Python, open-loop Poisson)           analysis/
  imports llm-inference-bench's workload       reads results/raw/ only
  generator                                    writes figures and RESULTS.md
        |
        v
   inference-router  (Rust, tokio and axum)
        |  /v1/completions, /v1/chat/completions   streaming passthrough
        |  /metrics (Prometheus), /admin/policy, /admin/fleet
        |  health checks, ejection and recovery, bounded admission
        v
   r0        r1        r2        r3        vLLM 0.27.1, one GPU each
   GPU0      GPU1      GPU2      GPU3      Qwen2.5-7B-Instruct, bf16
   NUMA3     NUMA1     NUMA7     NUMA5     pinned to local cores, full NVLink mesh
```

| component | file |
|---|---|
| routing trait and policy registry | `src/policy/mod.rs` |
| rendezvous hashing and affinity keys | `src/policy/hash.rs` |
| the four policies | `src/policy/{round_robin,least_loaded,prefix_affinity}.rs` |
| streaming passthrough proxy | `src/proxy.rs` |
| admission control and backpressure | `src/admission.rs` |
| health checks, ejection and recovery | `src/health.rs` |
| replica state and in-flight accounting | `src/replica.rs` |
| Prometheus metrics | `src/metrics.rs` |
| tracing with OpenTelemetry | `src/telemetry.rs` |
| open-loop load generation | `driver/ir_driver/loadgen.py` |
| phase orchestration and artifacts | `driver/ir_driver/experiment.py` |
| fairness metrics | `driver/ir_driver/stats.py` |

### Routing policies

| policy | rule | fallback |
|---|---|---|
| `round_robin` | rotate over the healthy replicas | none |
| `least_loaded` | fewest requests in flight or queued, with a rotating tiebreak | none |
| `prefix_affinity` | highest rendezvous score for the hash of the prompt's prefix | if the preferred replica is unhealthy, the next one in the ranking; with no prompt, the least loaded |
| `prefix_affinity_bounded` | as above, while the preferred replica's load is under `max(min_capacity, ceil(factor x mean_load))` | over the bound, move down the ranking; if all are over, the least loaded |

The rotating tiebreak in `least_loaded` matters more than it looks. On an idle fleet every replica reads zero load, so a tiebreak by lowest index would send all traffic to replica 0. That would look like a catastrophic fairness result, while being entirely an artefact of the tiebreak, so `src/policy/least_loaded.rs` has a regression test for it.

## Running it

Everything runs as Slurm batch jobs on a cluster's GPU partition. On the cluster I used, the login node had no module system, no `cargo` and only Python 3.6, so a batch job is the first chance to compile or run anything.

Firstly, copy `site.env.example` to `site.env` and set `IR_DATA` to a directory on project storage. It can be the same directory that llm-inference-bench uses, and that repository should be checked out next to this one, because this project reuses its vLLM image, its model weights and its lockfile. Then submit the jobs from the repository root, passing your Slurm account with `-A` or setting `SBATCH_ACCOUNT` once. Slurm writes each job's log to `logs/`.

```bash
cp site.env.example site.env                  # then edit it
sbatch -A <account> slurm/build.sh            # build, rustfmt, clippy -D warnings, tests
sbatch -A <account> slurm/run_full_experiment.sh   # streaming gate, density, fleet, calibration, matrix
sbatch -A <account> --time=03:30:00 slurm/run_matrix.sh --rates 8 32 48 \
    --ratios 0.6 0.9 --num-requests 960 --num-prefix-groups 216   # tenant mix
sbatch -A <account> slurm/run_failure_mode.sh 16   # kill and relaunch a replica, per policy
sbatch -A <account> slurm/run_analysis.sh     # regenerate every figure and table
```

`slurm/verify_single_replica.sh` is a real gate, not a formality. It checks four things that would invalidate every later number if they were silently false: that token-ID prompts survive the round trip through the router exactly, that `ignore_eos` is honoured, that responses genuinely stream, and that the prefix-cache endpoint and counters exist under the names vLLM 0.27.1 uses.

`run_full_experiment.sh` runs those gates and the density probe in the same allocation as the matrix, and then calibrates the rate ladder against the very fleet it's about to measure. `slurm/run_matrix.sh` refuses to run without a measured rate ladder, because the rates have to span underload to overload for this fleet, and that's a measured quantity, not a guess.

### Tracing with OpenTelemetry

The router can export a trace for every request it proxies. Tracing is off by default, so the router behaves and performs exactly as it did for the results below. It switches on when `--otlp-endpoint` (or `ROUTER_OTLP_ENDPOINT`) points at an OTLP/HTTP collector, for example `http://127.0.0.1:4318/v1/traces`, and `--service-name` keeps several routers apart.

With tracing on, every request gets a `router.request` span, which stays open until the last byte of the response reaches the client. As a result, its duration is the end-to-end latency that the router observes, and it records the chosen replica, the policy, the reason for the route, the admission wait and the time to the first byte. Child spans cover the admission queue and the round trip to the replica up to its response headers. In addition, the router continues the caller's W3C trace when the request carries a `traceparent` header, and it passes its own context on to the replica. That way, a load generator's span, the router's span and vLLM's own request span (from `vllm serve --otlp-traces-endpoint`) all land in one trace. With tracing off, the caller's `traceparent` and `tracestate` headers are still forwarded unchanged, so the replica can join a client's trace anyway.

One interoperability detail is worth knowing about. OpenTelemetry Python 1.45 sends `traceparent` flags of `03`, which means sampled plus the "random" bit from W3C Trace Context Level 2, and opentelemetry_sdk 0.31 for Rust rejected any such header without an error, so every request silently started a new trace. The router therefore requires opentelemetry 0.32 or newer, and `tests/trace_global.rs` sends exactly that header through the binary's own telemetry setup, so the problem can't come back unnoticed.

## Hardware

The measurements ran on the GPU partition of an HPC cluster. Each node has four NVIDIA A100-SXM4-40GB GPUs, 128 logical cores (four sockets of 16 cores with two threads each) and 480 GB of memory, and it's diskless. The experiment nodes reported driver 595.91.07, which every artifact records.

The fleet under test has four replicas, one vLLM replica per GPU at `gpu_memory_utilization=0.90`, each pinned to its GPU's local processor cores, all on one node. The router and the load generator run on the same node, over loopback. The router admits at most 64 in-flight requests per healthy replica and rejects the excess with HTTP 429. Every pair of GPUs is joined by NVLink, with no PCIe path between GPUs inside a node, so no NVLink-versus-PCIe contrast is visible here.

The backend and model identifiers are copied verbatim from llm-inference-bench's `provision.lock.json` into every artifact, and never retyped.

| | |
|---|---|
| backend | vLLM 0.27.1 (`sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`) |
| model | `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28` |
| precision | bf16 |

### Replica density limits the workload, not just the topology

I also needed to know whether two 7B replicas fit on one 40 GB card, and the answer matters for more than how many boxes appear in the diagram. On its own at a utilisation of 0.90 (job 5213921), one replica gets 19.09 GiB of KV cache, which is 357,520 tokens, enough for about 87 concurrent requests of 4,096 tokens, and it occupies 36,126 MiB of the card. Two replicas do fit on one card at a utilisation of 0.45 each, launched one after the other (job 5216059, `results/raw/density_probe_5216059.json`). However, each then gets a KV cache of only 24,608 tokens, enough for about six concurrent requests of 4,096 tokens. That's about 14.5 times less cache per replica, and the pair together holds about 7% of what one replica holds alone.

So a two-per-card fleet isn't a smaller version of the same fleet. It can't admit the same concurrency or the same context lengths, which makes the two topologies two different serving envelopes rather than two ways of serving one workload. Pooling them in one routing comparison would compare policies across configurations that differ in what they can physically admit, and part of any measured difference would come from that. Therefore, the experiment matrix runs on the one-replica-per-card fleet, and I report replica density separately as a capacity finding.

### Co-location

All four replicas share one node by design, because that's the topology under test. Until 2026-10-24, this section warned that co-location makes every absolute latency here a "fleet number", citing llm-inference-bench's measurement that packing four single-GPU replicas onto one node raised the median time to the first token by 150% and cut throughput by 34%. That measurement was withdrawn. Its packing script gave all four replicas the same port, so all four load generators measured one server on one GPU. When the same placement was measured again with each copy on its own port and with the same vLLM 0.27.1 image, four independent copies performed the same as one copy alone, within half a percent.

This repository was never affected by that bug, because every replica here has its own port (8100 plus its index) and its own log file. In addition, the policy comparison never depended on the caveat, because every policy ran against the same co-located fleet in the same job. So the routing results stand as they were, and co-location by itself doesn't inflate the absolute latencies. They still include the router and a load generator that shares the node, as the measurement caveats below describe.

## Results

The full tables are in [`results/RESULTS.md`](results/RESULTS.md) and the figures in `results/figures/`. Every number is the median over three repetitions, with the minimum and maximum in brackets, and the time to the first token is measured from each request's scheduled open-loop arrival.

### Calibration: where this fleet saturates

The rate calibration ran in the same allocation as the matrix (`results/calibration/5216059/`). The median time to the first token stays between 0.07 and 0.10 seconds, with no errors, up to 32 requests per second. At 48 requests per second, it reaches 0.34 seconds and the router rejects 19% of requests. So the matrix ran at 4, 16, 48 and 128 requests per second, which is underloaded, approaching saturation, at the saturation point and far past it. Past saturation, the latency percentiles describe only the requests that were admitted, so they must be read together with the error rate. For example, at 128 requests per second, plain prefix affinity shows a "faster" first token than round robin only because it rejects more requests (69% against 34% at a sharing ratio of 0.9).

### One hot prefix: affinity buys nothing and concentrates the load

This workload has 1,024-token prompts and 128-token outputs, and a fraction of the requests (the shared-prefix ratio) share one 512-token system prompt. These are the results at a ratio of 0.9.

| policy | fleet hit rate at 4 and 16 req/s | busiest replica's load over the mean | median first token at 16 req/s | error rate at 48 req/s |
|---|---|---|---|---|
| round robin | 0.440 / 0.444 | 1.00 | 0.067 s | 0.102 |
| least loaded | 0.440 / 0.444 | 1.00 to 1.01 | 0.067 s | 0.131 |
| prefix affinity | 0.448 / 0.448 | 3.69 to 3.72 | 0.151 s | 0.416 |
| prefix affinity, bounded | 0.440 / 0.444 | 1.26 to 1.90 | 0.070 s | 0.101 |

With a single prefix, round robin already reaches the best possible hit rate, because it pays four cold prefills instead of one out of thousands of requests. As a result, plain prefix affinity gains nothing on cache hits (0.448 against 0.444). What it does instead is send almost all traffic to one replica. The busiest replica carries 3.7 times the mean load, the median first token more than doubles at 16 requests per second (a load the fleet handles easily), and at the saturation point it sheds 42% of requests against round robin's 10%. That's the load-imbalance failure mode, measured.

The bounded policy contains it. Its busiest replica stays at 1.21 to 1.26 times the mean at 16 to 48 requests per second, its median first token at 16 requests per second is back to 0.070 seconds, and its error rate at 48 requests per second matches round robin's (0.101 against 0.102), although its median first token there is still higher (0.218 against 0.123 seconds). So for this workload, the expected headline that affinity helps at high sharing ratios is wrong, and the next experiment shows why.

### Many prefixes (a tenant mix): affinity wins on cache, within limits

This workload has 216 distinct 512-token prefixes and 960 requests per phase, so at a ratio of 0.9 each prefix is reused about four times. The best possible hit rate is 0.3375 at a ratio of 0.9 (and 0.1875 at 0.6). These are the results at 0.9.

| policy | fleet hit rate at 8 req/s | median first token at 8 req/s | median first token at 32 req/s | error rate at 48 req/s |
|---|---|---|---|---|
| round robin | 0.144 | 0.092 s | 0.110 s | 0.209 |
| least loaded | 0.140 | 0.093 s | 0.360 s | 0.222 |
| prefix affinity | 0.338 | 0.069 s | 0.148 s | 0.149 |
| prefix affinity, bounded | 0.263 | 0.090 s | 0.276 s | 0.184 |

Here, round robin scatters each prefix's handful of reuses across four replicas, and most of them land on a replica that has never seen that prefix. Prefix affinity reaches the best possible hit rate exactly (0.338) while the load stays even, with the busiest replica at 1.05 times the mean, and at 8 requests per second it cuts the median first token by 25%. So cache locality really is a property of the fleet, because the same engines with the same caches served 2.3 times more prefix tokens from cache purely because of where the requests were sent.

The latency win doesn't survive load, though. At 32 requests per second, affinity's median first token is 1.35 times round robin's, and the bounded policy, which moves requests to less loaded replicas and so loses some locality (hit rate 0.263 to 0.309), is slower still. At 48 requests per second, the medians swing widely between repetitions (round robin ranges from 0.37 to 2.87 seconds), so the ratios in `RESULTS.md` at that rate aren't a reliable ranking. However, affinity did reject fewer requests there (14.9% against 20.9%).

### Replica failure and recovery

I ran twelve trials (job 5216528, `results/RESULTS.md`) at 16 requests per second and a sharing ratio of 0.6. In each trial, one of the four replicas was killed (its whole process tree) 30 seconds in and relaunched 30 seconds later, with a fresh fleet for every trial. The router checked health every 500 ms and took a replica out after two consecutive failures.

| policy | detection (s) | relaunch to readmission (s) | error rate | killed replica's median share after readmission |
|---|---|---|---|---|
| round robin | 1.23 (0.99 to 1.52) | 109.8 (109.8 to 116.3) | 0.0018 | 0.250 |
| least loaded | 1.04 (0.99 to 1.12) | 113.9 (111.3 to 114.5) | 0.0034 | 0.251 |
| prefix affinity | 1.14 (1.04 to 1.18) | 112.8 (110.9 to 115.1) | 0.0006 | 0.099 |
| prefix affinity, bounded | 0.74 (0.70 to 0.75) | 113.2 (111.9 to 119.4) | 0.0025 | 0.292 |

Every failed request is attributable to the killed replica. Either its stream broke mid-response, or the router returned a 502 naming it, for a request sent between the kill and the moment the router took the replica out. No request failed after the replica was taken out or after it was readmitted, in any trial. Readmission is dominated by vLLM's cold start of about 110 seconds, and every policy gives the replica back its share from before the kill.

Those shares differ by policy, and that explains the error rates. Under plain prefix affinity, the killed replica isn't the shared prefix's preferred replica, so it only ever carries its hash share of the 40% of requests without the shared prefix (about 0.1 of all traffic, before and after the kill). It had little in flight to lose, hence the lowest error rate. By contrast, the bounded policy moves shared-prefix traffic onto it, so it carries about 0.29.

### Streaming under load

Every matrix phase delivered a median of 130 streamed chunks per 128-token request, at every rate. At 4 to 16 requests per second, the median ratio of the first-token time to the end-to-end latency is 0.035 to 0.056. At 128 requests per second it rises to between 0.30 and 0.88, because the queue wait dominates the first-token time, not because the proxy buffers, and the chunk count stays the same. The deterministic half of the check is `tests/streaming.rs`.

### Measurement caveats

Firstly, the client fell behind its own schedule. The Python load generator, which was unpinned and shared the node with four replicas, missed its 0.25 second dispatch tolerance on more than 2% of requests in most phases, even at 4 requests per second. I kept those phases and flagged them. `RESULTS.md` also reports the first-token time measured from the actual dispatch, along with the 95th percentile of the lag, and the medians quoted above barely move between the two clocks. At 128 requests per second the generator lags by seconds, so that rate means "at least heavily overloaded" rather than exactly 128.

Secondly, admission control shapes the overload. The 429 responses come from the router's own cap of 64 in-flight requests per replica, which is a configuration choice rather than an engine limit.

Lastly, job 5215612's eight matrix phases used an invalid rate ladder, as `docs/notes.md` explains. They're listed in `RESULTS.md` but not used.

### What I deliberately didn't do

I didn't run eight replicas across two nodes. That would make network latency visible in the routing, but at twice the node cost per phase, and a single node with four replicas already answers the questions above. It's unmeasured rather than estimated.

I also didn't run the matrix on the two-per-card topology, because its KV cache is about 14.5 times smaller per replica, so it serves a different workload, and pooling it with the one-per-card matrix would confuse the policy with the capacity.

In addition, I didn't root-cause the client's dispatch lag. The generator and router run unpinned on the same node as four pinned replicas, so pinning them to free cores is the obvious next step, but the conclusions above didn't need it. Lastly, I haven't put SGLang behind the router. The router speaks the OpenAI API, but I've only exercised it against vLLM 0.27.1, and I haven't written Kubernetes manifests, because I had no cluster to test them on.

## Reproducing

Every figure and table regenerates with one command, `sbatch -A <account> slurm/run_analysis.sh`, which reads `results/raw/` and writes `results/figures/` and `results/RESULTS.md`. Nothing in either is typed by hand.

Before publishing, I replaced site-specific strings in the raw and calibration artifacts, such as host names, file paths and account names, with neutral placeholders like `${IR_DATA}`, `${HOME}` and `node2061`, using `scripts/scrub_private.py`. A check over all 391 files confirmed that every number, every flag and the structure of every file are identical to what was written. `docs/notes.md` maps the commits recorded in the artifacts to this repository's history.
