"""Open-loop load generation against the router.

**Open loop, not closed loop.** The experiment axis is *request rate*, so
arrivals are drawn from a Poisson process and dispatched on schedule regardless
of whether earlier requests have finished. A closed-loop generator (N workers,
each sending the next request when the previous returns) cannot express
overload at all: as the fleet slows down, the offered load falls with it, and
the system self-throttles into a stable point. Every "overload" arm of the
matrix would silently become a different, lower load. The coordination-omission
problem is the same one Gil Tene describes for latency benchmarks, and the fix
is the same: schedule by wall clock, and measure latency from the *scheduled*
arrival time, not from when a worker happened to get around to sending.

Prompts come from P1's ``llm_inference_bench.workload`` generator, imported
rather than reimplemented, so the shared-prefix ratio here means exactly what it
meant in P1's prefix-caching measurements.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import httpx
import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class RequestRecord:
    """One request's outcome. Every field is measured, none derived here."""

    index: int
    prefix_group: int | None
    input_tokens: int
    requested_output_tokens: int

    # Wall-clock offsets from phase start, in seconds.
    scheduled_at: float
    sent_at: float
    first_token_at: float | None = None
    completed_at: float | None = None

    status: int | None = None
    error: str | None = None

    # Router-reported routing provenance, read from response headers.
    replica: str | None = None
    policy: str | None = None
    route_reason: str | None = None
    placement: str | None = None
    affinity_key: str | None = None
    queue_wait_us: int | None = None

    # Streaming detail.
    chunks: int = 0
    response_bytes: int = 0
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    # Arrival offsets of the first few chunks, kept so the non-buffering
    # property can be checked from the client side under real load rather than
    # only in a unit test.
    chunk_arrival_offsets: list[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300 and self.error is None

    @property
    def ttft(self) -> float | None:
        """Time to first token, measured from the *scheduled* arrival.

        Including the scheduling delay is what makes this honest under overload:
        if the generator itself falls behind, that delay is real latency the
        client experienced and hiding it would understate the tail exactly where
        it matters most.
        """
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.scheduled_at

    @property
    def e2e(self) -> float | None:
        if self.completed_at is None:
            return None
        return self.completed_at - self.scheduled_at

    @property
    def send_delay(self) -> float:
        """How late the generator was in dispatching this request.

        Reported per phase. A large value means the driver, not the fleet, was
        the bottleneck, which invalidates that phase's rate axis. Recorded so
        that can be detected rather than assumed away.
        """
        return self.sent_at - self.scheduled_at

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["ttft_s"] = self.ttft
        d["e2e_s"] = self.e2e
        d["send_delay_s"] = self.send_delay
        d["ok"] = self.ok
        return d


def poisson_offsets(rate: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Arrival offsets of a Poisson process at `rate` requests/second.

    Exponential inter-arrival gaps, cumulatively summed. A fixed-interval
    schedule would be a different (and easier) workload: bursts are what fill
    queues, and queue depth is the quantity the load-balancing policies differ
    on.
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    gaps = rng.exponential(scale=1.0 / rate, size=n)
    return np.cumsum(gaps)


def _parse_sse_usage(buf: str) -> tuple[int | None, int | None]:
    """Pull prompt/completion token counts out of a finished SSE stream.

    vLLM emits a final usage block when ``stream_options.include_usage`` is set.
    Using the server's own count rather than counting chunks is deliberate: P1
    established that a chunk is not always one token (n-gram speculation batches
    several per chunk), and deriving a token count from chunk count produced a
    per-token latency figure that was wrong by 3.3x and looked entirely
    plausible.
    """
    prompt = completion = None
    for line in buf.splitlines():
        line = line.strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            obj = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        usage = obj.get("usage")
        if isinstance(usage, dict):
            if usage.get("prompt_tokens") is not None:
                prompt = usage["prompt_tokens"]
            if usage.get("completion_tokens") is not None:
                completion = usage["completion_tokens"]
    return prompt, completion


async def _one_request(
    client: httpx.AsyncClient,
    router_url: str,
    rec: RequestRecord,
    prompt_token_ids: Sequence[int],
    model: str,
    phase_start: float,
    max_chunk_offsets: int,
) -> None:
    payload = {
        "model": model,
        # Token IDs, not text: the engine's prefix cache keys on the exact token
        # sequence, and a decode/re-tokenise round trip is not a fixed point.
        "prompt": list(prompt_token_ids),
        "max_tokens": rec.requested_output_tokens,
        "temperature": 0.0,
        # Controls the output-length axis: without it a sequence can stop early
        # and the decode span becomes uncontrolled.
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    rec.sent_at = time.perf_counter() - phase_start
    buf = []
    try:
        async with client.stream(
            "POST", f"{router_url}/v1/completions", json=payload
        ) as resp:
            rec.status = resp.status_code
            h = resp.headers
            rec.replica = h.get("x-router-replica")
            rec.policy = h.get("x-router-policy")
            rec.route_reason = h.get("x-router-route-reason")
            rec.placement = h.get("x-router-placement")
            rec.affinity_key = h.get("x-router-affinity-key")
            if h.get("x-router-queue-wait-us") is not None:
                try:
                    rec.queue_wait_us = int(h["x-router-queue-wait-us"])
                except ValueError:
                    rec.queue_wait_us = None

            if resp.status_code >= 300:
                body = await resp.aread()
                rec.error = body.decode("utf-8", "replace")[:500]
                rec.completed_at = time.perf_counter() - phase_start
                return

            async for chunk in resp.aiter_bytes():
                now = time.perf_counter() - phase_start
                if not chunk:
                    continue
                if rec.first_token_at is None:
                    rec.first_token_at = now
                if len(rec.chunk_arrival_offsets) < max_chunk_offsets:
                    rec.chunk_arrival_offsets.append(now)
                rec.chunks += 1
                rec.response_bytes += len(chunk)
                buf.append(chunk.decode("utf-8", "replace"))
            rec.completed_at = time.perf_counter() - phase_start
    except (httpx.HTTPError, asyncio.TimeoutError, OSError) as exc:
        rec.error = f"{type(exc).__name__}: {exc}"
        rec.completed_at = time.perf_counter() - phase_start
        return

    rec.prompt_tokens, rec.completion_tokens = _parse_sse_usage("".join(buf))


@dataclass
class PhaseResult:
    records: list[RequestRecord]
    wall_start_unix: float
    wall_end_unix: float
    elapsed_s: float

    @property
    def completed(self) -> list[RequestRecord]:
        return [r for r in self.records if r.ok and r.completed_at is not None]


async def run_phase(
    router_url: str,
    requests: Sequence[Any],
    rate: float,
    model: str,
    *,
    seed: int = 0,
    request_timeout_s: float = 600.0,
    max_connections: int = 4096,
    max_chunk_offsets: int = 8,
) -> PhaseResult:
    """Drive `requests` at `rate` req/s open-loop and collect every outcome.

    `requests` are `GeneratedRequest` objects from P1's workload generator.
    """
    rng = np.random.default_rng(seed)
    offsets = poisson_offsets(rate, len(requests), rng)

    limits = httpx.Limits(
        max_connections=max_connections, max_keepalive_connections=max_connections
    )
    # No total timeout on read: a streaming generation legitimately runs for a
    # long time under overload, and cutting it off would record a fleet that was
    # merely slow as a fleet that was broken.
    timeout = httpx.Timeout(
        connect=30.0, read=request_timeout_s, write=30.0, pool=request_timeout_s
    )

    records: list[RequestRecord] = []
    wall_start = time.time()
    phase_start = time.perf_counter()

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        tasks: list[asyncio.Task[None]] = []
        for i, (req, off) in enumerate(zip(requests, offsets, strict=True)):
            rec = RequestRecord(
                index=i,
                prefix_group=req.prefix_group,
                input_tokens=req.input_tokens,
                requested_output_tokens=req.output_tokens,
                scheduled_at=float(off),
                sent_at=float("nan"),
            )
            records.append(rec)

            # Sleep until this request's scheduled arrival. If we are already
            # late, do not sleep: dispatch immediately and let send_delay record
            # how late we were.
            now = time.perf_counter() - phase_start
            if off > now:
                await asyncio.sleep(off - now)

            tasks.append(
                asyncio.create_task(
                    _one_request(
                        client,
                        router_url,
                        rec,
                        req.prompt_token_ids,
                        model,
                        phase_start,
                        max_chunk_offsets,
                    )
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.perf_counter() - phase_start
    late = [r.send_delay for r in records if r.send_delay > 0.05]
    log.info(
        "phase_complete",
        n=len(records),
        ok=sum(1 for r in records if r.ok),
        rate_target=rate,
        elapsed_s=round(elapsed, 3),
        achieved_rate=round(len(records) / elapsed, 3) if elapsed > 0 else None,
        late_dispatches=len(late),
        max_send_delay_s=round(max(late), 3) if late else 0.0,
    )
    return PhaseResult(
        records=records,
        wall_start_unix=wall_start,
        wall_end_unix=time.time(),
        elapsed_s=elapsed,
    )
