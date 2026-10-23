//! OpenAI-compatible proxy handlers with true streaming passthrough.
//!
//! # The requirement, and why it is easy to get wrong
//!
//! The obvious proxy implementation is
//!
//! ```ignore
//! let body = upstream.text().await?;   // <-- wrong
//! Ok((StatusCode::OK, body))
//! ```
//!
//! and it *works*. Every request succeeds, the JSON is byte-identical, the tests
//! pass. What it destroys is time-to-first-token: the client receives nothing
//! until the replica has generated the **last** token, so TTFT collapses onto
//! end-to-end latency. For a 256-token generation at P1's measured 12.3 ms/token
//! that turns a ~70 ms first token into a ~3.2 s one, and every routing result
//! measured through it would be a measurement of decode time rather than of
//! routing. Since prefix-affinity's entire claimed benefit is a *prefill*
//! effect, a buffering proxy would erase exactly the signal this project exists
//! to measure, while still producing a complete and plausible dataset.
//!
//! # How this implementation avoids it
//!
//! The upstream response body is consumed as a [`futures_util::Stream`] of
//! [`Bytes`] and handed to [`axum::body::Body::from_stream`] without ever being
//! collected. Nothing in the path calls `.text()`, `.json()`, or
//! `.collect()` on the response body. The request body *is* buffered, and must
//! be: the router has to parse the prompt to compute an affinity key. Prompts
//! are bounded (~1 KB-40 KB here); responses are not.
//!
//! Three further details that a naive passthrough gets wrong:
//!
//! * **No total-request timeout** on the upstream client (see `state.rs`), only
//!   an inter-chunk read timeout. A total timeout severs long healthy
//!   generations.
//! * **In-flight accounting outlives the handler.** The [`InflightGuard`] is
//!   moved *into the stream's state*, so a replica counts as busy until its
//!   last byte reaches the client. Decrementing when the handler returns would
//!   report an idle fleet and make `least_loaded` route on a number that is
//!   always zero.
//! * **Hop-by-hop and length headers are dropped.** Forwarding upstream's
//!   `Content-Length` or `Transfer-Encoding` onto a re-framed body produces a
//!   response whose framing contradicts its headers.
//!
//! The non-buffering property is asserted by `tests/streaming.rs`, which stands
//! up a mock upstream that emits a chunk and then stalls, and fails if the first
//! chunk does not arrive through the router before the stall ends.

use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use futures_util::Stream;
use tracing::Instrument;
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::admission::AdmissionGuard;
use crate::error::{Result, RouterError};
use crate::metrics::SharedMetrics;
use crate::policy::AffinityKey;
use crate::replica::{InflightGuard, Replica};
use crate::state::SharedState;
use crate::telemetry;

/// Headers that must not be copied verbatim onto a re-framed response.
const HOP_BY_HOP: [&str; 8] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
];

pub async fn chat_completions(
    state: State<SharedState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    handle(state, headers, body, "/v1/chat/completions").await
}

pub async fn completions(state: State<SharedState>, headers: HeaderMap, body: Bytes) -> Response {
    handle(state, headers, body, "/v1/completions").await
}

async fn handle(
    State(state): State<SharedState>,
    headers: HeaderMap,
    body: Bytes,
    path: &'static str,
) -> Response {
    match dispatch(state.clone(), headers, body, path).await {
        Ok(resp) => resp,
        Err(e) => {
            let m = &state.metrics;
            m.responses_error_total.fetch_add(1, Ordering::Relaxed);
            match &e {
                RouterError::Overloaded { .. } => {
                    m.rejected_overload_total.fetch_add(1, Ordering::Relaxed);
                }
                RouterError::AdmissionTimeout { .. } => {
                    m.rejected_overload_total.fetch_add(1, Ordering::Relaxed);
                    m.admission_queue_timeout_total
                        .fetch_add(1, Ordering::Relaxed);
                }
                RouterError::NoHealthyReplica => {
                    m.rejected_no_healthy_total.fetch_add(1, Ordering::Relaxed);
                }
                RouterError::Upstream { .. } => {
                    m.upstream_error_total.fetch_add(1, Ordering::Relaxed);
                }
                _ => {}
            }
            e.into_response()
        }
    }
}

/// Extract the affinity key from an OpenAI-shaped request body.
///
/// Handles both prompt spellings the harness uses:
///
/// * `/v1/completions` with `"prompt": [int, ...]` — the exact token-ID form
///   P1's workload generator emits. This is the case that matters: the engine's
///   prefix cache keys on the exact token sequence, so hashing token IDs makes
///   the router's notion of "same prefix" identical to the engine's.
/// * `/v1/chat/completions` with `"messages": [...]` — text, so the key is the
///   leading characters of the concatenated messages. Coarser (character n is
///   not token n) but still correct in the sense that equal prefixes hash
///   equally.
///
/// Returns `None` when there is no prompt to hash; the caller routes by load
/// rather than rejecting.
pub fn affinity_key(body: &serde_json::Value, prefix_tokens: usize) -> Option<AffinityKey> {
    if let Some(prompt) = body.get("prompt") {
        // Token IDs: [1, 2, 3, ...]
        if let Some(arr) = prompt.as_array() {
            if arr.first().and_then(serde_json::Value::as_i64).is_some() {
                let ids: Vec<i64> = arr.iter().filter_map(serde_json::Value::as_i64).collect();
                if !ids.is_empty() {
                    return Some(AffinityKey::from_token_ids(&ids, prefix_tokens));
                }
            }
            // Batched text prompts: key on the first, so a batch stays together.
            if let Some(s) = arr.first().and_then(serde_json::Value::as_str) {
                return Some(AffinityKey::from_text(s, prefix_tokens));
            }
        }
        if let Some(s) = prompt.as_str() {
            return Some(AffinityKey::from_text(s, prefix_tokens));
        }
    }

    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        let mut buf = String::new();
        for msg in messages {
            if let Some(role) = msg.get("role").and_then(|r| r.as_str()) {
                buf.push_str(role);
                buf.push('\n');
            }
            match msg.get("content") {
                Some(serde_json::Value::String(s)) => buf.push_str(s),
                // Multimodal content parts: concatenate the text ones.
                Some(serde_json::Value::Array(parts)) => {
                    for p in parts {
                        if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                            buf.push_str(t);
                        }
                    }
                }
                _ => {}
            }
            buf.push('\n');
            // Stop once we have more than we will hash; a 100-message history
            // should not be walked in full on the hot path.
            if buf.len() > prefix_tokens.saturating_mul(4) {
                break;
            }
        }
        if !buf.is_empty() {
            return Some(AffinityKey::from_text(&buf, prefix_tokens));
        }
    }

    None
}

async fn dispatch(
    state: SharedState,
    headers: HeaderMap,
    body: Bytes,
    path: &'static str,
) -> Result<Response> {
    let start = Instant::now();
    let m = Arc::clone(&state.metrics);
    m.requests_total.fetch_add(1, Ordering::Relaxed);

    // One span per request, open until the last byte reaches the client (it is
    // moved into the relayed stream's state below). See src/telemetry.rs.
    let span = tracing::info_span!(
        "router.request",
        otel.kind = "server",
        http.route = path,
        router.replica = tracing::field::Empty,
        router.policy = tracing::field::Empty,
        router.route_reason = tracing::field::Empty,
        router.queue_wait_us = tracing::field::Empty,
        router.ttfb_ms = tracing::field::Empty,
        router.chunks = tracing::field::Empty,
        http.response.status_code = tracing::field::Empty,
        otel.status_code = tracing::field::Empty,
    );
    if telemetry::enabled() {
        let _ = span.set_parent(telemetry::parent_from(&headers));
    }
    let failed = |e: RouterError| {
        span.record("otel.status_code", "ERROR");
        e
    };

    let parsed: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| RouterError::BadRequest(format!("body is not JSON: {e}")))?;

    let streaming = parsed
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if streaming {
        m.requests_streaming_total.fetch_add(1, Ordering::Relaxed);
    }

    let policy = state.policy();
    let prefix_tokens = state.policy_config().prefix_hash_tokens;
    let key = affinity_key(&parsed, prefix_tokens);
    if key.is_none() {
        m.affinity_key_missing_total.fetch_add(1, Ordering::Relaxed);
    }

    // Admission first, routing second. Admitting after choosing a replica would
    // let a rejected request still perturb round-robin's cursor, so the rejection
    // rate would silently bias the load distribution of the policy under test.
    let n_healthy = state.fleet.healthy_indices().len();
    if n_healthy == 0 {
        return Err(RouterError::NoHealthyReplica);
    }
    let admission: AdmissionGuard = state
        .admission
        .acquire(n_healthy)
        .instrument(tracing::info_span!(parent: &span, "router.admission"))
        .await
        .map_err(failed)?;
    m.queue_wait.observe(admission.queue_wait.as_secs_f64());

    let decision = policy
        .select(&state.fleet, key)
        .ok_or(RouterError::NoHealthyReplica)
        .map_err(failed)?;
    let replica = Arc::clone(state.fleet.get(decision.replica_index));
    m.record_route(decision.replica_index, decision.reason);
    span.record("router.replica", replica.id.as_str());
    span.record("router.policy", policy.kind().as_str());
    span.record("router.route_reason", decision.reason.as_str());
    span.record(
        "router.queue_wait_us",
        admission.queue_wait.as_micros() as u64,
    );

    // Acquired before the request is sent and released when the response stream
    // is dropped. See the module docs.
    let inflight = InflightGuard::acquire(Arc::clone(&replica));

    let url = format!("{}{}", replica.url, path);
    let mut req = state
        .http
        .post(&url)
        .header(header::CONTENT_TYPE, "application/json")
        .body(body)
        .headers(telemetry::outgoing_headers(&span, &headers));
    // Pass the caller's authorization through untouched; vLLM may be running
    // with an API key and the router is not the place to mint or strip one.
    if let Some(auth) = headers.get(header::AUTHORIZATION) {
        req = req.header(header::AUTHORIZATION, auth.clone());
    }

    let upstream = req
        .send()
        .instrument(tracing::info_span!(
            parent: &span,
            "router.upstream_headers",
            router.replica = %replica.id
        ))
        .await
        .map_err(|source| {
            replica.note_upstream_error();
            failed(RouterError::Upstream {
                replica: replica.id.clone(),
                source,
            })
        })?;

    let status = upstream.status();
    span.record("http.response.status_code", status.as_u16());
    if !status.is_success() {
        span.record("otel.status_code", "ERROR");
    }
    if status.is_success() {
        m.responses_ok_total.fetch_add(1, Ordering::Relaxed);
    } else {
        m.responses_error_total.fetch_add(1, Ordering::Relaxed);
        replica.note_upstream_error();
        m.upstream_error_total.fetch_add(1, Ordering::Relaxed);
    }

    let mut out = Response::builder().status(status);
    {
        let hdrs = out.headers_mut().expect("builder has headers");
        for (name, value) in upstream.headers() {
            let lower = name.as_str().to_ascii_lowercase();
            // Content-Length is dropped along with the hop-by-hop set: the body
            // is re-framed by hyper, and a stale length header would contradict
            // the actual framing.
            if HOP_BY_HOP.contains(&lower.as_str()) || lower == "content-length" {
                continue;
            }
            hdrs.insert(name.clone(), value.clone());
        }
        // Routing provenance, read back per-request by the experiment driver.
        // Per-request attribution from the client side is what makes the fairness
        // metric checkable against the router's own counters instead of trusted.
        insert_str(hdrs, "x-router-replica", &replica.id);
        insert_str(hdrs, "x-router-policy", policy.kind().as_str());
        insert_str(hdrs, "x-router-route-reason", decision.reason.as_str());
        if let Some(p) = replica.placement.as_deref() {
            insert_str(hdrs, "x-router-placement", p);
        }
        insert_str(
            hdrs,
            "x-router-queue-wait-us",
            &admission.queue_wait.as_micros().to_string(),
        );
        if let Some(k) = key {
            insert_str(hdrs, "x-router-affinity-key", &format!("{:016x}", k.0));
        }
    }

    tracing::info!(
        path,
        replica = %replica.id,
        policy = policy.kind().as_str(),
        reason = decision.reason.as_str(),
        streaming,
        status = status.as_u16(),
        queue_wait_us = admission.queue_wait.as_micros() as u64,
        affinity_key = key.map(|k| k.0),
        "routed"
    );

    // The passthrough. `bytes_stream()` is lazy: nothing has been read from the
    // socket beyond the response headers at this point.
    let body_stream = upstream.bytes_stream();
    let relay = RelayStream {
        inner: Box::pin(body_stream),
        state: RelayState {
            metrics: m,
            _admission: admission,
            _inflight: inflight,
            replica,
            start,
            first_byte_at: None,
            completed: false,
            chunks: 0,
            bytes: 0,
            span,
        },
    };

    out.body(Body::from_stream(relay))
        .map_err(|e| RouterError::Other(anyhow::anyhow!("building response: {e}")))
}

fn insert_str(hdrs: &mut HeaderMap, name: &'static str, value: &str) {
    if let Ok(v) = HeaderValue::from_str(value) {
        hdrs.insert(HeaderName::from_static(name), v);
    }
}

/// State carried alongside the relayed body.
///
/// `Drop` is where completion is recorded, deliberately. A response stream can
/// end three ways — normal completion, upstream error, and the client hanging up
/// mid-generation — and only a destructor observes all three. Recording in the
/// `None` arm of the stream alone would leak the in-flight count on every client
/// disconnect, and under an overload arm (where clients time out and disconnect
/// constantly) the leaked counts would accumulate until the router believed the
/// fleet was permanently saturated.
struct RelayState {
    metrics: SharedMetrics,
    _admission: AdmissionGuard,
    _inflight: InflightGuard,
    replica: Arc<Replica>,
    start: Instant,
    first_byte_at: Option<Instant>,
    completed: bool,
    chunks: u64,
    bytes: u64,
    /// The request's span; it ends when this state is dropped, after the last byte.
    span: tracing::Span,
}

impl Drop for RelayState {
    fn drop(&mut self) {
        self.span.record("router.chunks", self.chunks);
        let e2e = self.start.elapsed().as_secs_f64();
        self.metrics.e2e.observe(e2e);
        self.metrics
            .sse_chunks_total
            .fetch_add(self.chunks, Ordering::Relaxed);
        self.metrics
            .bytes_streamed_total
            .fetch_add(self.bytes, Ordering::Relaxed);
        if !self.completed {
            self.metrics
                .client_disconnect_total
                .fetch_add(1, Ordering::Relaxed);
            tracing::debug!(
                replica = %self.replica.id,
                chunks = self.chunks,
                bytes = self.bytes,
                e2e_s = e2e,
                "stream_terminated_early"
            );
        }
    }
}

struct RelayStream<S> {
    inner: Pin<Box<S>>,
    state: RelayState,
}

impl<S> Stream for RelayStream<S>
where
    S: Stream<Item = std::result::Result<Bytes, reqwest::Error>>,
{
    type Item = std::result::Result<Bytes, reqwest::Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Safe: `self` is pinned, `inner` is already a `Pin<Box<_>>` (so moving
        // the struct never moves the stream), and `state` is `Unpin`.
        let this = unsafe { self.get_unchecked_mut() };
        match this.inner.as_mut().poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(Ok(chunk))) => {
                if this.state.first_byte_at.is_none() && !chunk.is_empty() {
                    let now = Instant::now();
                    this.state.first_byte_at = Some(now);
                    let ttft = now.duration_since(this.state.start).as_secs_f64();
                    // Router-observed TTFT: the moment the first body byte from
                    // the replica is handed downstream. It includes routing,
                    // admission wait and the replica's prefill, which is exactly
                    // the quantity a client experiences.
                    this.state
                        .metrics
                        .record_ttft(this.state.replica.index, ttft);
                    this.state.span.record("router.ttfb_ms", ttft * 1e3);
                }
                this.state.chunks += 1;
                this.state.bytes += chunk.len() as u64;
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => {
                this.state.replica.note_upstream_error();
                this.state
                    .metrics
                    .upstream_error_total
                    .fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    replica = %this.state.replica.id,
                    error = %e,
                    chunks_before_error = this.state.chunks,
                    "upstream_stream_error"
                );
                // Mark completed: the stream ended for an upstream reason, not
                // because the client left, and conflating the two would inflate
                // the client-disconnect count during a replica failure.
                this.state.completed = true;
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                this.state.completed = true;
                Poll::Ready(None)
            }
        }
    }
}

pub async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn token_id_prompts_produce_a_key_that_ignores_the_tail() {
        let a: Vec<i64> = (0..800).collect();
        let mut b = a.clone();
        for x in b.iter_mut().skip(512) {
            *x = 42;
        }
        let ka = affinity_key(&json!({ "prompt": a }), 512).unwrap();
        let kb = affinity_key(&json!({ "prompt": b }), 512).unwrap();
        assert_eq!(ka, kb, "shared 512-token prefix must map to one key");
    }

    #[test]
    fn token_id_prompts_differing_inside_the_prefix_differ() {
        let a: Vec<i64> = (0..800).collect();
        let mut b = a.clone();
        b[10] = -7;
        assert_ne!(
            affinity_key(&json!({ "prompt": a }), 512),
            affinity_key(&json!({ "prompt": b }), 512)
        );
    }

    #[test]
    fn text_prompts_are_keyed_too() {
        let k = affinity_key(&json!({ "prompt": "a long system prompt" }), 512);
        assert!(k.is_some());
    }

    #[test]
    fn chat_messages_share_a_key_when_the_system_prompt_matches() {
        let sys = "You are a helpful assistant with a very long preamble. ".repeat(20);
        let a = json!({"messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": "first question"},
        ]});
        let sys2 = "You are a helpful assistant with a very long preamble. ".repeat(20);
        let b = json!({"messages": [
            {"role": "system", "content": sys2},
            {"role": "user", "content": "a completely different question"},
        ]});
        assert_eq!(
            affinity_key(&a, 256),
            affinity_key(&b, 256),
            "identical system prompts longer than the hash window must collide"
        );
    }

    #[test]
    fn chat_messages_with_different_system_prompts_do_not_collide() {
        let a = json!({"messages": [{"role": "system", "content": "tenant A preamble"}]});
        let b = json!({"messages": [{"role": "system", "content": "tenant B preamble"}]});
        assert_ne!(affinity_key(&a, 256), affinity_key(&b, 256));
    }

    #[test]
    fn multimodal_content_parts_contribute_their_text() {
        let v = json!({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        ]});
        assert!(affinity_key(&v, 256).is_some());
    }

    #[test]
    fn a_body_with_no_prompt_yields_no_key() {
        assert!(affinity_key(&json!({"model": "x", "max_tokens": 5}), 512).is_none());
        assert!(affinity_key(&json!({"prompt": []}), 512).is_none());
    }

    #[test]
    fn batched_text_prompts_key_on_the_first_element() {
        let v = json!({"prompt": ["shared preamble here", "another"]});
        let w = json!({"prompt": ["shared preamble here"]});
        assert_eq!(affinity_key(&v, 512), affinity_key(&w, 512));
    }

    #[test]
    fn hop_by_hop_list_is_lowercase_so_the_comparison_works() {
        for h in HOP_BY_HOP {
            assert_eq!(h, h.to_ascii_lowercase());
        }
    }
}
