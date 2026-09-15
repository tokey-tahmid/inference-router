//! A mock inference replica, plus helpers to stand a router in front of it.
//!
//! The mock exists to test properties that a live vLLM replica cannot be made to
//! exhibit on demand: a deliberate stall between SSE chunks, a health endpoint
//! that can be failed at an exact instant, and a replica that reports which
//! requests it received. The live-replica verification is a separate,
//! complementary job (`slurm/verify_single_replica.sh`).

#![allow(dead_code)]

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use futures_util::stream;

use inference_router::config::{
    AdmissionConfig, Config, HealthConfig, PolicyConfig, ReplicaConfig, UpstreamConfig,
};
use inference_router::state::{AppState, SharedState};

#[derive(Debug)]
pub struct MockReplica {
    pub addr: SocketAddr,
    pub id: String,
    /// Flip to false to make `/health` return 503, simulating a dead replica
    /// without tearing down the socket.
    pub healthy: Arc<AtomicBool>,
    /// Flip to make the completion endpoint return 500.
    pub failing: Arc<AtomicBool>,
    pub requests: Arc<AtomicU64>,
    /// Milliseconds to wait before emitting the FIRST chunk (mock prefill).
    pub first_chunk_delay_ms: Arc<AtomicU64>,
    /// Milliseconds between subsequent chunks (mock decode).
    pub inter_chunk_delay_ms: Arc<AtomicU64>,
    pub chunks: Arc<AtomicU64>,
}

#[derive(Clone)]
struct MockState {
    healthy: Arc<AtomicBool>,
    failing: Arc<AtomicBool>,
    requests: Arc<AtomicU64>,
    first_chunk_delay_ms: Arc<AtomicU64>,
    inter_chunk_delay_ms: Arc<AtomicU64>,
    chunks: Arc<AtomicU64>,
    id: String,
}

impl MockReplica {
    pub async fn spawn(id: &str) -> Self {
        let healthy = Arc::new(AtomicBool::new(true));
        let failing = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(AtomicU64::new(0));
        let first_chunk_delay_ms = Arc::new(AtomicU64::new(0));
        let inter_chunk_delay_ms = Arc::new(AtomicU64::new(0));
        let chunks = Arc::new(AtomicU64::new(4));

        let st = MockState {
            healthy: healthy.clone(),
            failing: failing.clone(),
            requests: requests.clone(),
            first_chunk_delay_ms: first_chunk_delay_ms.clone(),
            inter_chunk_delay_ms: inter_chunk_delay_ms.clone(),
            chunks: chunks.clone(),
            id: id.to_string(),
        };

        let app = Router::new()
            .route("/v1/completions", post(complete))
            .route("/v1/chat/completions", post(complete))
            .route("/health", get(health))
            .with_state(st);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        Self {
            addr,
            id: id.to_string(),
            healthy,
            failing,
            requests,
            first_chunk_delay_ms,
            inter_chunk_delay_ms,
            chunks,
        }
    }

    pub fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn request_count(&self) -> u64 {
        self.requests.load(Ordering::Relaxed)
    }

    pub fn set_healthy(&self, v: bool) {
        self.healthy.store(v, Ordering::Relaxed);
    }

    pub fn set_first_chunk_delay(&self, ms: u64) {
        self.first_chunk_delay_ms.store(ms, Ordering::Relaxed);
    }

    pub fn set_inter_chunk_delay(&self, ms: u64) {
        self.inter_chunk_delay_ms.store(ms, Ordering::Relaxed);
    }

    pub fn set_chunks(&self, n: u64) {
        self.chunks.store(n, Ordering::Relaxed);
    }
}

async fn health(State(st): State<MockState>) -> Response {
    if st.healthy.load(Ordering::Relaxed) {
        (StatusCode::OK, "ok").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "dead").into_response()
    }
}

async fn complete(State(st): State<MockState>, _body: Bytes) -> Response {
    st.requests.fetch_add(1, Ordering::Relaxed);
    if st.failing.load(Ordering::Relaxed) {
        return (StatusCode::INTERNAL_SERVER_ERROR, "boom").into_response();
    }

    let first = st.first_chunk_delay_ms.load(Ordering::Relaxed);
    let inter = st.inter_chunk_delay_ms.load(Ordering::Relaxed);
    let n = st.chunks.load(Ordering::Relaxed);
    let id = st.id.clone();

    // A genuine lazy stream: each chunk is produced only when the consumer polls
    // for it, after the configured delay. If the router buffered, the client
    // would observe nothing until every one of these sleeps had elapsed.
    let s = stream::unfold(0u64, move |i| {
        let id = id.clone();
        async move {
            if i >= n {
                return None;
            }
            let delay = if i == 0 { first } else { inter };
            if delay > 0 {
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
            let payload = if i + 1 == n {
                "data: [DONE]\n\n".to_string()
            } else {
                format!(
                    "data: {{\"id\":\"{id}\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"t{i}\"}}}}]}}\n\n"
                )
            };
            Some((Ok::<Bytes, std::io::Error>(Bytes::from(payload)), i + 1))
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(s))
        .unwrap()
}

/// Build a router config pointing at the given mocks.
pub fn config_for(mocks: &[&MockReplica]) -> Config {
    Config {
        listen: "127.0.0.1:0".parse().unwrap(),
        replicas: mocks
            .iter()
            .enumerate()
            .map(|(i, m)| ReplicaConfig {
                id: m.id.clone(),
                url: m.url(),
                placement: Some(format!("mock/gpu{i}")),
            })
            .collect(),
        policy: PolicyConfig::default(),
        health: HealthConfig {
            enabled: false,
            interval_ms: 100,
            timeout_ms: 200,
            unhealthy_threshold: 2,
            healthy_threshold: 2,
            path: "/health".into(),
        },
        admission: AdmissionConfig::default(),
        upstream: UpstreamConfig::default(),
    }
}

/// Start a router on an ephemeral port. Returns its base URL and shared state.
pub async fn spawn_router(config: Config) -> (String, SharedState) {
    let state = AppState::new(config).unwrap();
    if state.config.health.enabled {
        inference_router::health::spawn(state.clone());
    }
    let listener = tokio::net::TcpListener::bind(state.config.listen)
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    let app = inference_router::app(state.clone());
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    // Let the acceptor reach its first poll before the test connects.
    tokio::time::sleep(Duration::from_millis(20)).await;
    (format!("http://{addr}"), state)
}

/// A token-ID prompt of `len` tokens whose first `shared` tokens are fixed.
///
/// Mirrors the shape P1's workload generator emits, which is what the affinity
/// key is designed around.
pub fn prompt_with_prefix(shared_group: u64, shared: usize, len: usize) -> Vec<i64> {
    let mut v: Vec<i64> = (0..shared)
        .map(|i| (shared_group * 100_000 + i as u64) as i64)
        .collect();
    let mut x = shared_group.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (len as u64);
    while v.len() < len {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push((x >> 33) as i64);
    }
    v
}
