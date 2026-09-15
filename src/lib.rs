//! A prefix-aware request router for a fleet of LLM inference replicas.
//!
//! See `README.md` for the experiment this exists to run, and `policy/mod.rs`
//! for why cache locality is a fleet-level property rather than a per-replica
//! one.

pub mod admission;
pub mod config;
pub mod error;
pub mod health;
pub mod metrics;
pub mod policy;
pub mod proxy;
pub mod replica;
pub mod state;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::config::{PolicyConfig, PolicyKind};
use crate::error::{Result, RouterError};
use crate::state::SharedState;

pub fn app(state: SharedState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(proxy::chat_completions))
        .route("/v1/completions", post(proxy::completions))
        .route("/health", get(proxy::healthz))
        .route("/metrics", get(metrics_handler))
        .route("/admin/policy", get(get_policy).post(set_policy))
        .route("/admin/fleet", get(fleet_status))
        .with_state(state)
}

async fn metrics_handler(
    State(state): State<SharedState>,
) -> ([(&'static str, &'static str); 1], String) {
    (
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        state.metrics.render(&state.fleet, state.policy_kind()),
    )
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PolicyView {
    pub policy: String,
    pub prefix_hash_tokens: usize,
    pub bounded_overload_factor: f64,
    pub bounded_min_capacity: usize,
}

impl From<PolicyConfig> for PolicyView {
    fn from(c: PolicyConfig) -> Self {
        Self {
            policy: c.kind.as_str().to_string(),
            prefix_hash_tokens: c.prefix_hash_tokens,
            bounded_overload_factor: c.bounded_overload_factor,
            bounded_min_capacity: c.bounded_min_capacity,
        }
    }
}

/// Body of `POST /admin/policy`. Every field but `policy` is optional and
/// defaults to the value currently in force, so an experiment can switch the
/// rule without restating the tuning.
#[derive(Debug, Deserialize)]
pub struct SetPolicyRequest {
    pub policy: String,
    pub prefix_hash_tokens: Option<usize>,
    pub bounded_overload_factor: Option<f64>,
    pub bounded_min_capacity: Option<usize>,
}

async fn get_policy(State(state): State<SharedState>) -> Json<PolicyView> {
    Json(state.policy_config().into())
}

async fn set_policy(
    State(state): State<SharedState>,
    Json(req): Json<SetPolicyRequest>,
) -> Result<Json<serde_json::Value>> {
    let kind = PolicyKind::parse(&req.policy)
        .ok_or_else(|| RouterError::UnknownPolicy(req.policy.clone()))?;
    let current = state.policy_config();
    let next = PolicyConfig {
        kind,
        prefix_hash_tokens: req.prefix_hash_tokens.unwrap_or(current.prefix_hash_tokens),
        bounded_overload_factor: req
            .bounded_overload_factor
            .unwrap_or(current.bounded_overload_factor),
        bounded_min_capacity: req
            .bounded_min_capacity
            .unwrap_or(current.bounded_min_capacity),
    };
    if next.bounded_overload_factor < 1.0 {
        return Err(RouterError::BadRequest(
            "bounded_overload_factor must be >= 1.0".into(),
        ));
    }
    let previous = state.set_policy(next.clone());
    state
        .metrics
        .policy_switches_total
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    tracing::info!(
        from = previous.as_str(),
        to = kind.as_str(),
        prefix_hash_tokens = next.prefix_hash_tokens,
        bounded_overload_factor = next.bounded_overload_factor,
        bounded_min_capacity = next.bounded_min_capacity,
        "policy_switched"
    );
    Ok(Json(serde_json::json!({
        "previous": previous.as_str(),
        "active": PolicyView::from(next),
    })))
}

#[derive(Debug, Serialize)]
pub struct ReplicaStatus {
    pub id: String,
    pub url: String,
    pub placement: Option<String>,
    pub healthy: bool,
    pub inflight: usize,
    pub queued: usize,
    pub dispatched_total: u64,
    pub upstream_errors_total: u64,
    pub ejections_total: u64,
    pub recoveries_total: u64,
    pub last_transition_unix_us: u64,
}

#[derive(Debug, Serialize)]
pub struct FleetStatus {
    pub policy: String,
    pub healthy_replicas: usize,
    pub total_replicas: usize,
    pub admitted_inflight: usize,
    pub admission_waiting: usize,
    pub capacity: usize,
    pub replicas: Vec<ReplicaStatus>,
}

/// Machine-readable fleet state, polled by the experiment driver between phases
/// to confirm the fleet is drained and fully healthy before a measurement
/// starts. A phase that begins with a replica still ejected or still draining is
/// not the configuration it claims to be, and the driver records the check
/// rather than assuming it.
async fn fleet_status(State(state): State<SharedState>) -> Json<FleetStatus> {
    let healthy = state.fleet.healthy_indices().len();
    Json(FleetStatus {
        policy: state.policy_kind().as_str().to_string(),
        healthy_replicas: healthy,
        total_replicas: state.fleet.len(),
        admitted_inflight: state.admission.inflight(),
        admission_waiting: state.admission.waiting(),
        capacity: state.admission.capacity(healthy),
        replicas: state
            .fleet
            .replicas
            .iter()
            .map(|r| ReplicaStatus {
                id: r.id.clone(),
                url: r.url.clone(),
                placement: r.placement.clone(),
                healthy: r.is_healthy(),
                inflight: r.inflight(),
                queued: r.queued(),
                dispatched_total: r.dispatched_total(),
                upstream_errors_total: r.upstream_errors_total(),
                ejections_total: r.ejections_total(),
                recoveries_total: r.recoveries_total(),
                last_transition_unix_us: r.last_transition_unix_us(),
            })
            .collect(),
    })
}
