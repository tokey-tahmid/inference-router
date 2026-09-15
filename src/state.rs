//! Shared router state.
//!
//! The active policy lives behind an `ArcSwap`-style `RwLock<Arc<dyn
//! RoutingPolicy>>` so it can be replaced at runtime by `/admin/policy` without
//! restarting the router.
//!
//! That endpoint is not a convenience. The experiment matrix is
//! policy x prefix-ratio x rate x repetition against a fleet of four vLLM
//! replicas that each take minutes to load 15 GB of weights from Lustre.
//! Restarting the fleet per policy would cost more node-hours than the
//! measurement itself and, far worse, would make the four policies a comparison
//! of four separately-warmed fleets rather than of four routing rules. Switching
//! the rule under one fleet holds everything else fixed.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;

use crate::admission::Admission;
use crate::config::{Config, PolicyConfig, PolicyKind};
use crate::metrics::{Metrics, SharedMetrics};
use crate::policy::{self, RoutingPolicy};
use crate::replica::Fleet;

pub struct AppState {
    pub fleet: Arc<Fleet>,
    pub metrics: SharedMetrics,
    pub admission: Arc<Admission>,
    pub http: reqwest::Client,
    pub config: Config,
    policy: RwLock<Arc<dyn RoutingPolicy>>,
    policy_cfg: RwLock<PolicyConfig>,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("replicas", &self.fleet.len())
            .field("policy", &self.policy_kind().as_str())
            .finish_non_exhaustive()
    }
}

impl AppState {
    pub fn new(config: Config) -> anyhow::Result<Arc<Self>> {
        let fleet = Arc::new(Fleet::new(&config.replicas));
        let metrics = Arc::new(Metrics::new(fleet.len()));
        let admission = Arc::new(Admission::new(config.admission.clone()));

        // No total-request timeout: a streaming completion legitimately runs for
        // tens of seconds and a total timeout would sever healthy generations
        // and record them as upstream failures. `read_timeout` is the right
        // shape — it fires on a *stall* between body chunks, which is what a
        // wedged replica actually looks like.
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(config.upstream.connect_timeout_ms))
            .read_timeout(Duration::from_millis(config.upstream.read_timeout_ms))
            .pool_max_idle_per_host(config.upstream.pool_max_idle_per_host)
            // Compression would force the proxy to decode and re-encode the
            // body, which defeats zero-copy passthrough. The replicas are on the
            // same node or the same IB fabric, so there is nothing to save.
            .no_gzip()
            .no_brotli()
            .no_deflate()
            .build()?;

        let policy = policy::build(&config.policy);
        Ok(Arc::new(Self {
            fleet,
            metrics,
            admission,
            http,
            policy: RwLock::new(policy),
            policy_cfg: RwLock::new(config.policy.clone()),
            config,
        }))
    }

    pub fn policy(&self) -> Arc<dyn RoutingPolicy> {
        Arc::clone(&self.policy.read())
    }

    pub fn policy_kind(&self) -> PolicyKind {
        self.policy.read().kind()
    }

    pub fn policy_config(&self) -> PolicyConfig {
        self.policy_cfg.read().clone()
    }

    /// Replace the active policy. Returns the previous kind.
    ///
    /// In-flight requests keep whatever placement they were already given; only
    /// subsequent routing decisions change. The experiment driver therefore
    /// drains the fleet and resets replica prefix caches around a switch, so a
    /// phase never mixes two policies' decisions.
    pub fn set_policy(&self, cfg: PolicyConfig) -> PolicyKind {
        let new = policy::build(&cfg);
        let mut guard = self.policy.write();
        let previous = guard.kind();
        *guard = new;
        *self.policy_cfg.write() = cfg;
        previous
    }
}

pub type SharedState = Arc<AppState>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AdmissionConfig, HealthConfig, ReplicaConfig, UpstreamConfig};

    fn cfg(n: usize) -> Config {
        Config {
            listen: "127.0.0.1:0".parse().unwrap(),
            replicas: (0..n)
                .map(|i| ReplicaConfig {
                    id: format!("r{i}"),
                    url: format!("http://127.0.0.1:{}", 8100 + i),
                    placement: None,
                })
                .collect(),
            policy: PolicyConfig::default(),
            health: HealthConfig::default(),
            admission: AdmissionConfig::default(),
            upstream: UpstreamConfig::default(),
        }
    }

    #[test]
    fn policy_can_be_swapped_at_runtime() {
        let st = AppState::new(cfg(4)).unwrap();
        assert_eq!(st.policy_kind(), PolicyKind::RoundRobin);
        let prev = st.set_policy(PolicyConfig {
            kind: PolicyKind::PrefixAffinityBounded,
            prefix_hash_tokens: 256,
            bounded_overload_factor: 1.5,
            bounded_min_capacity: 8,
        });
        assert_eq!(prev, PolicyKind::RoundRobin);
        assert_eq!(st.policy_kind(), PolicyKind::PrefixAffinityBounded);
        assert_eq!(st.policy_config().prefix_hash_tokens, 256);
        assert_eq!(st.policy_config().bounded_min_capacity, 8);
    }

    #[test]
    fn fleet_and_metrics_agree_on_replica_count() {
        let st = AppState::new(cfg(6)).unwrap();
        assert_eq!(st.fleet.len(), 6);
        let text = st.metrics.render(&st.fleet, st.policy_kind());
        for i in 0..6 {
            assert!(text.contains(&format!("replica=\"r{i}\"")));
        }
    }
}
