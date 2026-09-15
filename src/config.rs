//! Router configuration, loaded from TOML.
//!
//! Everything that changes between experiment arms is here rather than compiled
//! in, because the experiment matrix switches policy at runtime against one
//! long-lived replica fleet. Re-launching four vLLM replicas per arm would cost
//! several minutes of model load per point and, worse, would make the policy
//! comparison a comparison of four separately-warmed fleets.

use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

fn d_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "Config::default_listen")]
    pub listen: SocketAddr,

    /// Upstream replicas. Order is stable and defines replica indices, which are
    /// what the per-replica metrics and the `x-router-replica` response header
    /// report, so analysis can attribute every request to a specific GPU.
    pub replicas: Vec<ReplicaConfig>,

    #[serde(default)]
    pub policy: PolicyConfig,

    #[serde(default)]
    pub health: HealthConfig,

    #[serde(default)]
    pub admission: AdmissionConfig,

    #[serde(default)]
    pub upstream: UpstreamConfig,
}

impl Config {
    fn default_listen() -> SocketAddr {
        "0.0.0.0:8080".parse().expect("valid default listen addr")
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading config {}: {e}", path.display()))?;
        let cfg: Config = toml::from_str(&text)
            .map_err(|e| anyhow::anyhow!("parsing config {}: {e}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.replicas.is_empty() {
            anyhow::bail!("config declares no replicas");
        }
        let mut ids: Vec<&str> = self.replicas.iter().map(|r| r.id.as_str()).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        if ids.len() != before {
            anyhow::bail!("replica ids must be unique");
        }
        if self.policy.bounded_overload_factor < 1.0 {
            anyhow::bail!("bounded_overload_factor must be >= 1.0 (1.0 means perfectly even)");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaConfig {
    /// Stable identifier, e.g. `r0`. Used as a metrics label, so keep it short
    /// and keep the set fixed for the life of an experiment.
    pub id: String,
    /// Base URL including scheme and port, no trailing slash, e.g.
    /// `http://localhost:8100`.
    pub url: String,
    /// Free-form provenance recorded in `/metrics` help text and in the router's
    /// startup log: which GPU and which node this replica occupies. Not used for
    /// routing; it exists so a results artifact can state the topology exactly.
    #[serde(default)]
    pub placement: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyKind {
    RoundRobin,
    LeastLoaded,
    PrefixAffinity,
    PrefixAffinityBounded,
}

impl PolicyKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::RoundRobin => "round_robin",
            Self::LeastLoaded => "least_loaded",
            Self::PrefixAffinity => "prefix_affinity",
            Self::PrefixAffinityBounded => "prefix_affinity_bounded",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "round_robin" => Some(Self::RoundRobin),
            "least_loaded" => Some(Self::LeastLoaded),
            "prefix_affinity" => Some(Self::PrefixAffinity),
            "prefix_affinity_bounded" => Some(Self::PrefixAffinityBounded),
            _ => None,
        }
    }

    pub const ALL: [PolicyKind; 4] = [
        Self::RoundRobin,
        Self::LeastLoaded,
        Self::PrefixAffinity,
        Self::PrefixAffinityBounded,
    ];
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    #[serde(default = "PolicyConfig::default_kind")]
    pub kind: PolicyKind,

    /// How many leading prompt tokens (or characters, in text mode) define the
    /// affinity key.
    ///
    /// This must be **at most** the workload's shared prefix length, or two
    /// requests that genuinely share a 512-token system prompt but diverge after
    /// it would hash differently and affinity would degenerate to random
    /// placement. Set it to the shared-prefix length the workload uses.
    #[serde(default = "PolicyConfig::default_prefix_tokens")]
    pub prefix_hash_tokens: usize,

    /// Bounded-load guard for `prefix_affinity_bounded`.
    ///
    /// Capacity per replica is `ceil(factor * mean_inflight)` floored at
    /// `bounded_min_capacity`. 1.0 forces perfect balance and destroys affinity;
    /// a large value degenerates to plain `prefix_affinity`.
    #[serde(default = "PolicyConfig::default_overload_factor")]
    pub bounded_overload_factor: f64,

    /// Floor on the per-replica capacity bound.
    ///
    /// Without it, at low load the mean in-flight count is near zero, the bound
    /// rounds to 1, and the "bounded" policy spills on the second concurrent
    /// request, i.e. it becomes round-robin exactly where affinity is cheapest
    /// and most useful.
    #[serde(default = "PolicyConfig::default_min_capacity")]
    pub bounded_min_capacity: usize,
}

impl PolicyConfig {
    fn default_kind() -> PolicyKind {
        PolicyKind::RoundRobin
    }
    fn default_prefix_tokens() -> usize {
        512
    }
    fn default_overload_factor() -> f64 {
        1.25
    }
    fn default_min_capacity() -> usize {
        4
    }
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            kind: Self::default_kind(),
            prefix_hash_tokens: Self::default_prefix_tokens(),
            bounded_overload_factor: Self::default_overload_factor(),
            bounded_min_capacity: Self::default_min_capacity(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    #[serde(default = "d_true")]
    pub enabled: bool,
    /// Poll period. Detection time for a dead replica is bounded below by
    /// `interval_ms * unhealthy_threshold`, which is the quantity the
    /// failure-mode experiment measures against.
    #[serde(default = "HealthConfig::default_interval")]
    pub interval_ms: u64,
    #[serde(default = "HealthConfig::default_timeout")]
    pub timeout_ms: u64,
    /// Consecutive failures before ejection. >1 so a single dropped probe during
    /// a GC pause or a busy scheduler step does not eject a healthy replica.
    #[serde(default = "HealthConfig::default_unhealthy")]
    pub unhealthy_threshold: u32,
    /// Consecutive successes before a previously ejected replica is returned to
    /// service.
    #[serde(default = "HealthConfig::default_healthy")]
    pub healthy_threshold: u32,
    #[serde(default = "HealthConfig::default_path")]
    pub path: String,
}

impl HealthConfig {
    fn default_interval() -> u64 {
        500
    }
    fn default_timeout() -> u64 {
        1000
    }
    fn default_unhealthy() -> u32 {
        2
    }
    fn default_healthy() -> u32 {
        2
    }
    fn default_path() -> String {
        "/health".to_string()
    }
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_ms: Self::default_interval(),
            timeout_ms: Self::default_timeout(),
            unhealthy_threshold: Self::default_unhealthy(),
            healthy_threshold: Self::default_healthy(),
            path: Self::default_path(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverloadBehaviour {
    /// Return 429 immediately once the fleet is at capacity. Honest
    /// backpressure: the client learns at once and can shed or retry.
    Reject,
    /// Hold the request in a bounded queue until a slot frees or
    /// `queue_timeout_ms` elapses (then 429). Trades latency for goodput.
    Queue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionConfig {
    #[serde(default = "AdmissionConfig::default_behaviour")]
    pub overload_behaviour: OverloadBehaviour,
    /// Per-replica in-flight ceiling. Fleet capacity is this times the number of
    /// **healthy** replicas, so an ejection tightens the bound automatically
    /// instead of letting the survivors absorb the dead replica's share
    /// unbounded.
    #[serde(default = "AdmissionConfig::default_max_inflight")]
    pub max_inflight_per_replica: usize,
    /// Bound on the admission queue itself, in requests. Unbounded queues turn
    /// an overload into an unbounded latency excursion instead of a visible
    /// rejection, which is the classic bufferbloat failure.
    #[serde(default = "AdmissionConfig::default_queue_depth")]
    pub max_queue_depth: usize,
    #[serde(default = "AdmissionConfig::default_queue_timeout")]
    pub queue_timeout_ms: u64,
}

impl AdmissionConfig {
    fn default_behaviour() -> OverloadBehaviour {
        OverloadBehaviour::Reject
    }
    fn default_max_inflight() -> usize {
        64
    }
    fn default_queue_depth() -> usize {
        256
    }
    fn default_queue_timeout() -> u64 {
        10_000
    }
    pub fn queue_timeout(&self) -> Duration {
        Duration::from_millis(self.queue_timeout_ms)
    }
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            overload_behaviour: Self::default_behaviour(),
            max_inflight_per_replica: Self::default_max_inflight(),
            max_queue_depth: Self::default_queue_depth(),
            queue_timeout_ms: Self::default_queue_timeout(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamConfig {
    /// Time allowed to establish a connection and receive response *headers*.
    ///
    /// Note this is not a total-request timeout, and must not be: a streaming
    /// completion legitimately runs for tens of seconds, and a total timeout
    /// would sever long generations and record them as upstream failures.
    #[serde(default = "UpstreamConfig::default_connect")]
    pub connect_timeout_ms: u64,
    /// Idle timeout *between* body chunks. This is the right shape of timeout
    /// for a stream: it catches a wedged replica without penalising a long but
    /// healthy generation.
    #[serde(default = "UpstreamConfig::default_read")]
    pub read_timeout_ms: u64,
    #[serde(default = "UpstreamConfig::default_pool")]
    pub pool_max_idle_per_host: usize,
}

impl UpstreamConfig {
    fn default_connect() -> u64 {
        5_000
    }
    fn default_read() -> u64 {
        120_000
    }
    fn default_pool() -> usize {
        256
    }
}

impl Default for UpstreamConfig {
    fn default() -> Self {
        Self {
            connect_timeout_ms: Self::default_connect(),
            read_timeout_ms: Self::default_read(),
            pool_max_idle_per_host: Self::default_pool(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_kind_round_trips_through_str() {
        for k in PolicyKind::ALL {
            assert_eq!(PolicyKind::parse(k.as_str()), Some(k));
        }
        assert_eq!(PolicyKind::parse("nonsense"), None);
    }

    #[test]
    fn rejects_duplicate_replica_ids() {
        let cfg = Config {
            listen: Config::default_listen(),
            replicas: vec![
                ReplicaConfig {
                    id: "r0".into(),
                    url: "http://a".into(),
                    placement: None,
                },
                ReplicaConfig {
                    id: "r0".into(),
                    url: "http://b".into(),
                    placement: None,
                },
            ],
            policy: PolicyConfig::default(),
            health: HealthConfig::default(),
            admission: AdmissionConfig::default(),
            upstream: UpstreamConfig::default(),
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_empty_replica_set() {
        let cfg = Config {
            listen: Config::default_listen(),
            replicas: vec![],
            policy: PolicyConfig::default(),
            health: HealthConfig::default(),
            admission: AdmissionConfig::default(),
            upstream: UpstreamConfig::default(),
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn parses_minimal_toml() {
        let cfg: Config = toml::from_str(
            r#"
            listen = "127.0.0.1:9000"
            [[replicas]]
            id = "r0"
            url = "http://127.0.0.1:8100"
            "#,
        )
        .expect("minimal config parses");
        assert_eq!(cfg.replicas.len(), 1);
        assert_eq!(cfg.policy.kind, PolicyKind::RoundRobin);
        assert!(cfg.health.enabled);
    }
}
