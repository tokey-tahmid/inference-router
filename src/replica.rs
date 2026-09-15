//! Replica state: liveness, in-flight accounting, and queue depth.
//!
//! All counters are atomics rather than a lock-guarded struct. Routing happens
//! on every request on the hot path, and every policy reads the in-flight counts
//! of the whole fleet, so a mutex here would serialise the router across all
//! connections. The cost of relaxed atomics is that a policy sees a very
//! slightly stale view of load; that is fine, because the true load is stale by
//! the time the request reaches the replica anyway.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::ReplicaConfig;

/// A single upstream inference replica.
#[derive(Debug)]
pub struct Replica {
    /// Index into `Fleet::replicas`; used as the compact key in hot paths.
    pub index: usize,
    pub id: String,
    pub url: String,
    pub placement: Option<String>,

    /// Requests dispatched to this replica and not yet fully streamed back.
    inflight: AtomicUsize,
    /// Requests admitted by the router and waiting for a dispatch slot.
    queued: AtomicUsize,
    /// Lifetime dispatch count, for the fairness metric.
    dispatched_total: AtomicU64,
    /// Lifetime count of requests that failed upstream after dispatch.
    upstream_errors_total: AtomicU64,

    healthy: AtomicBool,
    consecutive_failures: AtomicU32,
    consecutive_successes: AtomicU32,
    /// Lifetime ejection/recovery counts and the wall-clock instant of the most
    /// recent transition, in microseconds since the Unix epoch.
    ///
    /// Wall clock rather than `Instant` deliberately: the failure-mode experiment
    /// correlates the router's ejection against the moment an external process
    /// killed the replica, and those two clocks have to be comparable.
    ejections_total: AtomicU64,
    recoveries_total: AtomicU64,
    last_transition_unix_us: AtomicU64,
}

fn now_unix_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

impl Replica {
    fn new(index: usize, cfg: &ReplicaConfig) -> Self {
        Self {
            index,
            id: cfg.id.clone(),
            url: cfg.url.trim_end_matches('/').to_string(),
            placement: cfg.placement.clone(),
            inflight: AtomicUsize::new(0),
            queued: AtomicUsize::new(0),
            dispatched_total: AtomicU64::new(0),
            upstream_errors_total: AtomicU64::new(0),
            // Optimistic start: a replica is assumed healthy until a probe says
            // otherwise. The alternative (start unhealthy, wait for a probe)
            // would make the router 503 for the first health interval after
            // launch, which in the experiment harness reads as a fleet outage.
            healthy: AtomicBool::new(true),
            consecutive_failures: AtomicU32::new(0),
            consecutive_successes: AtomicU32::new(0),
            ejections_total: AtomicU64::new(0),
            recoveries_total: AtomicU64::new(0),
            last_transition_unix_us: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn inflight(&self) -> usize {
        self.inflight.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn queued(&self) -> usize {
        self.queued.load(Ordering::Relaxed)
    }

    /// In-flight plus queued: the quantity `least_loaded` actually wants.
    ///
    /// In-flight alone under-counts a replica that is about to receive work, and
    /// during a burst that difference is exactly when the policy's decision
    /// matters.
    #[inline]
    pub fn load(&self) -> usize {
        self.inflight() + self.queued()
    }

    #[inline]
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    pub fn dispatched_total(&self) -> u64 {
        self.dispatched_total.load(Ordering::Relaxed)
    }
    pub fn upstream_errors_total(&self) -> u64 {
        self.upstream_errors_total.load(Ordering::Relaxed)
    }
    pub fn ejections_total(&self) -> u64 {
        self.ejections_total.load(Ordering::Relaxed)
    }
    pub fn recoveries_total(&self) -> u64 {
        self.recoveries_total.load(Ordering::Relaxed)
    }
    pub fn last_transition_unix_us(&self) -> u64 {
        self.last_transition_unix_us.load(Ordering::Relaxed)
    }

    pub fn enqueue(&self) {
        self.queued.fetch_add(1, Ordering::Relaxed);
    }
    pub fn dequeue(&self) {
        self.queued.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn note_upstream_error(&self) {
        self.upstream_errors_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a successful health probe, returning `true` if this probe caused
    /// the replica to be returned to service.
    pub fn record_health_success(&self, healthy_threshold: u32) -> bool {
        self.consecutive_failures.store(0, Ordering::Relaxed);
        let n = self.consecutive_successes.fetch_add(1, Ordering::Relaxed) + 1;
        if !self.is_healthy() && n >= healthy_threshold {
            self.healthy.store(true, Ordering::Relaxed);
            self.recoveries_total.fetch_add(1, Ordering::Relaxed);
            self.last_transition_unix_us
                .store(now_unix_us(), Ordering::Relaxed);
            return true;
        }
        false
    }

    /// Record a failed health probe, returning `true` if this probe caused the
    /// replica to be ejected.
    pub fn record_health_failure(&self, unhealthy_threshold: u32) -> bool {
        self.consecutive_successes.store(0, Ordering::Relaxed);
        let n = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        if self.is_healthy() && n >= unhealthy_threshold {
            self.healthy.store(false, Ordering::Relaxed);
            self.ejections_total.fetch_add(1, Ordering::Relaxed);
            self.last_transition_unix_us
                .store(now_unix_us(), Ordering::Relaxed);
            return true;
        }
        false
    }
}

/// RAII guard for the in-flight count.
///
/// This is the piece that makes streaming accounting correct. A request is
/// in flight for as long as its response body is still being streamed to the
/// client, which is long after the proxy handler returns. The guard is moved
/// into the response stream's state, so the decrement happens when the stream is
/// dropped: on normal completion, on upstream error, and on client disconnect
/// alike. Decrementing in the handler instead would report a fleet that looks
/// idle while every replica is saturated, and `least_loaded` would then route on
/// a number that is always zero.
#[derive(Debug)]
pub struct InflightGuard {
    replica: Arc<Replica>,
}

impl InflightGuard {
    pub fn acquire(replica: Arc<Replica>) -> Self {
        replica.inflight.fetch_add(1, Ordering::Relaxed);
        replica.dispatched_total.fetch_add(1, Ordering::Relaxed);
        Self { replica }
    }

    pub fn replica(&self) -> &Arc<Replica> {
        &self.replica
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.replica.inflight.fetch_sub(1, Ordering::Relaxed);
    }
}

/// The whole replica set plus derived views of it.
#[derive(Debug)]
pub struct Fleet {
    pub replicas: Vec<Arc<Replica>>,
}

impl Fleet {
    pub fn new(cfgs: &[ReplicaConfig]) -> Self {
        Self {
            replicas: cfgs
                .iter()
                .enumerate()
                .map(|(i, c)| Arc::new(Replica::new(i, c)))
                .collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.replicas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.replicas.is_empty()
    }

    /// Indices of currently healthy replicas.
    ///
    /// Returned as a `Vec` of indices rather than references so policies can
    /// index cheaply and so the healthy set is sampled exactly once per routing
    /// decision. Sampling it repeatedly inside a policy would let a replica be
    /// ejected halfway through a decision and produce an inconsistent choice.
    pub fn healthy_indices(&self) -> Vec<usize> {
        self.replicas
            .iter()
            .enumerate()
            .filter(|(_, r)| r.is_healthy())
            .map(|(i, _)| i)
            .collect()
    }

    pub fn total_inflight(&self) -> usize {
        self.replicas.iter().map(|r| r.inflight()).sum()
    }

    pub fn total_load(&self) -> usize {
        self.replicas.iter().map(|r| r.load()).sum()
    }

    pub fn get(&self, idx: usize) -> &Arc<Replica> {
        &self.replicas[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fleet(n: usize) -> Fleet {
        let cfgs: Vec<ReplicaConfig> = (0..n)
            .map(|i| ReplicaConfig {
                id: format!("r{i}"),
                url: format!("http://127.0.0.1:{}", 8100 + i),
                placement: Some(format!("gpu{i}")),
            })
            .collect();
        Fleet::new(&cfgs)
    }

    #[test]
    fn inflight_guard_increments_and_decrements() {
        let f = fleet(2);
        let r = f.get(0).clone();
        assert_eq!(r.inflight(), 0);
        {
            let _g = InflightGuard::acquire(r.clone());
            assert_eq!(r.inflight(), 1);
            let _g2 = InflightGuard::acquire(r.clone());
            assert_eq!(r.inflight(), 2);
        }
        assert_eq!(r.inflight(), 0);
        // dispatch count is cumulative and does not decrement
        assert_eq!(r.dispatched_total(), 2);
    }

    #[test]
    fn trailing_slash_is_stripped_from_url() {
        let f = Fleet::new(&[ReplicaConfig {
            id: "r0".into(),
            url: "http://127.0.0.1:8100/".into(),
            placement: None,
        }]);
        assert_eq!(f.get(0).url, "http://127.0.0.1:8100");
    }

    #[test]
    fn ejection_requires_threshold_consecutive_failures() {
        let f = fleet(1);
        let r = f.get(0);
        assert!(r.is_healthy());
        assert!(!r.record_health_failure(3));
        assert!(r.is_healthy());
        assert!(!r.record_health_failure(3));
        assert!(r.is_healthy());
        assert!(r.record_health_failure(3), "third failure ejects");
        assert!(!r.is_healthy());
        assert_eq!(r.ejections_total(), 1);
        assert!(r.last_transition_unix_us() > 0);
    }

    #[test]
    fn a_success_resets_the_failure_streak() {
        let f = fleet(1);
        let r = f.get(0);
        r.record_health_failure(3);
        r.record_health_failure(3);
        r.record_health_success(1);
        // streak reset: two more failures must not eject
        assert!(!r.record_health_failure(3));
        assert!(r.is_healthy());
    }

    #[test]
    fn recovery_requires_threshold_consecutive_successes() {
        let f = fleet(1);
        let r = f.get(0);
        r.record_health_failure(1);
        assert!(!r.is_healthy());
        assert!(!r.record_health_success(2));
        assert!(!r.is_healthy());
        assert!(r.record_health_success(2));
        assert!(r.is_healthy());
        assert_eq!(r.recoveries_total(), 1);
    }

    #[test]
    fn healthy_indices_excludes_ejected() {
        let f = fleet(4);
        f.get(1).record_health_failure(1);
        f.get(3).record_health_failure(1);
        assert_eq!(f.healthy_indices(), vec![0, 2]);
    }

    #[test]
    fn load_sums_inflight_and_queued() {
        let f = fleet(1);
        let r = f.get(0).clone();
        let _g = InflightGuard::acquire(r.clone());
        r.enqueue();
        r.enqueue();
        assert_eq!(r.inflight(), 1);
        assert_eq!(r.queued(), 2);
        assert_eq!(r.load(), 3);
        r.dequeue();
        assert_eq!(r.load(), 2);
    }
}
