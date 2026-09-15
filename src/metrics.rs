//! Prometheus metrics, exposed at `/metrics` in the text exposition format.
//!
//! Hand-rolled rather than pulled from a crate, for two reasons. The histogram
//! buckets need to be chosen for LLM serving (TTFT spans roughly 10 ms to 10 s,
//! which a default web-latency bucket set clips at both ends and reports as an
//! uninformative `+Inf` pile), and the whole thing is about 200 lines of
//! atomics. Everything is lock-free so scraping cannot perturb the serving path
//! that is being measured.
//!
//! All counters are **cumulative for the life of the process**, which is the
//! Prometheus convention. The experiment driver runs many phases against one
//! long-lived router, so it differences consecutive scrapes to get per-phase
//! values. This is the same discipline P1 had to adopt for vLLM's prefix-cache
//! counters after a phase-ordering artefact was traced to reading them as
//! absolute (see `../llm-inference-bench/docs/notes.md`, 2026-08-31).

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::config::PolicyKind;
use crate::policy::RouteReason;
use crate::replica::Fleet;

/// Bucket upper bounds in seconds, chosen for LLM serving latencies.
///
/// Covers a ~20 ms cached prefill through a ~120 s saturated long generation.
/// The bottom bucket is below the fastest TTFT P1 measured (20.7 ms at 128
/// input tokens) so the low end is resolved rather than collapsed.
pub const LATENCY_BUCKETS: [f64; 16] = [
    0.005, 0.010, 0.025, 0.050, 0.075, 0.100, 0.250, 0.500, 0.750, 1.0, 2.5, 5.0, 10.0, 30.0,
    60.0, 120.0,
];

#[derive(Debug, Default)]
pub struct Histogram {
    buckets: [AtomicU64; LATENCY_BUCKETS.len()],
    count: AtomicU64,
    /// Sum in microseconds, kept integral so the metric is exactly additive.
    /// Accumulating an f64 through an atomic would need a CAS loop and would
    /// drift; microseconds in a u64 do not overflow this side of 500,000 years.
    sum_us: AtomicU64,
}

impl Histogram {
    pub fn observe(&self, seconds: f64) {
        let us = (seconds * 1e6).max(0.0) as u64;
        self.sum_us.fetch_add(us, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        for (i, ub) in LATENCY_BUCKETS.iter().enumerate() {
            if seconds <= *ub {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn render(&self, out: &mut String, name: &str, labels: &str) {
        // Buckets are already cumulative because `observe` increments every
        // bucket whose bound the sample is under.
        for (i, ub) in LATENCY_BUCKETS.iter().enumerate() {
            let v = self.buckets[i].load(Ordering::Relaxed);
            let sep = if labels.is_empty() { "" } else { "," };
            let _ = writeln!(out, "{name}_bucket{{{labels}{sep}le=\"{ub}\"}} {v}");
        }
        let total = self.count.load(Ordering::Relaxed);
        let sep = if labels.is_empty() { "" } else { "," };
        let _ = writeln!(out, "{name}_bucket{{{labels}{sep}le=\"+Inf\"}} {total}");
        let lb = if labels.is_empty() {
            String::new()
        } else {
            format!("{{{labels}}}")
        };
        let sum = self.sum_us.load(Ordering::Relaxed) as f64 / 1e6;
        let _ = writeln!(out, "{name}_sum{lb} {sum}");
        let _ = writeln!(out, "{name}_count{lb} {total}");
    }
}

#[derive(Debug, Default)]
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub requests_streaming_total: AtomicU64,
    pub responses_ok_total: AtomicU64,
    pub responses_error_total: AtomicU64,
    pub rejected_overload_total: AtomicU64,
    pub rejected_no_healthy_total: AtomicU64,
    pub admission_queue_timeout_total: AtomicU64,
    pub upstream_error_total: AtomicU64,
    pub client_disconnect_total: AtomicU64,
    pub affinity_key_missing_total: AtomicU64,
    pub bytes_streamed_total: AtomicU64,
    pub sse_chunks_total: AtomicU64,
    pub policy_switches_total: AtomicU64,

    pub ttft: Histogram,
    pub e2e: Histogram,
    pub queue_wait: Histogram,

    /// Per-replica dispatch counts by route reason, flattened as
    /// `reason_index * n_replicas + replica_index`. Sized at construction.
    route_reason_counts: Vec<AtomicU64>,
    n_replicas: usize,
    /// Per-replica TTFT, so a hot replica's tail is visible without joining
    /// against the driver's artifacts.
    per_replica_ttft: Vec<Histogram>,
}

impl Metrics {
    pub fn new(n_replicas: usize) -> Self {
        Self {
            route_reason_counts: (0..n_replicas * RouteReason::ALL.len())
                .map(|_| AtomicU64::new(0))
                .collect(),
            per_replica_ttft: (0..n_replicas).map(|_| Histogram::default()).collect(),
            n_replicas,
            ..Default::default()
        }
    }

    pub fn record_route(&self, replica_index: usize, reason: RouteReason) {
        let r = RouteReason::ALL
            .iter()
            .position(|x| *x == reason)
            .unwrap_or(0);
        if let Some(c) = self.route_reason_counts.get(r * self.n_replicas + replica_index) {
            c.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_ttft(&self, replica_index: usize, seconds: f64) {
        self.ttft.observe(seconds);
        if let Some(h) = self.per_replica_ttft.get(replica_index) {
            h.observe(seconds);
        }
    }

    /// Render the full exposition.
    pub fn render(&self, fleet: &Fleet, policy: PolicyKind) -> String {
        let mut out = String::with_capacity(8192);

        macro_rules! counter {
            ($field:ident, $name:literal, $help:literal) => {
                let _ = writeln!(out, "# HELP {} {}", $name, $help);
                let _ = writeln!(out, "# TYPE {} counter", $name);
                let _ = writeln!(
                    out,
                    "{} {}",
                    $name,
                    self.$field.load(Ordering::Relaxed)
                );
            };
        }

        counter!(
            requests_total,
            "router_requests_total",
            "Requests accepted by the proxy handlers."
        );
        counter!(
            requests_streaming_total,
            "router_requests_streaming_total",
            "Subset of requests that asked for stream=true."
        );
        counter!(
            responses_ok_total,
            "router_responses_ok_total",
            "Responses whose upstream returned a 2xx."
        );
        counter!(
            responses_error_total,
            "router_responses_error_total",
            "Requests the router failed for any reason."
        );
        counter!(
            rejected_overload_total,
            "router_rejected_overload_total",
            "Requests rejected with 429 by the admission controller."
        );
        counter!(
            rejected_no_healthy_total,
            "router_rejected_no_healthy_total",
            "Requests rejected with 503 because no replica was healthy."
        );
        counter!(
            admission_queue_timeout_total,
            "router_admission_queue_timeout_total",
            "Requests that waited out queue_timeout_ms and were then rejected."
        );
        counter!(
            upstream_error_total,
            "router_upstream_error_total",
            "Dispatches that failed at the replica."
        );
        counter!(
            client_disconnect_total,
            "router_client_disconnect_total",
            "Streams terminated because the client went away."
        );
        counter!(
            affinity_key_missing_total,
            "router_affinity_key_missing_total",
            "Requests with no extractable prompt, routed by load instead."
        );
        counter!(
            bytes_streamed_total,
            "router_bytes_streamed_total",
            "Response body bytes forwarded downstream."
        );
        counter!(
            sse_chunks_total,
            "router_sse_chunks_total",
            "Body chunks forwarded downstream. Compare against tokens generated: a chunk count near 1 per request means the response was buffered, not streamed."
        );
        counter!(
            policy_switches_total,
            "router_policy_switches_total",
            "Runtime routing-policy changes via /admin/policy."
        );

        let _ = writeln!(out, "# HELP router_active_policy Active routing policy (1 for the active one, 0 otherwise).");
        let _ = writeln!(out, "# TYPE router_active_policy gauge");
        for k in PolicyKind::ALL {
            let _ = writeln!(
                out,
                "router_active_policy{{policy=\"{}\"}} {}",
                k.as_str(),
                u8::from(k == policy)
            );
        }

        let _ = writeln!(out, "# HELP router_ttft_seconds Time to first response byte, router-observed.");
        let _ = writeln!(out, "# TYPE router_ttft_seconds histogram");
        self.ttft.render(&mut out, "router_ttft_seconds", "");

        let _ = writeln!(out, "# HELP router_e2e_seconds End-to-end request duration, router-observed.");
        let _ = writeln!(out, "# TYPE router_e2e_seconds histogram");
        self.e2e.render(&mut out, "router_e2e_seconds", "");

        let _ = writeln!(out, "# HELP router_queue_wait_seconds Time spent in the admission queue before dispatch.");
        let _ = writeln!(out, "# TYPE router_queue_wait_seconds histogram");
        self.queue_wait.render(&mut out, "router_queue_wait_seconds", "");

        // -- per replica ----------------------------------------------------
        let _ = writeln!(out, "# HELP router_replica_inflight Requests dispatched and not yet fully streamed back.");
        let _ = writeln!(out, "# TYPE router_replica_inflight gauge");
        for r in &fleet.replicas {
            let _ = writeln!(
                out,
                "router_replica_inflight{{replica=\"{}\"}} {}",
                r.id,
                r.inflight()
            );
        }

        let _ = writeln!(out, "# HELP router_replica_queued Requests admitted and waiting for a dispatch slot.");
        let _ = writeln!(out, "# TYPE router_replica_queued gauge");
        for r in &fleet.replicas {
            let _ = writeln!(
                out,
                "router_replica_queued{{replica=\"{}\"}} {}",
                r.id,
                r.queued()
            );
        }

        let _ = writeln!(out, "# HELP router_replica_healthy 1 if the replica is in service, 0 if ejected.");
        let _ = writeln!(out, "# TYPE router_replica_healthy gauge");
        for r in &fleet.replicas {
            let _ = writeln!(
                out,
                "router_replica_healthy{{replica=\"{}\"}} {}",
                r.id,
                u8::from(r.is_healthy())
            );
        }

        let _ = writeln!(out, "# HELP router_replica_dispatched_total Requests dispatched to this replica. The fairness metric is computed from this.");
        let _ = writeln!(out, "# TYPE router_replica_dispatched_total counter");
        for r in &fleet.replicas {
            let placement = r.placement.as_deref().unwrap_or("unspecified");
            let _ = writeln!(
                out,
                "router_replica_dispatched_total{{replica=\"{}\",placement=\"{}\"}} {}",
                r.id,
                placement,
                r.dispatched_total()
            );
        }

        let _ = writeln!(out, "# HELP router_replica_upstream_errors_total Dispatches to this replica that failed.");
        let _ = writeln!(out, "# TYPE router_replica_upstream_errors_total counter");
        for r in &fleet.replicas {
            let _ = writeln!(
                out,
                "router_replica_upstream_errors_total{{replica=\"{}\"}} {}",
                r.id,
                r.upstream_errors_total()
            );
        }

        let _ = writeln!(out, "# HELP router_replica_ejections_total Health-check ejections of this replica.");
        let _ = writeln!(out, "# TYPE router_replica_ejections_total counter");
        for r in &fleet.replicas {
            let _ = writeln!(
                out,
                "router_replica_ejections_total{{replica=\"{}\"}} {}",
                r.id,
                r.ejections_total()
            );
        }

        let _ = writeln!(out, "# HELP router_replica_recoveries_total Health-check recoveries of this replica.");
        let _ = writeln!(out, "# TYPE router_replica_recoveries_total counter");
        for r in &fleet.replicas {
            let _ = writeln!(
                out,
                "router_replica_recoveries_total{{replica=\"{}\"}} {}",
                r.id,
                r.recoveries_total()
            );
        }

        let _ = writeln!(out, "# HELP router_replica_last_transition_unix_us Wall-clock microseconds of the last health transition, 0 if never. Used to measure ejection detection time against an externally recorded kill time.");
        let _ = writeln!(out, "# TYPE router_replica_last_transition_unix_us gauge");
        for r in &fleet.replicas {
            let _ = writeln!(
                out,
                "router_replica_last_transition_unix_us{{replica=\"{}\"}} {}",
                r.id,
                r.last_transition_unix_us()
            );
        }

        let _ = writeln!(out, "# HELP router_route_decisions_total Routing decisions by replica and reason. Spill and failover rates are the mechanism behind any latency difference between policies.");
        let _ = writeln!(out, "# TYPE router_route_decisions_total counter");
        for (ri, reason) in RouteReason::ALL.iter().enumerate() {
            for r in &fleet.replicas {
                let v = self
                    .route_reason_counts
                    .get(ri * self.n_replicas + r.index)
                    .map(|c| c.load(Ordering::Relaxed))
                    .unwrap_or(0);
                let _ = writeln!(
                    out,
                    "router_route_decisions_total{{replica=\"{}\",reason=\"{}\"}} {}",
                    r.id,
                    reason.as_str(),
                    v
                );
            }
        }

        let _ = writeln!(out, "# HELP router_replica_ttft_seconds Router-observed TTFT split by replica.");
        let _ = writeln!(out, "# TYPE router_replica_ttft_seconds histogram");
        for r in &fleet.replicas {
            if let Some(h) = self.per_replica_ttft.get(r.index) {
                h.render(
                    &mut out,
                    "router_replica_ttft_seconds",
                    &format!("replica=\"{}\"", r.id),
                );
            }
        }

        out
    }
}

/// Shared handle.
pub type SharedMetrics = Arc<Metrics>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ReplicaConfig;

    fn fleet(n: usize) -> Fleet {
        Fleet::new(
            &(0..n)
                .map(|i| ReplicaConfig {
                    id: format!("r{i}"),
                    url: format!("http://127.0.0.1:{}", 8100 + i),
                    placement: Some(format!("node0/gpu{i}")),
                })
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn histogram_buckets_are_cumulative_and_monotone() {
        let h = Histogram::default();
        for v in [0.001, 0.03, 0.3, 3.0, 300.0] {
            h.observe(v);
        }
        let mut s = String::new();
        h.render(&mut s, "x", "");
        let counts: Vec<u64> = s
            .lines()
            .filter(|l| l.starts_with("x_bucket"))
            .map(|l| l.rsplit(' ').next().unwrap().parse().unwrap())
            .collect();
        assert!(
            counts.windows(2).all(|w| w[0] <= w[1]),
            "bucket counts must be non-decreasing: {counts:?}"
        );
        assert_eq!(*counts.last().unwrap(), 5, "+Inf must hold every sample");
        assert!(s.contains("x_count 5"));
    }

    #[test]
    fn a_sample_above_every_bound_lands_only_in_inf() {
        let h = Histogram::default();
        h.observe(9_999.0);
        let mut s = String::new();
        h.render(&mut s, "x", "");
        assert!(s.contains("x_bucket{le=\"120\"} 0"));
        assert!(s.contains("x_bucket{le=\"+Inf\"} 1"));
    }

    #[test]
    fn exposition_is_well_formed_and_labels_every_replica() {
        let f = fleet(4);
        let m = Metrics::new(4);
        m.record_route(2, RouteReason::AffinityPrimary);
        m.record_ttft(2, 0.042);
        m.requests_total.fetch_add(7, Ordering::Relaxed);
        let text = m.render(&f, PolicyKind::PrefixAffinity);

        assert!(text.contains("router_requests_total 7"));
        assert!(text.contains("router_active_policy{policy=\"prefix_affinity\"} 1"));
        assert!(text.contains("router_active_policy{policy=\"round_robin\"} 0"));
        assert!(text.contains(
            "router_route_decisions_total{replica=\"r2\",reason=\"affinity_primary\"} 1"
        ));
        assert!(text.contains("router_replica_dispatched_total{replica=\"r0\",placement=\"node0/gpu0\"} 0"));
        assert!(text.contains("router_replica_ttft_seconds_count{replica=\"r2\"} 1"));

        // Every non-comment line must be `name value` or `name{labels} value`.
        for line in text.lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
            let (_, v) = line.rsplit_once(' ').expect("metric line has a value");
            assert!(
                v.parse::<f64>().is_ok(),
                "line is not valid exposition: {line}"
            );
        }
    }

    #[test]
    fn route_reason_labels_are_all_emitted() {
        let f = fleet(2);
        let m = Metrics::new(2);
        let text = m.render(&f, PolicyKind::RoundRobin);
        for reason in RouteReason::ALL {
            assert!(
                text.contains(&format!("reason=\"{}\"", reason.as_str())),
                "missing series for {reason:?}"
            );
        }
    }

    #[test]
    fn recording_an_out_of_range_replica_does_not_panic() {
        let m = Metrics::new(2);
        m.record_route(99, RouteReason::Rotation);
        m.record_ttft(99, 1.0);
    }
}
