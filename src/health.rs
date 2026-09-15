//! Active health checking with ejection and recovery.
//!
//! Active probing rather than passive circuit-breaking on request failures, for
//! one reason specific to this workload: under prefix affinity a replica can go
//! several seconds between requests at low rates, so a passive breaker would
//! have no signal at all until traffic happened to land on the dead replica.
//! Detection time would then depend on the routing policy under test, which
//! would make the failure-mode experiment measure the wrong thing entirely.
//!
//! Detection latency is bounded by `interval_ms * unhealthy_threshold` plus one
//! probe timeout. With the defaults (500 ms, 2) that is 1.0-2.0 s. The
//! failure-mode experiment measures the realised value against the externally
//! recorded kill timestamp rather than quoting this bound.

use std::sync::Arc;
use std::time::Duration;

use crate::state::SharedState;

pub fn spawn(state: SharedState) -> Option<tokio::task::JoinHandle<()>> {
    if !state.config.health.enabled {
        tracing::warn!("health checking disabled by config; replicas will never be ejected");
        return None;
    }
    Some(tokio::spawn(async move { run(state).await }))
}

async fn run(state: SharedState) {
    let hc = state.config.health.clone();
    let interval = Duration::from_millis(hc.interval_ms);
    let timeout = Duration::from_millis(hc.timeout_ms);

    // A dedicated client: health probes must not queue behind serving traffic in
    // a saturated connection pool, or a fully healthy but busy replica gets
    // ejected for being busy. That is a real and common production bug and it
    // would be indistinguishable in the results from a genuine ejection.
    let client = match reqwest::Client::builder()
        .connect_timeout(timeout)
        .timeout(timeout)
        .pool_max_idle_per_host(2)
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(error = %e, "health client build failed; health checking is off");
            return;
        }
    };

    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        ticker.tick().await;
        let mut probes = Vec::with_capacity(state.fleet.len());
        for replica in &state.fleet.replicas {
            let client = client.clone();
            let url = format!("{}{}", replica.url, hc.path);
            let replica = Arc::clone(replica);
            let hc = hc.clone();
            // Probe every replica concurrently: sequential probes would make
            // detection time depend on a replica's position in the list.
            probes.push(tokio::spawn(async move {
                let ok = matches!(client.get(&url).send().await, Ok(r) if r.status().is_success());
                if ok {
                    if replica.record_health_success(hc.healthy_threshold) {
                        tracing::info!(
                            replica = %replica.id,
                            placement = replica.placement.as_deref().unwrap_or("unspecified"),
                            transition_unix_us = replica.last_transition_unix_us(),
                            "replica_recovered"
                        );
                    }
                } else if replica.record_health_failure(hc.unhealthy_threshold) {
                    tracing::warn!(
                        replica = %replica.id,
                        placement = replica.placement.as_deref().unwrap_or("unspecified"),
                        transition_unix_us = replica.last_transition_unix_us(),
                        inflight_at_ejection = replica.inflight(),
                        "replica_ejected"
                    );
                }
            }));
        }
        for p in probes {
            let _ = p.await;
        }
    }
}
