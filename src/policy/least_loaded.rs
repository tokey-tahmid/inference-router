//! Least-loaded: send to the healthy replica with the fewest in-flight plus
//! queued requests.
//!
//! Ties are broken by rotating cursor, not by always taking the lowest index.
//! With a fleet that is idle most of the time (the underload arms of the
//! matrix), every replica reads zero and a lowest-index tiebreak would send
//! *all* traffic to replica 0 — which looks like a catastrophic fairness result
//! and is entirely an artefact of the tiebreak. This is the subtlest bug in the
//! four policies and the fairness metric is precisely what would have hidden it
//! behind a plausible story about load-based routing.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::config::PolicyKind;
use crate::policy::{AffinityKey, Decision, RouteReason, RoutingPolicy};
use crate::replica::Fleet;

#[derive(Debug, Default)]
pub struct LeastLoaded {
    cursor: AtomicUsize,
}

impl LeastLoaded {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Pick the minimum-load index from `candidates`, rotating among ties.
pub(crate) fn least_loaded_of(
    fleet: &Fleet,
    candidates: &[usize],
    cursor: &AtomicUsize,
) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }
    let start = cursor.fetch_add(1, Ordering::Relaxed) % candidates.len();
    let mut best = candidates[start];
    let mut best_load = fleet.get(best).load();
    for off in 1..candidates.len() {
        let idx = candidates[(start + off) % candidates.len()];
        let load = fleet.get(idx).load();
        // Strictly less: the first candidate encountered in rotated order wins a
        // tie, which is what makes the tiebreak rotate.
        if load < best_load {
            best = idx;
            best_load = load;
        }
    }
    Some(best)
}

impl RoutingPolicy for LeastLoaded {
    fn kind(&self) -> PolicyKind {
        PolicyKind::LeastLoaded
    }

    fn select(&self, fleet: &Fleet, _key: Option<AffinityKey>) -> Option<Decision> {
        let healthy = fleet.healthy_indices();
        let idx = least_loaded_of(fleet, &healthy, &self.cursor)?;
        Some(Decision {
            replica_index: idx,
            reason: RouteReason::LeastLoad,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::testutil::fleet;
    use crate::replica::InflightGuard;

    #[test]
    fn picks_the_replica_with_fewest_inflight() {
        let f = fleet(3);
        let _a = InflightGuard::acquire(f.get(0).clone());
        let _b = InflightGuard::acquire(f.get(0).clone());
        let _c = InflightGuard::acquire(f.get(2).clone());
        let p = LeastLoaded::new();
        assert_eq!(p.select(&f, None).unwrap().replica_index, 1);
    }

    #[test]
    fn counts_queue_depth_as_load_too() {
        let f = fleet(2);
        let _a = InflightGuard::acquire(f.get(1).clone());
        f.get(0).enqueue();
        f.get(0).enqueue();
        let p = LeastLoaded::new();
        // r0: 0 inflight + 2 queued = 2; r1: 1 inflight + 0 queued = 1
        assert_eq!(p.select(&f, None).unwrap().replica_index, 1);
    }

    #[test]
    fn an_idle_fleet_is_spread_evenly_rather_than_piled_on_replica_zero() {
        // The regression test for the lowest-index tiebreak bug described above.
        let f = fleet(4);
        let p = LeastLoaded::new();
        let mut counts = [0usize; 4];
        for _ in 0..4000 {
            // Guard dropped immediately, so the fleet is idle at every decision.
            counts[p.select(&f, None).unwrap().replica_index] += 1;
        }
        for (i, c) in counts.iter().enumerate() {
            assert_eq!(*c, 1000, "replica {i} took {c} of 4000 on an idle fleet");
        }
    }

    #[test]
    fn never_selects_an_unhealthy_replica_even_when_it_is_idlest() {
        let f = fleet(3);
        f.get(2).record_health_failure(1); // idle but dead
        let _a = InflightGuard::acquire(f.get(0).clone());
        let _b = InflightGuard::acquire(f.get(1).clone());
        let p = LeastLoaded::new();
        for _ in 0..20 {
            assert_ne!(p.select(&f, None).unwrap().replica_index, 2);
        }
    }

    #[test]
    fn reports_least_load_as_the_reason() {
        let f = fleet(2);
        let p = LeastLoaded::new();
        assert_eq!(p.select(&f, None).unwrap().reason, RouteReason::LeastLoad);
    }
}
