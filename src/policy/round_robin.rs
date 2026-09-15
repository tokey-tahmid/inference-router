//! Round robin over the healthy set. The baseline every other policy is
//! measured against.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::config::PolicyKind;
use crate::policy::{AffinityKey, Decision, RouteReason, RoutingPolicy};
use crate::replica::Fleet;

#[derive(Debug, Default)]
pub struct RoundRobin {
    cursor: AtomicUsize,
}

impl RoundRobin {
    pub fn new() -> Self {
        Self::default()
    }
}

impl RoutingPolicy for RoundRobin {
    fn kind(&self) -> PolicyKind {
        PolicyKind::RoundRobin
    }

    fn select(&self, fleet: &Fleet, _key: Option<AffinityKey>) -> Option<Decision> {
        let healthy = fleet.healthy_indices();
        if healthy.is_empty() {
            return None;
        }
        // Rotate over the *healthy* list rather than over all replica slots.
        // Rotating over all slots and skipping dead ones would give the
        // survivors an uneven share whenever the dead set is not a divisor of
        // the fleet size, which would show up in the fairness metric as a
        // property of the policy rather than of the failure.
        let n = self.cursor.fetch_add(1, Ordering::Relaxed);
        Some(Decision {
            replica_index: healthy[n % healthy.len()],
            reason: RouteReason::Rotation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::testutil::fleet;

    #[test]
    fn cycles_through_every_healthy_replica_in_order() {
        let f = fleet(4);
        let p = RoundRobin::new();
        let seq: Vec<usize> = (0..8)
            .map(|_| p.select(&f, None).unwrap().replica_index)
            .collect();
        assert_eq!(seq, vec![0, 1, 2, 3, 0, 1, 2, 3]);
    }

    #[test]
    fn distributes_evenly_over_survivors_after_an_ejection() {
        let f = fleet(4);
        f.get(1).record_health_failure(1);
        let p = RoundRobin::new();
        let mut counts = [0usize; 4];
        for _ in 0..300 {
            counts[p.select(&f, None).unwrap().replica_index] += 1;
        }
        assert_eq!(counts[1], 0, "ejected replica must receive nothing");
        assert_eq!(counts[0], 100);
        assert_eq!(counts[2], 100);
        assert_eq!(counts[3], 100);
    }

    #[test]
    fn reports_rotation_as_the_reason() {
        let f = fleet(2);
        let p = RoundRobin::new();
        assert_eq!(p.select(&f, None).unwrap().reason, RouteReason::Rotation);
    }

    #[test]
    fn ignores_the_affinity_key_entirely() {
        let f = fleet(3);
        let p = RoundRobin::new();
        let a = p.select(&f, Some(AffinityKey(1))).unwrap().replica_index;
        let b = p.select(&f, Some(AffinityKey(1))).unwrap().replica_index;
        assert_ne!(a, b, "same key must still rotate");
    }
}
