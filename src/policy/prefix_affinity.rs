//! Prefix affinity, in an unbounded and a bounded-load variant.
//!
//! Both share one implementation because they differ in exactly one rule, and
//! having them differ in exactly one rule is what makes the A/B interpretable:
//! any difference in the measured results is attributable to the load bound and
//! to nothing else.
//!
//! # Placement
//!
//! Replicas are ranked for a key by **rendezvous (highest-random-weight)
//! hashing** over their *stable* indices (see `hash.rs` for why not a modulo or
//! a ring).
//!
//! # Fallback rules, in order
//!
//! 1. **No key.** A request whose prompt could not be extracted has no affinity.
//!    It routes by least-load rather than failing: a 400 here would turn a
//!    parsing edge case into a measured error rate.
//! 2. **Unhealthy primary** (`AffinityFailover`). Walk down the rendezvous
//!    ranking to the highest-ranked *healthy* replica. Because the ranking is
//!    over stable indices, an ejection moves only the keys owned by the ejected
//!    replica, and returns them when it recovers.
//! 3. **Over the load bound** (`AffinitySpill`, bounded variant only). See
//!    below.
//! 4. **Every replica over the bound** (`AffinitySpillExhausted`). Fall back to
//!    the globally least loaded healthy replica. The bound is advisory: refusing
//!    to route would convert a transient imbalance into an error, and the
//!    admission layer already exists to reject genuine overload.
//!
//! # The load bound
//!
//! Consistent hashing with bounded loads (Mirrokni, Thorup, Zadimoghaddam).
//! Capacity per replica is
//!
//! ```text
//! cap = max(min_capacity, ceil(factor * total_load / n_healthy))
//! ```
//!
//! A request takes the highest-ranked healthy replica whose current load is
//! below `cap`, else spills down the ranking, else falls back to least-loaded.
//!
//! `factor` is the knob: 1.0 forces near-perfect balance and destroys affinity,
//! large values degenerate to the unbounded policy. `min_capacity` matters more
//! than it looks. Without it, an idle fleet has `total_load ≈ 0`, so `cap`
//! rounds to 1, and the bounded policy spills on the *second* concurrent request
//! to any key — that is, it becomes round-robin exactly in the underload regime
//! where affinity is cheapest and where the cache benefit should be largest. The
//! floor keeps the bound inactive until there is real contention to bound.

use std::sync::atomic::AtomicUsize;

use crate::config::PolicyKind;
use crate::policy::hash::rank_by_affinity;
use crate::policy::least_loaded::least_loaded_of;
use crate::policy::{AffinityKey, Decision, RouteReason, RoutingPolicy};
use crate::replica::Fleet;

#[derive(Debug)]
pub struct PrefixAffinity {
    bound: Option<LoadBound>,
    /// Tiebreak cursor for the least-loaded fallback paths, shared with the
    /// `least_loaded` implementation so the idle-fleet spread bug cannot exist
    /// in one policy and not the other.
    cursor: AtomicUsize,
}

#[derive(Debug, Clone, Copy)]
struct LoadBound {
    factor: f64,
    min_capacity: usize,
}

impl PrefixAffinity {
    pub fn unbounded() -> Self {
        Self {
            bound: None,
            cursor: AtomicUsize::new(0),
        }
    }

    pub fn bounded(factor: f64, min_capacity: usize) -> Self {
        Self {
            bound: Some(LoadBound {
                factor,
                // A capacity bound of zero would reject every replica on every
                // request and make the policy permanently least-loaded.
                min_capacity: min_capacity.max(1),
            }),
            cursor: AtomicUsize::new(0),
        }
    }

    /// Per-replica capacity for the current fleet load, or `None` when unbounded.
    fn capacity(&self, fleet: &Fleet, n_healthy: usize) -> Option<usize> {
        let b = self.bound?;
        debug_assert!(n_healthy > 0);
        let mean = fleet.total_load() as f64 / n_healthy as f64;
        let cap = (b.factor * mean).ceil() as usize;
        Some(cap.max(b.min_capacity))
    }
}

impl RoutingPolicy for PrefixAffinity {
    fn kind(&self) -> PolicyKind {
        if self.bound.is_some() {
            PolicyKind::PrefixAffinityBounded
        } else {
            PolicyKind::PrefixAffinity
        }
    }

    fn select(&self, fleet: &Fleet, key: Option<AffinityKey>) -> Option<Decision> {
        let healthy = fleet.healthy_indices();
        if healthy.is_empty() {
            return None;
        }

        // Rule 1: no prompt, no affinity.
        let Some(key) = key else {
            return least_loaded_of(fleet, &healthy, &self.cursor).map(|idx| Decision {
                replica_index: idx,
                reason: RouteReason::LeastLoad,
            });
        };

        // Rank over the *healthy* subset. Because scores are computed from
        // stable replica indices, this is the same ordering as ranking the full
        // fleet and then filtering — which is the property that makes ejection
        // remap only the ejected replica's keys.
        let ranked = rank_by_affinity(key, &healthy);
        let primary_is_top_of_fleet = {
            let all: Vec<usize> = (0..fleet.len()).collect();
            rank_by_affinity(key, &all).first().copied() == ranked.first().copied()
        };

        let Some(cap) = self.capacity(fleet, healthy.len()) else {
            // Unbounded: first healthy in rank order, full stop. Deliberately
            // blind to load — that blindness is the failure mode under test.
            let idx = ranked[0];
            return Some(Decision {
                replica_index: idx,
                reason: if primary_is_top_of_fleet {
                    RouteReason::AffinityPrimary
                } else {
                    RouteReason::AffinityFailover
                },
            });
        };

        // Bounded: walk the ranking for the first replica under capacity.
        for (rank, &idx) in ranked.iter().enumerate() {
            if fleet.get(idx).load() < cap {
                let reason = if rank > 0 {
                    RouteReason::AffinitySpill
                } else if primary_is_top_of_fleet {
                    RouteReason::AffinityPrimary
                } else {
                    RouteReason::AffinityFailover
                };
                return Some(Decision {
                    replica_index: idx,
                    reason,
                });
            }
        }

        // Rule 4: everything is at or over the bound.
        least_loaded_of(fleet, &healthy, &self.cursor).map(|idx| Decision {
            replica_index: idx,
            reason: RouteReason::AffinitySpillExhausted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::testutil::fleet;
    use crate::replica::InflightGuard;

    #[test]
    fn the_same_key_always_lands_on_the_same_replica() {
        let f = fleet(4);
        let p = PrefixAffinity::unbounded();
        let first = p.select(&f, Some(AffinityKey(777))).unwrap().replica_index;
        for _ in 0..100 {
            assert_eq!(
                p.select(&f, Some(AffinityKey(777))).unwrap().replica_index,
                first
            );
        }
    }

    #[test]
    fn different_keys_spread_over_the_fleet() {
        let f = fleet(4);
        let p = PrefixAffinity::unbounded();
        let mut counts = [0usize; 4];
        for k in 0..4000u64 {
            let key = AffinityKey(k.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            counts[p.select(&f, Some(key)).unwrap().replica_index] += 1;
        }
        for (i, c) in counts.iter().enumerate() {
            assert!(*c > 700 && *c < 1300, "replica {i} took {c} of 4000");
        }
    }

    #[test]
    fn unbounded_affinity_ignores_load_completely() {
        // This is the documented failure mode, asserted rather than described.
        let f = fleet(4);
        let p = PrefixAffinity::unbounded();
        let hot = AffinityKey(42);
        let target = p.select(&f, Some(hot)).unwrap().replica_index;
        let mut guards = Vec::new();
        for _ in 0..64 {
            guards.push(InflightGuard::acquire(f.get(target).clone()));
        }
        assert_eq!(f.get(target).inflight(), 64);
        // Every other replica is idle; the policy still picks the saturated one.
        let d = p.select(&f, Some(hot)).unwrap();
        assert_eq!(d.replica_index, target);
        assert_eq!(d.reason, RouteReason::AffinityPrimary);
    }

    #[test]
    fn bounded_affinity_spills_once_the_primary_is_over_capacity() {
        let f = fleet(4);
        let p = PrefixAffinity::bounded(1.25, 4);
        let hot = AffinityKey(42);
        let target = p.select(&f, Some(hot)).unwrap().replica_index;
        let mut guards = Vec::new();
        for _ in 0..64 {
            guards.push(InflightGuard::acquire(f.get(target).clone()));
        }
        let d = p.select(&f, Some(hot)).unwrap();
        assert_ne!(d.replica_index, target, "must have spilled off the hot replica");
        assert_eq!(d.reason, RouteReason::AffinitySpill);
    }

    #[test]
    fn bounded_affinity_keeps_affinity_while_the_fleet_is_idle() {
        // The min_capacity floor: at low load the bound must not be active, or
        // the policy is round-robin exactly where affinity is most valuable.
        let f = fleet(4);
        let p = PrefixAffinity::bounded(1.25, 4);
        let hot = AffinityKey(42);
        let target = p.select(&f, Some(hot)).unwrap().replica_index;
        let mut guards = Vec::new();
        for _ in 0..3 {
            guards.push(InflightGuard::acquire(f.get(target).clone()));
            let d = p.select(&f, Some(hot)).unwrap();
            assert_eq!(d.replica_index, target, "spilled too early at load {}", f.get(target).load());
            assert_eq!(d.reason, RouteReason::AffinityPrimary);
        }
    }

    #[test]
    fn bounded_affinity_falls_back_to_least_loaded_when_all_are_saturated() {
        let f = fleet(3);
        let p = PrefixAffinity::bounded(1.0, 1);
        // Load every replica heavily but unevenly.
        let mut guards = Vec::new();
        for (idx, n) in [(0usize, 30usize), (1, 20), (2, 25)] {
            for _ in 0..n {
                guards.push(InflightGuard::acquire(f.get(idx).clone()));
            }
        }
        // mean = 25, factor 1.0 -> cap 25. r0(30) and r2(25) are at/over cap,
        // r1(20) is under, so a spill to r1 is expected rather than exhaustion.
        let d = p.select(&f, Some(AffinityKey(9))).unwrap();
        assert_eq!(d.replica_index, 1);

        // Now push r1 over too; nothing is under cap and we must land on the
        // least loaded rather than erroring.
        for _ in 0..10 {
            guards.push(InflightGuard::acquire(f.get(1).clone()));
        }
        let d = p.select(&f, Some(AffinityKey(9))).unwrap();
        assert_eq!(d.reason, RouteReason::AffinitySpillExhausted);
        assert_eq!(d.replica_index, 1, "least loaded of 30/30/25 after the fills");
    }

    #[test]
    fn ejection_remaps_only_the_ejected_replica_s_keys() {
        let f = fleet(4);
        let p = PrefixAffinity::unbounded();
        const N: u64 = 5000;
        let before: Vec<usize> = (0..N)
            .map(|k| {
                p.select(&f, Some(AffinityKey(k.wrapping_mul(0x9E37_79B9_7F4A_7C15))))
                    .unwrap()
                    .replica_index
            })
            .collect();
        f.get(2).record_health_failure(1);
        let after: Vec<usize> = (0..N)
            .map(|k| {
                p.select(&f, Some(AffinityKey(k.wrapping_mul(0x9E37_79B9_7F4A_7C15))))
                    .unwrap()
                    .replica_index
            })
            .collect();
        for i in 0..N as usize {
            if before[i] != 2 {
                assert_eq!(before[i], after[i], "key {i} moved but did not own replica 2");
            } else {
                assert_ne!(after[i], 2);
            }
        }
    }

    #[test]
    fn recovery_restores_the_original_mapping_exactly() {
        let f = fleet(4);
        let p = PrefixAffinity::unbounded();
        let key = AffinityKey(31337);
        let home = p.select(&f, Some(key)).unwrap().replica_index;
        f.get(home).record_health_failure(1);
        let exiled = p.select(&f, Some(key)).unwrap();
        assert_ne!(exiled.replica_index, home);
        assert_eq!(exiled.reason, RouteReason::AffinityFailover);
        f.get(home).record_health_success(1);
        let back = p.select(&f, Some(key)).unwrap();
        assert_eq!(back.replica_index, home);
        assert_eq!(back.reason, RouteReason::AffinityPrimary);
    }

    #[test]
    fn a_request_with_no_key_routes_by_load() {
        let f = fleet(3);
        let p = PrefixAffinity::unbounded();
        let _a = InflightGuard::acquire(f.get(0).clone());
        let _b = InflightGuard::acquire(f.get(2).clone());
        let d = p.select(&f, None).unwrap();
        assert_eq!(d.replica_index, 1);
        assert_eq!(d.reason, RouteReason::LeastLoad);
    }

    #[test]
    fn kind_reflects_whether_a_bound_is_configured() {
        assert_eq!(PrefixAffinity::unbounded().kind(), PolicyKind::PrefixAffinity);
        assert_eq!(
            PrefixAffinity::bounded(1.25, 4).kind(),
            PolicyKind::PrefixAffinityBounded
        );
    }

    #[test]
    fn a_zero_min_capacity_is_clamped_to_one() {
        // cap of 0 would reject every replica on every request.
        let f = fleet(2);
        let p = PrefixAffinity::bounded(1.0, 0);
        let d = p.select(&f, Some(AffinityKey(5))).unwrap();
        assert_eq!(d.reason, RouteReason::AffinityPrimary);
    }
}
