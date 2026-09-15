//! Routing policies behind a single trait.
//!
//! ## Why prefix affinity is a fleet-level concern
//!
//! A serving engine's prefix cache is *per replica*. vLLM will happily reuse the
//! KV blocks of a 512-token system prompt it has already prefilled, but only on
//! the replica that prefilled it. A load balancer that spreads N requests
//! sharing one prefix evenly over R replicas therefore forces R cold prefills
//! instead of one, and the cache hit rate the engine reports is bounded above by
//! `1 - R/N` before eviction is even considered. Cache locality is a property of
//! the *routing decision*, not of the engine.
//!
//! ## The failure mode, stated up front
//!
//! Affinity and balance are in direct tension. If one prefix is much hotter than
//! the others (one popular system prompt, one large tenant), pure affinity sends
//! all of its traffic to one replica while the rest idle. The cache hit rate
//! goes up and the tail latency goes up with it, because a saturated replica
//! queues. `prefix_affinity_bounded` exists precisely to bound that, and the
//! experiment matrix is designed so that the crossover is visible rather than
//! asserted.

use std::sync::Arc;

use crate::config::{PolicyConfig, PolicyKind};
use crate::replica::Fleet;

mod hash;
mod least_loaded;
mod prefix_affinity;
mod round_robin;

pub use hash::{fnv1a64, AffinityKey};
pub use least_loaded::LeastLoaded;
pub use prefix_affinity::PrefixAffinity;
pub use round_robin::RoundRobin;

/// Why a policy chose the replica it chose.
///
/// Recorded per request (as the `x-router-route-reason` header and in the
/// structured log) because the *mechanism* is the finding. "Prefix affinity was
/// slower at rate 24" is an observation; "prefix affinity spilled 61% of its
/// decisions at rate 24" is an explanation, and only the second one survives an
/// interview.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteReason {
    /// Deterministic rotation (round_robin).
    Rotation,
    /// Chosen for lowest load (least_loaded, or a bounded-affinity fallback that
    /// exhausted every candidate).
    LeastLoad,
    /// The top-ranked replica for this prefix key was chosen.
    AffinityPrimary,
    /// The top-ranked replica was unhealthy; the next-ranked healthy replica for
    /// the same key was chosen. Stable under ejection: only the dead replica's
    /// share of keys is remapped.
    AffinityFailover,
    /// The top-ranked replica was over the load bound, so the request spilled to
    /// a lower-ranked one (prefix_affinity_bounded only).
    AffinitySpill,
    /// Every replica was over the bound; fell back to the least loaded.
    AffinitySpillExhausted,
}

impl RouteReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rotation => "rotation",
            Self::LeastLoad => "least_load",
            Self::AffinityPrimary => "affinity_primary",
            Self::AffinityFailover => "affinity_failover",
            Self::AffinitySpill => "affinity_spill",
            Self::AffinitySpillExhausted => "affinity_spill_exhausted",
        }
    }

    pub const ALL: [RouteReason; 6] = [
        Self::Rotation,
        Self::LeastLoad,
        Self::AffinityPrimary,
        Self::AffinityFailover,
        Self::AffinitySpill,
        Self::AffinitySpillExhausted,
    ];
}

#[derive(Debug, Clone, Copy)]
pub struct Decision {
    pub replica_index: usize,
    pub reason: RouteReason,
}

/// A routing policy.
///
/// `select` is called once per request, on the hot path, from many tasks at
/// once. Implementations must be `Send + Sync` and must not block. `&self`
/// rather than `&mut self` is deliberate: any state a policy keeps (round-robin
/// cursor) is an atomic, so the router never needs a write lock to route.
pub trait RoutingPolicy: Send + Sync + std::fmt::Debug {
    fn kind(&self) -> PolicyKind;

    /// Choose a replica for a request with the given affinity key.
    ///
    /// `key` is `None` when the request carried no extractable prompt, which
    /// happens for malformed or unusual bodies. A prefix policy must still
    /// return a decision in that case rather than failing the request: falling
    /// back to load-based placement is strictly better than a 400.
    ///
    /// Returns `None` only when there is no healthy replica at all.
    fn select(&self, fleet: &Fleet, key: Option<AffinityKey>) -> Option<Decision>;
}

/// Build a policy from config.
pub fn build(cfg: &PolicyConfig) -> Arc<dyn RoutingPolicy> {
    match cfg.kind {
        PolicyKind::RoundRobin => Arc::new(RoundRobin::new()),
        PolicyKind::LeastLoaded => Arc::new(LeastLoaded::new()),
        PolicyKind::PrefixAffinity => Arc::new(PrefixAffinity::unbounded()),
        PolicyKind::PrefixAffinityBounded => Arc::new(PrefixAffinity::bounded(
            cfg.bounded_overload_factor,
            cfg.bounded_min_capacity,
        )),
    }
}

#[cfg(test)]
pub(crate) mod testutil {
    use crate::config::ReplicaConfig;
    use crate::replica::Fleet;

    pub fn fleet(n: usize) -> Fleet {
        let cfgs: Vec<ReplicaConfig> = (0..n)
            .map(|i| ReplicaConfig {
                id: format!("r{i}"),
                url: format!("http://127.0.0.1:{}", 8100 + i),
                placement: None,
            })
            .collect();
        Fleet::new(&cfgs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PolicyConfig;

    #[test]
    fn build_returns_the_requested_kind() {
        for kind in PolicyKind::ALL {
            let cfg = PolicyConfig {
                kind,
                ..Default::default()
            };
            assert_eq!(build(&cfg).kind(), kind);
        }
    }

    #[test]
    fn every_policy_returns_none_on_an_entirely_unhealthy_fleet() {
        let fleet = testutil::fleet(3);
        for r in &fleet.replicas {
            r.record_health_failure(1);
        }
        for kind in PolicyKind::ALL {
            let p = build(&PolicyConfig {
                kind,
                ..Default::default()
            });
            assert!(
                p.select(&fleet, Some(AffinityKey(42))).is_none(),
                "{kind:?} must not route to a dead fleet"
            );
        }
    }

    #[test]
    fn every_policy_routes_without_an_affinity_key() {
        let fleet = testutil::fleet(3);
        for kind in PolicyKind::ALL {
            let p = build(&PolicyConfig {
                kind,
                ..Default::default()
            });
            assert!(
                p.select(&fleet, None).is_some(),
                "{kind:?} must still route a request with no extractable prompt"
            );
        }
    }

    #[test]
    fn no_policy_ever_selects_an_unhealthy_replica() {
        let fleet = testutil::fleet(4);
        fleet.get(0).record_health_failure(1);
        fleet.get(2).record_health_failure(1);
        for kind in PolicyKind::ALL {
            let p = build(&PolicyConfig {
                kind,
                ..Default::default()
            });
            for k in 0..200u64 {
                let d = p.select(&fleet, Some(AffinityKey(k))).expect("healthy left");
                assert!(
                    fleet.get(d.replica_index).is_healthy(),
                    "{kind:?} selected ejected replica {}",
                    d.replica_index
                );
            }
        }
    }
}
