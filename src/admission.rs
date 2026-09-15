//! Admission control: bounded concurrency with an explicit overload behaviour.
//!
//! The router must have a *stated* answer to "what happens when the fleet is
//! full", because the two reasonable answers produce opposite-looking
//! measurements and both are defensible:
//!
//! * **Reject (429).** Latency stays bounded and the overload shows up as an
//!   error rate. This is what a fleet behind an autoscaler wants: the signal is
//!   legible and the client can shed.
//! * **Queue.** Goodput stays high and the overload shows up as a latency
//!   excursion instead. Correct when the client cannot retry cheaply.
//!
//! The failure mode this module exists to prevent is the third, unstated
//! option: an unbounded queue. That converts overload into an ever-growing
//! latency with no error rate at all, so the system looks healthy in every
//! metric except the one nobody is watching. `max_queue_depth` is what makes the
//! queueing arm bounded rather than bufferbloated.
//!
//! Capacity is `max_inflight_per_replica * n_healthy`, recomputed per request
//! rather than fixed at startup: when a replica is ejected the fleet's capacity
//! genuinely shrinks, and admitting as though it had not would simply relocate
//! the queue into the surviving replicas where the router cannot see or bound it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::Notify;
use tokio::time::{timeout, Duration, Instant};

use crate::config::{AdmissionConfig, OverloadBehaviour};
use crate::error::{Result, RouterError};

#[derive(Debug)]
pub struct Admission {
    cfg: AdmissionConfig,
    admitted: AtomicUsize,
    waiting: AtomicUsize,
    slot_freed: Notify,
}

/// Held for as long as a request occupies a fleet slot. Releasing it wakes one
/// waiter, which is what keeps the queueing arm FIFO-ish rather than a
/// thundering herd.
#[derive(Debug)]
pub struct AdmissionGuard {
    admission: Arc<Admission>,
    /// How long this request spent queued before it was admitted.
    pub queue_wait: Duration,
}

impl Drop for AdmissionGuard {
    fn drop(&mut self) {
        self.admission.admitted.fetch_sub(1, Ordering::AcqRel);
        self.admission.slot_freed.notify_one();
    }
}

impl Admission {
    pub fn new(cfg: AdmissionConfig) -> Self {
        Self {
            cfg,
            admitted: AtomicUsize::new(0),
            waiting: AtomicUsize::new(0),
            slot_freed: Notify::new(),
        }
    }

    pub fn config(&self) -> &AdmissionConfig {
        &self.cfg
    }

    pub fn inflight(&self) -> usize {
        self.admitted.load(Ordering::Relaxed)
    }

    pub fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Relaxed)
    }

    pub fn capacity(&self, n_healthy: usize) -> usize {
        self.cfg.max_inflight_per_replica.saturating_mul(n_healthy)
    }

    fn try_take(self: &Arc<Self>, capacity: usize) -> Option<AdmissionGuard> {
        let mut cur = self.admitted.load(Ordering::Relaxed);
        loop {
            if cur >= capacity {
                return None;
            }
            match self.admitted.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Some(AdmissionGuard {
                        admission: Arc::clone(self),
                        queue_wait: Duration::ZERO,
                    })
                }
                Err(actual) => cur = actual,
            }
        }
    }

    /// Acquire a fleet slot, honouring the configured overload behaviour.
    pub async fn acquire(self: &Arc<Self>, n_healthy: usize) -> Result<AdmissionGuard> {
        let capacity = self.capacity(n_healthy);
        if let Some(g) = self.try_take(capacity) {
            return Ok(g);
        }

        match self.cfg.overload_behaviour {
            OverloadBehaviour::Reject => Err(RouterError::Overloaded {
                inflight: self.inflight(),
                limit: capacity,
            }),
            OverloadBehaviour::Queue => {
                // Bound the queue itself, not just the in-flight set.
                if self.waiting.load(Ordering::Relaxed) >= self.cfg.max_queue_depth {
                    return Err(RouterError::Overloaded {
                        inflight: self.inflight(),
                        limit: capacity,
                    });
                }
                self.waiting.fetch_add(1, Ordering::AcqRel);
                let started = Instant::now();
                let result = timeout(self.cfg.queue_timeout(), async {
                    loop {
                        // Register interest *before* re-checking, so a slot freed
                        // between the check and the await is not missed.
                        let notified = self.slot_freed.notified();
                        if let Some(g) = self.try_take(self.capacity(n_healthy)) {
                            return g;
                        }
                        notified.await;
                    }
                })
                .await;
                self.waiting.fetch_sub(1, Ordering::AcqRel);

                match result {
                    Ok(mut g) => {
                        g.queue_wait = started.elapsed();
                        Ok(g)
                    }
                    Err(_) => Err(RouterError::AdmissionTimeout {
                        waited_ms: started.elapsed().as_millis() as u64,
                    }),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(
        behaviour: OverloadBehaviour,
        per_replica: usize,
        depth: usize,
        to_ms: u64,
    ) -> AdmissionConfig {
        AdmissionConfig {
            overload_behaviour: behaviour,
            max_inflight_per_replica: per_replica,
            max_queue_depth: depth,
            queue_timeout_ms: to_ms,
        }
    }

    #[tokio::test]
    async fn admits_up_to_capacity_then_rejects() {
        let a = Arc::new(Admission::new(cfg(OverloadBehaviour::Reject, 2, 16, 1000)));
        // 2 replicas x 2 = capacity 4
        let mut held = Vec::new();
        for _ in 0..4 {
            held.push(a.acquire(2).await.expect("under capacity"));
        }
        let err = a.acquire(2).await.expect_err("at capacity");
        assert!(matches!(err, RouterError::Overloaded { limit: 4, .. }));
        assert_eq!(err.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn releasing_a_slot_allows_the_next_request() {
        let a = Arc::new(Admission::new(cfg(OverloadBehaviour::Reject, 1, 16, 1000)));
        let g = a.acquire(1).await.unwrap();
        assert!(a.acquire(1).await.is_err());
        drop(g);
        assert!(a.acquire(1).await.is_ok(), "slot must be reusable");
    }

    #[tokio::test]
    async fn capacity_shrinks_when_replicas_are_ejected() {
        let a = Arc::new(Admission::new(cfg(OverloadBehaviour::Reject, 4, 16, 1000)));
        assert_eq!(a.capacity(4), 16);
        assert_eq!(a.capacity(2), 8);
        assert_eq!(a.capacity(0), 0);
        // With no healthy replicas, nothing is admissible at all.
        assert!(a.acquire(0).await.is_err());
    }

    #[tokio::test]
    async fn queue_mode_waits_for_a_slot_instead_of_rejecting() {
        let a = Arc::new(Admission::new(cfg(OverloadBehaviour::Queue, 1, 16, 5000)));
        let g = a.acquire(1).await.unwrap();
        let a2 = Arc::clone(&a);
        let waiter = tokio::spawn(async move { a2.acquire(1).await });
        // Give the waiter a chance to park.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(a.waiting(), 1);
        drop(g);
        let guard = waiter.await.unwrap().expect("queued request is admitted");
        assert!(
            guard.queue_wait >= Duration::from_millis(40),
            "queue wait should be recorded, got {:?}",
            guard.queue_wait
        );
    }

    #[tokio::test]
    async fn queue_mode_times_out_rather_than_waiting_forever() {
        let a = Arc::new(Admission::new(cfg(OverloadBehaviour::Queue, 1, 16, 100)));
        let _g = a.acquire(1).await.unwrap();
        let err = a.acquire(1).await.expect_err("should time out");
        assert!(matches!(err, RouterError::AdmissionTimeout { .. }));
        assert_eq!(err.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(a.waiting(), 0, "waiter count must unwind on timeout");
    }

    #[tokio::test]
    async fn queue_mode_rejects_past_the_queue_depth_bound() {
        // The anti-bufferbloat check: the queue itself is finite.
        let a = Arc::new(Admission::new(cfg(OverloadBehaviour::Queue, 1, 2, 5000)));
        let _g = a.acquire(1).await.unwrap();
        let mut waiters = Vec::new();
        for _ in 0..2 {
            let a2 = Arc::clone(&a);
            waiters.push(tokio::spawn(async move { a2.acquire(1).await }));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(a.waiting(), 2);
        let err = a.acquire(1).await.expect_err("queue is full");
        assert!(matches!(err, RouterError::Overloaded { .. }));
        for w in waiters {
            w.abort();
        }
    }
}
