//! End-to-end routing, health and admission behaviour through a live HTTP
//! router in front of mock replicas.
//!
//! These complement the unit tests in `src/policy/`: those assert what a policy
//! *decides*, these assert that the decision actually reaches a replica and is
//! reported honestly in the response headers and in `/metrics`. Both matter,
//! because the experiment driver attributes every request to a replica using
//! the `x-router-replica` header, and a header that disagreed with the routing
//! decision would corrupt the fairness metric silently.

mod common;

use std::collections::HashMap;
use std::time::Duration;

use common::{config_for, prompt_with_prefix, spawn_router, MockReplica};
use inference_router::config::OverloadBehaviour;

async fn post_prompt(
    client: &reqwest::Client,
    router: &str,
    prompt: &[i64],
) -> (reqwest::StatusCode, HashMap<String, String>) {
    let resp = client
        .post(format!("{router}/v1/completions"))
        .json(&serde_json::json!({
            "model": "mock", "prompt": prompt, "max_tokens": 8, "stream": true
        }))
        .send()
        .await
        .expect("router responds");
    let status = resp.status();
    let headers = resp
        .headers()
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();
    // Drain so the in-flight guard is released before the next request.
    let _ = resp.bytes().await;
    (status, headers)
}

async fn set_policy(client: &reqwest::Client, router: &str, body: serde_json::Value) {
    let r = client
        .post(format!("{router}/admin/policy"))
        .json(&body)
        .send()
        .await
        .expect("policy switch accepted");
    assert!(
        r.status().is_success(),
        "policy switch failed: {:?}",
        r.text().await
    );
}

#[tokio::test]
async fn round_robin_spreads_evenly_and_reports_the_replica_in_a_header() {
    let mocks: Vec<MockReplica> = {
        let mut v = Vec::new();
        for i in 0..4 {
            v.push(MockReplica::spawn(&format!("r{i}")).await);
        }
        v
    };
    for m in &mocks {
        m.set_chunks(2);
    }
    let refs: Vec<&MockReplica> = mocks.iter().collect();
    let (router, _state) = spawn_router(config_for(&refs)).await;
    let client = reqwest::Client::new();

    let mut header_counts: HashMap<String, usize> = HashMap::new();
    for i in 0..40 {
        let p = prompt_with_prefix(i % 4, 32, 64);
        let (status, h) = post_prompt(&client, &router, &p).await;
        assert!(status.is_success());
        assert_eq!(h.get("x-router-policy").unwrap(), "round_robin");
        assert_eq!(h.get("x-router-route-reason").unwrap(), "rotation");
        *header_counts
            .entry(h.get("x-router-replica").unwrap().clone())
            .or_default() += 1;
    }

    assert_eq!(header_counts.len(), 4, "every replica should be used");
    for (id, n) in &header_counts {
        assert_eq!(*n, 10, "replica {id} got {n} of 40 under round robin");
    }
    // The header must agree with what the replica actually received.
    for m in &mocks {
        assert_eq!(
            m.request_count(),
            header_counts[&m.id] as u64,
            "replica {} disagrees with the x-router-replica header",
            m.id
        );
    }
}

#[tokio::test]
async fn prefix_affinity_pins_a_shared_prefix_to_one_replica() {
    let mocks: Vec<MockReplica> = {
        let mut v = Vec::new();
        for i in 0..4 {
            v.push(MockReplica::spawn(&format!("r{i}")).await);
        }
        v
    };
    for m in &mocks {
        m.set_chunks(2);
    }
    let refs: Vec<&MockReplica> = mocks.iter().collect();
    let (router, _state) = spawn_router(config_for(&refs)).await;
    let client = reqwest::Client::new();
    set_policy(
        &client,
        &router,
        serde_json::json!({"policy": "prefix_affinity", "prefix_hash_tokens": 32}),
    )
    .await;

    // One hot prefix group: every request shares the first 32 tokens.
    let mut seen: HashMap<String, usize> = HashMap::new();
    for i in 0..30 {
        let p = prompt_with_prefix(7, 32, 64 + i as usize);
        let (status, h) = post_prompt(&client, &router, &p).await;
        assert!(status.is_success());
        assert_eq!(h.get("x-router-policy").unwrap(), "prefix_affinity");
        assert_eq!(h.get("x-router-route-reason").unwrap(), "affinity_primary");
        *seen
            .entry(h.get("x-router-replica").unwrap().clone())
            .or_default() += 1;
    }
    assert_eq!(
        seen.len(),
        1,
        "a single shared prefix must land on exactly one replica, got {seen:?}"
    );

    // Four distinct prefixes should spread; this is the "tenant mix" case.
    let mut spread: HashMap<String, usize> = HashMap::new();
    for g in 0..16u64 {
        let p = prompt_with_prefix(g, 32, 64);
        let (_, h) = post_prompt(&client, &router, &p).await;
        *spread
            .entry(h.get("x-router-replica").unwrap().clone())
            .or_default() += 1;
    }
    assert!(
        spread.len() > 1,
        "16 distinct prefixes should not all collide onto one replica: {spread:?}"
    );
}

#[tokio::test]
async fn affinity_key_header_is_stable_for_a_shared_prefix() {
    let m = MockReplica::spawn("r0").await;
    let (router, _state) = spawn_router(config_for(&[&m])).await;
    let client = reqwest::Client::new();
    set_policy(
        &client,
        &router,
        serde_json::json!({"policy": "prefix_affinity", "prefix_hash_tokens": 32}),
    )
    .await;

    let (_, h1) = post_prompt(&client, &router, &prompt_with_prefix(3, 32, 64)).await;
    let (_, h2) = post_prompt(&client, &router, &prompt_with_prefix(3, 32, 128)).await;
    let (_, h3) = post_prompt(&client, &router, &prompt_with_prefix(4, 32, 64)).await;
    assert_eq!(
        h1.get("x-router-affinity-key"),
        h2.get("x-router-affinity-key"),
        "same prefix, different tail: key must match"
    );
    assert_ne!(
        h1.get("x-router-affinity-key"),
        h3.get("x-router-affinity-key"),
        "different prefix: key must differ"
    );
}

#[tokio::test]
async fn policy_is_switchable_at_runtime_and_reported_in_metrics() {
    let a = MockReplica::spawn("r0").await;
    let b = MockReplica::spawn("r1").await;
    let (router, state) = spawn_router(config_for(&[&a, &b])).await;
    let client = reqwest::Client::new();

    for kind in [
        "round_robin",
        "least_loaded",
        "prefix_affinity",
        "prefix_affinity_bounded",
    ] {
        set_policy(&client, &router, serde_json::json!({"policy": kind})).await;
        let view: serde_json::Value = client
            .get(format!("{router}/admin/policy"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(view["policy"], kind);

        let (_, h) = post_prompt(&client, &router, &prompt_with_prefix(1, 32, 64)).await;
        assert_eq!(h.get("x-router-policy").unwrap(), kind);

        let text = client
            .get(format!("{router}/metrics"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(
            text.contains(&format!("router_active_policy{{policy=\"{kind}\"}} 1")),
            "metrics do not report {kind} as active"
        );
    }
    assert!(
        state
            .metrics
            .policy_switches_total
            .load(std::sync::atomic::Ordering::Relaxed)
            >= 4
    );
}

#[tokio::test]
async fn an_unknown_policy_is_rejected_rather_than_silently_ignored() {
    let m = MockReplica::spawn("r0").await;
    let (router, state) = spawn_router(config_for(&[&m])).await;
    let client = reqwest::Client::new();
    let r = client
        .post(format!("{router}/admin/policy"))
        .json(&serde_json::json!({"policy": "magic_pixie_dust"}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::BAD_REQUEST);
    // A silently ignored switch would mean an entire experiment arm ran under
    // the wrong policy while its artifacts claimed otherwise.
    assert_eq!(state.policy_kind().as_str(), "round_robin");
}

#[tokio::test]
async fn an_ejected_replica_receives_nothing_and_recovers_when_it_returns() {
    let a = MockReplica::spawn("r0").await;
    let b = MockReplica::spawn("r1").await;
    let mut cfg = config_for(&[&a, &b]);
    cfg.health.enabled = true;
    cfg.health.interval_ms = 50;
    cfg.health.timeout_ms = 200;
    cfg.health.unhealthy_threshold = 2;
    cfg.health.healthy_threshold = 2;
    let (router, state) = spawn_router(cfg).await;
    let client = reqwest::Client::new();

    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(state.fleet.healthy_indices().len(), 2);

    b.set_healthy(false);
    // Detection bound is interval * threshold; allow generous slack for CI.
    tokio::time::sleep(Duration::from_millis(800)).await;
    assert_eq!(
        state.fleet.healthy_indices(),
        vec![0],
        "r1 should have been ejected"
    );
    assert_eq!(state.fleet.get(1).ejections_total(), 1);
    assert!(state.fleet.get(1).last_transition_unix_us() > 0);

    let before = b.request_count();
    for _ in 0..10 {
        let (status, h) = post_prompt(&client, &router, &prompt_with_prefix(1, 32, 64)).await;
        assert!(status.is_success());
        assert_eq!(h.get("x-router-replica").unwrap(), "r0");
    }
    assert_eq!(
        b.request_count(),
        before,
        "ejected replica must receive nothing"
    );

    b.set_healthy(true);
    tokio::time::sleep(Duration::from_millis(800)).await;
    assert_eq!(
        state.fleet.healthy_indices().len(),
        2,
        "r1 should have recovered"
    );
    assert_eq!(state.fleet.get(1).recoveries_total(), 1);
}

#[tokio::test]
async fn no_healthy_replica_yields_503_not_a_hang() {
    let a = MockReplica::spawn("r0").await;
    let mut cfg = config_for(&[&a]);
    cfg.health.enabled = true;
    cfg.health.interval_ms = 50;
    cfg.health.unhealthy_threshold = 1;
    let (router, state) = spawn_router(cfg).await;

    a.set_healthy(false);
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(state.fleet.healthy_indices().len(), 0);

    let r = reqwest::Client::new()
        .post(format!("{router}/v1/completions"))
        .json(&serde_json::json!({"model":"mock","prompt":[1,2,3]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
    let body: serde_json::Value = r.json().await.unwrap();
    assert_eq!(body["error"]["type"], "no_healthy_replica");
}

#[tokio::test]
async fn overload_rejects_with_429_when_configured_to_reject() {
    let m = MockReplica::spawn("r0").await;
    m.set_chunks(20);
    m.set_inter_chunk_delay(100); // hold the slot ~2 s

    let mut cfg = config_for(&[&m]);
    cfg.admission.overload_behaviour = OverloadBehaviour::Reject;
    cfg.admission.max_inflight_per_replica = 2;
    let (router, state) = spawn_router(cfg).await;
    let client = reqwest::Client::new();

    // Fill capacity with streams we deliberately do not drain.
    let mut held = Vec::new();
    for _ in 0..2 {
        held.push(
            client
                .post(format!("{router}/v1/completions"))
                .json(&serde_json::json!({"model":"mock","prompt":[1,2,3],"stream":true}))
                .send()
                .await
                .unwrap(),
        );
    }
    tokio::time::sleep(Duration::from_millis(100)).await;

    let r = client
        .post(format!("{router}/v1/completions"))
        .json(&serde_json::json!({"model":"mock","prompt":[1,2,3],"stream":true}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::TOO_MANY_REQUESTS);
    let body: serde_json::Value = r.json().await.unwrap();
    assert_eq!(body["error"]["type"], "overloaded");

    let text = state.metrics.render(&state.fleet, state.policy_kind());
    assert!(text.contains("router_rejected_overload_total 1"));
    drop(held);
}

#[tokio::test]
async fn fleet_status_reports_the_topology_the_driver_records() {
    let a = MockReplica::spawn("r0").await;
    let b = MockReplica::spawn("r1").await;
    let (router, _state) = spawn_router(config_for(&[&a, &b])).await;

    let s: serde_json::Value = reqwest::get(format!("{router}/admin/fleet"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(s["total_replicas"], 2);
    assert_eq!(s["healthy_replicas"], 2);
    assert_eq!(s["replicas"][0]["id"], "r0");
    assert_eq!(s["replicas"][0]["placement"], "mock/gpu0");
    assert_eq!(s["replicas"][1]["placement"], "mock/gpu1");
}

#[tokio::test]
async fn least_loaded_prefers_the_idle_replica_under_real_load() {
    let a = MockReplica::spawn("r0").await;
    let b = MockReplica::spawn("r1").await;
    a.set_chunks(30);
    a.set_inter_chunk_delay(60);
    b.set_chunks(2);

    let (router, state) = spawn_router(config_for(&[&a, &b])).await;
    let client = reqwest::Client::new();
    set_policy(
        &client,
        &router,
        serde_json::json!({"policy": "least_loaded"}),
    )
    .await;

    // Open several streams and leave them undrained. r0's mock emits 30 chunks
    // 60 ms apart, so streams landing there stay in flight for ~1.8 s; r1's
    // emits 2 chunks and drains into the socket buffer almost immediately. The
    // fleet therefore settles with load concentrated on r0.
    let mut held = Vec::new();
    for _ in 0..6 {
        held.push(
            client
                .post(format!("{router}/v1/completions"))
                .json(&serde_json::json!({"model":"mock","prompt":[1,2,3],"stream":true}))
                .send()
                .await
                .unwrap(),
        );
    }
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Read the actual in-flight split rather than assuming which side is busy:
    // which replica ends up loaded depends on the rotating tiebreak's starting
    // point, and hard-coding it would make this test pass or fail by luck.
    let loads: Vec<usize> = (0..2).map(|i| state.fleet.get(i).inflight()).collect();
    let busy = if loads[0] >= loads[1] { 0 } else { 1 };
    let idle = 1 - busy;
    assert!(
        loads[busy] > loads[idle],
        "test setup did not produce an imbalance: {loads:?}"
    );
    let idle_id = format!("r{idle}");

    let mut tail = HashMap::new();
    for _ in 0..6 {
        let r = client
            .post(format!("{router}/v1/completions"))
            .json(&serde_json::json!({"model":"mock","prompt":[1,2,3],"stream":true}))
            .send()
            .await
            .unwrap();
        let id = r
            .headers()
            .get("x-router-replica")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        *tail.entry(id).or_insert(0usize) += 1;
        let _ = r.bytes().await;
    }

    // Every tail request is drained before the next is sent, so at each decision
    // the idle replica is genuinely the least loaded and least_loaded must
    // choose it every time.
    //
    // An earlier version of this test asserted the opposite (`max < 6`, i.e. the
    // requests must be *spread*), which contradicts the policy's entire purpose:
    // concentrating on the idle replica is the correct behaviour, and the
    // assertion would have been satisfied only by a least_loaded that ignored
    // load.
    assert_eq!(
        tail.get(&idle_id).copied().unwrap_or(0),
        6,
        "least_loaded must send every request to the idle replica {idle_id} \
         (in-flight was {loads:?}), got {tail:?}"
    );
    drop(held);
}
