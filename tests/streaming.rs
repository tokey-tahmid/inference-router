//! The non-buffering proof.
//!
//! "SSE streaming verified non-buffering" is a definition-of-done item, and the
//! only honest way to verify it is to make the upstream *stall* and check that
//! the client has already received data. A buffering proxy passes every
//! correctness test — the bytes are identical — and fails only this one.

mod common;

use std::time::{Duration, Instant};

use common::{config_for, spawn_router, MockReplica};
use futures_util::StreamExt;

/// The headline test: the upstream emits one chunk, then stalls for 800 ms.
/// A non-buffering router delivers that first chunk immediately.
#[tokio::test]
async fn first_chunk_arrives_before_the_upstream_finishes() {
    let mock = MockReplica::spawn("r0").await;
    mock.set_chunks(4);
    mock.set_first_chunk_delay(0);
    mock.set_inter_chunk_delay(300); // 3 gaps -> ~900 ms total

    let (router, _state) = spawn_router(config_for(&[&mock])).await;
    let client = reqwest::Client::new();

    let start = Instant::now();
    let resp = client
        .post(format!("{router}/v1/completions"))
        .json(&serde_json::json!({
            "model": "mock", "prompt": [1, 2, 3], "max_tokens": 8, "stream": true
        }))
        .send()
        .await
        .expect("router responds");
    assert!(resp.status().is_success());

    let mut stream = resp.bytes_stream();
    let first = stream
        .next()
        .await
        .expect("a first chunk")
        .expect("no error");
    let ttfb = start.elapsed();
    assert!(!first.is_empty());

    // Drain the rest and time the whole thing.
    let mut chunks = 1usize;
    while let Some(c) = stream.next().await {
        assert!(c.is_ok());
        chunks += 1;
    }
    let total = start.elapsed();

    assert!(
        ttfb < Duration::from_millis(250),
        "first chunk took {ttfb:?}; a non-buffering proxy must deliver it immediately"
    );
    assert!(
        total > Duration::from_millis(700),
        "total was {total:?}; the upstream should have stalled ~900 ms, so the \
         mock is not exercising the stall this test depends on"
    );
    assert!(
        total > ttfb * 3,
        "first byte at {ttfb:?} but total {total:?}: the response was buffered"
    );
    assert!(
        chunks >= 2,
        "received {chunks} chunks; framing was collapsed"
    );
}

/// Chunk boundaries survive the proxy. If the router re-framed the body, a
/// client parsing SSE incrementally would see a different number of events than
/// the replica sent, and per-token timing measured downstream would be wrong.
#[tokio::test]
async fn chunk_boundaries_are_preserved_rather_than_coalesced() {
    let mock = MockReplica::spawn("r0").await;
    mock.set_chunks(6);
    mock.set_inter_chunk_delay(40);

    let (router, _state) = spawn_router(config_for(&[&mock])).await;
    let resp = reqwest::Client::new()
        .post(format!("{router}/v1/completions"))
        .json(&serde_json::json!({
            "model": "mock", "prompt": [1, 2, 3], "stream": true
        }))
        .send()
        .await
        .unwrap();

    let mut stream = resp.bytes_stream();
    let mut arrivals = Vec::new();
    let start = Instant::now();
    let mut body = Vec::new();
    while let Some(c) = stream.next().await {
        let c = c.unwrap();
        arrivals.push(start.elapsed());
        body.extend_from_slice(&c);
    }

    assert_eq!(arrivals.len(), 6, "expected one chunk per upstream chunk");
    // Arrivals must be spread over time, not all at the end.
    let spread = arrivals.last().unwrap().saturating_sub(arrivals[0]);
    assert!(
        spread > Duration::from_millis(120),
        "all chunks arrived within {spread:?}: they were buffered and released together"
    );
    let text = String::from_utf8(body).unwrap();
    assert!(text.contains("data: [DONE]"));
    assert_eq!(text.matches("data: ").count(), 6);
}

/// The in-flight count must track the *stream*, not the handler. This is what
/// `least_loaded` reads, and if it were decremented when the handler returned it
/// would be zero at every routing decision.
#[tokio::test]
async fn inflight_is_held_for_the_life_of_the_stream() {
    let mock = MockReplica::spawn("r0").await;
    mock.set_chunks(5);
    mock.set_inter_chunk_delay(150);

    let (router, state) = spawn_router(config_for(&[&mock])).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{router}/v1/completions"))
        .json(&serde_json::json!({"model": "mock", "prompt": [1, 2, 3], "stream": true}))
        .send()
        .await
        .unwrap();

    // Headers are back but the body is still streaming.
    let mut stream = resp.bytes_stream();
    let _first = stream.next().await.unwrap().unwrap();
    assert_eq!(
        state.fleet.get(0).inflight(),
        1,
        "replica must count as busy while its response is still streaming"
    );

    while stream.next().await.is_some() {}
    // The guard drops with the stream; give the runtime a tick to observe it.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        state.fleet.get(0).inflight(),
        0,
        "in-flight must return to zero once the stream completes"
    );
}

/// A client that hangs up mid-generation must release the slot, or an overload
/// arm (where clients time out constantly) leaks in-flight counts until the
/// router believes the fleet is permanently saturated.
#[tokio::test]
async fn client_disconnect_releases_the_inflight_slot() {
    let mock = MockReplica::spawn("r0").await;
    mock.set_chunks(50);
    mock.set_inter_chunk_delay(100);

    let (router, state) = spawn_router(config_for(&[&mock])).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{router}/v1/completions"))
        .json(&serde_json::json!({"model": "mock", "prompt": [1, 2, 3], "stream": true}))
        .send()
        .await
        .unwrap();
    let mut stream = resp.bytes_stream();
    let _ = stream.next().await.unwrap().unwrap();
    assert_eq!(state.fleet.get(0).inflight(), 1);

    drop(stream); // hang up mid-generation
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        state.fleet.get(0).inflight(),
        0,
        "abandoned stream must not leak an in-flight slot"
    );
    let text = state.metrics.render(&state.fleet, state.policy_kind());
    assert!(
        text.contains("router_client_disconnect_total 1"),
        "disconnect should be counted separately from a completed stream"
    );
}

/// Streaming must stay concurrent across replicas: N slow streams in flight must
/// not serialise. A proxy that awaited each body to completion inside the
/// handler would pass every other test here and fail this one.
#[tokio::test]
async fn concurrent_streams_do_not_serialise() {
    let a = MockReplica::spawn("r0").await;
    let b = MockReplica::spawn("r1").await;
    for m in [&a, &b] {
        m.set_chunks(4);
        m.set_inter_chunk_delay(150); // ~450 ms each
    }

    let (router, _state) = spawn_router(config_for(&[&a, &b])).await;
    let client = reqwest::Client::new();

    let start = Instant::now();
    let mut tasks = Vec::new();
    for _ in 0..8 {
        let client = client.clone();
        let router = router.clone();
        tasks.push(tokio::spawn(async move {
            let resp = client
                .post(format!("{router}/v1/completions"))
                .json(&serde_json::json!({"model":"mock","prompt":[1,2,3],"stream":true}))
                .send()
                .await
                .unwrap();
            let mut s = resp.bytes_stream();
            let mut n = 0;
            while let Some(c) = s.next().await {
                c.unwrap();
                n += 1;
            }
            n
        }));
    }
    let mut total_chunks = 0;
    for t in tasks {
        total_chunks += t.await.unwrap();
    }
    let elapsed = start.elapsed();

    assert_eq!(total_chunks, 8 * 4);
    // Serialised would be ~8 x 450 ms = 3.6 s. Concurrent is ~450 ms plus slop.
    assert!(
        elapsed < Duration::from_millis(1500),
        "8 concurrent 450 ms streams took {elapsed:?}: they were serialised"
    );
}

/// TTFT recorded by the router must reflect the replica's prefill delay, not the
/// full generation. This is the metric every routing result is built on.
#[tokio::test]
async fn router_observed_ttft_tracks_prefill_not_total_generation() {
    let mock = MockReplica::spawn("r0").await;
    mock.set_chunks(6);
    mock.set_first_chunk_delay(200); // "prefill"
    mock.set_inter_chunk_delay(100); // 5 more gaps -> +500 ms

    let (router, state) = spawn_router(config_for(&[&mock])).await;
    let resp = reqwest::Client::new()
        .post(format!("{router}/v1/completions"))
        .json(&serde_json::json!({"model":"mock","prompt":[1,2,3],"stream":true}))
        .send()
        .await
        .unwrap();
    let mut s = resp.bytes_stream();
    while let Some(c) = s.next().await {
        c.unwrap();
    }
    tokio::time::sleep(Duration::from_millis(50)).await;

    let text = state.metrics.render(&state.fleet, state.policy_kind());
    let ttft_sum = metric_value(&text, "router_ttft_seconds_sum").expect("ttft recorded");
    let e2e_sum = metric_value(&text, "router_e2e_seconds_sum").expect("e2e recorded");

    assert!(
        (0.15..0.45).contains(&ttft_sum),
        "TTFT was {ttft_sum}s, expected ~0.2s (the prefill delay)"
    );
    assert!(
        e2e_sum > ttft_sum + 0.3,
        "e2e {e2e_sum}s should exceed TTFT {ttft_sum}s by the decode span; if they \
         are equal the router buffered the response"
    );
}

fn metric_value(text: &str, name: &str) -> Option<f64> {
    text.lines()
        .find(|l| l.starts_with(name) && l.as_bytes().get(name.len()) == Some(&b' '))
        .and_then(|l| l.rsplit(' ').next())
        .and_then(|v| v.parse().ok())
}
