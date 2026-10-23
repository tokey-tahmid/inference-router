//! With tracing off (the default), the router passes the caller's trace headers
//! through untouched, so the replica's own spans can still join a client trace.

mod common;

use common::{config_for, spawn_router, MockReplica};

const CALLER: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

#[tokio::test]
async fn trace_headers_pass_through_unchanged_when_tracing_is_off() {
    let mock = MockReplica::spawn("r0").await;
    let (router, _state) = spawn_router(config_for(&[&mock])).await;
    let resp = reqwest::Client::new()
        .post(format!("{router}/v1/completions"))
        .header("traceparent", CALLER)
        .json(&serde_json::json!({
            "model": "mock", "prompt": [1, 2, 3], "max_tokens": 4, "stream": true
        }))
        .send()
        .await
        .expect("router responds");
    assert!(resp.status().is_success());
    let _ = resp.bytes().await.expect("body streams to the end");
    assert_eq!(mock.last_traceparent().as_deref(), Some(CALLER));
}
