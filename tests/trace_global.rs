//! The same check as `trace_on.rs`, but through `telemetry::init`, the global
//! subscriber and a multi-threaded runtime: the configuration the binary runs in.
//! (The OTLP endpoint is a closed port; exporting fails quietly, which is fine here.)

mod common;

use common::{config_for, spawn_router, MockReplica};

const CALLER: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

#[test]
fn the_callers_trace_continues_with_the_binarys_own_setup() {
    let telemetry = inference_router::telemetry::init(
        true,
        Some("http://127.0.0.1:9/v1/traces"),
        "router-test",
    )
    .expect("telemetry init");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let seen = rt.block_on(async {
        let mock = MockReplica::spawn("r0").await;
        let (router, _state) = spawn_router(config_for(&[&mock])).await;
        let resp = reqwest::Client::new()
            .post(format!("{router}/v1/chat/completions"))
            .header("traceparent", CALLER)
            .json(&serde_json::json!({
                "model": "mock", "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4, "stream": true
            }))
            .send()
            .await
            .expect("router responds");
        assert!(resp.status().is_success());
        let _ = resp.bytes().await.expect("body streams to the end");
        mock.last_traceparent()
    });
    drop(rt);
    telemetry.shutdown();
    let seen = seen.expect("the replica received a traceparent");
    assert_eq!(
        seen.split('-').nth(1),
        Some("4bf92f3577b34da6a3ce929d0e0e4736"),
        "the trace ID must be the caller's, got {seen}"
    );
}
