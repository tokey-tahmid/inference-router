//! With tracing on, the router continues the caller's W3C trace: the replica
//! receives the caller's trace ID, with the router's own span as its parent.
//! This is what lets one trace follow a request from the load generator through
//! the router into vLLM.
//!
//! Tracing state is process-global, so this file holds the only test that
//! switches it on; `trace_off.rs` checks the default in its own process.

mod common;

use common::{config_for, spawn_router, MockReplica};
use opentelemetry::trace::TracerProvider as _;
use tracing_subscriber::layer::SubscriberExt;

const CALLER: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

#[tokio::test]
async fn the_callers_trace_continues_through_the_router_to_the_replica() {
    // An SDK provider with no exporter still creates real, sampled spans.
    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder().build();
    let subscriber = tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));
    // The test runtime is single-threaded, so the router's tasks see this subscriber.
    let _guard = tracing::subscriber::set_default(subscriber);
    inference_router::telemetry::enable_propagation();

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

    let seen = mock
        .last_traceparent()
        .expect("the replica received a traceparent");
    let parts: Vec<&str> = seen.split('-').collect();
    assert_eq!(parts.len(), 4, "malformed traceparent {seen:?}");
    assert_eq!(
        parts[1], "4bf92f3577b34da6a3ce929d0e0e4736",
        "the trace ID must be the caller's"
    );
    assert_ne!(
        parts[2], "00f067aa0ba902b7",
        "the parent must be the router's span, not the caller's"
    );
    assert_eq!(parts[3], "01", "the sampled flag must survive");
}
