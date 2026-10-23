//! Optional OpenTelemetry tracing.
//!
//! Tracing is off unless an OTLP endpoint is given (`--otlp-endpoint`), so the
//! default router behaves, and performs, exactly as before. When it is on, every
//! proxied request gets a `router.request` span that:
//!
//! * continues the caller's W3C trace when the request carries a `traceparent`
//!   header, so a load generator's client span, the router and the replica end
//!   up in one trace;
//! * stays open until the last byte of the response reaches the client, because
//!   the span lives in the relayed stream's state, like the in-flight guard; its
//!   duration is therefore the router-observed end-to-end latency;
//! * has child spans for the admission queue and for the upstream round trip up
//!   to the response headers;
//! * is injected into the request sent to the replica, so vLLM's own spans
//!   (`--otlp-traces-endpoint`) join the same trace.
//!
//! With tracing off, the caller's `traceparent` and `tracestate` headers are
//! still forwarded unchanged, so the replica's spans can join a client trace
//! even when the router itself is not exporting.

use std::sync::atomic::{AtomicBool, Ordering};

use axum::http::{HeaderMap, HeaderName};
use opentelemetry::propagation::TextMapPropagator;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{fmt, EnvFilter, Layer};

static ENABLED: AtomicBool = AtomicBool::new(false);

/// True once trace propagation is on (exporting, or switched on by a test).
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Install the W3C trace-context propagator and mark tracing as on. [`init`]
/// calls this when an endpoint is configured; tests that build their own
/// subscriber call it directly.
pub fn enable_propagation() {
    opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
    ENABLED.store(true, Ordering::Relaxed);
}

/// Owns the tracer provider, so that spans still in the batch are flushed.
pub struct Telemetry {
    provider: Option<SdkTracerProvider>,
}

impl Telemetry {
    /// Flush and stop the exporter. Call this outside the async runtime: the
    /// OTLP/HTTP exporter uses a blocking client on its own thread.
    pub fn shutdown(self) {
        if let Some(p) = self.provider {
            if let Err(e) = p.shutdown() {
                eprintln!("telemetry shutdown: {e}");
            }
        }
    }
}

/// Set up logging (human-readable or JSON) and, when `otlp_endpoint` is given,
/// OpenTelemetry trace export over OTLP/HTTP to that exact URL, for example
/// `http://127.0.0.1:4318/v1/traces`.
pub fn init(
    log_json: bool,
    otlp_endpoint: Option<&str>,
    service_name: &str,
) -> anyhow::Result<Telemetry> {
    let filter = EnvFilter::try_from_env("ROUTER_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    let fmt_layer = if log_json {
        fmt::layer().json().with_current_span(false).boxed()
    } else {
        fmt::layer().boxed()
    };
    let provider = match otlp_endpoint {
        Some(endpoint) => {
            let exporter = opentelemetry_otlp::SpanExporter::builder()
                .with_http()
                .with_endpoint(endpoint)
                .build()?;
            Some(
                SdkTracerProvider::builder()
                    .with_batch_exporter(exporter)
                    .with_resource(
                        Resource::builder()
                            .with_service_name(service_name.to_owned())
                            .build(),
                    )
                    .build(),
            )
        }
        None => None,
    };
    let otel_layer = provider
        .as_ref()
        .map(|p| tracing_opentelemetry::layer().with_tracer(p.tracer("inference-router")));
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .with(otel_layer)
        .try_init()?;
    if provider.is_some() {
        enable_propagation();
    }
    Ok(Telemetry { provider })
}

/// The trace context carried by the caller's request headers (empty if none).
pub fn parent_from(headers: &HeaderMap) -> opentelemetry::Context {
    opentelemetry::global::get_text_map_propagator(|p| {
        p.extract(&opentelemetry_http::HeaderExtractor(headers))
    })
}

/// The headers that carry `span`'s trace context to the replica. With tracing
/// off, the caller's own `traceparent` and `tracestate` are passed through.
pub fn outgoing_headers(span: &tracing::Span, incoming: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    if enabled() {
        let cx = span.context();
        opentelemetry::global::get_text_map_propagator(|p| {
            p.inject_context(&cx, &mut opentelemetry_http::HeaderInjector(&mut out))
        });
    } else {
        for name in ["traceparent", "tracestate"] {
            if let Some(v) = incoming.get(name) {
                out.insert(HeaderName::from_static(name), v.clone());
            }
        }
    }
    out
}
