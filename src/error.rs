//! Error types for the router.
//!
//! The distinction that matters here is between *client* errors (the request was
//! malformed, or the fleet is over capacity and the configured overload policy is
//! to reject) and *upstream* errors (a replica failed). They map to different
//! status codes and, more importantly, they are counted separately: an
//! experiment that lumps 429s in with 502s cannot tell deliberate backpressure
//! apart from a broken replica, and those are opposite findings.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("no healthy replica available")]
    NoHealthyReplica,

    #[error("fleet at capacity: {inflight} in flight, limit {limit}")]
    Overloaded { inflight: usize, limit: usize },

    #[error("timed out waiting for admission after {waited_ms} ms")]
    AdmissionTimeout { waited_ms: u64 },

    #[error("invalid request body: {0}")]
    BadRequest(String),

    #[error("upstream replica {replica} failed: {source}")]
    Upstream {
        replica: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("unknown routing policy {0:?}")]
    UnknownPolicy(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl RouterError {
    /// Short stable label used for the Prometheus `error_kind` dimension.
    ///
    /// Deliberately a fixed, small set: a label whose cardinality grows with
    /// upstream error strings would blow up the metrics series count.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::NoHealthyReplica => "no_healthy_replica",
            Self::Overloaded { .. } => "overloaded",
            Self::AdmissionTimeout { .. } => "admission_timeout",
            Self::BadRequest(_) => "bad_request",
            Self::Upstream { .. } => "upstream",
            Self::UnknownPolicy(_) => "unknown_policy",
            Self::Other(_) => "internal",
        }
    }

    pub fn status(&self) -> StatusCode {
        match self {
            // 503, not 500: there is nothing wrong with the request, the fleet
            // simply has nowhere to put it right now.
            Self::NoHealthyReplica => StatusCode::SERVICE_UNAVAILABLE,
            Self::Overloaded { .. } | Self::AdmissionTimeout { .. } => {
                StatusCode::TOO_MANY_REQUESTS
            }
            Self::BadRequest(_) | Self::UnknownPolicy(_) => StatusCode::BAD_REQUEST,
            Self::Upstream { .. } => StatusCode::BAD_GATEWAY,
            Self::Other(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for RouterError {
    fn into_response(self) -> Response {
        let status = self.status();
        let body = json!({
            "error": {
                "message": self.to_string(),
                "type": self.kind(),
                "code": status.as_u16(),
            }
        });
        tracing::warn!(error_kind = self.kind(), status = status.as_u16(), error = %self, "request_failed");
        (status, axum::Json(body)).into_response()
    }
}

pub type Result<T> = std::result::Result<T, RouterError>;
