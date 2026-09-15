//! Router entry point.

use std::path::PathBuf;

use clap::Parser;
use inference_router::config::{Config, PolicyKind};
use inference_router::state::AppState;
use inference_router::{app, health};

#[derive(Debug, Parser)]
#[command(name = "inference-router", about = "Prefix-aware router for LLM inference replicas")]
struct Cli {
    /// Path to the TOML config.
    #[arg(short, long, env = "ROUTER_CONFIG")]
    config: PathBuf,

    /// Override the listen address from the config.
    #[arg(long, env = "ROUTER_LISTEN")]
    listen: Option<String>,

    /// Override the starting routing policy.
    #[arg(long, env = "ROUTER_POLICY")]
    policy: Option<String>,

    /// Emit newline-delimited JSON logs instead of human-readable ones.
    ///
    /// On by default under Slurm, where the log is a file that analysis reads
    /// rather than a terminal a human watches.
    #[arg(long, env = "ROUTER_LOG_JSON")]
    log_json: bool,

    /// Validate the config, print the resolved fleet, and exit without binding.
    ///
    /// The login node has no cargo and no GPUs, so the first chance to run
    /// anything is inside a batch job. This makes "is the config well formed"
    /// a cheap check at the top of a job rather than a failure five minutes
    /// into one.
    #[arg(long)]
    check: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let filter = tracing_subscriber::EnvFilter::try_from_env("ROUTER_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    if cli.log_json {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .with_current_span(false)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }

    let mut config = Config::load(&cli.config)?;
    if let Some(listen) = cli.listen {
        config.listen = listen.parse()?;
    }
    if let Some(p) = cli.policy {
        config.policy.kind = PolicyKind::parse(&p)
            .ok_or_else(|| anyhow::anyhow!("unknown routing policy {p:?}"))?;
    }
    config.validate()?;

    let state = AppState::new(config)?;

    tracing::info!(
        listen = %state.config.listen,
        replicas = state.fleet.len(),
        policy = state.policy_kind().as_str(),
        prefix_hash_tokens = state.config.policy.prefix_hash_tokens,
        overload_behaviour = ?state.config.admission.overload_behaviour,
        max_inflight_per_replica = state.config.admission.max_inflight_per_replica,
        health_interval_ms = state.config.health.interval_ms,
        unhealthy_threshold = state.config.health.unhealthy_threshold,
        "router_starting"
    );
    for r in &state.fleet.replicas {
        tracing::info!(
            replica = %r.id,
            url = %r.url,
            placement = r.placement.as_deref().unwrap_or("unspecified"),
            "replica_registered"
        );
    }

    if cli.check {
        tracing::info!("config check passed; exiting without binding");
        return Ok(());
    }

    let health_task = health::spawn(state.clone());

    let listener = tokio::net::TcpListener::bind(state.config.listen).await?;
    // Report the *bound* address: a config of port 0 is useful in tests and the
    // ephemeral port has to be discoverable.
    tracing::info!(bound = %listener.local_addr()?, "router_listening");

    let result = axum::serve(listener, app(state))
        .with_graceful_shutdown(shutdown_signal())
        .await;

    if let Some(h) = health_task {
        h.abort();
    }
    result?;
    tracing::info!("router_stopped");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(e) => tracing::error!(error = %e, "cannot install SIGTERM handler"),
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("shutdown: SIGINT"),
        _ = terminate => tracing::info!("shutdown: SIGTERM"),
    }
}
