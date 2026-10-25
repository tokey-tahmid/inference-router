#!/bin/bash
# Build, lint and test the router.
#
#   sbatch slurm/build.sh
#
# This runs on the `gpu` partition and uses no GPU. That is not a mistake: the
# allocation it ran under had no CPU hours at all, so a cpu submission was
# rejected outright. The login
# node has no `module` and no `cargo`, so a batch job is the *first* opportunity
# to compile anything at all.
#
#SBATCH --job-name=ir-build
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=logs/ir_build_%j.out

set -euo pipefail

if ! type module >/dev/null 2>&1; then
    set +eu
    source /etc/profile
    set -eu
fi

find_repo() {
    local c
    for c in "${IR_REPO:-}" "${SLURM_SUBMIT_DIR:-}"; do
        if [[ -n "$c" && -f "${c}/env.sh" ]]; then printf '%s' "$c"; return 0; fi
    done
    return 1
}
# BASH_SOURCE points at Slurm's spool copy of this script, not at the repo,
# so the repo has to be found via SLURM_SUBMIT_DIR.
HERE="$(find_repo)" || { echo "cannot locate repo root" >&2; exit 1; }
# shellcheck disable=SC1091
source "${HERE}/env.sh"

log() { printf '[build %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "host=$(hostname) job=${SLURM_JOB_ID} repo=${HERE}"

ir_load_modules

# Toolchain: rustup-managed stable, installed into project storage.
#
# The cluster exposes exactly one Rust module, 1.85.1, and the dependency tree has
# outgrown it: reqwest -> url -> idna -> icu_* now reaches `yoke-derive`, which
# calls `str::from_utf8` as an inherent associated function, and that needs a
# newer compiler (build 5214119: error[E0599] in yoke-derive).
#
# `resolver = "3"` with `rust-version = "1.85"` was the first attempt at this and
# it cannot be made airtight. The MSRV-aware resolver can only honour the
# `rust-version` that each UPSTREAM crate declares; a crate that omits it, or
# declares it wrongly, gives the resolver nothing to act on. So pinning our own
# floor protects against well-behaved dependencies only, and the failure moves
# to whichever transitive crate is least careful, one build at a time.
#
# Compute nodes have outbound internet on this cluster, and rustup needs no
# root, so installing a current toolchain removes the whole class of problem
# instead of chasing it crate by crate. The exact version installed is logged
# below and recorded in the build provenance, so "stable" never becomes an
# unrecorded moving target.
if [[ ! -x "${CARGO_HOME}/bin/rustup" ]]; then
    log "installing rustup into ${RUSTUP_HOME} / ${CARGO_HOME}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal \
            --default-toolchain stable --component clippy,rustfmt
fi
rustup toolchain install stable --profile minimal --component clippy,rustfmt >/dev/null
rustup default stable >/dev/null

log "toolchain (recorded as build provenance):"
cargo --version
rustc --version
cargo clippy --version
{
    echo "rustc=$(rustc --version)"
    echo "cargo=$(cargo --version)"
    echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "slurm_job_id=${SLURM_JOB_ID:-none}"
} > "${HERE}/toolchain.lock"
log "CARGO_HOME=${CARGO_HOME} RUSTUP_HOME=${RUSTUP_HOME} CARGO_TARGET_DIR=${CARGO_TARGET_DIR}"

cd "$HERE"

# Compute nodes have outbound internet here (the provision/run split is inverted
# on this cluster), so the registry fetch works from inside the job.
# rustfmt is applied here rather than only checked. The login node has no cargo,
# so there is no pre-submission hook that could have formatted the tree; a
# check-only gate would just fail every job until someone hand-formatted Rust
# they cannot run a formatter against. Apply, then verify the apply converged.
log "=== fmt (apply, then verify) ==="
cargo fmt --all
cargo fmt --all -- --check

# Resolve first and report what was selected. The MSRV-aware resolver only runs
# when there is no lock to reuse: a Cargo.lock generated before rust-version was
# pinned keeps its MSRV-incompatible selections and the build fails identically
# to having no resolver at all (job 5211452 failed exactly this way).
log "=== resolve (MSRV-aware) ==="
if [[ ! -f Cargo.lock ]]; then
    log "no Cargo.lock: resolving from scratch against rust-version"
fi
cargo generate-lockfile
log "selected versions for the MSRV-sensitive chain:"
grep -A1 -E '^name = "(idna|idna_adapter|url|icu_properties|icu_normalizer|icu_collections|icu_provider)"' Cargo.lock | sed 's/^/  /'

log "=== build ==="
cargo build --release --all-targets

log "=== clippy ==="
cargo clippy --all-targets --all-features -- -D warnings

log "=== test ==="
# --no-fail-fast: run EVERY test binary even when one fails. Without it cargo
# stops at the first failing binary, and in job 5214089 a single wrong assertion
# in tests/routing.rs meant tests/streaming.rs -- the SSE non-buffering proof,
# which is the load-bearing test in this repo -- never ran at all and its status
# was silently unknown.
cargo test --release --no-fail-fast -- --nocapture

log "=== binary ==="
ls -la "${CARGO_TARGET_DIR}/release/inference-router"
"${CARGO_TARGET_DIR}/release/inference-router" --help || true

log "BUILD COMPLETE"
