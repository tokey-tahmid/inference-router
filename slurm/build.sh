#!/bin/bash
# Build, lint and test the router.
#
#   sbatch slurm/build.sh
#
# This runs on the `gpu` partition and uses no GPU. That is not a mistake: the
# p201466 account has no CPU allocation at all (gres/cpun=0) and every `cpu`
# QOS carries DenyOnLimit, so a cpu submission is rejected outright. The login
# node has no `module` and no `cargo`, so a batch job is the *first* opportunity
# to compile anything at all.
#
#SBATCH --job-name=ir-build
#SBATCH --account=p201466
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/ir_build_%j.out

set -euo pipefail

if ! type module >/dev/null 2>&1; then
    set +eu
    source /etc/profile
    set -eu
fi

find_repo() {
    local c
    for c in "${IR_REPO:-}" "${SLURM_SUBMIT_DIR:-}" \
             "/home/users/u104240/apps/inference-work/inference-router"; do
        if [[ -n "$c" && -f "${c}/env.sh" ]]; then printf '%s' "$c"; return 0; fi
    done
    return 1
}
# BASH_SOURCE points at psslurm's spool copy of this script, not at the repo,
# so the repo has to be found via SLURM_SUBMIT_DIR or an explicit path.
HERE="$(find_repo)" || { echo "cannot locate repo root" >&2; exit 1; }
# shellcheck disable=SC1091
source "${HERE}/env.sh"

log() { printf '[build %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "host=$(hostname) job=${SLURM_JOB_ID} repo=${HERE}"

module load "$IR_MOD_ENV"
# Confirm the exact module name rather than trusting a remembered one.
log "available Rust modules:"
module -t avail Rust 2>&1 | sed 's/^/  /'
module load "$IR_MOD_RUST"

log "toolchain:"
cargo --version
rustc --version
log "CARGO_HOME=${CARGO_HOME} CARGO_TARGET_DIR=${CARGO_TARGET_DIR}"

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
