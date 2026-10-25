# Shared environment for inference-router.
# Source from every slurm script: `source "${HERE}/env.sh"`.
#
# Everything specific to one cluster (where bulk data lives, which modules
# provide Apptainer) comes from site.env, which is git-ignored. Copy
# site.env.example to site.env and fill it in once per cluster. The Slurm
# account is not set anywhere in the repo: pass it at submit time with
# `sbatch -A <account>` or set SBATCH_ACCOUNT in your shell.
#
# `module` is not initialised in a Slurm batch environment on some clusters, so
# callers `set +eu; source /etc/profile; set -eu` before loading modules.

IR_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export IR_REPO

if [[ -f "${IR_REPO}/site.env" ]]; then
    # shellcheck disable=SC1091
    source "${IR_REPO}/site.env"
fi

# Bulk data on project storage, shared with llm-inference-bench. Never in the
# git repo.
: "${IR_DATA:?IR_DATA is not set: copy site.env.example to site.env and set it}"
export IR_DATA
export IR_IMAGES="${IR_DATA}/images"
export IR_LOGS="${IR_DATA}/logs"
export IR_ENV="${IR_DATA}/env"

# llm-inference-bench's harness venv. This project imports
# llm_inference_bench.workload from there rather than reimplementing the
# workload generator: if the two diverged, the prefix-caching numbers there
# would stop being comparable to the routing numbers here, which is the entire
# point of the pairing.
export IR_VENV="${IR_ENV}/venv"

# vLLM image and model weights, provisioned by llm-inference-bench, which is
# expected to be checked out next to this repository. Digests, backend version
# and model revisions are in its provision.lock.json and are COPIED from there
# into every artifact, never retyped.
export IR_SIF_VLLM="${IR_IMAGES}/vllm.sif"
export IR_LOCKFILE="${IR_LOCKFILE:-$(cd "${IR_REPO}/.." && pwd)/llm-inference-bench/provision.lock.json}"

export HF_HOME="${IR_DATA}/hf_cache"
export HF_HUB_CACHE="${IR_DATA}/hf_cache/hub"

# Cargo's registry and build outputs go to project storage: $HOME is small and
# a release build of this dependency tree is well over a GB.
export CARGO_HOME="${IR_DATA}/cargo_home"
# rustup-managed toolchain, also on project storage. See slurm/build.sh for why
# a cluster Rust module is not used.
export RUSTUP_HOME="${IR_DATA}/rustup_home"
export PATH="${CARGO_HOME}/bin:${PATH}"
export IR_TARGET_DIR="${IR_DATA}/router_target"
export CARGO_TARGET_DIR="${IR_TARGET_DIR}"

# Environment modules, loaded in this order. IR_MOD_ENV is the meta-module that
# exposes the software tree, on clusters that have one; site.env.example says why
# it must come first.
export IR_MOD_ENV="${IR_MOD_ENV:-}"
export IR_MOD_APPTAINER="${IR_MOD_APPTAINER:-Apptainer}"

# Load the modules above, skipping any that are unset.
ir_load_modules() {
    local m
    for m in "$IR_MOD_ENV" "$IR_MOD_APPTAINER"; do
        [[ -n "$m" ]] && module load "$m"
    done
    return 0
}

mkdir -p "$IR_LOGS" "$CARGO_HOME" "$RUSTUP_HOME" "$IR_TARGET_DIR" "${IR_REPO}/logs" 2>/dev/null || true
