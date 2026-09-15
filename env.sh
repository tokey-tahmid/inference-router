# Shared environment for inference-router (P3) on MeluXina.
# Source from every slurm script: `source "${HERE}/env.sh"`.
#
# `module` is NOT initialised in a Slurm batch environment here; callers must
# `set +eu; source /etc/profile; set -eu` before loading modules.

# Compute is billed to p201466 (GPU headroom, no CPU allocation at all, so every
# job -- including a pure `cargo build` that needs no GPU -- runs on the `gpu`
# partition). Storage stays under p201362; filesystem access follows unix group
# membership, not the Slurm account. See ../CLAUDE.md.
export IR_ACCOUNT="p201466"

IR_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export IR_REPO

# Bulk data on project storage, shared with P1. Never in the git repo.
export IR_DATA="/mnt/tier2/project/p201362/ttahmid/inference-work"
export IR_IMAGES="${IR_DATA}/images"
export IR_LOGS="${IR_DATA}/logs"
export IR_ENV="${IR_DATA}/env"

# P1's harness venv. P3 imports llm_inference_bench.workload from here rather
# than reimplementing the workload generator: if the two diverged, P1's
# prefix-caching numbers would stop being comparable to P3's routing numbers,
# which is the entire point of the pairing.
export IR_VENV="${IR_ENV}/venv"

# vLLM image and model weights, provisioned by P1. Digests, backend version and
# model revision SHAs are in P1's provision.lock.json and are COPIED from there
# into every artifact -- never retyped.
export IR_SIF_VLLM="${IR_IMAGES}/vllm.sif"
export IR_LOCKFILE="/home/users/u104240/apps/inference-work/llm-inference-bench/provision.lock.json"

export HF_HOME="${IR_DATA}/hf_cache"
export HF_HUB_CACHE="${IR_DATA}/hf_cache/hub"

# Cargo's registry and build outputs go to project storage: $HOME is small and
# a release build of this dependency tree is well over a GB.
export CARGO_HOME="${IR_DATA}/cargo_home"
export IR_TARGET_DIR="${IR_DATA}/router_target"
export CARGO_TARGET_DIR="${IR_TARGET_DIR}"

# `module load env/release/2025.1` FIRST: the EasyBuild tree is only exposed
# behind that meta-module, and its absence gives the misleading Lmod error
# "exist but cannot be loaded as requested".
export IR_MOD_ENV="env/release/2025.1"
export IR_MOD_RUST="Rust/1.85.1-GCCcore-14.2.0"
export IR_MOD_APPTAINER="Apptainer/1.4.2-GCCcore-14.2.0"

mkdir -p "$IR_LOGS" "$CARGO_HOME" "$IR_TARGET_DIR" 2>/dev/null || true
