#!/bin/bash
# Measure how many 7B replicas actually fit on one A100-40GB.
#
#   sbatch slurm/probe_replica_density.sh
#
# The question is whether two replicas fit per card or one, and it has to be
# determined empirically: measure, don't assume. The arithmetic says 7B bf16 is ~15.2 GB
# of weights, so two on a 40 GB card leaves ~9.6 GB of KV cache between them at
# gpu_memory_utilization 0.47. Whether that is enough to *serve* is a different
# question from whether it *loads*, so this probe records the number of KV cache
# blocks vLLM actually allocates in each configuration, which is the quantity
# that determines how many concurrent sequences a replica can hold.
#
# Cost: one node for at most 30 minutes. Cheap against committing the whole
# matrix to a topology that turns out to thrash.
#
#SBATCH --job-name=ir-density
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=logs/ir_density_%j.out

set -uo pipefail

if ! type module >/dev/null 2>&1; then
    set +eu
    source /etc/profile
    set -u
fi

find_repo() {
    local c
    for c in "${IR_REPO:-}" "${SLURM_SUBMIT_DIR:-}"; do
        if [[ -n "$c" && -f "${c}/env.sh" ]]; then printf '%s' "$c"; return 0; fi
    done
    return 1
}
HERE="$(find_repo)" || { echo "cannot locate repo root" >&2; exit 1; }
# shellcheck disable=SC1091
source "${HERE}/env.sh"
# shellcheck disable=SC1091
source "${HERE}/slurm/lib_replicas.sh"

export IR_TAG="density"
ir_load_modules

ir_log "host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

# Model revision COPIED from P1's lockfile, never typed.
REV="$(python3 -c "import json;print(json.load(open('${IR_LOCKFILE}'))['models']['Qwen_Qwen2_5-7B-Instruct']['revision'])")"
REPO_US="Qwen--Qwen2.5-7B-Instruct"
MODEL="$(ir_model_path "$REPO_US" "$REV")" || exit 1
ir_log "model=${MODEL}"
ir_log "revision=${REV} (copied from ${IR_LOCKFILE})"

OUT="${IR_LOGS}/ir_density_${SLURM_JOB_ID}"
mkdir -p "$OUT"
RESULT="${HERE}/results/raw/density_probe_${SLURM_JOB_ID}.json"
mkdir -p "$(dirname "$RESULT")"

trap ir_teardown EXIT

# Extract the KV cache size vLLM reports at startup. This is the number that
# decides the topology; a replica that loads but has room for 3 sequences is not
# a usable replica.
kv_report() {
    grep -iE "GPU KV cache size|Maximum concurrency|Available KV cache memory|# GPU blocks" "$1" 2>/dev/null | head -8
}

declare -a FINDINGS=()

record() {
    FINDINGS+=("$1")
    ir_log "FINDING: $1"
}

# ---- Arm A: one replica per card at a high memory fraction (the baseline) ----
ir_log "=== Arm A: 1 replica on GPU0, gpu_memory_utilization=0.90 ==="
LOG_A="${OUT}/armA_gpu0.log"
PID_A=$(ir_launch_replica 0 8100 "$MODEL" qwen2.5-7b-instruct 0.90 256 "$LOG_A")
if ir_wait_healthy 900 8100; then
    ir_log "Arm A healthy"
    kv_report "$LOG_A" | sed 's/^/  A: /'
    curl -sf -m 10 http://127.0.0.1:8100/metrics -o "${OUT}/armA_metrics.txt" || true
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  A mem: /'
    record "arm_a_single_replica_util0.90=healthy"
else
    record "arm_a_single_replica_util0.90=FAILED"
    tail -40 "$LOG_A" | sed 's/^/  A: /'
fi
kill "$PID_A" 2>/dev/null || true
pkill -f "port 8100" 2>/dev/null || true
sleep 20

# ---- Arm B: two replicas on ONE card ----------------------------------------
# 0.47 each leaves a little headroom below 1.0 for allocator slop and the CUDA
# context each process creates (~500 MB, and there are two of them here).
ir_log "=== Arm B: 2 replicas both on GPU0, gpu_memory_utilization=0.47 each ==="
LOG_B1="${OUT}/armB_gpu0_a.log"; LOG_B2="${OUT}/armB_gpu0_b.log"
PID_B1=$(ir_launch_replica 0 8110 "$MODEL" qwen2.5-7b-instruct 0.47 128 "$LOG_B1")
sleep 30   # stagger: two concurrent 15 GB reads off Lustre contend badly
PID_B2=$(ir_launch_replica 0 8111 "$MODEL" qwen2.5-7b-instruct 0.47 128 "$LOG_B2")
if ir_wait_healthy 1200 8110 8111; then
    ir_log "Arm B healthy: TWO replicas fit on one A100-40GB"
    kv_report "$LOG_B1" | sed 's/^/  B1: /'
    kv_report "$LOG_B2" | sed 's/^/  B2: /'
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  B mem: /'
    curl -sf -m 10 http://127.0.0.1:8110/metrics -o "${OUT}/armB1_metrics.txt" || true
    curl -sf -m 10 http://127.0.0.1:8111/metrics -o "${OUT}/armB2_metrics.txt" || true
    record "arm_b_two_replicas_per_card_util0.47=healthy"
else
    ir_log "Arm B FAILED: two replicas do not both come up on one card"
    tail -40 "$LOG_B1" | sed 's/^/  B1: /'
    tail -40 "$LOG_B2" | sed 's/^/  B2: /'
    record "arm_b_two_replicas_per_card_util0.47=FAILED"
fi
kill "$PID_B1" "$PID_B2" 2>/dev/null || true
sleep 20
ir_teardown
sleep 20

# ---- Arm C: the intended production topology, 4 replicas on 4 cards ---------
ir_log "=== Arm C: 4 replicas, one per GPU, gpu_memory_utilization=0.90 ==="
declare -a PIDS_C=()
for g in 0 1 2 3; do
    p=$(( 8120 + g ))
    PIDS_C+=("$(ir_launch_replica "$g" "$p" "$MODEL" qwen2.5-7b-instruct 0.90 256 "${OUT}/armC_gpu${g}.log")")
    sleep 15
done
if ir_wait_healthy 1500 8120 8121 8122 8123; then
    ir_log "Arm C healthy: 4 replicas, 1 per GPU"
    for g in 0 1 2 3; do kv_report "${OUT}/armC_gpu${g}.log" | sed "s/^/  C${g}: /"; done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  C mem: /'
    record "arm_c_four_replicas_one_per_gpu=healthy"
else
    record "arm_c_four_replicas_one_per_gpu=FAILED"
    for g in 0 1 2 3; do tail -30 "${OUT}/armC_gpu${g}.log" | sed "s/^/  C${g}: /"; done
fi

ir_log "=== findings ==="
printf '%s\n' "${FINDINGS[@]}"
ir_log "logs and metrics under ${OUT}"
ir_log "DENSITY PROBE COMPLETE"
