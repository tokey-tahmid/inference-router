#!/bin/bash
# Kill a replica mid-load and measure detection, error rate and recovery,
# once per routing policy.
#
#   sbatch slurm/run_failure_mode.sh <rate>
#
# Why per policy: the policies differ in how a dead replica's share is
# redistributed, and that is the interesting part. Round robin simply rotates
# over a shorter list. Prefix affinity has to remap the dead replica's *keys*,
# and because placement is rendezvous hashing over stable indices, only those
# keys move -- the survivors keep their own warm prefixes instead of every
# replica's cache being invalidated at once.
#
# Detection time is measured against an externally recorded kill timestamp, not
# against the router's own belief about when the replica died. The router
# reports router_replica_last_transition_unix_us in wall-clock microseconds
# precisely so the two clocks are comparable.
#
#SBATCH --job-name=ir-failure
#SBATCH --account=p201466
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/ir_failure_%j.out

set -uo pipefail

if ! type module >/dev/null 2>&1; then
    set +eu
    source /etc/profile
    set -u
fi

find_repo() {
    local c
    for c in "${IR_REPO:-}" "${SLURM_SUBMIT_DIR:-}" \
             "/home/users/u104240/apps/inference-work/inference-router"; do
        if [[ -n "$c" && -f "${c}/env.sh" ]]; then printf '%s' "$c"; return 0; fi
    done
    return 1
}
HERE="$(find_repo)" || { echo "cannot locate repo root" >&2; exit 1; }
# shellcheck disable=SC1091
source "${HERE}/env.sh"
# shellcheck disable=SC1091
source "${HERE}/slurm/lib_replicas.sh"

export IR_TAG="failure"
module load "$IR_MOD_ENV"
module load "$IR_MOD_APPTAINER"

RATE="${1:?usage: run_failure_mode.sh <request_rate>}"
NREPLICAS="${IR_NREPLICAS:-4}"
KILL_IDX="${IR_KILL_IDX:-2}"

ir_log "host=$(hostname) job=${SLURM_JOB_ID} rate=${RATE} kill=r${KILL_IDX}"
ROUTER_BIN="${CARGO_TARGET_DIR}/release/inference-router"
[[ -x "$ROUTER_BIN" ]] || { ir_log "router binary missing"; exit 1; }

REV="$(python3 -c "import json;print(json.load(open('${IR_LOCKFILE}'))['models']['Qwen_Qwen2_5-7B-Instruct']['revision'])")"
MODEL="$(ir_model_path "Qwen--Qwen2.5-7B-Instruct" "$REV")" || exit 1

OUT="${IR_LOGS}/ir_failure_${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap ir_teardown EXIT

KILL_PORT=$(( 8100 + KILL_IDX ))
# Kill by port so the command is exact and re-runnable, and so it targets the
# replica process rather than anything else that happens to match "vllm".
KILL_SCRIPT="${OUT}/kill_r${KILL_IDX}.sh"
cat > "$KILL_SCRIPT" <<EOF
#!/bin/bash
# SIGKILL, not SIGTERM: the point is an abrupt failure. A graceful shutdown
# would let vLLM drain its in-flight requests, which is a much kinder
# experiment than the one intended.
pkill -9 -f "port ${KILL_PORT}"
EOF
chmod +x "$KILL_SCRIPT"

declare -a POLICIES=(round_robin least_loaded prefix_affinity prefix_affinity_bounded)

for POLICY in "${POLICIES[@]}"; do
  for REP in 1 2 3; do
    ir_log "=== ${POLICY} rep ${REP}: launching a fresh fleet ==="
    # A fresh fleet per trial, unlike the matrix: the killed replica has to come
    # back, and restarting one replica inside a live fleet would give it a cold
    # cache while its peers stayed warm -- an asymmetry that would show up in the
    # recovery numbers as though it were a routing effect.
    declare -a PORTS=() REPL_ARGS=() CFG_SPECS=()
    NODE="$(hostname)"
    for (( i=0; i<NREPLICAS; i++ )); do
        gpu=$(( i % 4 )); port=$(( 8100 + i ))
        PORTS+=("$port")
        REPL_ARGS+=("r${i}=http://127.0.0.1:${port}")
        CFG_SPECS+=("${port}:node=${NODE},gpu=${gpu}")
        ir_launch_replica "$gpu" "$port" "$MODEL" qwen2.5-7b-instruct 0.90 256 \
            "${OUT}/${POLICY}_rep${REP}_replica${i}.log" >/dev/null
        sleep 15
    done
    if ! ir_wait_healthy 1800 "${PORTS[@]}"; then
        ir_log "fleet failed to start for ${POLICY} rep ${REP}; recording and moving on"
        ir_teardown; sleep 20; continue
    fi

    CFG="${OUT}/router_${POLICY}_${REP}.toml"
    ir_write_router_config "$CFG" "127.0.0.1:8080" "$POLICY" 64 "reject" "${CFG_SPECS[@]}"
    "$ROUTER_BIN" --config "$CFG" --log-json > "${OUT}/router_${POLICY}_${REP}.log" 2>&1 &
    for _ in $(seq 1 60); do curl -sf -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done

    PYTHONPATH="${HERE}/driver" "${IR_VENV}/bin/python" -m ir_driver.cli failure \
        --router http://127.0.0.1:8080 \
        --replicas "${REPL_ARGS[@]}" \
        --model-path "$MODEL" \
        --served-name qwen2.5-7b-instruct \
        --lockfile "$IR_LOCKFILE" \
        --out-dir "${HERE}/results/raw" \
        --policy "$POLICY" \
        --rate "$RATE" \
        --ratio 0.6 \
        --repetition-index "$REP" \
        --kill "r${KILL_IDX}" \
        --kill-command "$KILL_SCRIPT" \
        --kill-after-s 30 \
        --num-requests $(python3 -c "print(int(${RATE}*90))") \
        --health-interval-ms 500 \
        --unhealthy-threshold 2 || ir_log "failure trial ${POLICY} rep ${REP} returned non-zero"

    ir_teardown
    sleep 25
  done
done

ir_log "FAILURE MODE EXPERIMENT COMPLETE"
