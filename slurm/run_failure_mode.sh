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
#SBATCH --time=04:00:00
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

PYTHONPATH="${HERE}/driver" "${IR_VENV}/bin/python" -c "import ir_driver.cli as c; c.build_parser().parse_args(['failure','--router','x','--replicas','r0=x','--model-path','x','--out-dir','x','--kill','r0','--kill-command','x','--rate','1','--restart-command','x'])" \
    || { ir_log "FATAL: driver does not import or parse"; exit 1; }

REV="$(python3 -c "import json;print(json.load(open('${IR_LOCKFILE}'))['models']['Qwen_Qwen2_5-7B-Instruct']['revision'])")"
MODEL="$(ir_model_path "Qwen--Qwen2.5-7B-Instruct" "$REV")" || exit 1

OUT="${IR_LOGS}/ir_failure_${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap ir_teardown EXIT

KILL_PORT=$(( 8100 + KILL_IDX ))
KILL_GPU=$(( KILL_IDX % 4 ))
RESTART_AFTER="${IR_RESTART_AFTER_S:-30}"
# Kill by port so the command targets this replica and nothing else that
# happens to match "vllm", then take the whole process tree down: vLLM's
# EngineCore child does not carry --port on its command line, and if it
# survived it would keep the GPU memory and make the restart fail for a reason
# that has nothing to do with routing.
KILL_SCRIPT="${OUT}/kill_r${KILL_IDX}.sh"
cat > "$KILL_SCRIPT" <<EOF
#!/bin/bash
# SIGKILL, not SIGTERM: the point is an abrupt failure. A graceful shutdown
# would let vLLM drain its in-flight requests, which is a much kinder
# experiment than the one intended.
tree() { local p; for p in \$(pgrep -P "\$1"); do tree "\$p"; done; echo "\$1"; }
roots=\$(pgrep -f -- "--port ${KILL_PORT}( |\$)")
[ -z "\$roots" ] && { echo "no process matched port ${KILL_PORT}"; exit 1; }
pids=\$(for r in \$roots; do tree "\$r"; done | sort -u)
echo "killing: \$(echo \$pids)"
kill -9 \$pids 2>/dev/null
exit 0
EOF
chmod +x "$KILL_SCRIPT"

# Relaunch into the same GPU and port once the dead tree has released the
# memory. Waiting on nvidia-smi rather than a fixed sleep: a relaunch that
# races the old allocation fails with "insufficient memory" and would be
# recorded as a recovery failure.
RESTART_SCRIPT="${OUT}/restart_r${KILL_IDX}.sh"
cat > "$RESTART_SCRIPT" <<EOF
#!/bin/bash
type module >/dev/null 2>&1 || source /etc/profile
source "${HERE}/env.sh"
source "${HERE}/slurm/lib_replicas.sh"
module load "${IR_MOD_ENV}" >/dev/null 2>&1
module load "${IR_MOD_APPTAINER}" >/dev/null 2>&1
for i in \$(seq 1 120); do
    used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${KILL_GPU} | tr -d ' ')
    [ "\$used" -lt 2000 ] && break
    sleep 0.5
done
echo "gpu ${KILL_GPU} memory.used=\${used}MiB before relaunch (waited \$i polls)"
ir_launch_replica ${KILL_GPU} ${KILL_PORT} "${MODEL}" qwen2.5-7b-instruct 0.90 256 \\
    "\${RESTART_LOG}" </dev/null
EOF
chmod +x "$RESTART_SCRIPT"

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

    export RESTART_LOG="${OUT}/${POLICY}_rep${REP}_replica${KILL_IDX}_restart.log"
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
        --restart-command "$RESTART_SCRIPT" \
        --restart-after-s "$RESTART_AFTER" \
        --poll-interval-ms 250 \
        `# 30 s before the kill, ~30 s to relaunch, a vLLM cold start, and at` \
        `# least a minute after readmission, all under load: 420 s of arrivals.` \
        --num-requests $(python3 -c "print(int(${RATE}*420))") \
        --health-interval-ms 500 \
        --unhealthy-threshold 2 || ir_log "failure trial ${POLICY} rep ${REP} returned non-zero"

    ir_teardown
    sleep 25
  done
done

ir_log "FAILURE MODE EXPERIMENT COMPLETE"
