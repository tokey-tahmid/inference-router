#!/bin/bash
# The experiment matrix: 4 policies x 4 shared-prefix ratios x 4 rates x N reps
# against ONE long-lived 4-replica fleet.
#
#   sbatch slurm/run_matrix.sh                       # full matrix
#   sbatch slurm/run_matrix.sh --rates 4 10 18 28    # override the rate ladder
#
# All arguments pass through to `ir_driver.cli matrix`, so the workload can be
# changed too, e.g. the many-prefix ("tenant mix") variant:
#   sbatch --time=03:30:00 slurm/run_matrix.sh --rates 8 32 48 \
#       --ratios 0.6 0.9 --num-requests 960 --num-prefix-groups 216
#
# The single-fleet design is the central experimental choice here. Each vLLM
# replica takes minutes to stream 15 GB of weights off Lustre, so relaunching
# per policy would cost more node-hours than the measurement and, far worse,
# would turn a comparison of four routing rules into a comparison of four
# separately-warmed fleets. The router's /admin/policy endpoint exists so the
# routing rule is the single independent variable.
#
# Cross-phase contamination is the hazard that design creates, and it is fatal
# rather than cosmetic for this project: a warm prefix cache is exactly what
# prefix affinity is claimed to produce, so a phase inheriting the previous
# phase's cache would post a flattering TTFT that had nothing to do with its own
# routing. Every phase therefore resets every replica's prefix cache, waits for
# the fleet to drain, and records whether both actually happened.
#
#SBATCH --job-name=ir-matrix
#SBATCH --account=p201466
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/ir_matrix_%j.out

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

export IR_TAG="matrix"
module load "$IR_MOD_ENV"
module load "$IR_MOD_APPTAINER"

NREPLICAS="${IR_NREPLICAS:-4}"
RATES_DEFAULT="${IR_RATES:-}"

ir_log "host=$(hostname) job=${SLURM_JOB_ID} replicas=${NREPLICAS}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
ir_log "GPU topology:"; nvidia-smi topo -m 2>/dev/null | head -8 | sed 's/^/  /'

ROUTER_BIN="${CARGO_TARGET_DIR}/release/inference-router"
[[ -x "$ROUTER_BIN" ]] || { ir_log "router binary missing; run slurm/build.sh first"; exit 1; }

REV="$(python3 -c "import json;print(json.load(open('${IR_LOCKFILE}'))['models']['Qwen_Qwen2_5-7B-Instruct']['revision'])")"
MODEL="$(ir_model_path "Qwen--Qwen2.5-7B-Instruct" "$REV")" || exit 1
ir_log "model=${MODEL}"
ir_log "revision=${REV} (copied verbatim from ${IR_LOCKFILE})"

OUT="${IR_LOGS}/ir_matrix_${SLURM_JOB_ID}"
mkdir -p "$OUT"

trap ir_teardown EXIT

# ---- fleet: one replica per GPU, pinned to its NUMA-local cores -------------
declare -a PORTS=() REPL_ARGS=() PLACE_ARGS=() CFG_SPECS=()
NODE="$(hostname)"
for (( i=0; i<NREPLICAS; i++ )); do
    gpu=$(( i % 4 ))
    port=$(( 8100 + i ))
    PORTS+=("$port")
    REPL_ARGS+=("r${i}=http://127.0.0.1:${port}")
    PLACE_ARGS+=("r${i}=node=${NODE},gpu=${gpu}")
    CFG_SPECS+=("${port}:node=${NODE},gpu=${gpu}")
    ir_log "launching replica r${i} on GPU${gpu} port ${port} (cores $(ir_cores_for_gpu "$gpu"))"
    ir_launch_replica "$gpu" "$port" "$MODEL" qwen2.5-7b-instruct 0.90 256 "${OUT}/replica${i}.log" >/dev/null
    # Stagger: N concurrent 15 GB reads off Lustre contend, and the contention
    # would land inside the startup path rather than being amortised.
    sleep 20
done

ir_wait_healthy 1800 "${PORTS[@]}" || {
    ir_log "FLEET FAILED TO START"
    for (( i=0; i<NREPLICAS; i++ )); do
        ir_log "--- replica ${i} tail ---"; tail -40 "${OUT}/replica${i}.log" | sed 's/^/  /'
    done
    exit 1
}
ir_log "fleet of ${NREPLICAS} replicas is healthy"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'

# ---- router ----------------------------------------------------------------
CFG="${OUT}/router.toml"
ir_write_router_config "$CFG" "127.0.0.1:8080" "round_robin" 64 "reject" "${CFG_SPECS[@]}"
ir_log "router config:"; sed 's/^/  /' "$CFG"

"$ROUTER_BIN" --config "$CFG" --log-json > "${OUT}/router.log" 2>&1 &
for _ in $(seq 1 60); do
    curl -sf -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break
    sleep 1
done
curl -sf -m 5 http://127.0.0.1:8080/health >/dev/null || { ir_log "router never came up"; tail -40 "${OUT}/router.log"; exit 1; }
ir_log "router is up"
curl -s http://127.0.0.1:8080/admin/fleet | python3 -m json.tool | sed 's/^/  /'

# ---- budget ----------------------------------------------------------------
# Stop cleanly with time to spare rather than being SIGKILLed mid-write: a
# truncated artifact is worse than a missing one. 25 minutes of slack covers
# fleet teardown and the final artifact flush.
# End time as an ISO timestamp (squeue %e). The earlier form fed squeue's %L
# ("5:59:30") to `date -d "+..."`, which cannot parse it, and a fallback hid
# that; see docs/notes.md, 2026-09-16.
END_ISO="$(squeue -j "${SLURM_JOB_ID}" -h -o %e 2>/dev/null | tr -d '[:space:]')"
if ! END_EPOCH=$(date -d "${END_ISO}" +%s 2>/dev/null) || [[ -z "${END_ISO}" ]]; then
    ir_log "FATAL: cannot determine job end time (squeue %e returned '${END_ISO}')"
    exit 1
fi
BUDGET=$(( END_EPOCH - $(date +%s) - 1500 ))
if (( BUDGET < 600 )); then
    ir_log "FATAL: driver budget would be ${BUDGET}s; not enough for a matrix"
    exit 1
fi
ir_log "driver budget: ${BUDGET}s"

RATE_ARGS=()
if [[ $# -gt 0 ]]; then
    RATE_ARGS=("$@")
elif [[ -n "$RATES_DEFAULT" ]]; then
    # shellcheck disable=SC2206
    RATE_ARGS=(--rates $RATES_DEFAULT)
else
    ir_log "ERROR: no rate ladder given. Run slurm/calibrate_rates.sh first and"
    ir_log "pass the measured saturation ladder; the rate axis must span"
    ir_log "underload to overload for THIS fleet, and that is a measured quantity."
    exit 2
fi

ir_log "=== running matrix ==="
PYTHONPATH="${HERE}/driver" "${IR_VENV}/bin/python" -m ir_driver.cli matrix \
    --router http://127.0.0.1:8080 \
    --replicas "${REPL_ARGS[@]}" \
    --placements "${PLACE_ARGS[@]}" \
    --model-path "$MODEL" \
    --served-name qwen2.5-7b-instruct \
    --lockfile "$IR_LOCKFILE" \
    --out-dir "${HERE}/results/raw" \
    --replicas-per-gpu 1 \
    --gpus-used 4 \
    --nodes-used 1 \
    --topology-note "4x A100-SXM4-40GB on ${NODE}, one TP=1 replica per GPU, each pinned to its NUMA-local cores; full NVLink NV4 mesh; replicas co-located on one node (P1 measured co-location alone costs +150% TTFT p50 / -34% throughput vs isolation, so absolute numbers here are fleet numbers, not isolated-replica numbers)" \
    --budget-s "$BUDGET" \
    "${RATE_ARGS[@]}"
RC=$?

ir_log "=== final router metrics ==="
curl -s http://127.0.0.1:8080/metrics > "${OUT}/router_metrics_final.txt" || true
grep -E "^router_replica_dispatched_total|^router_requests_total|^router_rejected" "${OUT}/router_metrics_final.txt" | sed 's/^/  /'

ir_log "artifacts: $(find "${HERE}/results/raw" -name '*.json' -newer "$CFG" | wc -l) new files"
ir_log "MATRIX COMPLETE (rc=${RC})"
exit $RC
