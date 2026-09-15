#!/bin/bash
# Gate: one live vLLM replica behind the router, end to end.
#
#   sbatch slurm/verify_single_replica.sh
#
# Nothing expensive runs until this passes. It checks the four things that would
# invalidate every subsequent number if they were silently false:
#
#   1. Token-ID prompts round-trip exactly THROUGH THE ROUTER. P1 verified this
#      against vLLM directly; the router re-serialises the body, so the property
#      has to be re-established across the proxy or the shared-prefix knob is
#      measuring nothing.
#   2. ignore_eos is honoured, so the output-length axis is controlled.
#   3. Responses actually stream. A buffering proxy would collapse TTFT onto
#      end-to-end latency and erase exactly the prefill effect the whole project
#      is built to measure, while producing a complete-looking dataset.
#   4. /reset_prefix_cache exists and the prefix-cache counters are present
#      under the names vLLM 0.27.1 actually uses.
#
#SBATCH --job-name=ir-verify
#SBATCH --account=p201466
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/ir_verify_%j.out

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

export IR_TAG="verify"
module load "$IR_MOD_ENV"
module load "$IR_MOD_APPTAINER"

ir_log "host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

ROUTER_BIN="${CARGO_TARGET_DIR}/release/inference-router"
[[ -x "$ROUTER_BIN" ]] || { ir_log "router binary missing at ${ROUTER_BIN}; run slurm/build.sh first"; exit 1; }

REV="$(python3 -c "import json;print(json.load(open('${IR_LOCKFILE}'))['models']['Qwen_Qwen2_5-7B-Instruct']['revision'])")"
MODEL="$(ir_model_path "Qwen--Qwen2.5-7B-Instruct" "$REV")" || exit 1
ir_log "model=${MODEL} revision=${REV}"

OUT="${IR_LOGS}/ir_verify_${SLURM_JOB_ID}"
mkdir -p "$OUT"

trap ir_teardown EXIT

ir_log "launching 1 replica on GPU0, port 8100"
ir_launch_replica 0 8100 "$MODEL" qwen2.5-7b-instruct 0.90 256 "${OUT}/replica0.log" >/dev/null
ir_wait_healthy 900 8100 || { ir_log "replica never became healthy"; tail -60 "${OUT}/replica0.log"; exit 1; }

CFG="${OUT}/router.toml"
ir_write_router_config "$CFG" "127.0.0.1:8080" "round_robin" 64 "reject" "8100:node=$(hostname),gpu=0"
ir_log "router config:"; sed 's/^/  /' "$CFG"

"$ROUTER_BIN" --config "$CFG" --log-json > "${OUT}/router.log" 2>&1 &
ROUTER_PID=$!
ir_log "router pid ${ROUTER_PID}"

for _ in $(seq 1 60); do
    curl -sf -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break
    sleep 1
done
curl -sf -m 5 http://127.0.0.1:8080/health >/dev/null || { ir_log "router never became healthy"; tail -40 "${OUT}/router.log"; exit 1; }
ir_log "router is up"
curl -s http://127.0.0.1:8080/admin/fleet | python3 -m json.tool | sed 's/^/  /'

ir_log "=== driver verification ==="
PYTHONPATH="${HERE}/driver" "${IR_VENV}/bin/python" -m ir_driver.cli verify \
    --router http://127.0.0.1:8080 \
    --replicas "r0=http://127.0.0.1:8100" \
    --model-path "$MODEL" \
    --served-name qwen2.5-7b-instruct \
    --lockfile "$IR_LOCKFILE" \
    --out-dir "${HERE}/results/raw" \
    --rate 2.0 --num-requests 32
RC=$?

ir_log "=== router metrics after verification ==="
curl -s http://127.0.0.1:8080/metrics | grep -E "^router_(requests|responses|ttft_seconds_count|e2e_seconds_count|sse_chunks|replica_dispatched)" | sed 's/^/  /'

ir_log "=== router structured log (last 20 routed lines) ==="
grep -c routed "${OUT}/router.log" | sed 's/^/  routed lines: /'
grep routed "${OUT}/router.log" | tail -5 | sed 's/^/  /'

if [[ $RC -eq 0 ]]; then
    ir_log "VERIFICATION PASSED"
else
    ir_log "VERIFICATION FAILED (rc=${RC})"
fi
ir_log "artifacts under ${HERE}/results/raw, logs under ${OUT}"
exit $RC
