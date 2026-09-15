#!/bin/bash
# Measure where this 4-replica fleet saturates, so the matrix's rate axis spans
# underload to overload for real rather than by guess.
#
#   sbatch slurm/calibrate_rates.sh
#
# The matrix asks for "4 request-rate points spanning underload to overload".
# That is only meaningful relative to a measured saturation point. An arithmetic
# guess would be wrong here in a specific and knowable way: P1 measured that
# co-locating four single-GPU replicas costs -34% throughput versus isolation,
# so a ladder derived from isolated-replica throughput would put every "overload"
# point well inside the stable region and the matrix would measure four policies
# that never contend.
#
# Method: a single rate ladder under round_robin at a middling sharing ratio,
# short phases, N=1. We are locating a knee, not measuring a latency.
#
#SBATCH --job-name=ir-calib
#SBATCH --account=p201466
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/ir_calib_%j.out

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

export IR_TAG="calib"
module load "$IR_MOD_ENV"
module load "$IR_MOD_APPTAINER"

NREPLICAS="${IR_NREPLICAS:-4}"
LADDER="${IR_LADDER:-1 2 4 8 12 16 24 32 48}"

ir_log "host=$(hostname) job=${SLURM_JOB_ID} replicas=${NREPLICAS} ladder=${LADDER}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

ROUTER_BIN="${CARGO_TARGET_DIR}/release/inference-router"
[[ -x "$ROUTER_BIN" ]] || { ir_log "router binary missing; run slurm/build.sh first"; exit 1; }

REV="$(python3 -c "import json;print(json.load(open('${IR_LOCKFILE}'))['models']['Qwen_Qwen2_5-7B-Instruct']['revision'])")"
MODEL="$(ir_model_path "Qwen--Qwen2.5-7B-Instruct" "$REV")" || exit 1
ir_log "model=${MODEL} revision=${REV}"

OUT="${IR_LOGS}/ir_calib_${SLURM_JOB_ID}"
mkdir -p "$OUT"
trap ir_teardown EXIT

declare -a PORTS=() REPL_ARGS=() PLACE_ARGS=() CFG_SPECS=()
NODE="$(hostname)"
for (( i=0; i<NREPLICAS; i++ )); do
    gpu=$(( i % 4 )); port=$(( 8100 + i ))
    PORTS+=("$port")
    REPL_ARGS+=("r${i}=http://127.0.0.1:${port}")
    PLACE_ARGS+=("r${i}=node=${NODE},gpu=${gpu}")
    CFG_SPECS+=("${port}:node=${NODE},gpu=${gpu}")
    ir_log "launching r${i} on GPU${gpu} port ${port}"
    ir_launch_replica "$gpu" "$port" "$MODEL" qwen2.5-7b-instruct 0.90 256 "${OUT}/replica${i}.log" >/dev/null
    sleep 20
done

ir_wait_healthy 1800 "${PORTS[@]}" || {
    ir_log "FLEET FAILED TO START"
    for (( i=0; i<NREPLICAS; i++ )); do tail -40 "${OUT}/replica${i}.log" | sed "s/^/  r${i}: /"; done
    exit 1
}
ir_log "fleet healthy"

CFG="${OUT}/router.toml"
ir_write_router_config "$CFG" "127.0.0.1:8080" "round_robin" 64 "reject" "${CFG_SPECS[@]}"
"$ROUTER_BIN" --config "$CFG" --log-json > "${OUT}/router.log" 2>&1 &
for _ in $(seq 1 60); do curl -sf -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break; sleep 1; done
curl -sf -m 5 http://127.0.0.1:8080/health >/dev/null || { ir_log "router never came up"; tail -40 "${OUT}/router.log"; exit 1; }
ir_log "router up"

# N=1 and no warmup: this locates a knee, it does not produce headline numbers.
# The artifacts are still written to results/raw/ with warmup/repetition marked
# honestly, so the calibration is auditable rather than a number in a log.
ir_log "=== rate ladder ==="
PYTHONPATH="${HERE}/driver" "${IR_VENV}/bin/python" -m ir_driver.cli matrix \
    --router http://127.0.0.1:8080 \
    --replicas "${REPL_ARGS[@]}" \
    --placements "${PLACE_ARGS[@]}" \
    --model-path "$MODEL" \
    --served-name qwen2.5-7b-instruct \
    --lockfile "$IR_LOCKFILE" \
    --out-dir "${HERE}/results/raw" \
    --policies round_robin \
    --ratios 0.6 \
    --rates $LADDER \
    --repetitions 1 \
    --no-warmup \
    --num-requests 150 \
    --replicas-per-gpu 1 --gpus-used 4 --nodes-used 1 \
    --topology-note "rate calibration ladder, ${NREPLICAS} replicas on ${NODE}, 1 per GPU" \
    --budget-s 4200
RC=$?

ir_log "=== achieved vs offered (the knee is where these diverge) ==="
"${IR_VENV}/bin/python" - "$HERE" <<'PYEOF'
import json, sys, pathlib
raw = pathlib.Path(sys.argv[1]) / "results" / "raw"
rows = []
for p in sorted(raw.glob("*round_robin__ratio0.6*.json")):
    o = json.loads(p.read_text())
    s = o.get("summary", {})
    ph = o.get("full_config_dict", {}).get("phase", {})
    if s.get("achieved_request_rate_per_s") is None:
        continue
    rows.append((
        ph.get("request_rate"), s["achieved_request_rate_per_s"],
        (s.get("ttft_s") or {}).get("p50"), (s.get("ttft_s") or {}).get("p95"),
        s.get("error_rate"), s.get("output_tokens_per_s"), o.get("status"),
    ))
rows.sort(key=lambda r: r[0] or 0)
print(f"{'offered':>8} {'achieved':>9} {'ttft p50':>9} {'ttft p95':>9} {'err':>6} {'tok/s':>8}  status")
for off, ach, t50, t95, err, tok, st in rows:
    f = lambda v, n=3: "    n/a" if v is None else f"{v:.{n}f}"
    print(f"{off:>8.1f} {ach:>9.2f} {f(t50):>9} {f(t95):>9} {f(err):>6} {f(tok,1):>8}  {st}")
print()
print("Pick 4 rates for the matrix: one clearly underloaded (achieved ~= offered,")
print("flat TTFT), one approaching the knee, one at it, one clearly past it")
print("(achieved < offered and/or a non-zero error rate).")
PYEOF

ir_log "CALIBRATION COMPLETE (rc=${RC})"
exit $RC
