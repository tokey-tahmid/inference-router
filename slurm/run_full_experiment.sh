#!/bin/bash
# The whole experiment in one node allocation, staged so each stage gates the
# next and every stage's output survives an abort.
#
#   sbatch slurm/run_full_experiment.sh
#
#   stage 1  verify SSE streaming is non-buffering against ONE live replica
#   stage 2  measure replica density: does a 2nd 7B fit on the same 40 GB card
#   stage 3  bring up 4 replicas, one per GPU
#   stage 4  calibrate the rate ladder against THIS fleet
#   stage 5  run the matrix with whatever budget remains
#
# One job rather than five because each stage needs the same 4-GPU node and the
# queue wait dominates the work. It also means the matrix runs against a fleet
# whose streaming behaviour and saturation point were measured minutes earlier
# on the same hardware, rather than inherited from a different allocation.
#
# Stage 1 is a hard gate. A buffering proxy would collapse TTFT onto end-to-end
# latency and erase the prefill effect the entire project exists to measure,
# while still producing a complete and plausible dataset. There is no point
# spending node-hours on the matrix if that is broken.
#
#SBATCH --job-name=ir-full
#SBATCH --account=p201466
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/ir_full_%j.out

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

export IR_TAG="full"
module load "$IR_MOD_ENV"
module load "$IR_MOD_APPTAINER"

JOB="${SLURM_JOB_ID}"
OUT="${IR_LOGS}/ir_full_${JOB}"
RAW="${HERE}/results/raw"
mkdir -p "$OUT" "$RAW"
NODE="$(hostname)"
PY="${IR_VENV}/bin/python"
export PYTHONPATH="${HERE}/driver"

ir_log "host=${NODE} job=${JOB}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
ir_log "GPU topology (first rows):"; nvidia-smi topo -m 2>/dev/null | head -7 | sed 's/^/  /'

ROUTER_BIN="${CARGO_TARGET_DIR}/release/inference-router"
[[ -x "$ROUTER_BIN" ]] || { ir_log "FATAL: router binary missing at ${ROUTER_BIN}"; exit 1; }

# Syntax gate, first thing. The login node's Python is 3.6 and cannot even parse
# this code, so a batch job is the first opportunity to find a typo. Without this
# the first syntax error would surface in stage 1, after a ~200 s replica cold
# start; here it costs seconds. (P1 learned the same lesson and opens its probe
# the same way.)
ir_log "=== python syntax gate ==="
"$PY" -m compileall -q "${HERE}/driver" || { ir_log "FATAL: driver does not compile"; exit 1; }
"$PY" -c "import ir_driver.cli, ir_driver.experiment, ir_driver.loadgen, ir_driver.pick_rates, ir_driver.stats, ir_driver.metrics_scrape, ir_driver.provenance; import llm_inference_bench.workload as w; print('  imports OK, P1 workload generator:', w.__file__)" \
    || { ir_log "FATAL: driver imports failed"; exit 1; }

REV="$("$PY" -c "import json;print(json.load(open('${IR_LOCKFILE}'))['models']['Qwen_Qwen2_5-7B-Instruct']['revision'])")"
MODEL="$(ir_model_path "Qwen--Qwen2.5-7B-Instruct" "$REV")" || exit 1
ir_log "model=${MODEL}"
ir_log "revision=${REV} (copied verbatim from ${IR_LOCKFILE}, never retyped)"

JOB_END=$(date -d "+$(squeue -j "${JOB}" -h -o %L)" +%s 2>/dev/null || echo 0)
remaining() { local n; n=$(date +%s); echo $(( JOB_END > n ? JOB_END - n : 0 )); }
ir_log "wall clock remaining: $(remaining)s"

trap ir_teardown EXIT

start_router() {  # start_router <policy> <tag> <cfgspec...>
    local policy="$1" tag="$2"; shift 2
    local cfg="${OUT}/router_${tag}.toml"
    ir_write_router_config "$cfg" "127.0.0.1:8080" "$policy" 64 "reject" "$@"
    "$ROUTER_BIN" --config "$cfg" --log-json > "${OUT}/router_${tag}.log" 2>&1 &
    for _ in $(seq 1 90); do
        curl -sf -m 2 http://127.0.0.1:8080/health >/dev/null 2>&1 && break
        sleep 1
    done
    curl -sf -m 5 http://127.0.0.1:8080/health >/dev/null || {
        ir_log "router did not come up (${tag})"; tail -40 "${OUT}/router_${tag}.log"; return 1; }
    ir_log "router up (${tag}, policy=${policy})"
}

############################################################ stage 1: SSE gate
ir_log "=============== STAGE 1: SSE non-buffering gate, 1 live replica ==============="
ir_launch_replica 0 8100 "$MODEL" qwen2.5-7b-instruct 0.90 256 "${OUT}/s1_replica0.log" >/dev/null
if ! ir_wait_healthy 1200 8100; then
    ir_log "FATAL: single replica never became healthy"
    tail -60 "${OUT}/s1_replica0.log" | sed 's/^/  /'
    exit 1
fi
grep -iE "GPU KV cache size|Maximum concurrency|Available KV cache memory" "${OUT}/s1_replica0.log" | sed 's/^/  /'

start_router round_robin s1 "8100:node=${NODE},gpu=0" || exit 1

"$PY" -m ir_driver.cli verify \
    --router http://127.0.0.1:8080 \
    --replicas "r0=http://127.0.0.1:8100" \
    --model-path "$MODEL" --served-name qwen2.5-7b-instruct \
    --lockfile "$IR_LOCKFILE" --out-dir "$RAW" \
    --rate 2.0 --num-requests 32
S1=$?
ir_log "stage 1 rc=${S1}"
if [[ $S1 -ne 0 ]]; then
    ir_log "FATAL: SSE / prompt-fidelity gate FAILED. Not spending node-hours on the"
    ir_log "matrix: a buffering proxy produces a complete-looking but meaningless dataset."
    exit 1
fi
ir_log "STAGE 1 PASSED"

############################################################ stage 2: density
ir_log "=============== STAGE 2: replica density on one 40 GB card ==============="
# SEQUENTIAL, unlike the first attempt (job 5213921), which launched both
# replicas 30 s apart and had one die with "No available memory for the cache
# blocks". vLLM sizes its KV cache from *currently free* memory at init, so two
# concurrent starts profile against each other and the loser sees a card that is
# already full. Waiting for the first to be fully healthy removes that race and
# separates "they contend while starting" from "a second one genuinely does not
# fit", which are different findings.
pkill -f "inference-router" 2>/dev/null || true
sleep 3
DENSITY_RESULT="not measured"
DENSITY_DETAIL=""
for UTIL in 0.45 0.40; do
    ir_log "--- trying a 2nd replica on GPU0 at gpu_memory_utilization=${UTIL} ---"
    ir_launch_replica 0 8101 "$MODEL" qwen2.5-7b-instruct "$UTIL" 128 "${OUT}/s2_replica1_u${UTIL}.log" >/dev/null
    if ir_wait_healthy 900 8101; then
        ir_log "TWO replicas are live on GPU0 (first at util 0.90, second at ${UTIL})"
        # A replica that loads but has room for 3 sequences is not a usable
        # replica, so record the KV cache size, not just that it started.
        grep -iE "GPU KV cache size|Maximum concurrency|Available KV cache memory" \
            "${OUT}/s2_replica1_u${UTIL}.log" | sed 's/^/  /'
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1 | sed 's/^/  GPU0 mem: /'
        DENSITY_RESULT="two_replicas_fit_at_util_${UTIL}_with_first_at_0.90"
        DENSITY_DETAIL="$(grep -iE 'GPU KV cache size|Maximum concurrency' "${OUT}/s2_replica1_u${UTIL}.log" | tr '\n' ';')"
        pkill -f "port 8101" 2>/dev/null || true
        sleep 10
        break
    fi
    ir_log "second replica at util=${UTIL} did NOT become healthy"
    grep -iE "ValueError|No available memory|out of memory|RuntimeError" \
        "${OUT}/s2_replica1_u${UTIL}.log" | tail -3 | sed 's/^/  /'
    DENSITY_RESULT="two_replicas_do_not_fit"
    DENSITY_DETAIL="$(grep -iE 'No available memory|out of memory' "${OUT}/s2_replica1_u${UTIL}.log" | tail -1)"
    pkill -f "port 8101" 2>/dev/null || true
    sleep 10
done
ir_log "DENSITY FINDING: ${DENSITY_RESULT}"

# Recorded as a raw artifact so the topology decision is traceable, not a log line.
"$PY" - "$RAW" "$JOB" "$NODE" "$DENSITY_RESULT" "$DENSITY_DETAIL" "$HERE" <<'PYEOF'
import json, sys, subprocess, datetime, pathlib
raw, job, node, result, detail, repo = sys.argv[1:7]
def sh(c):
    try: return subprocess.run(c, cwd=repo, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception: return None
sha = sh(["git","rev-parse","HEAD"])
if sha and sh(["git","status","--porcelain"]): sha += "-dirty"
lock = json.load(open("/home/users/u104240/apps/inference-work/llm-inference-bench/provision.lock.json"))
img, mdl = lock["images"]["vllm"], lock["models"]["Qwen_Qwen2_5-7B-Instruct"]
out = {
  "artifact_schema_version": 1, "kind": "replica_density_probe", "status": "ok",
  "run_id": f"density-{job}", "slurm_job_id": job, "hostname": node,
  "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
  "git_sha": sha, "repetition_index": 1, "warmup": False,
  "backend_name": "vllm", "backend_version": img["version"],
  "backend_image_digest": img["digest"],
  "model_id": mdl["repo"], "model_revision": mdl["revision"],
  "finding": result, "detail": detail,
  "full_config_dict": {"method": "sequential launch: first replica fully healthy at "
                       "gpu_memory_utilization=0.90 before the second is started, so the "
                       "two do not profile free memory against each other",
                       "card": "A100-SXM4-40GB", "model_dtype": "bfloat16",
                       "utils_tried": [0.45, 0.40], "max_model_len": 4096},
}
p = pathlib.Path(raw) / f"density_probe_{job}.json"
p.write_text(json.dumps(out, indent=2, sort_keys=True))
print(f"  wrote {p}")
PYEOF

############################################################ stage 3: 4 replicas
ir_log "=============== STAGE 3: 4 replicas, one per GPU ==============="
ir_teardown
sleep 20

declare -a PORTS=() REPL_ARGS=() PLACE_ARGS=() CFG_SPECS=()
for i in 0 1 2 3; do
    port=$(( 8100 + i ))
    PORTS+=("$port")
    REPL_ARGS+=("r${i}=http://127.0.0.1:${port}")
    PLACE_ARGS+=("r${i}=node=${NODE},gpu=${i}")
    CFG_SPECS+=("${port}:node=${NODE},gpu=${i}")
    ir_log "launching r${i} on GPU${i} port ${port} (NUMA-local cores $(ir_cores_for_gpu "$i"))"
    ir_launch_replica "$i" "$port" "$MODEL" qwen2.5-7b-instruct 0.90 256 "${OUT}/s3_replica${i}.log" >/dev/null
    sleep 20
done
if ! ir_wait_healthy 1800 "${PORTS[@]}"; then
    ir_log "FATAL: 4-replica fleet failed to start"
    for i in 0 1 2 3; do tail -30 "${OUT}/s3_replica${i}.log" | sed "s/^/  r${i}: /"; done
    exit 1
fi
ir_log "4-replica fleet healthy"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'

start_router round_robin s3 "${CFG_SPECS[@]}" || exit 1
curl -s http://127.0.0.1:8080/admin/fleet | "$PY" -m json.tool | sed 's/^/  /'

TOPO="4x A100-SXM4-40GB on ${NODE}, one TP=1 vLLM replica per GPU at gpu_memory_utilization=0.90, each pinned to its NUMA-local cores; full NVLink NV4 mesh; single node. Replicas are co-located by design: P1 job 5162712 measured co-location alone costing +150.7% TTFT p50 and -34.1% throughput vs an isolated replica, so absolute latencies here are fleet numbers, not isolated-replica numbers."

############################################################ stage 4: calibrate
ir_log "=============== STAGE 4: rate calibration against THIS fleet ==============="
CAL_LADDER="${IR_LADDER:-2 4 8 12 16 24 32}"
ir_log "ladder: ${CAL_LADDER}"
# N=1, no warmup, short phases: this locates a knee, it does not produce
# headline numbers. Artifacts are still written with repetition/warmup marked
# honestly so the calibration is auditable rather than a number in a log.
"$PY" -m ir_driver.cli matrix \
    --router http://127.0.0.1:8080 \
    --replicas "${REPL_ARGS[@]}" --placements "${PLACE_ARGS[@]}" \
    --model-path "$MODEL" --served-name qwen2.5-7b-instruct \
    --lockfile "$IR_LOCKFILE" --out-dir "${HERE}/results/calibration" \
    --policies round_robin --ratios 0.6 --rates $CAL_LADDER \
    --repetitions 1 --no-warmup --num-requests 120 \
    --replicas-per-gpu 1 --gpus-used 4 --nodes-used 1 \
    --topology-note "RATE CALIBRATION (N=1, not a headline measurement). ${TOPO}" \
    --budget-s 2400
ir_log "calibration rc=$?"

RATES="$("$PY" - "${HERE}/results/calibration" <<'PYEOF'
import sys, pathlib, json
sys.path.insert(0, "/home/users/u104240/apps/inference-work/inference-router/driver")
from ir_driver.pick_rates import load_calibration, pick, format_table
rows = load_calibration(pathlib.Path(sys.argv[1]))
res = pick(rows)
print(format_table(rows), file=sys.stderr)
print(f"REASON: {res['reason']}", file=sys.stderr)
print(" ".join(f"{r:g}" for r in res["rates"]))
PYEOF
)"
ir_log "selected rate ladder: ${RATES}"
if [[ -z "${RATES// }" ]]; then
    ir_log "FATAL: rate calibration produced no ladder; refusing to guess one"
    exit 1
fi

############################################################ stage 5: matrix
ir_log "=============== STAGE 5: experiment matrix ==============="
BUDGET=$(( $(remaining) - 900 ))
(( BUDGET < 600 )) && BUDGET=600
ir_log "matrix budget: ${BUDGET}s"

# Policy-vs-ratio is the headline interaction, so all 4 policies x all 4 ratios
# are kept and the rate axis is what gives if time runs short. The driver stops
# cleanly at the budget rather than being SIGKILLed mid-write.
"$PY" -m ir_driver.cli matrix \
    --router http://127.0.0.1:8080 \
    --replicas "${REPL_ARGS[@]}" --placements "${PLACE_ARGS[@]}" \
    --model-path "$MODEL" --served-name qwen2.5-7b-instruct \
    --lockfile "$IR_LOCKFILE" --out-dir "$RAW" \
    --policies round_robin least_loaded prefix_affinity prefix_affinity_bounded \
    --ratios 0.0 0.3 0.6 0.9 \
    --rates $RATES \
    --repetitions 3 \
    --num-requests 200 --input-tokens 1024 --output-tokens 128 \
    --shared-prefix-tokens 512 --num-prefix-groups 1 \
    --replicas-per-gpu 1 --gpus-used 4 --nodes-used 1 \
    --topology-note "$TOPO" \
    --budget-s "$BUDGET"
RC=$?

ir_log "=============== done ==============="
curl -s http://127.0.0.1:8080/metrics > "${OUT}/router_metrics_final.txt" || true
grep -E "^router_replica_dispatched_total|^router_requests_total|^router_rejected|^router_sse_chunks" \
    "${OUT}/router_metrics_final.txt" | sed 's/^/  /'
ir_log "artifacts in results/raw: $(ls -1 "$RAW" | wc -l)"
ir_log "DENSITY FINDING WAS: ${DENSITY_RESULT}"
ir_log "RATE LADDER WAS: ${RATES}"
ir_log "FULL EXPERIMENT COMPLETE (rc=${RC})"
exit $RC
