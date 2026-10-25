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
#SBATCH --partition=gpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --output=logs/ir_full_%j.out

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

export IR_TAG="full"
ir_load_modules

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
# Logic gate: exercise the knee picker on the committed calibration from job
# 5215612 (read-only) and the rate-scaled matrix expansion, so a bug in either
# fails here in seconds rather than after stages 1-4.
"$PY" - "${HERE}/results/calibration" <<'PYEOF' || { ir_log "FATAL: driver logic gate failed"; exit 1; }
import pathlib, sys
from ir_driver.pick_rates import load_calibration, pick, format_table
from ir_driver.experiment import build_matrix
rows = load_calibration(pathlib.Path(sys.argv[1]))
res = pick(rows)
print(format_table(res["table"]))
print("  prior calibration ->", res["rates"], "|", res["reason"])
ph = build_matrix(["round_robin"], [0.6], [4.0, 64.0], 1, warmup=False,
                  num_requests=200, min_duration_s=20)
got = [p.num_requests for p in ph]
assert got == [200, 1280], got
print("  build_matrix request scaling OK:", got)
PYEOF

REV="$("$PY" -c "import json;print(json.load(open('${IR_LOCKFILE}'))['models']['Qwen_Qwen2_5-7B-Instruct']['revision'])")"
MODEL="$(ir_model_path "Qwen--Qwen2.5-7B-Instruct" "$REV")" || exit 1
ir_log "model=${MODEL}"
ir_log "revision=${REV} (copied verbatim from ${IR_LOCKFILE}, never retyped)"

# Job end time, read as an ISO timestamp (squeue %e), which `date -d` parses.
#
# This previously read the time-LEFT field (%L, e.g. "5:59:30") and passed it to
# `date -d "+5:59:30"`. GNU date rejects that ("invalid date"), and a trailing
# `|| echo 0` turned the failure into JOB_END=0, so remaining() was 0 from the
# first second. A 600 s floor on the matrix budget then turned that obviously
# broken 0 into a plausible-looking 10 minutes. Job 5215612 ran 8 of 192 phases
# in a 6 h allocation and reported "FULL EXPERIMENT COMPLETE (rc=0)".
#
# Two silent defaults stacked: either alone would have been caught, together
# they produced a wrong answer that looked like a finished experiment. So this
# now fails loudly instead of defaulting.
END_ISO="$(squeue -j "${JOB}" -h -o %e 2>/dev/null | tr -d '[:space:]')"
if ! JOB_END=$(date -d "${END_ISO}" +%s 2>/dev/null) || [[ -z "${END_ISO}" ]]; then
    ir_log "FATAL: cannot determine job end time (squeue %e returned '${END_ISO}')"
    exit 1
fi
remaining() { local n; n=$(date +%s); echo $(( JOB_END > n ? JOB_END - n : 0 )); }
ir_log "job ends at ${END_ISO}; wall clock remaining: $(remaining)s"
if (( $(remaining) < 1800 )); then
    ir_log "FATAL: only $(remaining)s of wall clock at start; refusing to run a truncated experiment"
    exit 1
fi

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
# Tear the stage-1 replica down FIRST. It was started at gpu_memory_utilization
# 0.90, and density probe 5213921 measured it occupying 36,126 MiB of the card's
# 40 GB. Leaving it up and starting a "second" replica alongside it would not be
# a two-replicas-per-card test at all: there is no room for a 15.2 GB weight load
# in the ~4 GB left, so the answer would be a foregone "no" that says nothing
# about whether the topology is viable.
#
# The honest test is two replicas that each ask for HALF the card.
ir_teardown
sleep 20

# SEQUENTIAL, unlike the first attempt (job 5213921), which launched both
# replicas 30 s apart against a ~200 s weight load and had one die with
# "No available memory for the cache blocks". vLLM sizes its KV cache from
# memory that is free at init, so two concurrent starts profile against each
# other and the loser sees a card that is already nearly full. Waiting for the
# first to be fully healthy separates "they contend while starting" from "a
# second one genuinely does not fit", which are different findings.
DENSITY_RESULT="not measured"
DENSITY_DETAIL=""
for UTIL in 0.45 0.42; do
    ir_log "--- two replicas on GPU0, gpu_memory_utilization=${UTIL} EACH, launched sequentially ---"
    ir_teardown; sleep 15
    ir_launch_replica 0 8100 "$MODEL" qwen2.5-7b-instruct "$UTIL" 128 "${OUT}/s2_a_u${UTIL}.log" >/dev/null
    if ! ir_wait_healthy 900 8100; then
        ir_log "  first replica at util=${UTIL} did not come up; that is the finding"
        grep -iE "ValueError|No available memory|out of memory" "${OUT}/s2_a_u${UTIL}.log" | tail -2 | sed 's/^/    /'
        DENSITY_RESULT="first_replica_failed_at_util_${UTIL}"
        continue
    fi
    ir_log "  first replica healthy at util=${UTIL}; now starting the second"
    ir_launch_replica 0 8101 "$MODEL" qwen2.5-7b-instruct "$UTIL" 128 "${OUT}/s2_b_u${UTIL}.log" >/dev/null
    if ir_wait_healthy 900 8101; then
        ir_log "  TWO replicas are live on one A100-40GB at util=${UTIL} each"
        # A replica that loads but holds three sequences is not a usable replica,
        # so record the KV cache size, not merely that the process started.
        for f in a b; do
            grep -iE "GPU KV cache size|Maximum concurrency|Available KV cache memory" \
                "${OUT}/s2_${f}_u${UTIL}.log" | sed "s/^/    ${f}: /"
        done
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1 | sed 's/^/    GPU0 mem: /'
        DENSITY_RESULT="two_replicas_fit_at_util_${UTIL}_each"
        DENSITY_DETAIL="$(grep -ihE 'GPU KV cache size|Maximum concurrency' "${OUT}/s2_a_u${UTIL}.log" "${OUT}/s2_b_u${UTIL}.log" | tr '\n' ';')"
        break
    fi
    ir_log "  second replica at util=${UTIL} did NOT become healthy"
    grep -iE "ValueError|No available memory|out of memory|RuntimeError" \
        "${OUT}/s2_b_u${UTIL}.log" | tail -3 | sed 's/^/    /'
    DENSITY_RESULT="two_replicas_do_not_fit_at_util_${UTIL}_each"
    DENSITY_DETAIL="$(grep -ihE 'No available memory|out of memory' "${OUT}/s2_b_u${UTIL}.log" | tail -1)"
done
ir_log "DENSITY FINDING: ${DENSITY_RESULT}"

# Recorded as a raw artifact so the topology decision is traceable, not a log line.
"$PY" - "$RAW" "$JOB" "$NODE" "$DENSITY_RESULT" "$DENSITY_DETAIL" "$HERE" <<'DENSITY_PY'
import json, os, sys, subprocess, datetime, pathlib
raw, job, node, result, detail, repo = sys.argv[1:7]
def sh(c):
    try:
        return subprocess.run(c, cwd=repo, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None
sha = sh(["git", "rev-parse", "HEAD"])
if sha and sh(["git", "status", "--porcelain"]):
    sha += "-dirty"
lock = json.load(open(os.environ["IR_LOCKFILE"]))
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
    "full_config_dict": {
        "method": "both replicas on GPU0 at the SAME gpu_memory_utilization, launched "
                  "sequentially: the first must be fully healthy before the second starts, "
                  "so the two do not profile free memory against each other",
        "card": "A100-SXM4-40GB", "model_dtype": "bfloat16",
        "utils_tried_per_replica": [0.45, 0.42], "max_model_len": 4096,
        "baseline_single_replica": "job 5213921 arm A: util 0.90 -> 19.09 GiB KV cache, "
                                   "357,520 tokens, 87.29x max concurrency at 4096 tok/req, "
                                   "36,126 MiB of 40 GB used",
    },
}
p = pathlib.Path(raw) / f"density_probe_{job}.json"
p.write_text(json.dumps(out, indent=2, sort_keys=True))
print(f"  wrote {p}")
DENSITY_PY

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

TOPO="4x A100-SXM4-40GB on ${NODE}, one TP=1 vLLM replica per GPU at gpu_memory_utilization=0.90, each pinned to its NUMA-local cores; full NVLink NV4 mesh; single node. Replicas are co-located by design."

############################################################ stage 4: calibrate
ir_log "=============== STAGE 4: rate calibration against THIS fleet ==============="
# Job 5215612 ran 2..32 and never saturated the fleet (median TTFT flat at
# 0.07-0.10 s, zero errors), so the ladder now reaches well past that.
CAL_LADDER="${IR_LADDER:-4 8 16 32 48 64 96 128}"
# Per-job directory: pick_rates reads every artifact in the directory it is
# given, and mixing an earlier job's calibration into this one's knee search
# would locate a knee against a different fleet.
CAL_DIR="${HERE}/results/calibration/${SLURM_JOB_ID}"
mkdir -p "$CAL_DIR"
ir_log "ladder: ${CAL_LADDER}"
# N=1, no warmup, short phases: this locates a knee, it does not produce
# headline numbers. Artifacts are still written with repetition/warmup marked
# honestly so the calibration is auditable rather than a number in a log.
"$PY" -m ir_driver.cli matrix \
    --router http://127.0.0.1:8080 \
    --replicas "${REPL_ARGS[@]}" --placements "${PLACE_ARGS[@]}" \
    --model-path "$MODEL" --served-name qwen2.5-7b-instruct \
    --lockfile "$IR_LOCKFILE" --out-dir "$CAL_DIR" \
    --policies round_robin --ratios 0.6 --rates $CAL_LADDER \
    --repetitions 1 --no-warmup --num-requests 120 --min-duration-s 30 \
    --replicas-per-gpu 1 --gpus-used 4 --nodes-used 1 \
    --topology-note "RATE CALIBRATION (N=1, not a headline measurement). ${TOPO}" \
    --budget-s 3000
ir_log "calibration rc=$?"

RATES="$("$PY" - "$CAL_DIR" <<'PYEOF'
import os, sys, pathlib, json
sys.path.insert(0, os.path.join(os.environ["IR_REPO"], "driver"))
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
# No silent floor. A small or negative budget is a real problem to surface, not
# a number to round up into something that looks reasonable (see the note on
# JOB_END above for what that rounding once hid).
if (( BUDGET < 600 )); then
    ir_log "FATAL: matrix budget would be ${BUDGET}s after stages 1-4; not enough to run a meaningful matrix"
    exit 1
fi
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
    --num-requests 200 --min-duration-s 20 --input-tokens 1024 --output-tokens 128 \
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
