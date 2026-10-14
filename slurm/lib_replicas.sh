#!/bin/bash
# Shared helpers for launching a fleet of vLLM replicas and the router in front
# of them. Sourced by the experiment batch scripts; not executable on its own.
#
# Topology facts this encodes (measured, see the workspace spec, not re-derived here):
#   - 4x A100-SXM4-40GB per node, full NVLink NV4 mesh between every pair.
#   - NUMA-local core sets per GPU:
#       GPU0 -> 24-31,88-95   (NUMA 3)     GPU1 -> 8-15,72-79    (NUMA 1)
#       GPU2 -> 56-63,120-127 (NUMA 7)     GPU3 -> 40-47,104-111 (NUMA 5)
#   - Nodes are diskless: /tmp is tmpfs and dies with the job, /dev/shm is 428 GB.
#   - Lustre is READ-ONLY inside Apptainer unless bound :rw, and torch creates
#     its inductor cache at *import* time, so an unbound cache path makes
#     `import vllm` die with Errno 30 before argparse runs.

# NUMA-local logical cores per GPU index.
IR_CORES_GPU0="24-31,88-95"
IR_CORES_GPU1="8-15,72-79"
IR_CORES_GPU2="56-63,120-127"
IR_CORES_GPU3="40-47,104-111"

ir_cores_for_gpu() {
    case "$1" in
        0) printf '%s' "$IR_CORES_GPU0" ;;
        1) printf '%s' "$IR_CORES_GPU1" ;;
        2) printf '%s' "$IR_CORES_GPU2" ;;
        3) printf '%s' "$IR_CORES_GPU3" ;;
        *) return 1 ;;
    esac
}

ir_log() { printf '[%s %s] %s\n' "${IR_TAG:-fleet}" "$(date -u +%H:%M:%S)" "$*"; }

# Resolve the model snapshot directory from the HF cache, by revision.
#
# The revision is COPIED out of P1's provision.lock.json, never typed: an agent
# previously fabricated a revision SHA here by autocompleting a truncated hash.
ir_model_path() {
    local repo_us="$1" rev="$2"
    local p="${HF_HUB_CACHE}/models--${repo_us}/snapshots/${rev}"
    [[ -d "$p" ]] || { echo "model snapshot not found: $p" >&2; return 1; }
    printf '%s' "$p"
}

# Launch one vLLM replica.
#   ir_launch_replica <gpu_index> <port> <model_path> <served_name> <gpu_mem_util> <max_num_seqs> <logfile>
ir_launch_replica() {
    local gpu="$1" port="$2" model="$3" served="$4" util="$5" maxseqs="$6" logf="$7"
    local cores
    cores="$(ir_cores_for_gpu "$gpu")" || { echo "no core map for gpu $gpu" >&2; return 1; }

    mkdir -p "$(dirname "$logf")"

    # taskset pins the replica to its NUMA-local cores. This matters here more
    # than in a single-replica benchmark: four co-located replicas that float
    # across sockets contend for remote memory bandwidth on the prefill path,
    # and P1 measured that co-location alone already costs +150% TTFT p50 and
    # -34% throughput versus isolation. Pinning does not remove that cost, it
    # keeps it from varying run to run.
    taskset -c "$cores" \
    apptainer exec --nv --cleanenv \
        --bind "${IR_DATA}:${IR_DATA}:rw" \
        --env "HF_HOME=${HF_HOME}" \
        --env "HF_HUB_CACHE=${HF_HUB_CACHE}" \
        --env "HF_HUB_OFFLINE=1" \
        --env "TRANSFORMERS_OFFLINE=1" \
        --env "CUDA_VISIBLE_DEVICES=${gpu}" \
        --env "TMPDIR=/tmp" \
        --env "VLLM_CACHE_ROOT=/tmp/vllm_cache_${port}" \
        --env "TRITON_CACHE_DIR=/tmp/triton_${port}" \
        --env "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${port}" \
        --env "TORCH_HOME=/tmp/torch_${port}" \
        --env "XDG_CACHE_HOME=/tmp/xdg_${port}" \
        --env "OUTLINES_CACHE_DIR=/tmp/outlines_${port}" \
        `# /reset_prefix_cache is registered only under this flag in 0.27.1.` \
        `# Without it the endpoint 404s and every phase inherits the previous` \
        `# phase's warm cache -- fatal for a prefix-affinity experiment.` \
        --env "VLLM_SERVER_DEV_MODE=1" \
        "$IR_SIF_VLLM" \
        vllm serve "$model" \
            --served-model-name "$served" \
            --host 127.0.0.1 \
            --port "$port" \
            --tensor-parallel-size 1 \
            --dtype bfloat16 \
            --gpu-memory-utilization "$util" \
            --max-num-seqs "$maxseqs" \
            --max-model-len 4096 \
            --seed 1234 \
            `# V1 defaults prefix caching ON, but state it explicitly: the whole` \
            `# experiment is about exploiting it, so it must not depend on a default.` \
            --enable-prefix-caching \
            --no-enable-log-requests \
        > "$logf" 2>&1 &
    echo $!
}

# Wait for a set of "port:logfile" pairs to report healthy.
#   ir_wait_healthy <timeout_s> <port1> <port2> ...
ir_wait_healthy() {
    local timeout="$1"; shift
    local deadline=$(( SECONDS + timeout ))
    local pending=("$@")
    while (( SECONDS < deadline )); do
        local still=()
        for p in "${pending[@]}"; do
            if curl -sf -m 3 "http://127.0.0.1:${p}/health" >/dev/null 2>&1; then
                ir_log "replica on port ${p} is healthy after $(( SECONDS ))s"
            else
                still+=("$p")
            fi
        done
        pending=("${still[@]}")
        (( ${#pending[@]} == 0 )) && return 0
        sleep 5
    done
    ir_log "TIMEOUT: ports still unhealthy after ${timeout}s: ${pending[*]}"
    return 1
}

# Emit a router TOML config for a set of replicas.
#   ir_write_router_config <outfile> <listen> <policy> <max_inflight_per_replica> <overload> <port:placement> ...
ir_write_router_config() {
    local out="$1" listen="$2" policy="$3" maxinflight="$4" overload="$5"; shift 5
    {
        echo "# Generated by slurm/lib_replicas.sh -- do not edit by hand."
        echo "listen = \"${listen}\""
        echo
        echo "[policy]"
        echo "kind = \"${policy}\""
        # Must match the workload's shared_prefix_tokens. If the router hashed a
        # longer window than the workload actually shares, two requests with a
        # genuinely shared system prompt would hash differently and affinity
        # would silently degenerate to random placement.
        echo "prefix_hash_tokens = 512"
        echo "bounded_overload_factor = 1.25"
        echo "bounded_min_capacity = 4"
        echo
        echo "[health]"
        echo "enabled = true"
        echo "interval_ms = 500"
        echo "timeout_ms = 1000"
        echo "unhealthy_threshold = 2"
        echo "healthy_threshold = 2"
        echo
        echo "[admission]"
        echo "overload_behaviour = \"${overload}\""
        echo "max_inflight_per_replica = ${maxinflight}"
        echo "max_queue_depth = 512"
        echo "queue_timeout_ms = 60000"
        echo
        echo "[upstream]"
        echo "connect_timeout_ms = 5000"
        echo "read_timeout_ms = 300000"
        echo "pool_max_idle_per_host = 512"
        echo
        local i=0
        for spec in "$@"; do
            local port="${spec%%:*}" placement="${spec#*:}"
            echo "[[replicas]]"
            echo "id = \"r${i}\""
            echo "url = \"http://127.0.0.1:${port}\""
            echo "placement = \"${placement}\""
            echo
            i=$(( i + 1 ))
        done
    } > "$out"
}

# Kill every child process and wait for the GPUs to be released.
ir_teardown() {
    ir_log "tearing down fleet"
    pkill -P $$ 2>/dev/null || true
    # vLLM spawns worker children that outlive a bare SIGTERM to the parent.
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "inference-router" 2>/dev/null || true
    sleep 5
}
