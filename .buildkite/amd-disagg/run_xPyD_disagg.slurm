#!/bin/bash
#SBATCH --job-name=vllm-disagg-pd
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/spur-%j.out
#SBATCH --error=/tmp/spur-%j.err

set -euo pipefail


# ---- inputs (env > default; forwarded by run-slurm-disagg-test.sh via --export)
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:nightly}"
WIDE_EP_MODE="${WIDE_EP_MODE:-0}"
xP="${xP:-1}"
yD="${yD:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

SHARED_MOUNT="${SHARED_MOUNT:-/data}"
LOG_ROOT="${LOG_ROOT:-/data/${USER:-$(id -un)}/disagg_logs}"
RUN_AFTER_HEALTH="${RUN_AFTER_HEALTH:-accuracy}"   # bench | accuracy
DRY_RUN="${DRY_RUN:-0}"
MORIIO_READ_MODE="${MORIIO_READ_MODE:-0}"   # 0 -> prefill writes (default); 1 -> decode reads KV (read_mode)

# ---- router / gateway (host-side; also forwarded into the container) --------
# vllm-router -> external `vllm/vllm-router` container started on the rank-0 node (default)
# proxy       -> in-container MoRIIO proxy (no extra container)
ROUTER_TYPE="${ROUTER_TYPE:-vllm-router}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
ROUTER_POLICY="${ROUTER_POLICY:-round_robin}"
VLLM_ROUTER_IMAGE="${VLLM_ROUTER_IMAGE:-vllm/vllm-router:nightly}"
PROXY_PING_PORT="${PROXY_PING_PORT:-36367}"   # must equal vllm-router --vllm-discovery-address port
PROXY_PORT="${PROXY_PORT:-10001}"
# Intra-node data-parallel size the router uses to route the MoRIIO KV-notify to
# the right DP rank. Required for wideEP DP (else decode hangs: "remote blocks
# never arrived"). EP -> GPUS_PER_NODE (matches --data-parallel-size-local); TP -> 1.
if [[ "${WIDE_EP_MODE}" == "1" ]]; then
    ROUTER_DP_LOCAL="${ROUTER_DP_LOCAL:-${GPUS_PER_NODE}}"
else
    ROUTER_DP_LOCAL="${ROUTER_DP_LOCAL:-1}"
fi
if [[ "${ROUTER_TYPE}" == "vllm-router" ]]; then
    GATEWAY_PORT="${GATEWAY_PORT:-${ROUTER_PORT}}"
else
    GATEWAY_PORT="${GATEWAY_PORT:-${PROXY_PORT}}"
fi

# This line is a recommended template from infra side.
exec > "$LOG_ROOT/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log" 2>&1

# Where the disagg scripts live ON THE HOST (bind-mounted into the container).
# Spur runs the job body from /tmp, so $(pwd)/SLURM_SUBMIT_DIR aren't reliable;
# default to the known host scripts dir. Override via env if you relocate it.
DISAGG_SCRIPTS_DIR="${DISAGG_SCRIPTS_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
echo "DISAGG SCRIPTS DIR is ----"
echo "$DISAGG_SCRIPTS_DIR"

# In-container path the scripts are mounted to (and sourced from).
CONTAINER_SCRIPTS="${CONTAINER_SCRIPTS:-/vllm-workspace/.buildkite/amd-disagg}"
CLUSTER_ENV_IN="${CLUSTER_ENV_IN:-${CONTAINER_SCRIPTS}/cluster.sh}"


# Health endpoints (TP-mode ports from cluster.sh defaults).
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-3600}"

# EP-mode ports, mirrored from cluster.sh purely so the preflight below can test
# them host-side; the in-container values still come from cluster.sh.
SERVE_PORT="${SERVE_PORT:-20005}"
RPC_PORT="${RPC_PORT:-13345}"
KV_PORT="${KV_PORT:-9711}"
LOCAL_PING_PORT="${LOCAL_PING_PORT:-61555}"
# vllm-router binds this for Prometheus in addition to ROUTER_PORT, and panics
# at startup if it is taken.
ROUTER_METRICS_PORT="${ROUTER_METRICS_PORT:-29000}"

# Node preflight (see preflight_node): 0 disables, VRAM% above which a GPU counts
# as busy, and how long to let a reaped container's VRAM drain before judging.
PREFLIGHT="${PREFLIGHT:-1}"
PREFLIGHT_MAX_VRAM_PCT="${PREFLIGHT_MAX_VRAM_PCT:-10}"
PREFLIGHT_GPU_SETTLE_S="${PREFLIGHT_GPU_SETTLE_S:-30}"


NUM_NODES=$((xP + yD))
echo "Calculated NUM_NODES: $NUM_NODES (xP=$xP + yD=$yD) WIDE_EP_MODE=$WIDE_EP_MODE"

# ------------------------
# Extract NUM_NODES from SLURM allocation and select nodes
# ------------------------
echo "Original SLURM allocation:"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"

# scontrol is available on classic Slurm but not in the Spur job body. Use it to
# expand compressed ranges when present, else split the (already comma-separated,
# full-hostname) SLURM_JOB_NODELIST on commas.
if command -v scontrol >/dev/null 2>&1; then
    FULL_NODELIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
else
    FULL_NODELIST=$(echo "$SLURM_JOB_NODELIST" | tr ',' '\n')
fi
AVAILABLE_COUNT=$(echo "$FULL_NODELIST" | wc -l)

if (( AVAILABLE_COUNT < NUM_NODES )); then
    echo "ERROR: Need ${NUM_NODES} nodes (xP=${xP} + yD=${yD}) but only ${AVAILABLE_COUNT} allocated." >&2
    exit 1
fi

# Pick the first NUM_NODES nodes and order them to match how IPs get assigned:
#   * classic Slurm: sort alphabetically -- srun assigns PROCID=0 to the
#     alphabetically-first hostname, so IPS[] (built per node below) lines up.
#   * Spur: preserve RANK order (SPUR_NODELIST), so the hostnames pair 1:1 with
#     the rank-ordered IPs in SPUR_PEER_NODES; an alphabetical sort here would
#     desync MASTER_NODE / MASTER_ADDR / IPADDRS from the actual ranks.
# IPADDRS[0..xP-1] = prefill, IPADDRS[xP..] = decode, so this order must match.
if command -v srun >/dev/null 2>&1; then
    SELECTED_NODES=$(echo "$FULL_NODELIST" | head -n "$NUM_NODES" | sort)
elif [[ -n "${SPUR_NODELIST:-}" ]]; then
    SELECTED_NODES=$(echo "${SPUR_NODELIST}" | tr ',' '\n' | sed '/^$/d' | head -n "$NUM_NODES")
else
    SELECTED_NODES=$(echo "$FULL_NODELIST" | head -n "$NUM_NODES")
fi
SELECTED_NODELIST_STR=$(echo "$SELECTED_NODES" | tr '\n' ',' | sed 's/,$//')

# Create new nodelist in SLURM format
# This is a simplified approach - for complex ranges, you might need more sophisticated parsing
NEW_SLURM_NODELIST=$(echo "$SELECTED_NODES" | paste -sd, | sed 's/,/,/g')

# Update SLURM environment variables
export SLURM_NNODES=$NUM_NODES
export SLURM_NTASKS=$NUM_NODES
export SLURM_JOB_NUM_NODES=$NUM_NODES
export SLURM_NPROCS=$NUM_NODES
export SLURM_JOB_NODELIST="$NEW_SLURM_NODELIST"
export SLURM_NODELIST="$NEW_SLURM_NODELIST"

# Keep other SLURM variables as they were or set defaults
export SLURM_TASKS_PER_NODE="1(x$NUM_NODES)"
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${HOME}}"
export SLURM_CLUSTER_NAME="${SLURM_CLUSTER_NAME:-vllm-di-ci}"
export SLURM_JOB_CPUS_PER_NODE="${SLURM_JOB_CPUS_PER_NODE:-1}"
export SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-amd-rccl}"
export SLURM_JOBID="${SLURM_JOBID:-$SLURM_JOB_ID}"
export SLURM_JOB_QOS="${SLURM_JOB_QOS:-normal}"
export SLURM_JOB_ACCOUNT="${SLURM_JOB_ACCOUNT:-amd-rccl}"
export SLURM_NTASKS_PER_NODE=1
export SLURM_SUBMIT_HOST="${SLURM_SUBMIT_HOST:-$(hostname)}"
export SLURM_JOB_ID="${SLURM_JOB_ID:-}"
export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-${xP}p${yD}d_bench-serving}"

echo ""
echo "Updated SLURM Environment Variables:"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_QOS: $SLURM_JOB_QOS"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_ACCOUNT: $SLURM_JOB_ACCOUNT"
echo "SLURM_NPROCS: $SLURM_NPROCS"
echo "SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "SLURM_CONF: $SLURM_CONF"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "SLURM_CLUSTER_NAME: $SLURM_CLUSTER_NAME"
echo "ulimit: $(ulimit -a)"
echo ""
echo "Selected nodes for execution:"
echo "$SELECTED_NODES"
echo ""

# Node information
MASTER_NODE=$(echo "$SELECTED_NODES" | head -n 1)
MASTER_PORT=39566  # Choose an open port

IPS=()
if command -v srun >/dev/null 2>&1; then
    # Classic Slurm: resolve each node's IP via srun (original behavior).
    for NODE in $SELECTED_NODES; do
        IP=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$NODE" bash -c 'hostname -I' | awk 'NR==1 {print $1}')
        IPS+=("$IP")
    done
else
    # Spur job body (no srun): use the rank-ordered IPs Spur provides in
    # SPUR_PEER_NODES ("ip:port,ip:port,..."). These are the cross-node-routable
    # bootstrap addresses; hostnames resolve to public IPs and the local node to
    # 127.0.1.1, so DNS is not usable here.
    mapfile -t IPS < <(echo "${SPUR_PEER_NODES:-}" | tr ',' '\n' | cut -d: -f1 | sed '/^$/d')
    if (( ${#IPS[@]} == 0 )); then
        echo "ERROR: no srun and SPUR_PEER_NODES is empty; cannot determine node IPs." >&2
        exit 1
    fi
fi
MASTER_ADDR="${IPS[0]}"

IPADDRS="${IPS[*]}"
IPADDRS="${IPADDRS// /,}"
echo "Selected node IPs: ${IPADDRS}"

NNODES=$NUM_NODES

echo "MASTER_NODE is ${MASTER_NODE}"
echo "MASTER_ADDR is ${MASTER_ADDR}"
echo "MASTER_PORT is ${MASTER_PORT}"
echo "NNODES is ${NNODES}"

# Per-job log folder (MAD-style): all per-role server/proxy/bench logs
export LOG_PATH="${LOG_ROOT}/${SLURM_JOB_ID}"
mkdir -p "$LOG_PATH"
echo "[disagg-pd] LOG_PATH=$LOG_PATH"

# ----------------------------------------------------------------------------
# Launch one container per node via a single srun and hand off to
# vllm_disagg.sh node (rank-based self-select). 
# ----------------------------------------------------------------------------
# IPADDRS is the ordered node IP list (prefill IPs first, then decode). IPS[]
# was built in SELECTED_NODES (alphabetical) order — the same order srun assigns
# SLURM_PROCID — so the rank->role mapping lines up:
#   rank 0           -> prefill master + MoRIIO proxy + orchestrator
#   rank 1 .. xP-1   -> prefill
#   rank xP .. end   -> decode
export IPADDRS

export WIDE_EP_MODE xP yD GPUS_PER_NODE RUN_AFTER_HEALTH HEALTH_TIMEOUT_S DRY_RUN MORIIO_READ_MODE
export IMAGE SHARED_MOUNT CONTAINER_SCRIPTS DISAGG_SCRIPTS_DIR
export CLUSTER_ENV="${CLUSTER_ENV_IN}"
export DOCKER_CONT_NAME="vllm_disagg_${MODEL_NAME:-model}_${SLURM_JOB_ID}"

# Router env: forwarded into the main container (cluster.sh honors these) and used
# host-side to launch the external router container on rank 0.
export ROUTER_TYPE ROUTER_PORT ROUTER_POLICY VLLM_ROUTER_IMAGE PROXY_PING_PORT GATEWAY_PORT ROUTER_DP_LOCAL
export ROUTER_CONT_NAME="vllm_router_${SLURM_JOB_ID}"

# Consumed by preflight_node on each node (srun forwards the environment).
export PREFILL_PORT DECODE_PORT PROXY_PORT ROUTER_METRICS_PORT
export SERVE_PORT RPC_PORT KV_PORT LOCAL_PING_PORT
export PREFLIGHT PREFLIGHT_MAX_VRAM_PCT PREFLIGHT_GPU_SETTLE_S

echo "[disagg-pd] IPADDRS=${IPADDRS}"
echo "[disagg-pd] launching ${NUM_NODES} containers (1/node) -> vllm_disagg.sh node"

# Refuse a node that is not actually free. $1 = this node's global rank.
#
# `#SBATCH --exclusive` only keeps out other SLURM jobs. A plain `docker run` on
# a node is invisible to the scheduler, so we get allocated nodes whose GPUs and
# ports are already in use. Without this the job pulls the image, loads the model
# and dies minutes later inside vLLM ("Free memory on device cuda:N ... is less
# than desired GPU memory utilization"); worse, on rank 0 a foreign router
# already listening on ROUTER_PORT can satisfy the health gate and take the
# accuracy traffic. Called after the stale-container reap, with a settle window
# so VRAM the reap is still releasing is not counted against us.
preflight_node() {
    local rank="$1"
    if [ "${PREFLIGHT:-1}" != "1" ]; then
        echo "Rank $rank preflight: skipped (PREFLIGHT=0)"
        return 0
    fi

    local host max_pct settle problems=""
    local -a issues=()          # one short tag per fault, for the one-line summary
    host="$(hostname)"
    max_pct="${PREFLIGHT_MAX_VRAM_PCT:-10}"
    settle="${PREFLIGHT_GPU_SETTLE_S:-30}"

    # GPUs: vLLM asks for a fraction of TOTAL, so memory already resident is
    # subtracted from what it can get and pushes it under its own threshold.
    if command -v rocm-smi >/dev/null 2>&1; then
        local csv busy nbusy ncards deadline
        deadline=$(( $(date +%s) + settle ))
        while :; do
            csv="$(rocm-smi --showmeminfo vram --csv 2>/dev/null || true)"
            busy="$(printf '%s\n' "$csv" |
                    awk -F, -v m="$max_pct" '/^card/ && $2+0 > 0 {
                        pct = $3 * 100 / $2
                        if (pct > m) printf "%s=%d%%(%.0fGiB) ", $1, pct, $3/1073741824
                    }' || true)"
            if [ -z "$busy" ]; then break; fi
            if [ "$(date +%s)" -ge "$deadline" ]; then break; fi
            sleep 5
        done
        if [ -n "$busy" ]; then
            nbusy="$(printf '%s' "$busy" | tr ' ' '\n' | grep -c 'card' || true)"
            ncards="$(printf '%s\n' "$csv" | grep -c '^card' || true)"
            issues+=("${nbusy}/${ncards} GPUs hold VRAM above ${max_pct}%")
            problems="${problems}  GPU VRAM in use above ${max_pct}%: ${busy}"$'\n'
        fi
    else
        echo "Rank $rank preflight: no rocm-smi on $host, skipping GPU check"
    fi

    # Ports this rank will bind. Every container here is --network host, so a
    # foreign listener either collides or silently serves our traffic.
    local -a want=()
    if [ "${WIDE_EP_MODE:-0}" = "1" ]; then
        want+=("$SERVE_PORT" "$RPC_PORT" "$KV_PORT" "$LOCAL_PING_PORT")
    elif [ "$rank" -lt "${xP:-1}" ]; then
        want+=("$PREFILL_PORT")
    else
        want+=("$DECODE_PORT")
    fi
    if [ "$rank" = "0" ]; then
        want+=("$PROXY_PING_PORT")
        if [ "$ROUTER_TYPE" = "vllm-router" ]; then
            want+=("$ROUTER_PORT" "$ROUTER_METRICS_PORT")
        else
            want+=("$PROXY_PORT")
        fi
    fi

    if command -v ss >/dev/null 2>&1; then
        local bound taken="" p
        bound="$(ss -lnt 2>/dev/null | awk '{ n = split($4, a, ":"); print a[n] }' | sort -u || true)"
        for p in "${want[@]}"; do
            if printf '%s\n' "$bound" | grep -qx -- "$p"; then
                taken="${taken}${p} "
            fi
        done
        if [ -n "$taken" ]; then
            issues+=("ports already bound: ${taken% }")
            problems="${problems}  ports already bound: ${taken}"$'\n'
        fi
    else
        echo "Rank $rank preflight: no ss on $host, skipping port check"
    fi

    if [ -n "$problems" ]; then
        mkdir -p "$LOG_PATH" 2>/dev/null || true
        local summary
        summary="$(printf '%s; ' "${issues[@]}")"
        summary="${summary%; }"
        # Every node writes the shared job log concurrently, so this diagnostic can
        # be overwritten there. Keep a single-writer copy per node as well, or the
        # job fails in seconds with no surviving explanation. The submitter reads
        # the PREFLIGHT-REJECTED line back out of it to use as the failure reason,
        # so keep that prefix in sync with run-slurm-disagg-test.sh. It must not
        # contain "FAIL:" — that is the accuracy gate's pattern.
        {
            echo "PREFLIGHT-REJECTED: rank ${rank} on ${host}: ${summary}"
            printf '%s' "$problems"
            echo "  containers here: $(docker ps --format '{{.Names}} ({{.Image}})' 2>/dev/null | paste -sd'; ' - || true)"
            echo "  --exclusive does not exclude non-SLURM workloads; clear the node or exclude it from the allocation."
            echo "  Set PREFLIGHT=0 to submit anyway."
        } | tee -a "$LOG_PATH/preflight_NODE${rank}.log" 2>/dev/null || true
        # Unblock the other ranks: they park on this sentinel, and the submitter
        # polls it. Without it a peer waits out HEALTH_TIMEOUT_S for an endpoint
        # that is never coming.
        echo 1 > "$LOG_PATH/.disagg_done" 2>/dev/null || true
        return 1
    fi

    echo "Rank $rank preflight ok on $host: GPUs idle, ports free (${want[*]})"
    return 0
}

# Per-node container launch, auto-dispatched by srun availability:
# Launch this node's disagg container and hand off to vllm_disagg.sh. $1 = this
# node's global rank. Relies on the env exported above (forwarded by srun on
# classic Slurm; already in scope when called directly on Spur).
run_node_container() {
    local rank="$1"
    set -o pipefail   # propagate docker run exit code through the tee pipe
    echo "Rank $rank on $(hostname)"
    local -a stale=()
    mapfile -t stale < <(docker ps -aq --filter name=vllm_disagg_ --filter name=vllm_router_ 2>/dev/null || true)
    if (( ${#stale[@]} > 0 )); then
        echo "Reaping stale containers on $(hostname): ${stale[*]}"
        docker stop -t 30 "${stale[@]}" >/dev/null 2>&1 || true
        docker rm "${stale[@]}" >/dev/null 2>&1 || true
    fi

    preflight_node "$rank" || return 1

    # Bind-mount the host AMD Pensando (ionic) RoCE userspace provider into the
    # container IF present: the image ships an ABI-incompatible copy and the NICs
    # are invisible without the host libs. Each entry is added only when the host
    # file exists, so this is a no-op on non-ionic (e.g. mlx5) clusters.
    local IONIC_MOUNTS=()
    local host_ionic
    host_ionic=$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1 2>/dev/null || true)
    [ -n "$host_ionic" ] && IONIC_MOUNTS+=(-v "$host_ionic:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro")
    [ -f /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so ] && \
        IONIC_MOUNTS+=(-v "/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro")
    [ -f /etc/libibverbs.d/ionic.driver ] && \
        IONIC_MOUNTS+=(-v "/etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro")
    echo "Rank $rank ionic mounts: ${IONIC_MOUNTS[*]:-<none>}"

    # Pull latest container.
    if [ "$DRY_RUN" != "1" ]; then
        echo "Rank $rank pulling image on $(hostname): $IMAGE"
        docker pull "$IMAGE" || echo "WARN: docker pull failed on $(hostname) for $IMAGE (using local copy if present)"
        if [ "$ROUTER_TYPE" = "vllm-router" ] && [ "$rank" = "0" ]; then
            echo "Rank $rank pulling router image on $(hostname): $VLLM_ROUTER_IMAGE"
            docker pull "$VLLM_ROUTER_IMAGE" || echo "WARN: docker pull failed on $(hostname) for $VLLM_ROUTER_IMAGE (using local copy if present)"
        fi
    fi

    # Start the external vLLM router as a SEPARATE container on the rank-0 node.
    if [ "$ROUTER_TYPE" = "vllm-router" ] && [ "$rank" = "0" ] && [ "$DRY_RUN" = "1" ]; then
        echo "DRY_RUN: would start vllm-router on $(hostname): $VLLM_ROUTER_IMAGE (discovery :$PROXY_PING_PORT, port :$ROUTER_PORT, policy $ROUTER_POLICY, dp_local=$ROUTER_DP_LOCAL)"
    elif [ "$ROUTER_TYPE" = "vllm-router" ] && [ "$rank" = "0" ]; then
        mkdir -p "$LOG_PATH" 2>/dev/null || true
        {
            echo "Starting vllm-router on $(hostname): $VLLM_ROUTER_IMAGE (discovery :$PROXY_PING_PORT, port :$ROUTER_PORT, policy $ROUTER_POLICY, dp_local=$ROUTER_DP_LOCAL)"
            docker run -d \
                --name "$ROUTER_CONT_NAME" \
                --network host \
                --user "$(id -u):$(id -g)" \
                -v "$LOG_PATH":/run_logs \
                -v "$SHARED_MOUNT":"$SHARED_MOUNT" \
                -v "/data/$(id -un):/workspace" \
                -e HOME=/workspace \
                -e USER="$(id -un)" \
                -e LOGNAME="$(id -un)" \
                "$VLLM_ROUTER_IMAGE" \
                bash -lc "exec vllm-router --vllm-pd-disaggregation --kv-connector moriio --vllm-discovery-address 0.0.0.0:$PROXY_PING_PORT --port $ROUTER_PORT --host 0.0.0.0 --intra-node-data-parallel-size $ROUTER_DP_LOCAL --policy $ROUTER_POLICY --prefill-policy $ROUTER_POLICY --decode-policy $ROUTER_POLICY --log-level info >> $LOG_PATH/vllm_router.log 2>&1" \
              && echo "vllm-router container started ok" \
              || echo "WARN: failed to start vllm-router container on $(hostname)"
        } 2>&1 | tee -a "$LOG_PATH/vllm_router_start.log"
    fi

    docker run --rm \
	--user "$(id -u):$(id -g)" \
        --device /dev/dri \
        --device /dev/kfd \
        --device /dev/infiniband \
        --network host \
        --ipc host \
        --group-add "$(getent group video  | cut -d: -f3)" \
        --group-add "$(getent group render | cut -d: -f3)" \
        --cap-add SYS_PTRACE \
        --cap-add IPC_LOCK \
        --security-opt seccomp=unconfined \
        --shm-size 64G \
        --ulimit nofile=1048576:1048576 \
        -v "$SHARED_MOUNT":"$SHARED_MOUNT" \
        -v "$DISAGG_SCRIPTS_DIR":"$CONTAINER_SCRIPTS" \
        -v "/data/$(id -un):/workspace" \
        "${IONIC_MOUNTS[@]}" \
        --ulimit memlock=-1:-1 \
        -e HF_TOKEN \
	-e HOME=/workspace \
        -e USER="$(id -un)" \
        -e LOGNAME="$(id -un)" \
        -e TRITON_CACHE_DIR=/tmp/triton_cache \
        -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache \
        -e VLLM_CACHE_ROOT=/tmp/vllm_cache \
        -e AITER_JIT_DIR=/tmp/aiter_jit \
        -e FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl_cache \
        -e SLURM_JOB_ID="$SLURM_JOB_ID" \
        -e SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST" \
        -e NODE_RANK="$rank" \
        -e IPADDRS="$IPADDRS" \
        -e WIDE_EP_MODE="$WIDE_EP_MODE" \
        -e xP="$xP" -e yD="$yD" \
        -e GPUS_PER_NODE="$GPUS_PER_NODE" \
        -e CLUSTER_ENV="$CLUSTER_ENV" \
        -e MODEL_NAME="${MODEL_NAME:-}" \
        -e MODEL_DIR="${MODEL_DIR:-}" \
        -e LOG_PATH="$LOG_PATH" \
        -e RUN_AFTER_HEALTH="$RUN_AFTER_HEALTH" \
        -e HEALTH_TIMEOUT_S="$HEALTH_TIMEOUT_S" \
        -e DRY_RUN="$DRY_RUN" \
        -e MORIIO_READ_MODE="$MORIIO_READ_MODE" \
        -e ROUTER_TYPE="$ROUTER_TYPE" \
        -e ROUTER_PORT="$ROUTER_PORT" \
        -e ROUTER_POLICY="$ROUTER_POLICY" \
        -e GATEWAY_PORT="$GATEWAY_PORT" \
        --name "$DOCKER_CONT_NAME" \
        --entrypoint /bin/bash \
        "$IMAGE" -c "bash $CONTAINER_SCRIPTS/vllm_disagg.sh node" \
      2>&1 | tee "$LOG_PATH/main_NODE${rank}.log"
    local node_rc=${PIPESTATUS[0]}   # docker run exit code (not tee), for rank-0 pass/fail
    # Stop the external router (rank-0 node) once the main container returns.
    if [ "$ROUTER_TYPE" = "vllm-router" ] && [ "$rank" = "0" ]; then
        docker stop "$ROUTER_CONT_NAME" 2>/dev/null || true; docker rm "$ROUTER_CONT_NAME" 2>/dev/null || true; 
    fi
    return "$node_rc"
}

# Teardown of the main + external router containers on every node ($1 = docker
# stop timeout). `docker run` containers outlive the client that started them,
# so on scancel the srun tasks die but the containers keep the GPUs until the
# walltime expires unless they are stopped explicitly.
teardown_containers() {
    export TEARDOWN_STOP_T="${1:-30}"
    if command -v srun >/dev/null 2>&1; then
        # shellcheck disable=SC2016  # must expand on the remote node, not here
        srun --nodelist="$SELECTED_NODELIST_STR" \
             --nodes="$NUM_NODES" --ntasks="$NUM_NODES" --ntasks-per-node=1 \
          bash -c 'docker stop -t "${TEARDOWN_STOP_T:-30}" "$DOCKER_CONT_NAME" >/dev/null 2>&1 || true; docker rm "$DOCKER_CONT_NAME" >/dev/null 2>&1 || true
                   [ "$ROUTER_TYPE" = "vllm-router" ] && { docker stop -t "${TEARDOWN_STOP_T:-30}" "$ROUTER_CONT_NAME" >/dev/null 2>&1 || true; docker rm "$ROUTER_CONT_NAME" >/dev/null 2>&1 || true; }' || true
    else
        docker stop -t "$TEARDOWN_STOP_T" "$DOCKER_CONT_NAME" >/dev/null 2>&1 || true
        docker rm "$DOCKER_CONT_NAME" >/dev/null 2>&1 || true
        if [ "$ROUTER_TYPE" = "vllm-router" ]; then
            docker stop -t "$TEARDOWN_STOP_T" "$ROUTER_CONT_NAME" >/dev/null 2>&1 || true
            docker rm "$ROUTER_CONT_NAME" >/dev/null 2>&1 || true
        fi
    fi
    return 0
}

# SLURM sends SIGTERM then SIGKILL after KillWait (30s by default), so the trap
# uses a short stop timeout to fit inside that window.
trap 'echo "[disagg-pd] signal received, tearing down containers"; teardown_containers 10; exit 143' TERM INT

RC=0
if command -v srun >/dev/null 2>&1; then
    # classic Slurm: fan out one task/node; ship the fn and call it with the
    # srun-supplied per-task SLURM_PROCID (escaped so the remote shell expands it).
    srun --label \
         --nodelist="$SELECTED_NODELIST_STR" \
         --nodes="$NUM_NODES" --ntasks="$NUM_NODES" --ntasks-per-node=1 \
      bash -c "$(declare -f preflight_node run_node_container); run_node_container \"\$SLURM_PROCID\"" || RC=$?
else
    # Spur: the body already runs once per node; this node's rank is
    # SPUR_TASK_OFFSET (fall back to NODE_RANK, then 0 for a single-node run).
    run_node_container "${SPUR_TASK_OFFSET:-${NODE_RANK:-0}}" || RC=$?
fi

teardown_containers 30

echo "[disagg-pd] launch finished rc=${RC}; logs in ${LOG_PATH}"
exit "${RC}"
