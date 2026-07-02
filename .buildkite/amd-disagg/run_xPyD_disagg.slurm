#!/bin/bash
#SBATCH --job-name=vllm-disagg-pd
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=01:30:00
#SBATCH --output=/shared_inference/%u/disagg_logs/slurm-%j.log

set -euo pipefail

# ---- inputs (env > default; forwarded by run-slurm-disagg-test.sh via --export)
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:nightly}"
WIDE_EP_MODE="${WIDE_EP_MODE:-0}"
xP="${xP:-1}"
yD="${yD:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
SHARED_MOUNT="${SHARED_MOUNT:-/shared_inference}"
LOG_ROOT="${LOG_ROOT:-${SHARED_MOUNT}/${USER:-$(whoami)}/disagg_logs}"
RUN_AFTER_HEALTH="${RUN_AFTER_HEALTH:-accuracy}"   # bench | accuracy
DRY_RUN="${DRY_RUN:-0}"
MORIIO_READ_MODE="${MORIIO_READ_MODE:-0}"   # 0 -> prefill writes (default); 1 -> decode reads KV (read_mode)

# ---- router / gateway (host-side; also forwarded into the container) --------
# toy         -> in-container MoRIIO toy proxy (default; no extra container)
# vllm-router -> external `vllm/vllm-router` container started on the rank-0 node
ROUTER_TYPE="${ROUTER_TYPE:-toy}"
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

# Where the disagg scripts live ON THE HOST (this dir, i.e. the sbatch submit
# dir)
DISAGG_SCRIPTS_DIR="${DISAGG_SCRIPTS_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

# In-container path the scripts are mounted to (and sourced from).
CONTAINER_SCRIPTS="${CONTAINER_SCRIPTS:-/vllm-workspace/.buildkite/amd-disagg}"
CLUSTER_ENV_IN="${CLUSTER_ENV_IN:-${CONTAINER_SCRIPTS}/cluster.sh}"


# Health endpoints (TP-mode ports from cluster.sh defaults).
PREFILL_PORT="${PREFILL_PORT:-8100}"
DECODE_PORT="${DECODE_PORT:-8200}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-2400}"


NUM_NODES=$((xP + yD))
echo "Calculated NUM_NODES: $NUM_NODES (xP=$xP + yD=$yD) WIDE_EP_MODE=$WIDE_EP_MODE"

# ------------------------
# Extract NUM_NODES from SLURM allocation and select nodes
# ------------------------
echo "Original SLURM allocation:"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"

FULL_NODELIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
AVAILABLE_COUNT=$(echo "$FULL_NODELIST" | wc -l)

if (( AVAILABLE_COUNT < NUM_NODES )); then
    echo "ERROR: Need ${NUM_NODES} nodes (xP=${xP} + yD=${yD}) but only ${AVAILABLE_COUNT} allocated." >&2
    exit 1
fi

# Pick the first NUM_NODES nodes from the allocation and sort alphabetically.
# srun assigns PROCID=0 to the alphabetically-first hostname regardless of --nodelist order.
# IPADDRS[0..xP-1] = prefill, IPADDRS[xP..] = decode, so the sorted order must match.
SELECTED_NODES=$(echo "$FULL_NODELIST" | head -n "$NUM_NODES" | sort)
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
export SLURM_JOB_CPUS_PER_NODE="${SLURM_JOB_CPUS_PER_NODE}"
export SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-amd-rccl}"
export SLURM_JOBID="${SLURM_JOBID:-$SLURM_JOB_ID}"
export SLURM_JOB_QOS="${SLURM_JOB_QOS:-normal}"
export SLURM_JOB_ACCOUNT="${SLURM_JOB_ACCOUNT:-amd-rccl}"
export SLURM_NTASKS_PER_NODE=1
export SLURM_SUBMIT_HOST="${SLURM_SUBMIT_HOST}"
export SLURM_JOB_ID="${SLURM_JOB_ID}"
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
USER_NAME=$(whoami)
MASTER_NODE=$(echo "$SELECTED_NODES" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$MASTER_NODE" bash -c 'hostname -I')
MASTER_ADDR=$(echo "$MASTER_ADDR" | awk 'NR==1 {print $1}')
MASTER_PORT=39566  # Choose an open port

IPS=()

for NODE in $SELECTED_NODES; do
    IP=$(srun --nodes=1 --ntasks=1 --time=00:20:00 --nodelist="$NODE" bash -c 'hostname -I')
    IP=$(echo "$IP" | awk 'NR==1 {print $1}')
    IPS+=("$IP")
done

echo "Selected node IPs: ${IPS[*]}" | sed 's/ /,/g'

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
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
export IPADDRS="$(echo "${IPS[*]}" | sed 's/ /,/g')"

export WIDE_EP_MODE xP yD GPUS_PER_NODE RUN_AFTER_HEALTH HEALTH_TIMEOUT_S DRY_RUN MORIIO_READ_MODE
export IMAGE SHARED_MOUNT CONTAINER_SCRIPTS DISAGG_SCRIPTS_DIR
export CLUSTER_ENV="${CLUSTER_ENV_IN}"
export DOCKER_CONT_NAME="vllm_disagg_${MODEL_NAME:-model}_${SLURM_JOB_ID}"

# Router env: forwarded into the main container (cluster.sh honors these) and used
# host-side to launch the external router container on rank 0.
export ROUTER_TYPE ROUTER_PORT ROUTER_POLICY VLLM_ROUTER_IMAGE PROXY_PING_PORT GATEWAY_PORT ROUTER_DP_LOCAL
export ROUTER_CONT_NAME="vllm_router_${SLURM_JOB_ID}"

echo "[disagg-pd] IPADDRS=${IPADDRS}"
echo "[disagg-pd] launching ${NUM_NODES} containers (1/node) -> vllm_disagg.sh node"

# One task per node; SLURM supplies the global rank via $SLURM_PROCID. --label
# prefixes each line with its task id so interleaved orchestration logs stay
# readable in the job log that run-slurm-disagg-test.sh streams to CI. rank 0's
# exit code (the workload pass/fail) propagates as srun's exit code.
RC=0
srun --label \
     --nodelist="$SELECTED_NODELIST_STR" \
     --nodes="$NUM_NODES" --ntasks="$NUM_NODES" --ntasks-per-node=1 \
  bash -c '
    set -o pipefail   # propagate docker run exit code through the tee pipe
    echo "Rank $SLURM_PROCID on $(hostname)"
    docker rm -f "$DOCKER_CONT_NAME" 2>/dev/null || true
    stale=$(docker ps -aq --filter name=vllm_disagg_ 2>/dev/null) || true
    [ -n "$stale" ] && { echo "Reaping stale containers on $(hostname): $stale"; docker rm -f $stale 2>/dev/null || true; }

    # Start the external vLLM router as a SEPARATE container on the rank-0 node.
    # It shares the host network with the main container (both --network host):
    # prefill/decode register to :$PROXY_PING_PORT (discovery) and clients hit
    # :$ROUTER_PORT. Only started when ROUTER_TYPE=vllm-router.
    if [ "$ROUTER_TYPE" = "vllm-router" ] && [ "$SLURM_PROCID" = "0" ] && [ "$DRY_RUN" = "1" ]; then
        echo "DRY_RUN: would start vllm-router on $(hostname): $VLLM_ROUTER_IMAGE (discovery :$PROXY_PING_PORT, port :$ROUTER_PORT, policy $ROUTER_POLICY, dp_local=$ROUTER_DP_LOCAL)"
    elif [ "$ROUTER_TYPE" = "vllm-router" ] && [ "$SLURM_PROCID" = "0" ]; then
        docker rm -f "$ROUTER_CONT_NAME" 2>/dev/null || true
        echo "Starting vllm-router on $(hostname): $VLLM_ROUTER_IMAGE (discovery :$PROXY_PING_PORT, port :$ROUTER_PORT, policy $ROUTER_POLICY, dp_local=$ROUTER_DP_LOCAL)"
        mkdir -p "$LOG_PATH" 2>/dev/null || true
        docker run -d \
            --name "$ROUTER_CONT_NAME" \
            --network host \
            -v "$LOG_PATH":/run_logs \
            -v "$SHARED_MOUNT":"$SHARED_MOUNT" \
            "$VLLM_ROUTER_IMAGE" \
            bash -lc "exec vllm-router --vllm-pd-disaggregation --kv-connector moriio --vllm-discovery-address 0.0.0.0:$PROXY_PING_PORT --port $ROUTER_PORT --host 0.0.0.0 --intra-node-data-parallel-size $ROUTER_DP_LOCAL --policy $ROUTER_POLICY --prefill-policy $ROUTER_POLICY --decode-policy $ROUTER_POLICY --log-level info > $LOG_PATH/vllm_router.log 2>&1" \
          || echo "WARN: failed to start vllm-router container on $(hostname)"
    fi

    docker run --rm \
        --device /dev/dri \
        --device /dev/kfd \
        --device /dev/infiniband \
        --network host \
        --ipc host \
        --group-add video \
        --cap-add SYS_PTRACE \
        --cap-add IPC_LOCK \
        --security-opt seccomp=unconfined \
        --privileged \
        --shm-size 64G \
        --ulimit nofile=1048576:1048576 \
        -v "$HOME":"$HOME" \
        -v "$SHARED_MOUNT":"$SHARED_MOUNT" \
        -v "$DISAGG_SCRIPTS_DIR":"$CONTAINER_SCRIPTS" \
        -e HF_TOKEN \
        -e SLURM_JOB_ID="$SLURM_JOB_ID" \
        -e SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST" \
        -e NODE_RANK="$SLURM_PROCID" \
        -e IPADDRS="$IPADDRS" \
        -e WIDE_EP_MODE="$WIDE_EP_MODE" \
        -e xP="$xP" -e yD="$yD" \
        -e GPUS_PER_NODE="$GPUS_PER_NODE" \
        -e CLUSTER_ENV="$CLUSTER_ENV" \
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
      2>&1 | tee "$LOG_PATH/main_NODE${SLURM_PROCID}.log"
    node_rc=${PIPESTATUS[0]}   # docker run exit code (not tee), for rank-0 pass/fail

    # Stop the external router (rank-0 node) once the main container returns, so
    # other ranks do not block and the node is left clean.
    if [ "$ROUTER_TYPE" = "vllm-router" ] && [ "$SLURM_PROCID" = "0" ]; then
        docker rm -f "$ROUTER_CONT_NAME" 2>/dev/null || true
    fi
    exit "$node_rc"
  ' || RC=$?

# Best-effort teardown of any lingering containers (main + external router)
srun --nodelist="$SELECTED_NODELIST_STR" \
     --nodes="$NUM_NODES" --ntasks="$NUM_NODES" --ntasks-per-node=1 \
  bash -c 'docker rm -f "$DOCKER_CONT_NAME" 2>/dev/null || true
           [ "$ROUTER_TYPE" = "vllm-router" ] && docker rm -f "$ROUTER_CONT_NAME" 2>/dev/null || true' || true

echo "[disagg-pd] srun finished rc=${RC}; logs in ${LOG_PATH}"
exit "${RC}"
