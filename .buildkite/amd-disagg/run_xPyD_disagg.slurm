#!/bin/bash
#SBATCH --job-name=vllm-disagg-pd
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/spur-%j.out
#SBATCH --error=/tmp/spur-%j.err

set -euo pipefail


# ---- inputs (env > default; forwarded by run-slurm-disagg-test.sh via --export)
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:v0.23.0}"
WIDE_EP_MODE="${WIDE_EP_MODE:-0}"
xP="${xP:-1}"
yD="${yD:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

SHARED_MOUNT="${SHARED_MOUNT:-/data}"
LOG_ROOT="${LOG_ROOT:-/data/${USER:-$(id -un)}/disagg_logs}"
RUN_AFTER_HEALTH="${RUN_AFTER_HEALTH:-accuracy}"   # bench | accuracy
DRY_RUN="${DRY_RUN:-0}"

exec > "$LOG_ROOT/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log" 2>&1

# Where the disagg scripts live ON THE HOST (bind-mounted into the container).
# Spur runs the job body from /tmp, so $(pwd)/SLURM_SUBMIT_DIR aren't reliable;
# default to the known host scripts dir. Override via env if you relocate it.
DISAGG_SCRIPTS_DIR="${DISAGG_SCRIPTS_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
echo "DISAGG SCRIPTS DIR is ----"
echo $DISAGG_SCRIPTS_DIR

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
USER_NAME=$(whoami)
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

export WIDE_EP_MODE xP yD GPUS_PER_NODE RUN_AFTER_HEALTH HEALTH_TIMEOUT_S DRY_RUN
export IMAGE SHARED_MOUNT CONTAINER_SCRIPTS DISAGG_SCRIPTS_DIR
export CLUSTER_ENV="${CLUSTER_ENV_IN}"
export DOCKER_CONT_NAME="vllm_disagg_${MODEL_NAME:-model}_${SLURM_JOB_ID}"

echo "[disagg-pd] IPADDRS=${IPADDRS}"
echo "[disagg-pd] launching ${NUM_NODES} containers (1/node) -> vllm_disagg.sh node"

# Per-node container launch, auto-dispatched by srun availability:
#   * classic Slurm: the batch body runs ONCE; srun fans out one task/node and
#     supplies the global rank via $SLURM_PROCID. The launcher fn is shipped to
#     each remote shell via `declare -f`. --label prefixes each line with the
#     task id so interleaved orchestration logs stay readable in CI.
#   * Spur: there is no srun in the job body, and the body ALREADY runs once per
#     node; this node's global rank is $SPUR_TASK_OFFSET, so the launcher is
#     invoked directly (no fan-out).
# rank->role (rank 0 = prefill master + MoRIIO proxy + orchestrator; 1..xP-1 =
# prefill; xP..end = decode) is resolved inside vllm_disagg.sh. rank 0's exit
# code (the workload pass/fail) propagates out as this job's exit code.

# Launch this node's disagg container and hand off to vllm_disagg.sh. $1 = this
# node's global rank. Relies on the env exported above (forwarded by srun on
# classic Slurm; already in scope when called directly on Spur).
run_node_container() {
    local rank="$1"
    set -o pipefail   # propagate docker run exit code through the tee pipe
    echo "Rank $rank on $(hostname)"
    docker rm -f "$DOCKER_CONT_NAME" 2>/dev/null || true
    local stale
    stale=$(docker ps -aq --filter name=vllm_disagg_ 2>/dev/null) || true
    [ -n "$stale" ] && { echo "Reaping stale containers on $(hostname): $stale"; docker rm -f $stale 2>/dev/null || true; }
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
    docker run --rm \
	--user $(id -u):$(id -g) \
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
        -e SLURM_JOB_ID="$SLURM_JOB_ID" \
        -e SLURM_JOB_NODELIST="$SLURM_JOB_NODELIST" \
        -e NODE_RANK="$rank" \
        -e IPADDRS="$IPADDRS" \
        -e WIDE_EP_MODE="$WIDE_EP_MODE" \
        -e xP="$xP" -e yD="$yD" \
        -e GPUS_PER_NODE="$GPUS_PER_NODE" \
        -e CLUSTER_ENV="$CLUSTER_ENV" \
        -e LOG_PATH="$LOG_PATH" \
        -e RUN_AFTER_HEALTH="$RUN_AFTER_HEALTH" \
        -e HEALTH_TIMEOUT_S="$HEALTH_TIMEOUT_S" \
        -e DRY_RUN="$DRY_RUN" \
        --name "$DOCKER_CONT_NAME" \
        --entrypoint /bin/bash \
        "$IMAGE" -c "bash $CONTAINER_SCRIPTS/vllm_disagg.sh node" \
      2>&1 | tee "$LOG_PATH/main_NODE${rank}.log"
}

RC=0
if command -v srun >/dev/null 2>&1; then
    # classic Slurm: fan out one task/node; ship the fn and call it with the
    # srun-supplied per-task SLURM_PROCID (escaped so the remote shell expands it).
    srun --label \
         --nodelist="$SELECTED_NODELIST_STR" \
         --nodes="$NUM_NODES" --ntasks="$NUM_NODES" --ntasks-per-node=1 \
      bash -c "$(declare -f run_node_container); run_node_container \"\$SLURM_PROCID\"" || RC=$?
else
    # Spur: the body already runs once per node; this node's rank is
    # SPUR_TASK_OFFSET (fall back to NODE_RANK, then 0 for a single-node run).
    run_node_container "${SPUR_TASK_OFFSET:-${NODE_RANK:-0}}" || RC=$?
fi

# Best-effort teardown of any lingering containers (same dispatch split).
if command -v srun >/dev/null 2>&1; then
    srun --nodelist="$SELECTED_NODELIST_STR" \
         --nodes="$NUM_NODES" --ntasks="$NUM_NODES" --ntasks-per-node=1 \
      bash -c 'docker rm -f "$DOCKER_CONT_NAME" 2>/dev/null || true' || true
else
    docker rm -f "$DOCKER_CONT_NAME" 2>/dev/null || true
fi

echo "[disagg-pd] launch finished rc=${RC}; logs in ${LOG_PATH}"
exit "${RC}"
