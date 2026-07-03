#!/bin/bash
# =============================================================================
# cluster.sh — single, generic config for the vLLM disaggregated (P/D) launcher.
# -----------------------------------------------------------------------------
# This script is *sourced* by vllm_disagg.sh.
#
# Every value is `${VAR:-default}`, so the environment always wins:
#       environment variable  >  built-in default below
#
# So you can override anything inline:
#   PREFILL_IP=10.0.0.1 DECODE_IP=10.0.0.2 ./vllm_disagg.sh prefill
#
# Site-specific values (model dir, IPs, NIC list, partition) are the defaults
# in the "site defaults" sections below — edit those for a new cluster.
# Model-SPECIFIC perf flags live in models.yaml, NOT here.
# =============================================================================

_CLUSTER_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------- model / mode
# MODEL_NAME indexes into models.yaml; MODEL_DIR is the parent dir holding it.
# MODEL_PATH (resolved by the launcher) = ${MODEL_DIR}/${MODEL_NAME}.   [site]
export MODEL_NAME="${MODEL_NAME:-DeepSeek-V3}"
export MODEL_DIR="${MODEL_DIR:-/data/models2}"

# Shared NFS root (5 TB, visible on every node): model weights + per-run logs. [site]
export SHARED_MOUNT="${SHARED_MOUNT:-/data}"

# Parallelism mode (launcher derives PARALLEL_MODE tp|ep from this):
#   WIDE_EP_MODE=0  tp : each node is an independent TP server (TP8, 1P1D)
#   WIDE_EP_MODE=1  ep : data-parallel + expert-parallel across xP/yD nodes (wideep)
export WIDE_EP_MODE="${WIDE_EP_MODE:-0}"

# ----------------------------------------------------------------- topology
# xP prefill nodes + yD decode nodes. IPADDRS is the ordered, comma-separated
# node list (prefill IPs first, then decode IPs). NODE_RANK is this node's global
# 0-based rank; under SLURM it defaults to $SLURM_PROCID. Leave IPADDRS empty to
# use the PREFILL_IP/DECODE_IP fallback defaults below (1P1D only).
export xP="${xP:-1}"
export yD="${yD:-1}"
export IPADDRS="${IPADDRS:-}"
export NODE_RANK="${NODE_RANK:-${SLURM_PROCID:-}}"
export PREFILL_IP="${PREFILL_IP:-10.0.0.1}"
export DECODE_IP="${DECODE_IP:-10.0.0.2}"

# Per-node GPU count and TP degree
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export TP_SIZE="${TP_SIZE:-${GPUS_PER_NODE}}"

# ----------------------------------------------------------------- ports
# TP-mode (WIDE_EP_MODE=0) server ports.
export PREFILL_PORT="${PREFILL_PORT:-8100}"
export DECODE_PORT="${DECODE_PORT:-8200}"

# EP-mode (WIDE_EP_MODE=1) ports: API serve port, DP RPC port, KV transfer port, and
# per-node MoRIIO local ping port (must differ from PROXY_PING_PORT).
export SERVE_PORT="${SERVE_PORT:-20005}"
export RPC_PORT="${RPC_PORT:-13345}"
export KV_PORT="${KV_PORT:-9711}"
export LOCAL_PING_PORT="${LOCAL_PING_PORT:-61555}"

# MoRIIO proxy: HTTP port clients/benchmark hit, plus the connector control ports.
# PROXY_PING_PORT MUST be 36367 — the toy proxy hardcodes its zmq service-discovery
# socket on that port; prefill/decode register to PROXY_IP:PROXY_PING_PORT.
export PROXY_IP="${PROXY_IP:-${PREFILL_IP}}"
export PROXY_PORT="${PROXY_PORT:-10001}"
export PROXY_PING_PORT="${PROXY_PING_PORT:-36367}"
export HANDSHAKE_PORT="${HANDSHAKE_PORT:-6301}"
export NOTIFY_PORT="${NOTIFY_PORT:-61005}"

export PROXY_SCRIPT="${PROXY_SCRIPT:-${_CLUSTER_SH_DIR}/moriio_toy_proxy_server.py}"

# MoRIIO KV transfer direction (injected into --kv-transfer-config by the launcher):
#   0 -> omit read_mode     (default; MoRIIO write mode: prefill pushes to decode)
#   1 -> "read_mode": true  (decode pulls KV from prefill; matches upstream disagg)
export MORIIO_READ_MODE="${MORIIO_READ_MODE:-0}"
export MORI_GPU_ARCHS="${MORI_GPU_ARCHS:-gfx950}"

# ----------------------------------------------------------------- router / gateway
# Selection for client (bench/accuracy) traffic:
#   toy         -> the in-container MoRIIO toy proxy started by the launcher (default)
#   vllm-router -> an external `vllm/vllm-router` container started by the SLURM job
#                  on the rank-0 node
# Both use the SAME MoRIIO discovery mechanism (prefill/decode register to
# PROXY_IP:PROXY_PING_PORT=36367); only the client HTTP front door differs.
export ROUTER_TYPE="${ROUTER_TYPE:-toy}"
export ROUTER_PORT="${ROUTER_PORT:-30000}"
export ROUTER_POLICY="${ROUTER_POLICY:-round_robin}"
export VLLM_ROUTER_IMAGE="${VLLM_ROUTER_IMAGE:-vllm/vllm-router:nightly}"
# Single client-facing port bench/accuracy target: the router port when routing,
# else the toy proxy port. Env override always wins.
if [[ "${ROUTER_TYPE}" == "vllm-router" ]]; then
    export GATEWAY_PORT="${GATEWAY_PORT:-${ROUTER_PORT}}"
else
    export GATEWAY_PORT="${GATEWAY_PORT:-${PROXY_PORT}}"
fi

# Where per-run logs / benchmark results are written. A $SLURM_JOB_ID subdir is
# appended so each CI run is self-scoped (falls back to 'local' off-SLURM).  [site]
_LOG_BASE="${LOG_BASE:-/data/${USER:-$(id -un)}/disagg_logs}"
export LOG_PATH="${LOG_PATH:-${_LOG_BASE}/${SLURM_JOB_ID:-local}}"

# ----------------------------------------------------------------- vLLM runtime
# Engine/platform/transport-level env (NOT model-specific). Model-architecture
# AITER kernel toggles live in models.yaml under each model's `env:` block.
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"

#export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
#export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
#
#export HOME=/tmp
#export HF_HOME="${HF_HOME:-/tmp/hf_home}"
#export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/.cache}"
# MoRIIO read-mode: decode pulls KV from prefill (matches the toy proxy READ path").
export VLLM_MORIIO_CONNECTOR_READ_MODE="${VLLM_MORIIO_CONNECTOR_READ_MODE:-1}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"

# ----------------------------------------------------------------- patch / scale
# MoRIIO multi-node DP patch (vLLM PR #39276). auto = apply only when WIDE_EP_MODE=1
# and xP>1 or yD>1; set 1 to force, 0 to skip.
export APPLY_MORIIO_PATCH="${APPLY_MORIIO_PATCH:-auto}"
# DP/EP group formation timeout across nodes (seconds).
export DISTRIBUTED_TIMEOUT_SECONDS="${DISTRIBUTED_TIMEOUT_SECONDS:-7200}"

# ----------------------------------------------------------------- RDMA / NCCL
# AMD Pensando AINIC RoCE fabric: 8 NICs exposed as ionic_0..7 (netdevs eth2..9),
# each rail on its own /24. GID index 1 + traffic class 104 are the
# DigitalOcean-validated tunables (also preset cluster-wide in /etc/rccl.conf).
# eth1 (VPC, 10.128.0.0/20) is the bootstrap/OOB socket; transport is RDMA over
# the ionic devices. Matches /data/templates/spur-multinode-rccl-template.sh. [site]
_IB_DEVICES="${IB_DEVICES:-ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}"
_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-1}"
export IB_DEVICES="${IB_DEVICES:-${_IB_DEVICES}}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-${_IB_DEVICES}}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-${_IB_GID_INDEX}}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
# In-box RCCL net transport (no external plugin); pin bootstrap to the VPC iface.
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-0}"
export NCCL_NET_DISABLE_INTRA="${NCCL_NET_DISABLE_INTRA:-1}"
export NCCL_IB_TC="${NCCL_IB_TC:-104}"
export NCCL_IB_FIFO_TC="${NCCL_IB_FIFO_TC:-192}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-1}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-12}"
# MoRI uses the same NIC set as NCCL.
export MORI_RDMA_DEVICES="${MORI_RDMA_DEVICES:-${_IB_DEVICES}}"
export MORI_IB_GID_INDEX="${MORI_IB_GID_INDEX:-${_IB_GID_INDEX}}"
# MoRI symmetric (shmem) heap. The 4G static default OOMs for DeepSeek-class EP
# all2all ("Out of static heap memory! Requested: ~1.75GB"); 16G matches the VMM
# default and leaves headroom for EP16. Accepts suffixes (e.g. 16G, 512M).
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-16G}"
# Pin the JIT target arch. mori auto-detects the GPU arch for its shmem/all2all
# kernels and on MI350X/MI355X (CDNA4 = gfx950) mis-selects gfx942 (MI300X); the
# resulting shmem_kernels.hsaco then fails to load with "device kernel image is
# invalid" and every EP worker dies at ep-group init (wide-EP only; TP is
# unaffected). Forcing gfx950 makes mori JIT for the real ISA.
# Ref: ROCm vLLM-MoRI docs (MORI_GPU_ARCHS=gfx950 for MI355X).
export MORI_GPU_ARCHS="${MORI_GPU_ARCHS:-gfx950}"

# ----------------------------------------------------------------- benchmark
# Space-separated ISL/OSL pairs and concurrency levels for the bench sweep.
# Defaults are tuned for the CI gate; override inline for a fuller prod sweep.
export BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-1024/128 2048/128}"
export BENCHMARK_CON="${BENCHMARK_CON:-32 64}"
# num-prompts per point = NUM_PROMPTS_FACTOR * concurrency (min BENCHMARK_MIN_PROMPTS).
export NUM_PROMPTS_FACTOR="${NUM_PROMPTS_FACTOR:-2}"
export BENCHMARK_MIN_PROMPTS="${BENCHMARK_MIN_PROMPTS:-32}"

# ----------------------------------------------------------------- accuracy
# `accuracy` role runs lm_eval (local-completions backend) against the proxy.
export ACCURACY_TASKS="${ACCURACY_TASKS:-gsm8k}"
export ACCURACY_NUM_CONCURRENT="${ACCURACY_NUM_CONCURRENT:-64}"
export ACCURACY_MAX_RETRIES="${ACCURACY_MAX_RETRIES:-3}"
# Correctness gate: the run FAILS if the parsed metric is below this threshold.
# ACCURACY_METRIC is matched against the lm_eval metric name (before the comma,
# e.g. "exact_match" for gsm8k); the max across filters/tasks is compared.
export ACCURACY_METRIC="${ACCURACY_METRIC:-exact_match}"
export ACCURACY_THRESHOLD="${ACCURACY_THRESHOLD:-0.90}"

# ----------------------------------------------------------------- SLURM (submit)
# Used by run-slurm-disagg-test.sh on the login node (harmless to export here). [site]
export SLURM_PARTITION="${SLURM_PARTITION:-default}"
