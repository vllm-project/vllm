#!/usr/bin/env bash
set -euo pipefail

export RANK=${OMPI_COMM_WORLD_RANK:?}
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:?}
export LOCAL_RANK=0
export MASTER_ADDR=${VLLM_MPI_MASTER_ADDR:-172.31.12.228}
export MASTER_PORT=${VLLM_MPI_MASTER_PORT:-29600}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp39s0}
export VLLM_HOST_IP=${VLLM_HOST_IP:-$(hostname -I | awk '{print $1}')}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_CPU_OMP_THREADS_BIND=0-1
export VLLM_CPU_INT4_W4A8=1
export ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX
export OMP_NUM_THREADS=2
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:/home/ubuntu/vllm-venv/lib/libiomp5.so

exec /home/ubuntu/vllm-venv/bin/python /home/ubuntu/legomem/vllm_mpi_kv_bench.py "$@"
