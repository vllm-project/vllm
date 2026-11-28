#!/bin/bash
# Multi-node vLLM server on a Slurm cluster with Apptainer/Singularity.
# One srun task per node; vLLM's multiprocessing backend joins them (no Ray).
#
#   sbatch --nodes=2 serve_multinode.sbatch          # TP=GPUs per node, PP=nodes
#   TP=16 PP=2 sbatch --nodes=4 serve_multinode.sbatch
#
# Adjust the #SBATCH lines (account, partition, GPU request) to your site.
#SBATCH --job-name=vllm-multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=vllm-multinode-%j.out
set -euo pipefail

export SIF=${SIF:?set SIF to a vLLM image, e.g. apptainer pull vllm.sif docker://vllm/vllm-openai}
export MODEL=${MODEL:?set MODEL to a model id or path}
export WORKDIR=${WORKDIR:-$PWD}              # holds hf_cache/ and cache/; must be on a shared filesystem
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=$SLURM_JOB_NUM_NODES
export TP=${TP:-$GPUS_PER_NODE} PP=${PP:-$NNODES}
export HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29511}
export API_PORT=${API_PORT:-8000}

if [ $((TP * PP)) -ne $((GPUS_PER_NODE * NNODES)) ]; then
  echo "TP*PP must equal GPUS_PER_NODE*NNODES" >&2; exit 1
fi

srun --ntasks="$NNODES" --ntasks-per-node=1 bash "$SLURM_SUBMIT_DIR/node.sh" &
SERVER=$!

# The API server runs on the first node only.
until curl -sf "http://$HEAD:$API_PORT/v1/models" > /dev/null; do
  kill -0 "$SERVER" 2> /dev/null || { echo "vLLM exited during startup" >&2; exit 1; }
  sleep 10
done
echo "vLLM is serving $MODEL at http://$HEAD:$API_PORT"
wait "$SERVER"
