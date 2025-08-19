#!/bin/bash

### SBATCH options
#SBATCH --account=hw_nresearch_snoise
#SBATCH --job-name=vllm_simple_generate
#SBATCH --partition=batch_short
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8


NODES=${SLURM_JOB_NUM_NODES:-1}
echo "Allocated nodes: $SLURM_JOB_NUM_NODES"
# echo "NODES = $NODES"

export ONE_LOGGER_JOB_CATEGORY=test
export PYTHONWARNINGS="ignore"

export HF_HOME=/lustre/fsw/portfolios/hw/users/sshrestha/.cache/huggingface
export HF_HUB_CACHE=/lustre/fsw/portfolios/hw/users/sshrestha/.cache/huggingface/hub
export HF_DATASETS_CACHE=/lustre/fsw/portfolios/hw/users/sshrestha/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/lustre/fsw/portfolios/hw/users/sshrestha/.cache/huggingface/transformers
export PIP_CACHE_DIR=/lustre/fsw/portfolios/hw/users/sshrestha/.cache/pip

# Expert parallel configs
export VLLM_ALL2ALL_BACKEND=deepep_low_latency
export VLLM_USE_DEEP_GEMM=1

# Configurables
# IMAGE=/lustre/fsw/portfolios/hw/users/sshrestha/nvresearch-vllm-v2.sqsh
IMAGE=/lustre/fsw/portfolios/hw/users/sshrestha/vllm+vllm-openai+gptoss.sqsh

# original workspace mount
WORK_MOUNT=/home/sshrestha/workspace/vllm-distributed/vllm-inference-bench:/mounted_ws/
CONTAINER_WORKDIR=/mounted_ws  
# mount your home directory
HOME_MOUNT=${HOME}:${HOME}

# mount an additional Lustre filesystem (replace with actual path)
EXTRA_FS_MOUNT=/lustre/fsw/portfolios/hw/users/sshrestha/:/lustre/fsw/portfolios/hw/users/sshrestha/
CONTAINER_MOUNTS="${WORK_MOUNT},${HOME_MOUNT},${EXTRA_FS_MOUNT}"

RESULTS_DIR=/home/sshrestha/workspace/vllm-distributed/vllm-inference-bench/sbatch_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_NODES_${NODES}
mkdir -p "${RESULTS_DIR}"

# SAVED_RESULTS_DIR=/home/sshrestha/workspace/vllm-distributed/vllm-inference-bench/benchmark_results/
# mkdir -p "${SAVED_RESULTS_DIR}"

SCRIPT="simple_generate.py"

TP_SIZE=8
PP_SIZE=1
DP_SIZE=1
TOKEN_PARALLEL_SIZE=1
# MODEL=deepseek-ai/DeepSeek-V2-Chat-0628
# MODEL="deepseek-ai/DeepSeek-V3-0324"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
# MODEL="meta-llama/Llama-4-Scout-17B-16E-Instruct"

# Arguments for simple_generate.py
ARGS="--tensor-parallel-size ${TP_SIZE} \
    --pipeline-parallel-size ${PP_SIZE} \
    --data-parallel-size ${DP_SIZE} \
    --token-parallel-size ${TOKEN_PARALLEL_SIZE} \
    --enable-expert-parallel \
    --model ${MODEL} \
    --max-model-len 32768 \
    --seed 1"

# Set distributed environment variables
export WORLD_SIZE=$((NODES * 8))  # Total number of processes (nodes * tasks-per-node)
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=12345

# Update the srun command to properly set RANK for each process
srun --mpi=pmix                                                            \
    --nodes=$NODES                                                        \
    --ntasks-per-node=8                                                   \
    --container-image=$IMAGE                                              \
    --container-mounts=$CONTAINER_MOUNTS                                  \
    --container-workdir=$CONTAINER_WORKDIR                                \
    --container-writable                                                  \
    --no-container-mount-home                                             \
    --output="${RESULTS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_NODES_${NODES}.out"      \
    --error="${RESULTS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_NODES_${NODES}.err"      \
    bash -c "export RANK=\$SLURM_PROCID; export LOCAL_RANK=\$SLURM_LOCALID; python3 $SCRIPT $ARGS $INFERENCE_ARGS"
