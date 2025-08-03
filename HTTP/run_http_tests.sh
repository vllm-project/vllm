#!/bin/bash

### SBATCH options
#SBATCH --account=hw_nresearch_snoise
#SBATCH --job-name=http_tests
#SBATCH --partition=batch
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=2
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

# Configurables
IMAGE=/lustre/fsw/portfolios/hw/users/sshrestha/nvresearch-vllm.sqsh

# original workspace mount
WORK_MOUNT=/home/sshrestha/workspace/vllm-distributed:/mounted_ws/
CONTAINER_WORKDIR=/mounted_ws  
# mount your home directory
HOME_MOUNT=${HOME}:${HOME}

# mount an additional Lustre filesystem (replace with actual path)
EXTRA_FS_MOUNT=/lustre/fsw/portfolios/hw/users/sshrestha/:/lustre/fsw/portfolios/hw/users/sshrestha/
CONTAINER_MOUNTS="${WORK_MOUNT},${HOME_MOUNT},${EXTRA_FS_MOUNT}"

RESULTS_DIR=/home/sshrestha/workspace/vllm-distributed/HTTP/sbatch_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_NODES_${NODES}
mkdir -p "${RESULTS_DIR}"

SCRIPT="HTTP/test_http.py"

TP_SIZE=8
PP_SIZE=1
DP_SIZE=1
TKNP_SIZE=2
ARGS="--tensor-parallel-size ${TP_SIZE} \
    --pipeline-parallel-size ${PP_SIZE} \
    --data-parallel-size ${DP_SIZE} \
    --token-parallel-size ${TKNP_SIZE}"

INFERENCE_ARGS="--enable-token-parallel"

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
    bash -c "python $SCRIPT $ARGS $INFERENCE_ARGS"
