#!/bin/bash

### SBATCH options
#SBATCH --account=hw_nresearch_snoise
#SBATCH --job-name=tp_flashattention
#SBATCH --partition=batch
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

# Configurables
IMAGE=/lustre/fsw/portfolios/hw/users/sshrestha/nemo-container.sqsh

# original workspace mount
# WORK_MOUNT=/home/sshrestha/workspace/comms:/mounted_ws
# WORK_MOUNT=/home/sshrestha/workspace/HybridTensor:/mounted_ws/HybridTensor
WORK_MOUNT=/home/sshrestha/workspace/HybridTensor:/mounted_ws/
WORK_MOUNT=/home/sshrestha/workspace/HybridTensor:/mounted_ws  
CONTAINER_WORKDIR=/mounted_ws  
# mount your home directory
HOME_MOUNT=${HOME}:${HOME}

# mount an additional Lustre filesystem (replace with actual path)
EXTRA_FS_MOUNT=/lustre/fsw/portfolios/hw/users/sshrestha/:/lustre/fsw/portfolios/hw/users/sshrestha/
CONTAINER_MOUNTS="${WORK_MOUNT},${HOME_MOUNT},${EXTRA_FS_MOUNT}"

RESULTS_DIR=/home/sshrestha/workspace/HybridTensor/HAMP/sbatch_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_NODES_${NODES}
mkdir -p "${RESULTS_DIR}"


SCRIPT="HAMP.tensorparallel_attention"

INFERENCE_ARGS="--hidden_size 8192 \
                --batch_size 256 \
                --seq_len 8192"

srun --mpi=pmix                                                            \
    --nodes=$NODES                                                        \
    --ntasks-per-node=8                                                   \
    --container-image=$IMAGE                                              \
    --container-mounts=$CONTAINER_MOUNTS                                  \
    --container-workdir=$CONTAINER_WORKDIR                                \
    --container-writable                                                  \
    --no-container-mount-home                                             \
    --output="${RESULTS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"      \
    --error="${RESULTS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"      \
    bash -c "python -m $SCRIPT $INFERENCE_ARGS"
