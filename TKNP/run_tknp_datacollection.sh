#!/bin/bash

### SBATCH options
#SBATCH --account=hw_nresearch_snoise
#SBATCH --job-name=hamp_datacollection
#SBATCH --partition=batch_short
#SBATCH --time=01:40:00
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

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR=/home/sshrestha/workspace/HybridTensor/HAMP/sbatch_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_NODES_${NODES}_${TIMESTAMP}
BENCHMARK_RESULTS_FILE=/home/sshrestha/workspace/HybridTensor/HAMP/results/distributed_attention_benchmark_results.csv
mkdir -p "${RESULTS_DIR}"
mkdir -p "$(dirname "${BENCHMARK_RESULTS_FILE}")"

# SCRIPT="/home/sshrestha/workspace/Megatron-LM/tensor_broadcast.py"
# SCRIPT="/home/sshrestha/workspace/HybridTensor/HAMP/process_group_test.py"
SCRIPT="HAMP.test_hamp_attention"

# Parameter arrays
PP_SIZE=1

TP_SIZES=(1 2 4 8)
BATCH_SIZES=(64 128 256 512 1024)
SEQ_LENS=(8192 16384 32768)

# Counter for tracking progress
total_runs=0
completed_runs=0

# Calculate total number of runs
for tp_size in "${TP_SIZES[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        for seq_len in "${SEQ_LENS[@]}"; do
            ((total_runs++))
        done
    done
done

echo "Starting benchmark suite with $total_runs total runs"
echo "TP_SIZES: ${TP_SIZES[*]}"
echo "PP_SIZE: $PP_SIZE"
echo "BATCH_SIZES: ${BATCH_SIZES[*]}"
echo "SEQ_LENS: ${SEQ_LENS[*]}"
echo "Results will be saved to: ${BENCHMARK_RESULTS_FILE}"

# Run benchmarks for all parameter combinations
for tp_size in "${TP_SIZES[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        for seq_len in "${SEQ_LENS[@]}"; do
            ((completed_runs++))
            
            echo "=== Run $completed_runs/$total_runs ==="
            echo "TP_SIZE: $tp_size, PP_SIZE: $PP_SIZE, BATCH_SIZE: $batch_size, SEQ_LEN: $seq_len"
            
            ARGS="--tensor_model_parallel_size ${tp_size} \
                --pipeline_model_parallel_size ${PP_SIZE} \
                --benchmark"

            INFERENCE_ARGS="--hidden_size 8192 \
                            --batch_size ${batch_size} \
                            --seq_len ${seq_len} \
                            --results_file ${BENCHMARK_RESULTS_FILE} \
                            --num_nodes ${NODES}"

            # Run the benchmark
            srun --mpi=pmix                                                            \
                --nodes=$NODES                                                        \
                --ntasks-per-node=${tp_size}                                          \
                --container-image=$IMAGE                                              \
                --container-mounts=$CONTAINER_MOUNTS                                  \
                --container-workdir=$CONTAINER_WORKDIR                                \
                --container-writable                                                  \
                --no-container-mount-home                                             \
                --output="${RESULTS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_NODES_${NODES}_tp${tp_size}_bs${batch_size}_seq${seq_len}.out"      \
                --error="${RESULTS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_NODES_${NODES}_tp${tp_size}_bs${batch_size}_seq${seq_len}.err"      \
                bash -c "python -m $SCRIPT $ARGS $INFERENCE_ARGS"
            
            # Check if the run was successful
            if [ $? -eq 0 ]; then
                echo "✓ Run $completed_runs/$total_runs completed successfully"
            else
                echo "✗ Run $completed_runs/$total_runs failed"
            fi
            
            echo ""
        done
    done
done

echo "Benchmark suite completed! All $total_runs runs finished."
echo "Results saved to: ${BENCHMARK_RESULTS_FILE}" 