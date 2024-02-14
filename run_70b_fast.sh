#!/bin/bash
set -e
BASE_DIR=/workspace
VLLM_DIR=$BASE_DIR/vllm-private
GRAD_DIR=/trees/gradlib
RPD_DIR=/workspace/rocmProfileData
MODEL=/data/llama2-70b-chat
MODEL_SIZE=`echo $MODEL | sed 's/.*\(.[0-9][bB]\).*/\1/'`

export VLLM_TUNE_GEMM=0
export VLLM_UNTUNE_FILE="/tmp/vllm_untuned.csv"
export VLLM_TUNE_FILE=$VLLM_DIR/"tuned.csv"

#Flag to use Triton Flash Attention vs CK
export VLLM_USE_TRITON=1

#Flag to use old torch.multinomial
#export VLLM_USE_TORCH_MULTINOMIAL=1

#Delete tuned gemms before running.
#DELETE_TUNED_CSV=1

#Flag to disable MSCCL
#export RCCL_MSCCL_ENABLE=0

#HIPGraph performance flags
export HIP_FORCE_DEV_KERNARG=1
export DEBUG_CLR_GRAPH_PACKET_CAPTURE=1

#Enable full decoder graph mode
HIP_GRAPH=--use-cuda-graph

#Use top of tree build of RCCL
export LD_LIBRARY_PATH=/workspace/rccl/build/

#Enable either flag to create a profile trace (rocprof, or rocpd)
#RPD_PROFILE="--profile"
#ROCPROF_PROFILE="rocprof --hip-trace"

#TP="1 2 4 8"
TP=8
GEN_LEN="1,32"
INPUT_LEN="512 1024 2048 3072"
#INPUT_LEN="512,1024,2048,3072,4096,6144,8192,16384"
BATCH_SIZE="1"
ITER=10

rm -f $VLLM_UNTUNE_FILE
for tp in $TP;
do
    cd $VLLM_DIR
    export VLLM_TUNE_GEMM=1
    echo "================================= WARMING UP $MODEL ==============================================="
    $ROCPROF_PROFILE torchrun --standalone --nnodes=1 --nproc-per-node=$tp benchmarks/benchmark_latency.py --model $MODEL  --batch-size $BATCH_SIZE --input-len $INPUT_LEN --output-len $GEN_LEN \
    --tensor-parallel-size $tp --num-iters 1 --warmup-only

    if [ -f $VLLM_UNTUNE_FILE ]; then
        echo "=============================== Tuning ======================================"
        python $GRAD_DIR/gemm_tuner.py --tuned_file $VLLM_TUNE_FILE --input_file $VLLM_UNTUNE_FILE
        echo "File does not exist."
    fi
    echo "================================= TUNED GEMMS  $tuned_file ==============================================="
    cat $VLLM_TUNE_FILE

    export VLLM_TUNE_GEMM=0
    echo "================================= RUNNING $MODEL ==============================================="
    $ROCPROF_PROFILE torchrun --standalone --nnodes=1 --nproc-per-node=$tp benchmarks/benchmark_latency.py --model $MODEL  --batch-size $BATCH_SIZE --input-len $INPUT_LEN --output-len $GEN_LEN \
    --tensor-parallel-size $tp --num-iters $ITER --report --report-file=$VLLM_DIR/report.csv $HIP_GRAPH
done