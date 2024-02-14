#!/bin/bash
BASE_DIR=/workspace
VLLM_DIR=$BASE_DIR/vllm-private
GRAD_DIR=$BASE_DIR/gradlib
RPD_DIR=/workspace/rocmProfileData
MODEL=/data/llama2-70b-chat
MODEL_SIZE=`echo $MODEL | sed 's/.*\(.[0-9][bB]\).*/\1/'`
#MODEL=/data/llama-2-13b-chat-hf
GEMM_TUNER=1
#TP="1 2 4 8"
TP=8
#Flag to use Triton Flash Attention vs CK
export VLLM_USE_TRITON=1

#Gemm tuner flags
export VLLM_TUNE_GEMM=0
export VLLM_UNTUNE_FILE="/tmp/vllm_untuned.csv"
export VLLM_TUNE_FILE=$VLLM_DIR"/tuned.csv"

#Flag to use old torch.multinomial
#export VLLM_USE_TORCH_MULTINOMIAL=1

#Delete tuned gemms before running.
DELETE_TUNED_CSV=1
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
GEN_LEN="1 32"
#INPUT_LEN="512 1024 2048 3072"
INPUT_LEN="512 1024 2048 3072 4096 6144 8192 16384"
ITER=10
# pring usage of the parameters
usage() {
    echo "Usage: $0 [--tp <n>] [--model <path>]"
    exit 1
}
# parse parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tp) TP="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --notune) GEMM_TUNER=0; shift ;;
        *) usage ;; # Any other argument will show usage information.
    esac
    shift # Move to next argument
done
for tp in $TP;
do
    if (( $GEMM_TUNER ));
    then
      echo "tuned_gemm_csv: ./tuned_tp$tp.csv" > $VLLM_DIR/tuned_perf_tp$tp.yaml
      tuned_file=$VLLM_DIR/tuned_tp$tp.csv
      if [[ $DELETE_TUNED_CSV == 1 || ! -f $VLLM_DIR/tuned_tp$tp.csv ]];
      echo "tuned_gemm_csv: "$VLLM_TUNE_FILE > $VLLM_DIR/tuned_perf.yaml
      if [[ $DELETE_TUNED_CSV == 1 ]];
      then
              rm -rf $tuned_file
              echo "INFO: Generating Tuned Gemm configs"
              cd $GRAD_DIR
              python gemm_tuner.py --model_dir $MODEL --output $tuned_file --tp $tp
      fi
      export VLLM_PERF_YAML=./tuned_perf_tp$tp.yaml
      echo "INFO: Generating Tuned Gemm configs"
      cd $GRAD_DIR
      python gemm_tuner.py --model_dir $MODEL --output $VLLM_TUNE_FILE --tp $tp


      echo "================================= TUNED GEMMS  $tuned_file ==============================================="
      cat $tuned_file

    fi

    cd $VLLM_DIR
    for gen_len in $GEN_LEN;
    do
        for input_len in $INPUT_LEN;
        do
            if [[ -v RPD_PROFILE ]] ;
            then
                rm /workspace/trace.rpd
                python -m rocpd.schema --create /workspace/trace.rpd
            fi
            echo "================================= RUNNING $MODEL $input_len $gen_len ==============================================="
            $ROCPROF_PROFILE torchrun --standalone --nnodes=1 --nproc-per-node=$tp benchmarks/benchmark_latency.py --model $MODEL  --batch-size 1 --input-len $input_len --output-len $gen_len \
            --tensor-parallel-size $tp --num-iters $ITER $HIP_GRAPH $RPD_PROFILE
            if [[ -v ROCPROF_PROFILE ]] ;
            then
                TRACE_FILE=$BASE_DIR/trace_${MODEL_SIZE}_${input_len}_${gen_len}.json
                echo "INFO: Creating Trace JSON file $TRACE_FILE"
                mv $VLLM_DIR/results.json $TRACE_FILE
            fi
            if [[ -v RPD_PROFILE ]] ;
            then
                TRACE_FILE=$BASE_DIR/trace_${MODEL_SIZE}_${input_len}_${gen_len}.json
                echo "INFO: Creating Trace JSON file $TRACE_FILE"
                python $RPD_DIR/tools/rpd2tracing.py --format object $BASE_DIR/trace.rpd $TRACE_FILE
            fi
        done
    done
done