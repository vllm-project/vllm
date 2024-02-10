#!/bin/bash
BASE_DIR=/trees/
VLLM_DIR=$BASE_DIR/vllm
GRAD_DIR=$BASE_DIR/gradlib
RPD_DIR=/workspace/rocmProfileData
MODEL=/data/llama2-70b-chat
#MODEL=/data/Llama-2-13B-Chat-fp16
#MODEL=/data/llama-2-13b-chat-hf
MODEL_SIZE=`echo $MODEL | sed 's/.*\(.[0-9][bB]\).*/\1/'`

GEN_LEN="8"
TP=8
INPUT_LEN=2048
ITER=1
cd $VLLM_DIR

    echo "tuned_gemm_csv: ./tuned_tp$TP.csv" > $VLLM_DIR/tuned_perf_tp$TP.yaml
    tuned_file=$VLLM_DIR/tuned_tp$TP.csv
export VLLM_PERF_YAML=./tuned_perf_tp$TP.yaml

for tp in $TP;
do
	for gen_len in $GEN_LEN;
	do
		for input_len in $INPUT_LEN;
		do

python benchmarks/benchmark_latency.py --model $MODEL  --batch-size 1    --input-len $input_len --output-len $gen_len \
		            --tensor-parallel-size $tp --num-iters $ITER
    done
done
done
