#!/bin/bash
tp_parrallel=8
bs=32
in_len=1024
out_len=1024
multi_step=1
total_len=$((in_len + out_len))
VLLM_DECODE_BLOCK_BUCKET_MIN=$((in_len * bs / 128))
VLLM_DECODE_BLOCK_BUCKET_MAX=$((total_len * bs / 128 + 128))

# model="/data/models/DeepSeek-R1/"
# tokenizer="/data/models/DeepSeek-R1/"
model="/software/data/DeepSeek-R1/"
tokenizer="/software/data/DeepSeek-R1/"
model_name="DeepSeek-R1"

HABANA_VISIBLE_DEVICES="ALL" \
VLLM_MOE_N_SLICE=8 \
PT_HPU_ENABLE_LAZY_COLLECTIVES="true" \
VLLM_RAY_DISABLE_LOG_TO_DRIVER="1" \
RAY_IGNORE_UNHANDLED_ERRORS="1" \
VLLM_PROMPT_BS_BUCKET_MIN=1 \
VLLM_PROMPT_BS_BUCKET_MAX=${bs} \
VLLM_PROMPT_SEQ_BUCKET_MIN=${in_len} \
VLLM_PROMPT_SEQ_BUCKET_MAX=${total_len} \
VLLM_DECODE_BS_BUCKET_MIN=${bs} \
VLLM_DECODE_BS_BUCKET_MAX=${bs} \
VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN} \
VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX} \
python -m vllm.entrypoints.openai.api_server \
    --port 8080 \
    --model ${model} \
    --tensor-parallel-size ${tp_parrallel} \
    --max-num-seqs ${bs} \
    --disable-log-requests \
    --dtype bfloat16 \
    --use-v2-block-manager \
    --num_scheduler_steps ${multi_step}\
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --distributed_executor_backend ray \
    --gpu_memory_utilization 0.9 \
    --trust_remote_code 2>&1 | tee benchmark_logs/serving.log &
pid=$(($!-1))

until [[ "$n" -ge 100 ]] || [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Uvicorn running on" benchmark_logs/serving.log; then
        break
    fi
    sleep 5s
done
sleep 5s
echo ${pid}

num_prompts=32
request_rate=1
start_time=$(date +%s)
echo "Start to benchmark"
python benchmarks/benchmark_serving.py --backend vllm --model ${model} --tokenizer ${tokenizer} --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --request-rate ${request_rate} --num-prompts ${num_prompts} --port 8080 --sonnet-input-len ${in_len} --sonnet-output-len ${out_len} --sonnet-prefix-len 100 \
--save-result 2>&1 | tee benchmark_logs/static-online-gaudi3-0.9util-TPparallel${tp_parrallel}-multistep${multi_step}_nprompt${num_prompts}_rrate${request_rate}_bs${bs}_i${in_len}_o${out_len}_prepad.log
end_time=$(date +%s)
echo "Time elapsed: $((end_time - start_time))s"

sleep 10

kill ${pid}
#--backend openai-chat --endpoint "v1/chat/completions"