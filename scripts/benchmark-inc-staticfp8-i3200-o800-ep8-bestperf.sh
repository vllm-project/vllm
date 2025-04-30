#!/bin/bash
tp_parrallel=8
in_len=3200
out_len=800
multi_step=1
total_len=$((in_len + out_len))
# if total_len is not multiple of 128, round up to the next multiple of 128
if [ $((total_len % 128)) -ne 0 ]; then
    echo 'round up for 128'
    total_len=$(((total_len / 128 +  1) * 128 ))
fi
ep_size=8
moe_n_slice=1
gpu_utils=0.98
bs=256
num_prompts=256
request_rate=128
log_name="[inc-staticquant-scalar-fp8matmul-split]online-gaudi3-${gpu_utils}util-TPparallel${tp_parrallel}-EP${ep_size}-loop${moe_n_slice}moegroups-multistep${multi_step}_nprompt${num_prompts}_rrate${request_rate}_bs${bs}_i${in_len}_o${out_len}_mdllen${total_len}"

in_len_aligned=$((in_len + 127 / 128 * 128))
total_len_aligend=$((total_len + 127 / 128 * 128))
VLLM_DECODE_BLOCK_BUCKET_MIN=$((in_len_aligned * bs / 128))
VLLM_DECODE_BLOCK_BUCKET_MIN=$(((VLLM_DECODE_BLOCK_BUCKET_MIN + 127) / 128 * 128))
VLLM_DECODE_BLOCK_BUCKET_MAX=$((total_len_aligend * bs / 128))
VLLM_DECODE_BLOCK_BUCKET_MAX=$(((VLLM_DECODE_BLOCK_BUCKET_MAX + 127) / 128 * 128))

# model="/data/models/DeepSeek-R1-static/"
# tokenizer="/data/models/DeepSeek-R1-static/"
model="/data/models/DeepSeek-R1/"
tokenizer="/data/models/DeepSeek-R1/"
model_name="DeepSeek-R1"


QUANT_CONFIG="scripts/quant_configs/inc_quant_with_fp8kv_config.json" \
VLLM_REQUANT_FP8_INC=1 \
VLLM_USE_FP8_MATMUL=true \
VLLM_ENABLE_RUNTIME_DEQUANT=1 \
VLLM_GRAPH_RESERVED_MEM=0.05 \
VLLM_DELAYED_SAMPLING=true \
HABANA_VISIBLE_DEVICES="ALL" \
VLLM_MOE_N_SLICE=${moe_n_slice} \
VLLM_EP_SIZE=${ep_size} \
VLLM_MLA_DISABLE_REQUANTIZATION=1 \
PT_HPU_ENABLE_LAZY_COLLECTIVES=true \
PT_HPU_WEIGHT_SHARING=0 \
VLLM_PROMPT_BS_BUCKET_MIN=1 \
VLLM_PROMPT_BS_BUCKET_MAX=16 \
VLLM_PROMPT_SEQ_BUCKET_MIN=${in_len_aligned} \
VLLM_PROMPT_SEQ_BUCKET_MAX=${in_len_aligned} \
VLLM_DECODE_BS_BUCKET_MIN=${bs} \
VLLM_DECODE_BS_BUCKET_MAX=${bs} \
VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN} \
VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX} \
python -m vllm.entrypoints.openai.api_server \
    --port 18080 \
    --model ${model} \
    --tensor-parallel-size ${tp_parrallel} \
    --max-num-seqs ${bs} \
    --disable-log-requests \
    --dtype bfloat16 \
    --use-v2-block-manager \
    --num_scheduler_steps ${multi_step}\
    --max-model-len 12800 \
    --max-num-batched-tokens 12800\
    --distributed_executor_backend mp \
    --gpu_memory_utilization ${gpu_utils} \
    --kv_cache_dtype "fp8_inc" \
    --trust_remote_code 2>&1 | tee benchmark_logs/${log_name}_serving.log &
pid=$(($!-1))

until [[ "$n" -ge 1000 ]] || [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Started server process" benchmark_logs/${log_name}_serving.log; then
        break
    fi
    sleep 5s
done
sleep 10s
echo ${pid}


start_time=$(date +%s)
echo "Start to benchmark"
python benchmarks/benchmark_serving.py --backend vllm --model ${model} --tokenizer ${tokenizer} --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --request-rate ${request_rate}  --ignore-eos --num-prompts ${num_prompts} --port 18080 --sonnet-input-len ${in_len} --sonnet-output-len ${out_len} --sonnet-prefix-len 100 2>&1 | tee benchmark_logs/${log_name}_run1.log
end_time=$(date +%s)
echo "Time elapsed: $((end_time - start_time))s"

sleep 10

kill ${pid}