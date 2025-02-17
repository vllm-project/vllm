#!/bin/bash
tp_parrallel=16
in_len=1024
out_len=1024
multi_step=1
total_len=$((in_len + out_len))
ep_size=16
moe_n_slice=8
gpu_utils=0.98
bs=128
num_prompts=300
request_rate=inf
ratio=1.0
log_name="static-online-gaudi3-${gpu_utils}util-TPparallel${tp_parrallel}-EP${ep_size}-loop${moe_n_slice}moegroups-multistep${multi_step}_nprompt${num_prompts}_rrate${request_rate}_bs${bs}_i${in_len}_o${out_len}"

# model="/data/models/DeepSeek-R1/"
# tokenizer="/data/models/DeepSeek-R1/"
#model="/data/DeepSeek-R1-G2/"
#tokenizer="/data/DeepSeek-R1-G2/"
model="/data/DeepSeek-R1-Dynamic-full-FP8"
tokenizer="/data/DeepSeek-R1-Dynamic-full-FP8"
model_name="DeepSeek-R1"

python -m vllm.entrypoints.openai.api_server \
    --port 8081 \
    --model ${model} \
    --tensor-parallel-size ${tp_parrallel} \
    --max-num-seqs ${bs} \
    --disable-log-requests \
    --dtype bfloat16 \
    --use-v2-block-manager \
    --num_scheduler_steps ${multi_step}\
    --max-model-len 2048 \
    --distributed_executor_backend ray \
    --gpu_memory_utilization ${gpu_utils} \
    --trust_remote_code 2>&1 | tee benchmark_logs/${log_name}_serving.log &
pid=$(($!-1))

until [[ "$n" -ge 100 ]] || [[ $ready == true ]]; do
    n=$((n+1))
    if grep -q "Uvicorn running on" benchmark_logs/${log_name}_serving.log; then
        break
    fi
    sleep 5s
done
sleep 10s
echo ${pid}

hl-smi -l > benchmark_logs/${log_name}_smi.log &
hl_pid=$(($!-1))


start_time=$(date +%s)
echo "Start to benchmark"
python benchmarks/benchmark_serving.py --backend vllm --model ${model} --tokenizer ${tokenizer} --dataset-name sonnet --dataset-path benchmarks/sonnet.txt --request-rate ${request_rate} --max-concurrency ${bs} --num-prompts ${num_prompts} --port 8081 --sonnet-input-len ${in_len} --sonnet-output-len ${out_len} --sonnet-prefix-len 100 --random-range-ratio ${ratio} 2>&1 | tee benchmark_logs/${log_name}_run1.log
end_time=$(date +%s)
echo "Time elapsed: $((end_time - start_time))s"

sleep 10

kill ${pid}
kill ${hl_pid}
#--backend openai-chat --endpoint "v1/chat/completions"
