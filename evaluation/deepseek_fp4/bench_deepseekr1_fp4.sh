model=/data/pretrained-models/amd/DeepSeek-R1-MXFP4-Preview
echo "benchmarking $model"

input_len=3584
output_len=1024
concurrency=64
prompts=128

vllm bench serve \
    --host localhost \
    --port 9000 \
    --model ${model} \
    --dataset-name random \
    --random-input-len ${input_len} \
    --random-output-len ${output_len} \
    --max-concurrency ${concurrency} \
    --num-prompts ${prompts} \
    --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos \
    --seed 123 \
    2>&1 | tee log.client.log

# --profile
