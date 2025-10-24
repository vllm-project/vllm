addr=localhost
port=9000
url=http://${addr}:${port}/v1/completions
model=/data/pretrained-models/amd/DeepSeek-R1-MXFP4-Preview
bs=50
task=gsm8k

echo "url=${url}"
echo "model=${model}"
echo "task=${task}"

lm_eval \
    --model local-completions \
    --tasks ${task} \
    --model_args model=${model},base_url=${url} \
    --batch_size ${bs} \
    --seed 123 \
    2>&1 | tee log.lmeval.log
