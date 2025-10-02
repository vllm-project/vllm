export CUDA_LAUNCH_BLOCKING=1
nohup python examples/offline_inference/spec_decode.py \
    --model-dir Qwen/Qwen3-1.7B \
    --draft-model Qwen/Qwen3-1.7B \
    --method draft_model \
    --num_spec_tokens 3 \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --num_prompts 80 \
    --temp 0.0 \
    --gpu-memory-utilization 0.9 \
    --enforce-eager > al.log 2>&1 &