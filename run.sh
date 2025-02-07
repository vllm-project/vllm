# Run config for DeepSeek-R1 on a single 8xH200 node
# Using one MTP module for speculative execution,
# Called recursively for k=2 speculative tokens.
# Expected draft acceptance rate is ~70%
# (~80% for token 1, ~60% for token 2 due to accuracy decay)
python3 \
  -m vllm.entrypoints.openai.api_server \
  --disable-log-requests \
  --gpu-memory-utilization 0.85 \
  --quantization fp8 \
  --max-model-len 65536 \
  --max-num-seqs 128 \
  --seed 0 \
  --tensor-parallel-size 8 \
  --swap-space 0 \
  --block-size 32 \
  --model deepseek-ai/DeepSeek-R1 \
  --distributed-executor-backend=mp \
  --trust-remote-code \
  --num-speculative-tokens 2 \
  --speculative-model DeepSeekV3MTP
