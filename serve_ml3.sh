vllm serve mistralai/mistral-large-3 \
  --tokenizer_mode mistral --config_format mistral \
  --load_format mistral --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt '{"image":10}' \
  --tensor-parallel-size 16 \
  --max_model_len 65536 \
  --max_num_seqs 128 \
  --enforce_eager \
  --data-parallel-size-local 1 \
  --data-parallel-backend=ray \
  --data-parallel-size 2

