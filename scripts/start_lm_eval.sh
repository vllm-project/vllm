export no_proxy="localhost, 127.0.0.1, ::1"
lm_eval --model local-completions \
    --tasks gsm8k \
    --model_args model=/mnt/disk5/hf_models/DeepSeek-R1-BF16,tokenizer_backend=huggingface,base_url=http://localhost:8858/v1/completions \
    --batch_size 4 \
    --log_samples \
    --limit 16 \
    --output_path ./lm_eval_output_gsm8k_bs4 2>&1 | tee benchmark_logs/lm_eval_output_gsm8k_bs4_kvcache.log
