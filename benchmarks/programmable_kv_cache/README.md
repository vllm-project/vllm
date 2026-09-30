# Programmable KV cache validation

`evaluate.py` establishes a model baseline using downloaded GSM8K data. It runs
the same seeded questions with forced prefix recomputation and with prefix-cache
reads enabled, recording per-question outputs, numeric correctness, cache hits,
truncation and output differences. The shared few-shot prefix must reach the
model's actual cache-group boundaries.

```bash
CUDA_VISIBLE_DEVICES=0,3 OMP_NUM_THREADS=4 \
VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python benchmarks/programmable_kv_cache/evaluate.py \
  --model /home/youwei/bzh/model/Qwen/Qwen3.8-27B \
  --dataset /home/youwei/bzh/dataset/openai/gsm8k \
  --num-samples 128 --num-fewshot 8 --max-tokens 1024 \
  --tensor-parallel-size 2 --kv-cache-memory-bytes 4294967296 \
  --output /tmp/programmable-kv-gsm8k-baseline-results.json
```

The local baseline uses model revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and dataset revision
`740312add88f781978c0658806c59bc2815b9866`. Model and data preparation used the
user-supplied download functions. Other environments can pass their own local
paths.

## Recorded local baseline

The compressed JSON under `results/` contains the configuration and all 256
generation records. Both modes score 125/128 (97.66%); neither truncates an answer.
Forced recomputation has zero cached tokens; the cache-enabled run has 99,568.
Final numeric answers match on all questions, while token sequences differ on 27.

This run leaves the new programmable controller disconnected from serving.
It validates the existing model/cache baseline and does not demonstrate native
hint execution, backend transfers, policy benefits or a serving speedup.
The complete serving/backend/policy matrix remains in the OpenSpec task list.
