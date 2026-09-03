# MRCR Long-Context Accuracy Evaluation

Smoke test for long-context behavior using OpenAI's public [`openai/mrcr`](https://huggingface.co/datasets/openai/mrcr) dataset. The model sees a long chat with several near-duplicate "needles" and must reproduce a specific earlier assistant turn verbatim, prepended with a random anti-guessing string.

**Scoring:** if the response doesn't start with `random_string_to_prepend`, score is 0; otherwise the prefix is stripped and the mean `SequenceMatcher.ratio()` against the reference answer is reported.

## Usage

```bash
# Pytest (spawns the server)
pytest -s -v tests/evals/mrcr/test_mrcr_correctness.py \
    --config-list-file=configs/models-small.txt

# Standalone (server already running; model and context auto-discovered)
vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3 --port 8000
python tests/evals/mrcr/mrcr_eval.py --port 8000
```

## Configuration

```yaml
model_name: "Qwen/Qwen3-0.6B"
# Per-needle thresholds catch bucket-specific regressions (sliding window,
# chunked prefill, prefix cache) that an aggregate can hide. A scalar
# (e.g. `match_ratio_threshold: 0.20`) is also accepted and checked against
# the mean match ratio.
match_ratio_threshold:
  2: 0.30
  4: 0.15
  8: 0.10
num_samples: 30
needles: [2, 4, 8]
# max_prompt_tokens: 32768       # Optional; defaults to server max_model_len - max_tokens - 256
max_tokens: 2048
concurrency: 8
server_args: "--max-model-len 32768 --reasoning-parser qwen3"
```

## Notes

- Samples stream from three parquet shards (`{N}needle/{N}needle_0.parquet`); only the first few row groups are fetched, not the full 1.4 GB repo.
- `max_prompt_tokens` defaults to `max_model_len - max_tokens - 256`, i.e. fills whatever context the server advertises. Set `--max-model-len` on the server to control the smoke-test context length; override `--max-prompt-tokens` on the client to cap below that.
- Sample length is pre-filtered by `n_chars × 4 ≤ max_prompt_tokens`, then verified via the server's `/tokenize` endpoint under the actual chat template.
- Reasoning models: start the server with `--reasoning-parser <name>` (e.g. `qwen3`, `deepseek_r1`) so `<think>` goes to `message.reasoning_content` and doesn't contaminate the scored answer.
