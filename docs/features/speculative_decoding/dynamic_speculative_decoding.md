# Dynamic Speculative Decoding

## Why is Dynamic SD needed?

SD methods need to verify K tokens for each sequence during decoding. As BS increases, the effective BS becomes BS\*K which increases the compute requirement during verification. When this BS\*K goes beyond a critical BS then SD negatively impacts the decode speed (TPOT). DSD helps by tuning the K to an optimal value such that we continue to reap the benefits from SD.

## Use cases

* Variable concurrency workload using same deployment. K would decrease as concurrency increases.
* During RL rollout where we start off with high BS but then end up with small BS due to very few long tail request which end up generating a lot of tokens stalling the progress of the current rollout. Here K would go up during the end of rollout.

## `--speculative-config` schema

To use Dynamic SD, add `num_speculative_tokens_per_batch_size` to the config of an SD method which is a list of list. Here, an entry is `[start_bs, end_bs, optimal_K]` which means when the concurrency is within range `[start_bs, end_bs]` then `optimal_K` number of draft tokens are used.

Every `optimal_K` must be less than or equal to `num_speculative_tokens`, which defines the maximum draft window.

```bash
--speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

implies that:

* K=3 will be used when the concurrency is in range [1, 64]
* K=1 will be used when the concurrency is in range [65, 128]
* K=0 will be used when the concurrency is in range [129, 512], i.e., no draft tokens will be produced.

## Adaptive verification

Adaptive verification is supported with the Model Runner V2 DSpark implementation. Enable it by setting `adaptive_verification` to `true` in the speculative config.

In this mode, `num_speculative_tokens` is the maximum per-request draft length. The drafter returns `num_speculative_tokens` tokens per request, and the scheduler behaves like ordinary fixed-length speculative decoding, including reserving KV-cache capacity for that full window. The worker uses `num_speculative_tokens_per_batch_size` as a cost model. For `N` decode requests, it selects a total verification budget of `N * optimal_K` and distributes that budget across draft prefixes using the DSpark confidence head. Only the allocated draft positions, plus one bonus position per request, run through the target model. Batches containing prefills use the largest-batch schedule entry as a conservative fixed-K fallback.

## Online Examples

### Adaptive Dynamic SD with DSpark

```bash
vllm serve Qwen/Qwen3-4B-FP8 \
  --trust-remote-code \
  --max-model-len 4096 \
  --attention-backend FLASH_ATTN \
  --speculative-config '{
    "method": "dspark",
    "model": "deepseek-ai/dspark_qwen3_4b_block7",
    "num_speculative_tokens": 11,
    "num_speculative_tokens_per_batch_size": [
      [1, 32, 7],
      [33, 128, 5],
      [129, 512, 3]
    ],
    "adaptive_verification": true,
    "draft_sample_method": "probabilistic"
  }'
```

Here the drafter produces 11 draft tokens for each request, but uses a different total verification budget depending on the batch size.

### Dynamic SD Eagle Drafter

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

### Dynamic SD Eagle3 Drafter

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle3",
    "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 5,
    "num_speculative_tokens_per_batch_size": [
      [1, 16, 5],
      [17, 32, 4],
      [33, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'

```

## Limitations

* Tested with Eagle, Eagle-3, and DFlash. Other SD methods may or may not work out of the box
* Full Cudagraph only works with Model Runner V2. MRv1 only supports piece-wise cuda graph with this feature
* Dynamic SD is not compatible with data parallelism (`--data-parallel-size > 1`). vLLM disables its batch-size schedule and falls back to `num_speculative_tokens`. Adaptive verification instead requires DP, DCP, and PP sizes of one.
