# Dynamic Speculative Decoding

## Why is Dynamic SD needed?

SD methods need to verify K tokens for each sequence during decoding. As BS increases, the effective BS becomes BS\*K which increases the compute requirement during verification. When this BS\*K goes beyond a critical BS then SD negatively impacts the decode speed (TPOT). DSD helps by tuning the K to an optimal value such that we continue to reap the benefits from SD.

## Use cases

* Variable concurrency workload using same deployment. K would decrease as concurrency increases.
* During RL rollout where we start off with high BS but then end up with small BS due to very few long tail request which end up generating a lot of tokens stalling the progress of the current rollout. Here K would go up during the end of rollout.

## `--speculative-config` schema

To use Dynamic SD, add `num_speculative_tokens_per_batch_size` to the config of an SD method which is a list of list. Here, an entry is `[start_bs, end_bs, optimal_K]` which means when the concurrency is within range `[start_bs, end_bs]` then `optimal_K` number of draft tokens are used. For e.g.,

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

## Online Examples

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
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 16, 5],
      [17, 32, 4],
      [33, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'

```

### Dynamic SD PARD Drafter

Dynamic SD works with [PARD](../parallel_draft_model.md) using
`method: "draft_model"` and `parallel_drafting: true`. Set
`num_speculative_tokens` to the maximum K and use
`num_speculative_tokens_per_batch_size` to pick a lower runtime K at higher
concurrency.

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve Qwen/Qwen3-1.7B \
  --speculative-config '{
    "method": "draft_model",
    "model": "amd/PARD-Qwen3-0.6B",
    "num_speculative_tokens": 3,
    "parallel_drafting": true,
    "num_speculative_tokens_per_batch_size": [
      [1, 32, 3],
      [33, 100, 1]
    ]
  }'
```

### Dynamic SD P-Eagle Drafter

Dynamic SD also works with P-Eagle (parallel EAGLE drafting) using
`method: "eagle"` or `"eagle3"` and `parallel_drafting: true`. The draft
model must expose `ptd_token_id` in its Hugging Face config.

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "parallel_drafting": true,
    "num_speculative_tokens_per_batch_size": [
      [1, 32, 3],
      [33, 100, 1]
    ]
  }'
```

## Limitations

* tested with Eagle, Eagle-3, PARD (`draft_model` with `parallel_drafting:
  true`), and P-Eagle (`eagle` / `eagle3` with `parallel_drafting: true`).
  Other SD methods may or may not work out of the box
* only usable with Model Runner V1
* not compatible with full cuda graph so we force piece-wise cuda graph with this feature

We are working on enabling it on MRv2 with full cuda graph support.
