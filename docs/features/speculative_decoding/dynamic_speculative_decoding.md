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

## Acceptance-adaptive K

The batch-size schedule adapts K to *load*. `adaptive_num_speculative_tokens`
adapts K to *drafter quality*: the scheduler tracks the unconditional acceptance
rate of every draft position from verification results and only drafts the
leading positions whose rate is at least `adaptive_acceptance_threshold`.
Each extra draft position costs one drafter forward pass and returns
`acceptance[pos]` expected tokens, so positions below the threshold are net
negative and are dropped. `num_speculative_tokens` becomes the upper bound.

```bash
--speculative-config '{
    "method": "draft_model",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "num_speculative_tokens": 5,
    "adaptive_num_speculative_tokens": true,
    "adaptive_acceptance_threshold": 0.4,
    "adaptive_min_num_speculative_tokens": 0,
    "adaptive_window_drafts": 256,
    "adaptive_probe_interval": 64
  }'
```

* `adaptive_acceptance_threshold` (default `0.4`): break-even acceptance for a
  position to stay enabled. With a measured per-position curve of
  `[0.70, 0.45, 0.35, 0.25, 0.20]` this yields K=2.
* `adaptive_min_num_speculative_tokens` (default `0`): floor on K. `0` lets the
  controller switch speculation off entirely when even position 0 is below the
  threshold.
* `adaptive_window_drafts` (default `256`): drafts averaged over (EMA decay
  `1/window`) and the warm-up before adaptation starts; until then the static
  `num_speculative_tokens` is used.
* `adaptive_probe_interval` (default `64`): every N scheduler steps the full
  `num_speculative_tokens` is drafted once so positions that were switched off
  keep receiving samples and can be re-enabled when the workload changes. `0`
  disables probing (K can then only shrink).
* `adaptive_hysteresis` (default `0.05`): a switched-off position is re-enabled
  only once its rate reaches `threshold + hysteresis`, so K does not flap when a
  position hovers at the threshold.

K changes are logged at INFO with the per-position acceptance that caused them.

Both mechanisms can be combined; the scheduler then uses
`min(schedule_k, adaptive_k)` — the schedule is the load cap, the controller
the quality cap. CUDA graphs are captured for every K the combination can
produce.

## Limitations

* Tested with Eagle, Eagle-3, and DFlash. Other SD methods may or may not work out of the box
* Full Cudagraph only works with Model Runner V2. MRv1 only supports piece-wise cuda graph with this feature
* Not compatible with data parallelism (`--data-parallel-size > 1`). Each DP rank schedules independently, so ranks can pick different K values, causing DP collective divergence and deadlocks. When DP is enabled, vLLM automatically disables `num_speculative_tokens_per_batch_size` and `adaptive_num_speculative_tokens` and falls back to the static `num_speculative_tokens` value.
