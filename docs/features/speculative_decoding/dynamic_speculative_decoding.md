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

## Measuring a table

The K values above are examples, not recommendations: the best K at a given concurrency depends on the model, drafter, GPU and workload, and a table copied from these docs can lose double-digit throughput on a different setup. Measure a table for your own deployment instead:

```bash
vllm bench sweep tune --target <model> --drafter <drafter> \
    --prompts-file prompts.jsonl -o my-tune
```

The tuner takes minutes and needs no code changes. For each draft length K it times the decode step over a grid of batch sizes on the serving path, with every draft token rejected (vLLM's synthetic rejection sampler), so the batch stays fixed while the drafter and verifier do the full work of K tokens. It then measures per-position acceptance on prompts like your traffic (`--prompts-file`, JSONL with one `{"prompt": ...}` per line) and prints the measured `num_speculative_tokens_per_batch_size` ready to paste into `--speculative-config`. See `vllm bench sweep tune --help` for the options (batch grid, prompt length, longest K, tensor parallelism).

K=0 is not measured by default, because a K=0 step still pays a drafter forward pass on current builds; pass `--include-k0` to measure it.

## Keeping the table current (opt-in controller)

Acceptance changes with the traffic, so a measured table can go stale. The controller rebuilds the table while serving: it reads the acceptance counts vLLM already computes, blends them with the tuner's prior, and re-fits the table every few hundred steps. It also holds K steady while the batch hovers near a table edge, so K does not flip every step. It only uses K values that are already in the table, so it never needs new CUDA graph shapes, and changing K never changes the output tokens (verification is lossless).

The controller ships disabled. Nothing changes unless you set `VLLM_DSD_CONTROLLER=1` when serving with a measured table, pointing at the tuner's output:

```bash
VLLM_DSD_CONTROLLER=1 \
VLLM_DSD_COSTS=my-tune/costs_k*.json \
VLLM_DSD_PRIOR=my-tune/acceptance.json \
VLLM_DSD_HYSTERESIS=8 \
VLLM_DSD_REFIT_STEPS=200 \
vllm serve <model> --speculative-config '{ ... the measured table ... }'
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_DSD_CONTROLLER` | off | enables the controller |
| `VLLM_DSD_COSTS` | – | step-cost files from `vllm bench sweep tune` (required for re-fit) |
| `VLLM_DSD_PRIOR` | – | the tuner's `acceptance.json` (required for re-fit) |
| `VLLM_DSD_HYSTERESIS` | 4 | hold K while the batch is within this many requests of a table edge |
| `VLLM_DSD_REFIT_STEPS` | 500 | re-fit the table every this many steps (0 = never, hysteresis only) |
| `VLLM_DSD_PRIOR_WEIGHT` | 2000 | how much the tuner's prior counts, in observations |
| `VLLM_DSD_DECAY` | 0.5 | how fast live acceptance evidence fades between re-fits |
| `VLLM_DSD_LOG_STEPS` | 0 | log controller stats every this many steps |

If re-fit is on and fails three times in a row (for example a malformed cost file), the controller turns re-fitting off and keeps the current table; serving never stops. With only `VLLM_DSD_CONTROLLER=1` and no cost/prior files, the controller does hysteresis only.

## Relation to adaptive verification

Adaptive verification and the Dynamic SD table do different jobs. Adaptive verification decides how many drafted tokens to verify. It does not set how many tokens the engine drafts. The Dynamic SD table sets that number. The two features compose: the table caps drafting at each batch size, and adaptive verification trims verification inside that cap.

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

## Limitations

* Tested with Eagle, Eagle-3, and DFlash. Other SD methods may or may not work out of the box
* Full Cudagraph only works with Model Runner V2. MRv1 only supports piece-wise cuda graph with this feature
* On some builds a table that steps from K=2 to K=1 can stall decode (piece-wise cuda graph capture)
* Not compatible with data parallelism (`--data-parallel-size > 1`). Each DP rank schedules independently, so ranks can pick different K values, causing DP collective divergence and deadlocks. When DP is enabled, vLLM automatically disables `num_speculative_tokens_per_batch_size` and falls back to the static `num_speculative_tokens` value.
