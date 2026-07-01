# Adaptive K (Online)

Adaptive K dynamically selects the optimal number of speculative tokens (`K`)
at every step using per-position conditional acceptance rates and the DSD
goodput formulation. It works alongside DSD's batch-size schedule as a runtime
fine-tuning layer.

## Cost Model

Adaptive K maximises goodput (accepted length / inter-token latency):

```text
ITL(K, BS) = 1  +  K * c_draft  +  verify_overhead
            \_/   \_________/     \_______________/
        target    draft cost       per-step constant
        forward   (K forwards)     (padded to max K)

E_acc(K) = 1 + Sigmai=1..K Pij=1..i alphaj
verify_overhead = bs_penalty * batch_size
```

where:

- alphaj is a per-position EMA of the **conditional** acceptance rate at position j
  (DISCO-correct, avoids uniform-alpha assumption)
- `c_draft` is the profiled ratio of draft-forward time to target-forward time

K=0 disables speculation when goodput falls below 1.0.

## Profiling Requirement

Adaptive K requires `c_draft` to be profiled once on target hardware:

```bash
# Profile using the vLLM benchmark script:
python benchmarks/benchmark_serving.py \
  --model <target-model> \
  --speculative-model <draft-model> \
  --num-prompts 100
# Compute c_draft = draft_step_latency / target_step_latency
# Set this as --speculative-config '{"adaptive_k_c_draft": <ratio>}'
```

The `bs_penalty` parameter corrects for batch-size scaling. The default (0.002)
works across most hardware; tune only if goodput is consistently suboptimal.

The default `c_draft` (0.1144) is profiled for 0.5B->7B AWQ on RTX 4050.
**Always re-profile on your own hardware.**

## Usage

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-model meta-llama/Llama-3.2-3B-Instruct \
  --num-speculative-tokens 10 \
  --enable-adaptive-k
```

Or via `--speculative-config`:

```bash
--speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 10,
    "enable_adaptive_k": true
}'
```

## Configuration Options

| Option | Default | Description |
| ------ | ------- | ----------- |
| `enable_adaptive_k` | `False` | Enable online adaptive K selection |
| `adaptive_k_ema_alpha` | `0.3` | EMA smoothing factor (0 = slow, 1 = raw) |
| `adaptive_k_c_draft` | `0.1144` | Profiled draft:target forward time ratio |
| `adaptive_k_min_tokens` | `0` | Minimum K (0 allows disabling speculation) |
| `adaptive_k_cooldown_steps` | `2` | Steps to wait after a K change |
| `adaptive_k_bs_penalty` | `0.002` | Verification overhead per request at batch size |
| `adaptive_k_alpha_prior` | `0.85` | Prior acceptance rate for untracked positions |

## When to Use Adaptive K vs DSD

- **Adaptive K** -- requires only one profiled `c_draft` value, adapts per-step
  using online acceptance tracking
- **DSD** (`num_speculative_tokens_per_batch_size`) -- requires exhaustive
  per-batch-size profiling but guarantees floor performance at each BS

The two coexist naturally: DSD sets the coarse K range per batch size,
Adaptive K fine-tunes within that range per step using measured acceptance.

## Model compatibility

Adaptive K works with all speculative decoding methods supported by vLLM V1
(draft-model, EAGLE, MTP, and n-gram). It requires no new GPU<->CPU sync points
and introduces negligible scheduler overhead (~10 float operations per step).
