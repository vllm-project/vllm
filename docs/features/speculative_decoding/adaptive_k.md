# Adaptive K (Online)

Adaptive K dynamically selects the optimal number of speculative tokens (`K`)
at every step using per-position conditional acceptance rates and the DSD
goodput formulation. It works alongside DSD's batch-size schedule as a runtime
fine-tuning layer.

## Cost Model

Adaptive K maximises goodput (accepted length / inter-token latency):

```text
goodput(K) = E_acc(K) / ITL(K, BS)

ITL(K, BS) = K · c_draft · scale  +  1  +  K · c_verify · bs_penalty · BS · scale
            \_________  __________/      \______________  __________________/
                       \/                                  \/
              draft cost (K forwards)          verify cost (target processes K+1 tokens)

E_acc(K) = 1 + Σ_{i=1}^{K} Π_{j=1}^{i} α_j
```

where:

- α_j is a per-position EMA of the **conditional** acceptance rate at position j
  (DISCO-correct, avoids uniform-alpha assumption; Mamou et al., NeurIPS 2024 §3)
- `c_draft` is the profiled ratio of draft-forward time to target-forward time
- `c_verify` captures the additional per-token verification overhead at batch size BS
- `scale` is an online correction factor (measured draft:target step ratio)

K=0 disables speculation when goodput falls below 1.0
(Cascade framework; Saxena et al., MLSys 2026, Theorem 4.2).

## Profiling Requirement

Like DSD, Adaptive K requires `c_draft` to be profiled once:

```bash
# Measure draft and target forward times for your hardware:
# draft_forward_time / target_forward_time → set adaptive_k_c_draft
```

The online `scale` factor and batch-size penalty `bs_penalty` then correct for
runtime conditions, reducing the need for exhaustive offline profiling. The
`bs_penalty` parameter is typically small (0.002) and works well at the default
across hardware.

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
| `adaptive_k_c_draft` | `0.05` | Profiled draft:target forward time ratio |
| `adaptive_k_min_tokens` | `0` | Minimum K (0 allows disabling speculation) |
| `adaptive_k_cooldown_steps` | `4` | Steps to wait after a K change |
| `adaptive_k_bs_penalty` | `0.002` | Verification overhead per request at batch size |

## When to Use Adaptive K vs DSD

- **Adaptive K** — requires only one profiled `c_draft` value, adapts per-step
  using online acceptance tracking
- **DSD** (`num_speculative_tokens_per_batch_size`) — requires exhaustive
  per-batch-size profiling but guarantees floor performance at each BS

The two coexist naturally: DSD sets the coarse K range per batch size,
Adaptive K fine-tunes within that range per step using measured acceptance.

## Model compatibility

Adaptive K works with all speculative decoding methods supported by vLLM V1
(draft-model, EAGLE, MTP, and n-gram). It requires no new GPU↔CPU sync points
and introduces negligible scheduler overhead (~10 float operations per step).
