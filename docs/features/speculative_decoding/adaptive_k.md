# Adaptive K (Online)

Adaptive K is a **zero-configuration online** alternative to Dynamic SD's
static batch-size lookup. Instead of requiring a profiling step, it
automatically selects the optimal number of speculative tokens (`K`) at every
step using per-position conditional acceptance rates and the Leviathan (2023)
cost model.

When enabled, Adaptive K picks the K that maximises:

```
U(K) = E_acc(K) / (K · c_draft + 1)

E_acc(K) = 1 + Σ_{i=1}^{K} Π_{j=1}^{i} α_j
```

where α_j is a per-position exponential moving average (EMA) of the
**conditional** acceptance rate at position j — the ratio of tokens accepted
at j to tokens that *reached* j. This is mathematically correct and avoids
the uniform-alpha assumption that the original Leviathan formula relies upon
(see DISCO / Mamou et al., NeurIPS 2024, §3).

Adaptive K also supports `K=0` (disabling speculation entirely) when utility
falls below 1.0, per the Cascade framework (Saxena et al.,
MLSys 2026, Theorem 4.2).

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
|--------|---------|-------------|
| `enable_adaptive_k` | `False` | Enable online adaptive K selection |
| `adaptive_k_ema_alpha` | `0.3` | EMA smoothing factor (0 = slow, 1 = raw) |
| `adaptive_k_c_draft` | `0.05` | Draft-forward to target-forward cost ratio |
| `adaptive_k_min_tokens` | `0` | Minimum K (0 allows disabling speculation) |
| `adaptive_k_cooldown_steps` | `4` | Steps to wait after a K change |
| `adaptive_k_bs_penalty` | `0.002` | Per-request penalty on cost |

## When to Use Adaptive K vs DSD

- **Adaptive K** — zero profiling required, adapts per-step, works with any
  draft/target pair
- **DSD** (`num_speculative_tokens_per_batch_size`) — requires profiling but
  guarantees floor performance at each batch size

The two can coexist: DSD sets the coarse K range per batch size, Adaptive K
fine-tunes within that range per step.

## Model compatibility

Adaptive K works with all speculative decoding methods supported by vLLM V1
(draft-model, EAGLE, MTP, and n-gram). It requires no new GPU↔CPU sync points
and introduces negligible scheduler overhead (~10 float operations per step).
