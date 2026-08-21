# Recirculation

!!! warning
    Recirculation support is experimental. Compatible model implementations
    must opt in to the engine capability; the optimized wavefront path has the
    additional restrictions listed below.

[Recirculation](https://arxiv.org/abs/2608.17981) feeds a norm-matched deep
residual-stream activation back into a shallower layer. vLLM returns the logits
from the token block's normal pass, then reruns the layers above the destination
with the mixed residual. The rerun overwrites the block's upper-layer KV cache,
so later tokens attend to the recirculated representations.

Enable fixed recirculation through `--hf-overrides`. For Gemma 3 1B PT, the
paper's perplexity configuration is:

```bash
vllm serve google/gemma-3-1b-pt \
  --hf-overrides '{
    "recirculation_config": {
      "source_layer": 11,
      "destination_layer": 4,
      "alpha": 0.15,
      "ramp_tokens": 10
    }
  }' \
  --long-prefill-token-threshold 1
```

When `beta` is omitted, vLLM uses the convex coefficient
`beta = 1 - alpha`. The ramp scales `alpha` from zero to its configured value
over the first `ramp_tokens` positions and adjusts the convex `beta`
accordingly.

An identity configuration with `alpha = 0` and `beta` omitted or set to `1`
disables Recirculation. It therefore agrees exactly with the baseline and does
not incur a redundant upper-layer pass.

## Model capability

The scheduler, recurrent-state buffers, KV-slot remapping, and CUDA-graph
specialization are engine-level features. Model implementations opt in through
the `SupportsRecirculation` interface and use a shared residual-decoder
execution mixin. This keeps scheduling behavior common while allowing a model
family to override its layer execution with a specialized implementation.

The reviewed text adapters cover the following residual-decoder families:

| Execution | Families |
| --- | --- |
| Serial and wavefront | Gemma 3, Gemma 4, Llama and direct aliases, Llama 4, Mistral, Mixtral, Qwen 2, Qwen 3, Qwen 3 MoE, GLM-4 MoE, MiniMax-M2, MiMo-V2, Step-3.5 |
| Serial only | DeepSeek V2/V3/V3.2, GLM-4.7-Flash, GPT-OSS, MiniMax-M3, Qwen3-Next, Qwen 3.5 dense and MoE |

Serial-only status is deliberate when the architecture uses an attention
backend that cannot consume the engine's two-token FlashAttention wavefront,
or a recurrent state that needs a pre-block snapshot. Qwen3-Next and Qwen 3.5
snapshot the active convolution and GDN state slots before the normal upper
stack, then restore those slots before the recirculated rerun. This prevents a
second in-place recurrent update from incorrectly consuming the current
token's already-updated state.

An unreviewed subclass does not inherit support automatically and fails during
model loading when Recirculation is requested. DeepSeek-V4 is intentionally
not opted in: its multi-stream hyperconnection state does not expose the
standard residual boundary assumed by this algorithm. Diffusion and other
non-causal models are also rejected.

## Wavefront execution

Set `"wavefront": true` to execute exact tokenwise Recirculation as a
two-token wavefront. After the first-token warmup, the layers above the
destination process the previous token's recurrent state and the current
token's normal state in one layer call. Each upper attention layer first
overwrites the previous token's KV entry, so the current token attends to the
same recurrent cache that the serial implementation would have produced.

```bash
vllm serve google/gemma-3-1b-pt \
  --hf-overrides '{
    "recirculation_config": {
      "source_layer": 11,
      "destination_layer": 4,
      "alpha": 0.15,
      "ramp_tokens": 10,
      "wavefront": true
    }
  }' \
  --max-num-seqs 1 \
  --long-prefill-token-threshold 1 \
  --no-enable-prefix-caching
```

Wavefront mode captures a dedicated one-token CUDA graph whose upper stack has
the internal two-token batch. Torch compilation remains enabled unless
`--enforce-eager` is also set.

The paper reports the following fixed configurations for its pretrained-model
perplexity evaluation:

| Model | Source | Destination | Alpha | Beta | Ramp tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 3 1B PT | 11 | 4 | 0.15 | `1 - alpha` | 10 |
| Gemma 3 4B PT | 18 | 9 | 0.15 | 1.0 | 0 |
| Gemma 3 12B PT | 35 | 16 | 0.15 | 1.0 | 0 |

Set `"beta": 1.0` explicitly for the non-convex 4B and 12B configurations.

## Exact and blockwise execution

Each forward call recirculates the scheduled tokens as one block. A one-token
block implements the paper's tokenwise recurrence. Larger prefill chunks use
the blockwise approximation proposed in the paper: logits within a block come
from the normal pass, and the recirculated cache affects later blocks.

For exact tokenwise evaluation, set `--long-prefill-token-threshold 1` and do
not enable speculative decoding. For throughput experiments, increase the
threshold to sweep the block size and measure the quality-throughput tradeoff.

## No-download adapter validation

`benchmarks/validate_recirculation_adapters.py` creates a four-layer local
configuration, initializes dummy weights, skips tokenizer initialization, and
runs token-ID generation through the real GPU engine. It does not download
model weights or tokenizers. Run each family in a fresh process, for example:

```bash
.venv/bin/python benchmarks/validate_recirculation_adapters.py \
  --family qwen3.5-moe --mode serial \
  --output results/qwen3.5-moe-recirculation.json

.venv/bin/python benchmarks/validate_recirculation_adapters.py \
  --family gemma4 --mode wavefront \
  --output results/gemma4-recirculation.json

.venv/bin/python benchmarks/validate_recirculation_adapters.py \
  --family qwen3 --mode wavefront --compile \
  --output results/qwen3-compiled-recirculation.json
```

This is an integration smoke for adapter signatures, attention/MoE kernels,
cache allocation, recurrent-state handling, and generation. Random dummy
weights cannot establish model quality; use the perplexity harness with real
weights when storage and hardware permit.

The validator accepts `baseline`, `no-op`, `serial`, and `wavefront` modes.
Use the same `--seed` across separate processes to compare generated token IDs
and selected-token log probabilities. Pass `--ramp-tokens 10` to exercise
position-dependent mixing, including MRoPE text positions.

## RTX 3080 validation snapshot

The following results were collected on an NVIDIA GeForce RTX 3080 10 GB after
rebasing onto `main` on August 20, 2026. The dummy-model runs use four layers,
BF16, random weights, seed 1234, eight decode tokens, and no downloaded weights
or tokenizer. Their elapsed times include process startup and JIT work, so they
are compatibility evidence rather than performance measurements.

For the compiled Gemma 3 dummy model, no-op output was bit-identical to the
baseline. Serial and wavefront generated the same token IDs; their maximum
selected-token log-probability difference was `1.235e-4`, consistent with BF16
execution-order differences.

| Mode | Token IDs | Log-probability sum | Peak PyTorch GPU MiB | Elapsed s |
| --- | --- | ---: | ---: | ---: |
| Baseline | 8 x 227 | -44.174433 | 145.590 | 25.386 |
| No-op | 8 x 227 | -44.174433 | 145.590 | 6.997 |
| Serial | 8 x 227 | -44.168809 | 145.594 | 12.571 |
| Wavefront | 8 x 227 | -44.169186 | 145.584 | 15.139 |

Every documented adapter also completed generation through the real GPU
engine. `ramp=10` was used for both MRoPE Qwen 3.5 adapters.

| Family | Mode | Ramp | Result |
| --- | --- | ---: | --- |
| DeepSeek V3 | Serial | 0 | Pass |
| Gemma 3 | Wavefront, compiled | 0 | Pass |
| Gemma 4 | Wavefront | 0 | Pass |
| GLM-4 MoE | Wavefront | 0 | Pass |
| GLM-4 MoE Lite | Serial | 0 | Pass |
| GPT-OSS | Serial | 0 | Pass |
| Llama | Wavefront | 0 | Pass |
| Llama 4 | Wavefront | 0 | Pass |
| MiMo-V2 | Wavefront | 0 | Pass |
| MiniMax-M2 | Wavefront | 0 | Pass |
| MiniMax-M3 | Serial | 0 | Pass |
| Mistral | Wavefront | 0 | Pass |
| Mixtral | Wavefront | 0 | Pass |
| Qwen 2 | Wavefront | 0 | Pass |
| Qwen 3 | Wavefront | 0 | Pass |
| Qwen 3 MoE | Wavefront | 0 | Pass |
| Qwen3-Next | Serial | 0 | Pass |
| Qwen 3.5 | Serial | 10 | Pass |
| Qwen 3.5 MoE | Serial | 10 | Pass |
| Step-3.5 | Wavefront | 0 | Pass |

MiniMax-M3 sparse attention was disabled in its tiny fixture, so this validates
its residual/MoE adapter but not that optional custom sparse-attention kernel.

A real-weight, compiled Gemma 3 1B PT comparison used the same 16 fixed C4
windows (16,368 scored tokens) for both modes:

| Mode | Perplexity | Mean prefill latency s | Decode tokens/s | Peak GPU delta GiB |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 19.1576 | 7.9496 | 194.97 | 7.661 |
| Wavefront | 16.9613 | 8.4007 | 181.90 | 7.593 |

For this small sample, wavefront reduced perplexity by 11.46%, increased mean
prefill latency by 5.67%, and reduced decode throughput by 6.70%. Treat the
quality delta as promising validation rather than a statistically complete
model evaluation.

## Reproducible evaluation

`benchmarks/benchmark_recirculation.py` scores deterministic, pre-tokenized
1024-token C4 windows. It excludes the first token in each independent window,
reports token-weighted negative log-likelihood and perplexity, and records
latency, decode throughput, peak GPU memory, revisions, and window hashes.

Install the optional dataset dependency, then reuse the same window file for
both quality runs:

```bash
uv pip install 'datasets>=3.3.0,<=3.6.0'

.venv/bin/python benchmarks/benchmark_recirculation.py \
  --mode baseline \
  --windows-file results/recirculation-windows.json \
  --output results/baseline-exact.json

.venv/bin/python benchmarks/benchmark_recirculation.py \
  --mode recirculation \
  --wavefront \
  --windows-file results/recirculation-windows.json \
  --output results/recirculation-wavefront.json
```

The harness keeps baseline and Recirculation on Model Runner V1 for a fair
comparison. Peak memory uses `nvidia-ml-py` when available and otherwise falls
back to PyTorch accelerator peak-memory statistics.

For the paper's Gemma 3 4B configuration, select the model explicitly and pass
the non-convex beta. Revisions are model-specific; omit `--model-revision` to
use the repository default or provide a checkpoint revision explicitly.

```bash
.venv/bin/python benchmarks/benchmark_recirculation.py \
  --mode recirculation \
  --model google/gemma-3-4b-pt \
  --source-layer 18 --destination-layer 9 \
  --alpha 0.15 --beta 1.0 --ramp-tokens 0 \
  --windows-file results/gemma-3-4b-windows.json \
  --output results/gemma-3-4b-recirculation.json
```

Use `--num-windows 50` for a larger sample. For a normal-scheduler baseline
performance measurement, add `--long-prefill-token-threshold 0
--performance-only`. The exact runs default to one sequence, compiled BF16,
`max_model_len=1024`, no prefix caching, and no speculative decoding. Add
`--enforce-eager` only when debugging; it disables torch.compile and CUDA graphs
and can be substantially slower.

For faster approximate prefill, increase `--long-prefill-token-threshold` to
the desired Recirculation block size. For example:

```bash
.venv/bin/python benchmarks/benchmark_recirculation.py \
  --mode recirculation \
  --long-prefill-token-threshold 16 \
  --windows-file results/recirculation-windows.json \
  --output results/recirculation-block16.json
```

A threshold of `1` is exact tokenwise Recirculation. Values greater than `1`
process that many prefill tokens together: the first-pass logits within each
block do not depend on that block's recirculated cache, so this is an explicit
quality-throughput tradeoff rather than an exact optimization. Autoregressive
decode remains tokenwise. Sweep block sizes on the target workload instead of
assuming one value is universally optimal.

## Current restrictions

- Multimodal wrappers, DeepSeek-V4 hyperconnections, and non-causal diffusion
  backbones are not supported.
- Pipeline parallelism is not supported.
- Model Runner V2 does not yet implement Recirculation. Recirculation selects
  Model Runner V1 automatically; explicitly setting
  `VLLM_USE_V2_MODEL_RUNNER=1` is rejected.
- Only fixed scalar coefficients and source norm matching are implemented.
- All Recirculation execution rejects speculative decoding, EAGLE/DFlash
  auxiliary hidden-state extraction, and other speculative configurations.
  Defining recurrence over draft-token blocks and auxiliary states requires a
  separate design. Qwen3-Next and Qwen 3.5 additionally reject
  sequence-parallel MoE execution; their recurrent-state adapter is serial
  only.
- MiniMax-M3 currently requires tensor-parallel size 1. Its sparse-attention
  backend is serial only.
- Gemma 4 YOCO fast prefill is unsupported. Gemma 4 per-layer embeddings are
  serial only because the engine does not retain the previous token's
  per-layer input stack for a wavefront.
- Wavefront execution currently requires one sequence, one scheduled token per
  step, FlashAttention, no prefix caching or speculative decoding, and no data,
  decode-context, sequence, or pipeline parallelism.
- Serial execution remains available when `"wavefront"` is omitted or false.
