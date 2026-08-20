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

The reviewed adapters currently cover text-only Gemma 3, the native Llama
implementation and its direct architecture aliases, and Mistral. An
unreviewed subclass does not inherit support automatically and fails during
model loading when Recirculation is requested.

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

- Reviewed adapters currently cover text-only Gemma 3, Llama, and Mistral
  implementations. Multimodal wrappers are not yet supported.
- Pipeline parallelism is not supported.
- Only fixed scalar coefficients and source norm matching are implemented.
- Wavefront execution currently requires one sequence, one scheduled token per
  step, FlashAttention, no prefix caching or speculative decoding, and no data,
  decode-context, sequence, or pipeline parallelism.
- Serial execution remains available when `"wavefront"` is omitted or false.
