# Recirculation

!!! warning
    Recirculation support is experimental. The current implementation is a
    correctness-first, serial implementation for text-only Gemma 3 models.

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

## Current restrictions

- Only `Gemma3ForCausalLM` is supported. Multimodal Gemma 3 models are not.
- Pipeline parallelism is not supported.
- Only fixed scalar coefficients and source norm matching are implemented.
- Layers above the destination run serially a second time. Wavefront execution
  that overlaps the normal and recirculated stacks is not yet implemented.
