# LongSpark

[LongSpark](https://arxiv.org/abs/2609.37029) is a parallel drafter for
long-context speculative decoding. Its implementation and checkpoints are
available from the [LongSpark project](https://github.com/Hao-Yuan-He/LongSpark).
The published Qwen3 checkpoints condition each seven-token draft block on three
bounded views:

- Global16, an incrementally updated summary of the committed target prefix;
- Raw256, the latest target K/V rows; and
- Local7, the current draft block.

The Model Runner V2 integration runs the target and drafter in one process and
keeps the target as the sole owner of persistent K/V. The first prefill scans
the committed prefix to initialize Global16. Later verification steps merge
only newly committed target rows into that state.

## Offline inference

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-4B",
    dtype="bfloat16",
    enable_prefix_caching=False,
    speculative_config={
        "method": "longspark",
        "model": "Hehy/LongSpark-Qwen3-4B",
    },
)

outputs = llm.generate(
    ["Explain why the night sky is dark."],
    SamplingParams(temperature=0, max_tokens=128),
)
print(outputs[0].outputs[0].text)
```

`method` and `num_speculative_tokens` may be omitted. A checkpoint with
`model_type="draftfreekv"` and architecture `DraftFreeKVModel` is detected as
LongSpark, and its `block_size` selects the checkpoint's draft width.

The equivalent server command is:

```bash
vllm serve Qwen/Qwen3-4B \
    --dtype bfloat16 \
    --no-enable-prefix-caching \
    --speculative-config '{
      "method": "longspark",
      "model": "Hehy/LongSpark-Qwen3-4B"
    }'
```

## Supported configuration

The initial native integration deliberately fails closed outside the
checkpoint's validated execution contract:

- CUDA with Model Runner V2 and the `FLASH_ATTN` backend;
- BF16 Qwen3 target and LongSpark draft weights, with a BF16 target KV cache;
- a Qwen3 target with a LongSpark checkpoint trained for the same hidden,
  vocabulary, attention, MLP, layer-count, and RoPE configuration;
- TP, PP, DCP, PCP, and draft TP all equal to 1;
- text-only requests, without target LoRA or prefix caching; and
- `num_speculative_tokens` matching the checkpoint's `block_size`.

The fused Triton kernels are shape-parameterized. Query/KV head counts, GQA
ratio, head dimension, global/raw/local window sizes, Markov rank, delta width,
and runtime batch size come from the checkpoint configuration or input tensor.
Consequently, the implementation is not tied to Qwen3-4B tensor dimensions;
another Qwen3 LongSpark pair can use the same path when its checkpoint declares
matching dimensions. The draft network itself is still Qwen3-style, so this
does not by itself enable checkpoints trained for unrelated model families.

Target-model CUDA graphs remain available. For greedy, unwatermarked decoding,
the LongSpark draft backbone and Markov sampler use vLLM's configured CUDA Graph
capture sizes and pad to the nearest captured batch, just like the target model.
Batches above the largest configured capture size and sampling modes that need
draft logits use the parameterized eager draft path. Raw-window rows remain
transient and are never stored in a second persistent K/V cache. These
constraints are checked during configuration or startup so unsupported
combinations do not silently take another speculative path.

For a token-for-token greedy correctness comparison against non-speculative
decoding, set `VLLM_BATCH_INVARIANT=1` for both runs. Without batch-invariant
kernels, the target's different verification batch shapes can change a greedy
choice at near-tied logits even though speculative verification remains
lossless.

Available checkpoint pairs are listed in the
[LongSpark collection](https://huggingface.co/collections/Hehy/longspark).
