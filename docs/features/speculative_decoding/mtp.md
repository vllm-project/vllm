# MTP (Multi-Token Prediction)

MTP is a speculative decoding method where the target model includes native
multi-token prediction capability. Unlike draft-model-based methods, you do not
need to provide a separate draft model.

MTP is useful when:

- Your model natively supports MTP.
- You want model-based speculative decoding with minimal extra configuration.

## Gemma 4 Assistant Models

Gemma 4 assistant checkpoints use vLLM's Gemma 4 MTP path. They are not generic
draft models, even though they are passed through the `model` field in
`--speculative-config`.

Use `"method": "mtp"` when serving Gemma 4 with an assistant checkpoint:

```bash
vllm serve google/gemma-4-E2B-it \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --speculative-config '{"method":"mtp","model":"gg-hf-am/gemma-4-E2B-it-assistant","num_speculative_tokens":1}'
```

The E2B, E4B, 12B, 26B-A4B, and 31B Gemma 4 IT assistant checkpoints are supported.
Tower-based variants use `model_type: gemma4_assistant` and the encoder-free
Gemma 4 Unified variant (12B) uses `model_type: gemma4_unified_assistant`.
vLLM maps both to `Gemma4MTPModel` internally and wires the assistant layers
to share KV cache with the target model.

If an older vLLM release logs `SpeculativeConfig(method='draft_model', ...)`
for a Gemma 4 assistant checkpoint, that release is treating the assistant as a
generic draft model and may fail during initialization for multimodal Gemma 4
targets. Upgrade to a version with Gemma 4 MTP support instead.

## Offline Example

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="XiaomiMiMo/MiMo-7B-Base",
    tensor_parallel_size=1,
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Online Example

```bash
vllm serve XiaomiMiMo/MiMo-7B-Base \
    --tensor-parallel-size 1 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

## Reduced draft vocabulary

MTP heads usually share the target model's lm_head, so every draft token pays
for a projection onto the full vocabulary. For large vocabularies on
bandwidth-bound GPUs this can be a large share of the drafting cost. The
`draft_token_map` key restricts the drafter's head to a list of frequent token
ids (the [FR-Spec](https://arxiv.org/abs/2502.14856) idea):

```bash
vllm serve <mtp-model> \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3,
                           "draft_token_map": "draft_vocab.pt"}'
```

- Requires Model Runner V2 (the default on CUDA).
- Only drafting changes. The target still verifies with its full lm_head, so a
  token outside the list costs a rejected draft, never a different output
  distribution. With probabilistic drafting
  (`draft_sample_method: "probabilistic"`), the proposal is a distribution over
  the full vocabulary with zero mass outside the list, so rejection sampling
  stays exact.
- The file is SGLang's `--speculative-token-map` format (a `.pt` list of ids)
  or a JSON list. EOS ids are always added.
- Drafters whose model code builds a `DraftVocab` support it (Qwen3.5 and
  Qwen4Exp MTP); others raise an error. A quantized lm_head (for example a
  ModelOpt NVFP4 checkpoint) keeps its checkpoint format: the listed rows are
  cut from the checkpoint tensors and run on the same quantized kernel.
- Acceptance drops when the list misses tokens your traffic uses, so build it
  from representative text, ideally the model's own outputs, and include all
  special tokens. [`build_draft_token_map.py`](../../../examples/features/speculative_decoding/build_draft_token_map.py)
  ranks token ids by frequency over a corpus and reports held-out coverage.
  Measure the real acceptance length (see
  [acceptance metrics](acceptance_metrics.md)) with and without the list.

A static list loses acceptance on traffic it was not built from (for example
other languages). `draft_token_map_dynamic_rows` adds that many tokens per
draft step on top of the list: the rows outside the list are scored with a
rank-`draft_token_map_dynamic_rank` (default 256) projection of the lm_head and
the best ones get exact logits.

```bash
vllm serve <mtp-model> \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3,
                           "draft_token_map": "draft_vocab_32k.pt",
                           "draft_token_map_dynamic_rows": 16384}'
```

It costs more per draft token than the list alone, and that cost grows with
batch size, so measure decode speed at your concurrency. Tensor parallel size 1
only.

`draft_token_map_quantization` (`"fp8"` or `"nvfp4"`) stores the listed and
dynamic rows weight-only quantized, which cuts the bytes read per draft token
further; the dynamic rows' scorer is then stored in FP8. Only the acceptance
length can change. It is rejected when the lm_head is already quantized in the
checkpoint. Dynamic rows on a quantized lm_head need the ModelOpt NVFP4 format.

## Notes

- MTP only works for model families that support MTP in vLLM.
- `num_speculative_tokens` controls speculative depth. A small value like `1`
  is a good default to start with.
- If your model does not support MTP, use another method such as EAGLE or draft
  model speculation.
