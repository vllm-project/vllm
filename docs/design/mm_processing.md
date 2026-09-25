# Multi-Modal Data Processing

To enable various optimizations in vLLM such as [chunked prefill](../configuration/optimization.md#chunked-prefill) and [prefix caching](../features/automatic_prefix_caching.md), we use [BaseMultiModalProcessor][vllm.multimodal.processing.BaseMultiModalProcessor] to provide the correspondence between placeholder feature tokens (e.g. `<image>`) and multi-modal inputs (e.g. the raw input image) based on the outputs of HF processor.

In vLLM's rendering pipeline (see [BaseRenderer][vllm.renderers.base.BaseRenderer]), tokenization is performed as a separate step before multi-modal processing. Therefore, `BaseMultiModalProcessor` needs to recreate the output of calling HF processor end-to-end, while not being able to see the original text. This is achieved through **Dummy Input Text** and **Prompt Update Detection**.

## Dummy Input Text

Since Transformers 5.10, `ProcessorMixin` now allows multi-modal inputs to be passed by themselves. However, certain subclasses (such as `ChameleonProcessor`) and older out-of-tree implementations may still define their own `__call__` method that assumes the presence of text with corresponding placeholder tokens. This causes a problem as we don't have the original text anymore to pass to these processors.

To work around this, each model defines how to generate dummy text based on the number of multi-modal inputs, via [get_dummy_text][vllm.multimodal.processing.BaseDummyInputsBuilder.get_dummy_text], which its override of [_get_hf_mm_text][vllm.multimodal.processing.BaseMultiModalProcessor._get_hf_mm_text] returns. [_get_hf_mm_inputs][vllm.multimodal.processing.BaseMultiModalProcessor._get_hf_mm_inputs] adds this text and `truncation=False` to the normalized HF inputs. The truncation setting is attached only to calls that use dummy text, so unrelated processor calls do not receive a text-specific option.

HF processing is split into four overridable stages:

1. [_get_hf_mm_inputs][vllm.multimodal.processing.BaseMultiModalProcessor._get_hf_mm_inputs] extracts and normalizes processor data, kwargs, and passthrough data. It centrally renames vLLM's `audios` field to the standard HF `audio` keyword; processors that require another spelling can adapt it in an override.
2. [_call_hf_processor][vllm.multimodal.processing.BaseMultiModalProcessor._call_hf_processor] invokes the HF processor. Models can override it when they use a different callable or need separate construction and invocation kwargs.
3. [_finalize_hf_mm_data][vllm.multimodal.processing.BaseMultiModalProcessor._finalize_hf_mm_data] merges passthrough fields into the processed output. It lazily creates an empty output when there is no processor data.
4. [_postprocess_hf_mm_data][vllm.multimodal.processing.BaseMultiModalProcessor._postprocess_hf_mm_data] performs model-specific output transformations after passthrough fields have been merged.

This division lets most models customize one stage without reimplementing [_apply_hf_processor_main][vllm.multimodal.processing.BaseMultiModalProcessor._apply_hf_processor_main]. Implementations that genuinely require multiple processor calls or manually construct their complete output may still override that method, but must use `_get_hf_mm_inputs` and `_finalize_hf_mm_data` to retain the common input and output behavior.

## Prompt Update Detection

One of the main responsibilities of HF processor is to update the prompt with placeholder tokens. For example:

- Insert feature placeholder tokens (e.g. `<image><image>...<image>`, the number of which equals to the feature size) at the start of the string.
- Replace existing input placeholder tokens (e.g. `<image>` for a single image) with feature placeholder tokens (e.g. `<image><image>...<image>`, the number of which equals to the feature size).

The information about which tokens have been updated is key to finding the correspondence between placeholder feature tokens and multi-modal inputs.

Since we call HF processor without the input text, we have to perform this update by ourselves. In vLLM, we represent the necessary information using [PromptUpdate][vllm.multimodal.processing.PromptUpdate] in [_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates], and apply them via [_apply_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._apply_prompt_updates].

Some HF processors additionally transform the prompt itself regardless of the multi-modal inputs (such as `ChameleonProcessor` appending a sep token for chat mode). Since the prompt tokens likewise bypass the HF processor, such transformations are replicated via [_postprocess_prompt][vllm.multimodal.processing.BaseMultiModalProcessor._postprocess_prompt] before the prompt updates are located or applied.

## Processor Output Caching

Some HF processors, such as the one for Qwen2-VL, are [very slow](https://github.com/vllm-project/vllm/issues/9238). To alleviate this problem, we cache the multi-modal outputs of HF processor to avoid processing the same multi-modal input (e.g. image) again.

When new data is passed in, we first check which items are in the cache, and which ones are missing. The missing items are passed into the HF processor in a single batch and cached, before being merged with the existing items in the cache.

## Speeding Up Multi‑Modal Data Processing

### Fused Normalisation on the Device

To accelerate the multi‑modal data pipeline (decoding, resizing, normalisation, and rescaling), we offload the **normalisation and rescaling** steps from the CPU to the device and optimises data movement.

#### Fusing Normalisation and Rescaling on the GPU

Traditionally, the CPU would divide pixel values by 255, then subtracts the mean and divides by the standard deviation. We fuse these steps into a single per-channel affine transform on the device.

`FusedMMInputNorm` dispatches to `fused_mm_input_norm_triton` on CUDA, which applies:

    y = x * weight[c] + bias[c]
      = (x * rescale_factor - image_mean[c]) / image_std[c]

with:

    weight[c] = rescale_factor / image_std[c]
    bias[c]   = -image_mean[c] / image_std[c]

The kernel accepts `uint8`, `float16`, `bfloat16` and `float32` inputs. The `uint8` path is the primary fast path: raw bytes travel to the device unprocessed, and the rescale factor is folded into `weight` — no separate divide-by-255 step. Compute is always done in fp32 inside the kernel.

#### Optimized Data Path

    Before: CPU decode → CPU resize → CPU rescale (÷255) → CPU normalize
            → cast to bf16 → PCIe (2 B/elem) → GPU

    After:  CPU decode → CPU resize → uint8 pixel_values
            → PCIe (1 B/elem) → GPU fused affine (fp32 compute) → bf16

The transfer path **Entrypoint (API server / offline LLM) → Engine Core(scheduler + workers) → Device Memory** stays in `uint8`. On the device,`fused_mm_input_norm_triton` computes in fp32 internally and writes the requested `visual_dtype` (a per-call argument to `forward_*`, commonly `bf16`) directly, without materialising a global fp32 intermediate.

#### Toggle: `mm_device_do_normalize`

GPU-side fusion is controlled by `multimodal_config.mm_device_do_normalize`:

- When `True`, normalisation and rescaling are done on the GPU using the `FusedMMInputNorm` layer; when `False`, we fall back to the old CPU‑side path.
- The flag is **enabled by default** for all models that support it.
- Currently, it’s on by default for these architectures:

| name         | Architecture                         | Example HF Models                   |
|--------------|--------------------------------------|-------------------------------------|
| `qwen2-vl`   | `Qwen2VLForConditionalGeneration`    | `Qwen/Qwen2-VL-2B-Instruct`, etc.   |
| `qwen2.5-vl` | `Qwen2_5_VLForConditionalGeneration` | `Qwen/Qwen2.5-VL-3B-Instruct`, etc. |

#### Key Properties and Gains

- **CPU offload & 50% PCIe savings** — normalisation/rescaling leaves the CPU entirely; sending `uint8` (1 byte) instead of `bf16` (2 bytes) halves transfer volume.
- **Single-pass fusion** — `y = x * weight[c] + bias[c]` in one kernel launch via `fused_mm_input_norm_triton`; input is read in its native dtype (`uint8`).
- **float32 compute for free** — arithmetic runs in fp32 inside the kernel regardless of I/O dtypes. Accuracy matches fp32, with no extra bandwidth: intermediates stay in registers, and no global fp32 tensor is materialised.
- **Preallocated output with batch-dim padding** — callers can pass a larger buffer; only the leading `N` rows are written, so buffers are reusable across calls.
