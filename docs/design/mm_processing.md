# Multi-Modal Data Processing

To enable various optimizations in vLLM such as [chunked prefill](../configuration/optimization.md#chunked-prefill) and [prefix caching](../features/automatic_prefix_caching.md), we use [BaseMultiModalProcessor][vllm.multimodal.processing.BaseMultiModalProcessor] to provide the correspondence between placeholder feature tokens (e.g. `<image>`) and multi-modal inputs (e.g. the raw input image) based on the outputs of HF processor.

In vLLM's rendering pipeline (see [BaseRenderer][vllm.renderers.base.BaseRenderer]), tokenization is performed as a separate step before multi-modal processing. Therefore, `BaseMultiModalProcessor` needs to recreate the output of calling HF processor end-to-end, while not being able to see the original text. This is achieved through **Dummy Input Text** and **Prompt Update Detection**.

## Dummy Input Text

Since Transformers 5.10, `ProcessorMixin` now allows multi-modal inputs to be passed by themselves. However, certain subclasses (such as `ChameleonProcessor`) and older out-of-tree implementations may still define their own `__call__` method that assumes the presence of text with corresponding placeholder tokens. This causes a problem as we don't have the original text anymore to pass to these processors.

To work around this, each model defines how to generate dummy text based on the number of multi-modal inputs, via [get_dummy_text][vllm.multimodal.processing.BaseDummyInputsBuilder.get_dummy_text], which its override of [_get_hf_processor_text][vllm.multimodal.processing.BaseMultiModalProcessor._get_hf_processor_text] returns so that [_apply_hf_processor_main][vllm.multimodal.processing.BaseMultiModalProcessor._apply_hf_processor_main] passes it to the HF processor together with the multi-modal inputs to obtain the processed multi-modal data.

Similarly, since the multi-modal data extracted by vLLM may not match what a specific HF processor expects, [_apply_hf_processor_main][vllm.multimodal.processing.BaseMultiModalProcessor._apply_hf_processor_main] allows each model to adapt the inputs via [_preprocess_hf_mm_data][vllm.multimodal.processing.BaseMultiModalProcessor._preprocess_hf_mm_data] (e.g. renaming keys like `audios` to `audio` or injecting extra keyword arguments such as `sampling_rate`) and the outputs via [_postprocess_hf_mm_data][vllm.multimodal.processing.BaseMultiModalProcessor._postprocess_hf_mm_data], without having to reimplement the entire method.

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

To accelerate the multi‑modal data pipeline (decoding, resizing, normalisation, and rescaling), we offload the heavy numerical preprocessing from the CPU to the GPU and optimise data movement.

#### Fusing Normalisation and Rescaling on the GPU

Traditionally, the CPU would divide pixel values by 255, then subtract the mean and divide by the standard deviation. We fuse these steps into one operation and run it entirely on the GPU.

**How it works**: We use a dedicated `FusedInputNorm` module that bakes the rescale factor (1/255) directly into the layer's `weight` and `bias` parameters. Instead of performing three separate steps (scale, subtract, divide), the module does everything in a single affine transformation: `y = x * weight + bias`.

The parameters are set as follows:

- `weight` controls both the standard deviation and the rescale factor
- `bias` centers the data using the mean and the same rescale factor

This means the module takes raw pixel values (0–255) and outputs properly normalised values without ever explicitly dividing by 255 as a separate step.

#### Optimized Data Path for Fused Normalisation

Performing fused normalisation directly on the device allows us to keep the entire transfer path—from **Entrypoint** through **Engine Core** to **GPU memory**—in **`uint8`**. This halves PCIe bandwidth and reduces CPU memory footprint.

Only after data reaches GPU memory do we cast to `fp32` for the `FusedInputNorm` layer (to ensure numerical accuracy), then cast to `bf16` for subsequent layers—all within the GPU, avoiding any host‑side conversions.

Overall path: **`Entrypoint (uint8) → Engine Core (uint8) → GPU Memory (uint8)`** → GPU‑local `fp32` `FusedInputNorm` → `bf16` output.

#### Toggle: `mm_device_do_normalize`

This GPU‑side fusion is controlled by a config flag called **`mm_device_do_normalize`**.

- When `True`, normalisation and rescaling are done on the GPU using the `FusedInputNorm` layer; when `False`, we fall back to the old CPU‑side path.
- The flag is **enabled by default** for all models that support it.
- Currently, it’s on by default for these architectures:

| name         | Architecture                         | Example HF Models                   |
|--------------|--------------------------------------|-------------------------------------|
| `qwen2-vl`   | `Qwen2VLForConditionalGeneration`    | `Qwen/Qwen2-VL-2B-Instruct`, etc.   |
| `qwen2.5-vl` | `Qwen2_5_VLForConditionalGeneration` | `Qwen/Qwen2.5-VL-3B-Instruct`, etc. |

#### What We Gain Overall

- **CPU offload**: The arithmetic for normalisation and rescaling is completely gone from the CPU.
- **PCIe savings**: Sending `uint8` (1 byte) instead of `bf16` (2 bytes) slashes data transfer volume by **50%** .
- **GPU overhead**: The fused kernel is very lightweight and can often be merged with subsequent CUDA operations, so it hardly adds any extra cost.
