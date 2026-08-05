# Multi-Modal Data Processing

To enable various optimizations in vLLM such as [chunked prefill](../configuration/optimization.md#chunked-prefill) and [prefix caching](../features/automatic_prefix_caching.md), we use [BaseMultiModalProcessor][vllm.multimodal.processing.BaseMultiModalProcessor] to provide the correspondence between placeholder feature tokens (e.g. `<image>`) and multi-modal inputs (e.g. the raw input image) based on the outputs of HF processor.

Here are the main features of [BaseMultiModalProcessor][vllm.multimodal.processing.BaseMultiModalProcessor]:

## Prompt Update Detection

One of the main responsibilities of HF processor is to update the prompt with placeholder tokens. For example:

- Insert feature placeholder tokens (e.g. `<image><image>...<image>`, the number of which equals to the feature size) at the start of the string.
- Replace existing input placeholder tokens (e.g. `<image>` for a single image) with feature placeholder tokens (e.g. `<image><image>...<image>`, the number of which equals to the feature size).

The information about which tokens have been updated is key to finding the correspondence between placeholder feature tokens and multi-modal inputs.

In vLLM, this information is specified using [PromptUpdate][vllm.multimodal.processing.PromptUpdate] in [_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates]. We can automatically detect whether HF has updated the prompt by checking the existence of the updated tokens.

## Tokenized Prompt Inputs

To enable tokenization in a separate process, we support passing input token IDs alongside multi-modal data.

### The problem

Consider that HF processors follow these main steps:

1. Tokenize the text
2. Process multi-modal inputs
3. Perform prompt updates

And we require that:

- For text + multi-modal inputs, apply all steps 1--3.
- For tokenized + multi-modal inputs, apply only steps 2--3.

How can we achieve this without rewriting HF processors? We can try to call the HF processor several times on different inputs:

- For text + multi-modal inputs, simply call the HF processor directly.
- For tokenized + multi-modal inputs, call the processor only on the multi-modal inputs.

While HF processors support text + multi-modal inputs natively, this is not so for tokenized + multi-modal inputs: an error is thrown if the number of input placeholder tokens do not correspond to the number of multi-modal inputs.

Moreover, since the tokenized text has not passed through the HF processor, we have to apply Step 3 by ourselves to keep the output tokens and multi-modal data consistent with each other.

### Dummy text

We work around the first issue by requiring each model to define how to generate dummy text based on the number of multi-modal inputs, via [get_dummy_text][vllm.multimodal.processing.BaseDummyInputsBuilder.get_dummy_text]. This lets us generate dummy text corresponding to the multi-modal inputs and input them together to obtain the processed multi-modal data.

### Automatic prompt updating

We address the second issue by implementing model-agnostic code in
[_apply_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._apply_prompt_updates] to automatically update the prompt with feature placeholder tokens based on the specification outputted by [_get_prompt_updates][vllm.multimodal.processing.BaseMultiModalProcessor._get_prompt_updates].

### Summary

With the help of dummy text and automatic prompt updating, our multi-modal processor can finally accept both text and token prompts with multi-modal data. The detailed logic is shown in [_apply_hf_processor_main][vllm.multimodal.processing.BaseMultiModalProcessor._apply_hf_processor_main].

## Processor Output Caching

Some HF processors, such as the one for Qwen2-VL, are [very slow](https://github.com/vllm-project/vllm/issues/9238). To alleviate this problem, we cache the multi-modal outputs of HF processor to avoid processing the same multi-modal input (e.g. image) again.

When new data is passed in, we first check which items are in the cache, and which ones are missing. The missing items are passed into the HF processor in a single batch and cached, before being merged with the existing items in the cache.

Since we only process the missing multi-modal data items, the number of input placeholder tokens no longer corresponds to the number of the multi-modal inputs, so they can't be passed alongside the text prompt to HF processor. Therefore, we process the text and multi-modal inputs separately, using [dummy text](#dummy-text) to avoid HF errors. Since this skips HF's prompt updating code, we apply [automatic prompt updating](#automatic-prompt-updating) afterwards to keep the output tokens and multi-modal data consistent with each other.

## Speeding Up Multi‑Modal Data Processing

### Fused Normalisation on the Device

To accelerate the multi‑modal data pipeline (decoding, resizing, normalisation, and rescaling), we offload the heavy numerical preprocessing from the CPU to the GPU and optimise data movement.

#### Fusing Normalisation and Rescaling on the GPU

Traditionally, the CPU would divide pixel values by 255, then subtract the mean and divide by the standard deviation. We fuse these steps into one operation and run it entirely on the GPU.

- **How it works**: We use a helper called `make_input_norm` (backed by `nn.BatchNorm1d(3, eps=0.0)`) and bake the rescale factor (typically 1/255) directly into the mean and standard deviation:
  - Effective mean = `image_mean * (1/rescale_factor)`
  - Effective std  = `image_std  * (1/rescale_factor)`
- **At runtime**: The layer takes raw uint8 pixel values (0–255) and does the full normalised mapping in a single GPU kernel—no CPU involvement.

#### Optimized Data Path for Fused Normalisation

Performing fused normalisation directly on the device allows us to keep the entire transfer path—from **Entrypoint** through **Engine Core** to **GPU memory**—in **`uint8`**. This halves PCIe bandwidth and reduces CPU memory footprint.

Only after data reaches GPU memory do we cast to `fp32` for `BatchNorm1d` (to ensure numerical accuracy), then cast to `bf16` for subsequent layers—all within the GPU, avoiding any host‑side conversions.

Overall path: **`Entrypoint (uint8) → Engine Core (uint8) → GPU Memory (uint8)`** → GPU‑local `fp32` BN → `bf16` output.

#### Toggle: `mm_device_do_normalize`

This GPU‑side fusion is controlled by a config flag called **`mm_device_do_normalize`**.

- When `True`, normalisation and rescaling are done on the GPU using the fused layer; when `False`, we fall back to the old CPU‑side path.
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
