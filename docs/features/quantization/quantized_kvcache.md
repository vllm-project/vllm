---
title: Quantized KV Cache
---
[](){ #quantized-kvcache }

## FP8 KV Cache

Quantizing the KV cache to FP8 reduces its memory footprint. This increases the number of tokens that can be stored in the cache, improving throughput.

### FP8 Formats

[OCP (Open Compute Project)](https://www.opencompute.org) specifies two common 8-bit floating point data formats:

- E5M2 (5 exponent bits and 2 mantissa bits)
- E4M3FN (4 exponent bits and 3 mantissa bits, often shortened as E4M3)

The E4M3 format offers higher precision compared to E5M2. However, due to its small dynamic range (±240.0), E4M3 typically requires a higher-precision (FP32) scaling factor alongside each quantized tensor.

### Current Limitations

For now, only per-tensor (scalar) scaling factors are supported. Development is ongoing to support scaling factors of a finer granularity (e.g. per-channel).

### Performance Impact

The current FP8 KV cache implementation primarily benefits throughput by allowing approximately double the amount of space for KV cache allocation. This enables either:

- Processing longer context lengths for individual requests, or
- Handling more concurrent request batches

However, there are currently no latency improvements as the implementation does not yet include fused dequantization and attention operations. Future releases will support quantized attention with hardware acceleration, which should provide additional performance benefits. While the most recent silicon offerings (e.g. AMD MI300, NVIDIA Hopper or later) support native hardware conversion between FP8 and other formats (fp32, fp16, bf16), this benefit is not yet fully realized.

Studies have shown that FP8 E4M3 quantization typically only minimally degrades inference accuracy, making it a practical choice for throughput optimization.

## Usage Example

Here is an example of how to enable FP8 quantization:

??? Code

    ```python
    # To calculate kv cache scales on the fly enable the calculate_kv_scales
    # parameter

    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
    llm = LLM(model="meta-llama/Llama-2-7b-chat-hf",
            kv_cache_dtype="fp8",
            calculate_kv_scales=True)
    prompt = "London is the capital of"
    out = llm.generate(prompt, sampling_params)[0].outputs[0].text
    print(out)
    ```

The `kv_cache_dtype` argument specifies the data type for KV cache storage:
- `"auto"`: Uses the model's default "unquantized" data type
- `"fp8"` or `"fp8_e4m3"`: Supported on CUDA 11.8+ and ROCm (AMD GPU)
- `"fp8_e5m2"`: Supported on CUDA 11.8+

## Calibrated Scales for Better Accuracy

For optimal model quality when using FP8 KV Cache, we recommend using calibrated scales tuned to representative inference data. [LLM Compressor](https://github.com/vllm-project/llm-compressor/) is the recommended tool for this process.

### Installation

First, install the required dependencies:

```bash
pip install llmcompressor
```

### Example Usage

Here's a complete example using `meta-llama/Llama-3.1-8B-Instruct` (most models can use this same pattern):

??? Code

    ```python
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor.transformers import oneshot

    # Select model and load it
    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Select calibration dataset
    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"

    # Configure calibration parameters
    NUM_CALIBRATION_SAMPLES = 512  # 512 samples is a good starting point
    MAX_SEQUENCE_LENGTH = 2048

    # Load and preprocess dataset
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    def process_and_tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return tokenizer(
            text,
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)

    # Configure quantization settings
    recipe = """
    quant_stage:
        quant_modifiers:
            QuantizationModifier:
                kv_cache_scheme:
                    num_bits: 8
                    type: float
                    strategy: tensor
                    dynamic: false
                    symmetric: true
    """

    # Apply quantization
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # Save quantized model: Llama-3.1-8B-Instruct-FP8-KV
    SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-KV"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    ```

The above script will create a folder in your current directory containing your quantized model (e.g., `Llama-3.1-8B-Instruct-FP8-KV`) with calibrated scales.

When running the model you must specify `kv_cache_dtype="fp8"` in order to enable the kv cache quantization and use the scales.

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(model="Llama-3.1-8B-Instruct-FP8-KV", kv_cache_dtype="fp8")
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```
