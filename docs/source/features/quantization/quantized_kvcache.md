(quantized-kvcache)=

# Quantized KV Cache

## FP8 KV Cache

Quantizing the KV cache to FP8 reduces its memory footprint. This increases the number of tokens that can be stored in the cache,
improving throughput. OCP (Open Compute Project www.opencompute.org) specifies two common 8-bit floating point data formats: E5M2
(5 exponent bits and 2 mantissa bits) and E4M3FN (4 exponent bits and 3 mantissa bits), often shortened as E4M3. One benefit of
the E4M3 format over E5M2 is that floating point numbers are represented in higher precision. However, the small dynamic range of
FP8 E4M3 (±240.0 can be represented) typically necessitates the use of a higher-precision (typically FP32) scaling factor alongside
each quantized tensor.

For now, only per-tensor (scalar) scaling factors are supported. Development is ongoing to support scaling
factors of a finer granularity (e.g. per-channel).

Studies have shown that FP8 E4M3 quantization typically only minimally degrades inference accuracy. The most recent silicon
offerings e.g. AMD MI300, NVIDIA Hopper or later support native hardware conversion to and from fp32, fp16, bf16, etc.
Thus, LLM inference is greatly accelerated with minimal accuracy loss.

Here is an example of how to enable this feature:

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=1.3, top_p=0.8)
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", kv_cache_dtype="fp8")
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)

# output w/ scaling factors:  England, the United Kingdom, and one of the world's leading financial,
# output w/o scaling factors:  England, located in the southeastern part of the country. It is known
```

The argument `kv_cache_dtype` is the data type for kv cache storage. If `"auto"`, will use the model's default "unquantized" data type. CUDA 11.8+ supports `"fp8"` (=`"fp8_e4m3"`) and `"fp8_e5m2"`. ROCm (AMD GPU) supports `"fp8"` (=`"fp8_e4m3"`)

### Calibrated scales for better accuracy preservation

For the best model quality when using FP8 KV Cache, we recommend producing calibrated scales tuned to representative data that the model will see at inference time. We recommend using [LLM Compressor](https://github.com/vllm-project/llm-compressor/) for this.

First install the dependencies:

```console
pip install llmcompressor
```

Then, run this example made for `meta-llama/Llama-3.1-8B-Instruct`. Most models should be able to just substitute in:

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.transformers import oneshot

# Select model and load it.
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
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

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the kv cache to fp8 with per-tensor scales
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

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-KV"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

This will produce a folder in your current directory with your new model (in this case `Llama-3.1-8B-Instruct-FP8-KV`) with calibrated scales!
