# Quantized KV Cache

## FP8 KV Cache Overview

Efficient memory usage is crucial for working with large language models. Quantizing the KV (Key-Value) cache to FP8 format can significantly reduce its memory footprint. This optimization enables you to store more tokens in memory, leading to improved throughput and support for longer context windows.

> **Note:** When using the Flash Attention 3 backend with FP8 KV cache, attention operations are also performed in the quantized (FP8) domain. In this configuration, queries are quantized to FP8 in addition to keys and values.

### Supported FP8 KV-Cache Quantization Schemes

vLLM supports two main quantization strategies for the FP8 KV-cache:

- **Per-tensor quantization:**  
  A single scale is applied for each Q, K, and V tensor individually. (`q/k/v_scale = [1]`)
- **Per-attention-head quantization:**  
  Each scale corresponds to an attention head: `q_scale = [num_heads]`, `k/v_scale = [num_kv_heads]`.

> **Note:**  
> Per-attention-head quantization is currently available **only with the Flash Attention backend** and requires the calibration pathway provided by **llm-compressor**.

### Scale Calibration Approaches

You can configure how the quantization scales are computed in vLLM using three different approaches:

1. **No calibration (default scales):**  
   All quantization scales are set to `1.0`.  
   _Configure with:_  
   ```python
   kv_cache_dtype="fp8"
   calculate_kv_scales=False
   ```

2. **Random token calibration (on-the-fly):**  
   Scales are automatically estimated from a single batch of random tokens during warmup and then fixed.  
   _Configure with:_  
   ```python
   kv_cache_dtype="fp8"
   calculate_kv_scales=True
   ```

3. **[Recommended] Calibration with a dataset (via llm-compressor):**  
   Scales are estimated using a curated calibration dataset for maximum accuracy.  
   This requires the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.  
   _See example below!_

#### Additional `kv_cache_dtype` Options

- `kv_cache_dtype="auto"`: Use the model's default data type
- `kv_cache_dtype="fp8_e4m3"`: Supported on CUDA 11.8+ and ROCm (AMD GPUs)
- `kv_cache_dtype="fp8_e5m2"`: Supported on CUDA 11.8+

---

## Examples

### 1. No Calibration (`kv_cache_dtype="fp8"`, `calculate_kv_scales=False`)

All quantization scales are set to 1.0.

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
    calculate_kv_scales=False,
)
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```

---

### 2. Random Token Calibration (`kv_cache_dtype="fp8"`, `calculate_kv_scales=True`)

Scales are automatically estimated from a single batch of tokens during warmup.

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
    calculate_kv_scales=True,
)
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```

---

### 3. **[Recommended] Calibration Using a Dataset (with `llm-compressor`)**

For the highest-quality quantization, we recommend calibrating against a dataset using `llm-compressor`. This enables advanced strategies such as per-attention-head quantization.

#### Install the required package

```bash
pip install llmcompressor
```

#### Example: Quantize Llama Attention & KV Cache to FP8

```python
"""
Quantize Llama attention + KV cache to FP8 (choose either 'tensor' or 'attn_head' strategy)
using llm-compressor one-shot calibration.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
STRATEGY = "tensor"       # or "attn_head"
NUM_CALIB_SAMPLES = 512   # Good starting value
MAX_SEQ_LEN = 2048

# -----------------------------
# Helpers
# -----------------------------
def process_and_tokenize(example, tokenizer: AutoTokenizer):
    """Convert chat messages to tokens."""
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        add_special_tokens=False,
    )

def build_recipe(strategy: str) -> QuantizationModifier:
    fp8_args = QuantizationArgs(num_bits=8, type="float", strategy=strategy)
    return QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=["LlamaAttention"],  # Quantize queries: q_scale
                input_activations=fp8_args,
            )
        },
        kv_cache_scheme=fp8_args,           # Quantize KV cache: k/v_scale
    )

# -----------------------------
# Main
# -----------------------------
def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIB_SAMPLES}]")
    ds = ds.shuffle(seed=42)
    ds = ds.map(
        lambda ex: process_and_tokenize(ex, tokenizer),
        remove_columns=ds.column_names,
    )

    recipe = build_recipe(STRATEGY)
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=NUM_CALIB_SAMPLES,
    )

    save_dir = f"{MODEL_ID.rstrip('/').split('/')[-1]}-kvattn-fp8-{STRATEGY}"
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    main()
```

For more detailed and up-to-date examples, see the [`llm-compressor` official examples](https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_kv_cache).

---

## OSCAR INT2 KV Cache (2-bit)

OSCAR (Offline Spectral Covariance-Aware Rotation,
[arXiv:2605.17757](https://arxiv.org/abs/2605.17757)) quantizes the KV cache to
~2 bits per element, cutting KV memory by roughly 5x relative to BF16. It applies
a _calibrated_ per-layer orthogonal rotation to keys and values before asymmetric
INT2 quantization, so quantization noise lands in the directions attention is
least sensitive to. The rotation is weight-preserving — model weights are loaded
unchanged.

Select it with `kv_cache_dtype="oscar_int2"`. Unlike FP8, OSCAR needs an
auxiliary set of per-layer rotation matrices, supplied via environment variables
(rather than baked into the checkpoint):

| Variable | Meaning | Recipe default |
| -------- | ------- | -------------- |
| `VLLM_OSCAR_K_ROTATION_PATH` | Key rotation `.pt` (per-layer `[head_dim, head_dim]`) | — |
| `VLLM_OSCAR_V_ROTATION_PATH` | Value rotation `.pt` | — |
| `VLLM_OSCAR_K_CLIP_RATIO` | Key percentile clip (0 disables) | `0.96` |
| `VLLM_OSCAR_V_CLIP_RATIO` | Value percentile clip (0 disables) | `0.92` |

Pre-computed rotations for several models (Qwen3-4B/8B/32B, GLM-4.7-FP8, …) are
published at [`Zhongzhu/OSCAR-RotationZoo`](https://huggingface.co/Zhongzhu/OSCAR-RotationZoo).
If no rotation path is set, OSCAR degrades gracefully to identity rotation (naive
clipped INT2), which loses most of the accuracy benefit.

### Example: Qwen3-32B with OSCAR INT2

```bash
# Download the calibrated rotations for this model.
hf download Zhongzhu/OSCAR-RotationZoo \
    "Qwen3-32B/seq16000_prompt69_group128/*" --local-dir ./oscar_rotations

export VLLM_OSCAR_K_ROTATION_PATH=./oscar_rotations/Qwen3-32B/seq16000_prompt69_group128/k_rotation_qqt_r_h_pbr.pt
export VLLM_OSCAR_V_ROTATION_PATH=./oscar_rotations/Qwen3-32B/seq16000_prompt69_group128/v_rotation_sst_r_h_pbr.pt
export VLLM_OSCAR_K_CLIP_RATIO=0.96
export VLLM_OSCAR_V_CLIP_RATIO=0.92
```

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-32B",
    kv_cache_dtype="oscar_int2",
)
out = llm.generate("London is the capital of", SamplingParams(temperature=0))
print(out[0].outputs[0].text)
```

Or for online serving (read the same environment variables before launching):

```bash
vllm serve Qwen/Qwen3-32B --kv-cache-dtype oscar_int2
```

!!! note
    The first and last two attention layers are kept in the native dtype
    (boundary protection), so the effective memory saving is slightly below the
    nominal 8x. OSCAR is a memory/throughput optimization: it preserves accuracy
    (e.g. near-zero gap to BF16 on Qwen3-32B) and enables much longer contexts /
    higher concurrency, but the INT2 decode kernel adds per-token latency at
    small batch sizes.
