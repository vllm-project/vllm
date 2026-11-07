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
   ```

2. **[Recommended] Calibration with a dataset (via llm-compressor):**  
   Scales are estimated using a curated calibration dataset for maximum accuracy.  
   This requires the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.  
   _See example below!_

#### Additional `kv_cache_dtype` Options

- `kv_cache_dtype="auto"`: Use the model's default data type
- `kv_cache_dtype="fp8_e4m3"`: Supported on CUDA 11.8+ and ROCm (AMD GPUs)
- `kv_cache_dtype="fp8_e5m2"`: Supported on CUDA 11.8+

### Skipping Specific Layers from KV-Cache Quantization

Some attention layer types (e.g. sliding-window) are more sensitive to KV-cache quantization. The `--kv-cache-dtype-skip-layers` flag leaves the specified layers at the model's native dtype while keeping the rest of the layers under the chosen quantized dtype. The flag accepts either layer indices or layer-type names:

```bash
# Skip every sliding-window attention layer.
vllm serve <model> \
  --kv-cache-dtype fp8 \
  --kv-cache-dtype-skip-layers sliding_window

# Skip specific layer indices.
vllm serve <model> \
  --kv-cache-dtype fp8 \
  --kv-cache-dtype-skip-layers 0 1 23
```

When skipped layers create a mixed-precision KV cache, vLLM automatically clears newly allocated attention blocks before use. With pipeline parallelism, this behavior is coordinated across all stages, including stages whose local KV cache uses only one precision.

Programmatic usage:

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="fp8",
    kv_cache_dtype_skip_layers=["sliding_window"],
)
```

---

## Examples

### 1. No Calibration (`kv_cache_dtype="fp8"`)

All quantization scales are set to 1.0.

```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.7, top_p=0.8)
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    kv_cache_dtype="fp8",
)
prompt = "London is the capital of"
out = llm.generate(prompt, sampling_params)[0].outputs[0].text
print(out)
```

---

### 2. **[Recommended] Calibration Using a Dataset (with `llm-compressor`)**

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

## OSCAR KV Cache (TurboQuant backend)

`--kv-cache-dtype turboquant_k3v3_oscar | turboquant_k3v2_oscar | turboquant_k2v2_oscar`
stores K and V as 3- or 2-bit codes and keeps each request's most recent
tokens at higher precision. It runs on the TurboQuant attention backend and
supports `head_dim` 128 and 256.

Per head and token, K and V are each Hadamard-rotated, clipped at a per-row
`|x|` quantile, quantized to asymmetric uniform codes, and stored with an
fp16 scale/min that also folds in a row-norm correction. Decode attends
quantized tokens without dequantizing them (the codes are fed to the tensor
cores directly). The last `recent` tokens of every request are attended from
an int8 copy instead, which removes most of the quantization error that
matters for the next token.

| Preset | Bytes / token / KV head (head_dim 256) | Recent window |
|---|---|---|
| `turboquant_4bit_nc` (reference) | 262 | — |
| `turboquant_k3v3_oscar` | 200 | 256 |
| `turboquant_k3v2_oscar` | 168 | 1024 |
| `turboquant_k2v2_oscar` | 136 | 1024 |

Notes:

- The recent-window copy is kept per physical KV page and is validated
  against the quantized record before use, so prefix caching, block reuse and
  KV offloading (`OffloadingConnector`) stay correct; when the copy is
  missing or stale the quantized value is used.
- The window copy is a fixed cost per KV cache group: about
  `windows x (recent + sink) x layers x kv_heads x 2 x head_dim` bytes, for up to
  8 concurrent windows (fewer with a small `--max-num-seqs`). It is capped at 10%
  of the free GPU memory at startup and allocated before the KV cache is sized;
  requests beyond the cap use the quantized values. On small GPUs this fixed
  cost can cancel the capacity gain.
- On hybrid (Mamba/GDN + attention) models a smaller KV page raises the
  attention block size chosen to match the Mamba page size, which makes
  prefix-cache hits coarser.
- The first 16 tokens of each request are also kept in the window
  (attention sinks), which matters for models such as Qwen3.
- Accuracy of the 2-bit keys is strongly model dependent. Validate
  `turboquant_k2v2_oscar` on your model before using it:

| Model | `turboquant_4bit_nc` | `k3v3_oscar` | `k3v2_oscar` | `k2v2_oscar` |
|---|---|---|---|---|
| Qwen3-4B, GSM8K (bf16 KV: 0.861) | 0.824 | 0.831 | 0.763 | 0.204 |
| Ternary-Bonsai-2-27B, long-context retrieval (144 q) | 141 | — | 138 | 135–137 |

Bonsai was measured with `oscar_sink` set to 0; GSM8K prompts (~1K tokens)
mostly fit in the 1024-token window of the `k3v2`/`k2v2` presets.
