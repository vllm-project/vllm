(quark)=

# Quark for ROCm

[Quark](https://github.com/ROCm/quark) is AMD's quantization tool designed specifically for ROCm platforms. It supports various quantization schemes including FP8, INT8, and UINT4, and can optimize models for running on AMD GPUs such as the MI300x.

## Installation

```console
pip install quark
```

For the latest version, you can install from source:

```console
git clone https://github.com/ROCm/quark.git
cd quark
pip install -e .
```

## Quantization Schemes

Quark supports multiple quantization schemes for vLLM:

- **FP8 Weight & Activation**: `w_fp8_a_fp8`
- **FP8 Weight, Activation & Output**: `w_fp8_a_fp8_o_fp8`
- **FP8 with KV Cache**: Add `--kv_cache_dtype fp8` to enable FP8 KV cache
- **INT8**: `w_int8_a_int8_per_tensor_sym`
- **4-bit Weight Only**: `w_uint4_per_group_asym` or `w_int4_per_group_asym` with AWQ/GPTQ algorithms

## Example: Quantizing a Model with Quark

To quantize a Llama model to FP8 and export it in a vLLM-compatible format:

```console
python -m quark.examples.torch.language_modeling.quantize_quark \
    --model_dir meta-llama/Llama-3-8B-hf \
    --output_dir llama3-8b-fp8 \
    --quant_scheme w_fp8_a_fp8 \
    --num_calib_data 128 \
    --model_export vllm_adopted_safetensors
```

For 4-bit AWQ quantization:

```console
python -m quark.examples.torch.language_modeling.quantize_quark \
    --model_dir meta-llama/Llama-3-8B-hf \
    --output_dir llama3-8b-awq \
    --quant_scheme w_uint4_per_group_asym \
    --quant_algo awq \
    --num_calib_data 128 \
    --group_size 32 \
    --model_export quark_safetensors
```

## Loading Quark-quantized Models in vLLM

```python
from vllm import LLM

model = LLM("llama3-8b-fp8")

outputs = model.generate("What is artificial intelligence?")
```

For more information about Quark's capabilities and advanced options, refer to the [Quark documentation](https://rocm.docs.amd.com/projects/quark/en/latest/).
