# LLM Compressor

[LLM Compressor](https://docs.vllm.ai/projects/llm-compressor/en/latest/) is a library for optimizing models for deployment with vLLM.
It provides a comprehensive set of quantization algorithms, including support for techniques such as FP4, FP8, INT8, and INT4 quantization.

## Why use LLM Compressor?

Modern LLMs often contain billions of parameters stored in 16-bit or 32-bit floating point, requiring substantial GPU memory and limiting deployment options.
Quantization lowers memory requirements while maintaining inference output quality by reducing the precision of model weights and activations to smaller data types.

LLM Compressor provides the following benefits:

- **Reduced memory footprint**: Run larger models on smaller GPUs.
- **Lower inference costs**: Serve more concurrent users per GPU, directly reducing the cost per query in production deployments.
- **Faster inference**: Smaller data types mean less memory bandwidth consumed, which often translates to higher throughput, especially for memory-bound workloads.

LLM Compressor handles the complexity of quantization, calibration, and format conversion, producing models ready for immediate use with vLLM.

## Key features

- **Multiple Quantization Algorithms**: Support for AWQ, GPTQ, AutoRound, and Round-to-Nearest.
Also includes support for QuIP and SpinQuant-style transforms as well as KV cache and attention quantization.
- **Multiple Quantization Methods**: Support for FP8, INT8, INT4, NVFP4, MXFP4, and mixed-precision quantization
- **One-Shot Quantization**: Quantize models quickly with minimal calibration data
- **vLLM Integration**: Seamlessly deploy quantized models with vLLM using the compressed-tensors format
- **Hugging Face Compatibility**: Works with models from the Hugging Face Hub

## Resources

- [LLM Compressor examples](https://github.com/vllm-project/llm-compressor/tree/main/examples)
- [GitHub Repository](https://github.com/vllm-project/llm-compressor)

## Experimental Rubin LUT-B evaluation

The `lut_b` linear backend is a calibration-free accuracy prototype for dense
NVFP4 linear layers. During checkpoint loading, it dequantizes each NVFP4
weight, fits an eight-entry E4M3 codebook for every 8x64 tile, and packs one
3-bit index per weight. The resulting representation uses 3.125 bits per
weight.

The reference forward fully reconstructs the BF16/FP16 weight before
`torch.nn.functional.linear`. It does not use the Rubin MMA and is intended
only for accuracy evaluation:

```bash
vllm serve RedHatAI/Qwen3-8B-NVFP4 \
  --linear-backend lut_b \
  --enforce-eager \
  --port 8000
```

For a useful GSM8K comparison, run these four configurations:

| Input checkpoint | Options | Measurement |
| ---------------- | ------- | ----------- |
| `Qwen/Qwen3-8B` | `--quantization lut_b --enforce-eager` | Direct BF16-to-LUT-B weight-only loss |
| `RedHatAI/Qwen3-8B-NVFP4` | Default options | NVFP4 weight-and-activation quantization |
| `RedHatAI/Qwen3-8B-NVFP4` | `--linear-backend marlin` | Existing NVFP4 weights with unquantized activations |
| `RedHatAI/Qwen3-8B-NVFP4` | `--linear-backend lut_b --enforce-eager` | NVFP4-to-LUT-B weight-only conversion |

Run the default case on hardware with a native NVFP4 W4A4 backend and confirm
the selected backend in the server log. On older hardware, automatic selection
can fall back to Marlin and duplicate the explicit weight-only measurement.

Run the internal GSM8K harness against each server, changing the result
filename:

```bash
.venv/bin/python tests/evals/gsm8k/gsm8k_eval.py \
  --port 8000 \
  --num-questions 1319 \
  --num-shots 5 \
  --temperature 0 \
  --max-concurrency 32 \
  --save-results lut-b-qwen3-8b.json
```

The prototype supports dense NVFP4 linear weights whose sharded K dimension
is divisible by 64 and whose logical output widths are divisible by 8. MoE
weights and the hardware-specific GMEM/SMEM/TMEM layouts are not included.
