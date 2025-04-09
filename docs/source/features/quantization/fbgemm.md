# FBGEMM GenAI

FBGEMM GenAI (FBGEMM Generative AI Kernels Library)

# **1. Overview**

FBGEMM FP8 rowwise quantization kernels have been officially adopted in the [Llama3.1 release](https://fb.workplace.com/groups/221503021668016/permalink/1900301927121442/). FP8 has been applied across Llama3 models with 8 B, 70 B, and 405 B. Notably, for the 405 B model, FP8 enables the inference on a single node, achieving a 2x throughput improvement over the baseline BF16 running on two nodes with pipeline parallelism. Externally, it has been mentioned in [Llama3 paper](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) & [repo](https://github.com/meta-llama/llama-toolchain/tree/main/llama_toolchain/inference/quantization), [HuggingFace](https://huggingface.co/docs/transformers/main/quantization/fbgemm_fp8), [vLLM](https://blog.vllm.ai/2024/07/23/llama31.html), and [TensorRT-LLM](https://developer.nvidia.com/blog/supercharging-llama-3-1-across-nvidia-platforms/).

FBGEMM GenAI FP8 supports a variety of configurations:

* GEMM Operators: {CUTLASS, CK, Triton} x {BF16, FP8} x {tensor-wise, row-wise, block-wise} x {Nvidia H100, AMD MI300x}.
* High/low Precision Conversion Kernels: (FP32 / BF16 <-> FP8) with scaling options {tensor-wise, row-wise, block-wise} across hardware platforms {Nvidia H100, AMD MI300x} and programming options of {Triton, CUDA/HIP}.

Besides FP8 support, FBGEMM GenAI operators also support:

* Customized AllReduce communications (reduce latency for small message sizes).
* GQA: optimized specifically for decoding cases, as detailed in PyTorch's blog on [INT4 decoding](https://pytorch.org/blog/int4-decoding/).
* KV cache quantizations.
* Rotary Positional Embedding (RoPE).

## **1.1 FP8 core API functions**

```python
# Rowwise quantize (channel wise) the weight from BF16 to FP8
wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
# Rowwise quantize the activation (token wise) from BF16 to FP8
xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(
    x, num_tokens, activation_scale_ub
)
# Rowwise quantize GEMM with FP8 input and BF16 output
y = torch.ops.fbgemm.f8f8bf16_rowwise(
    xq,
    wq,
    x_scale,
    w_scale,
    use_fast_accum=True,
)
```

## **1.2 How to install**

```bash
# Full FBGEMM library
pip install fbgemm-gpu==1.1.0
pip install fbgemm-gpu==1.1.0 --index-url https://download.pytorch.org/whl/cu126

# FBGEMM library with GenAI operator only
pip install fbgemm-gpu-genai
```

# 2. **External Coverage**

## 2.1 **Llama3 Paper**

[Llama3 paper](https://arxiv.org/pdf/2407.21783)

> We perform experiments leveraging the native FP8 support of H100 GPUs to perform low-precision inference. To enable low-precision inference, we apply FP8 quantization to most matrix multiplications inside the model. In particular, we quantize most parameters and activations in the feedforward network layers in the model, which account for roughly 50% of the inference compute time. We do not quantize parameters in the self-attention layers of the model. We leverage dynamic scaling factors for better accuracy (Xiao et al., 2024b), optimizing our CUDA kernels to reduce the overhead of calculating the scales.

> Our FP8 kernels are available at https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/gen_ai. We provide usage examples at https://github.com/meta-llama/llama-agentic-system.

## 2.2 **Llama3 Repo**

* [Llama Toolchain](https://github.com/meta-llama/llama-toolchain/tree/main/llama_toolchain/inference/quantization)
* [Llama Agentic System](https://github.com/meta-llama/llama-agentic-system/tree/main?tab=readme-ov-file#running-fp8)

## 2.3 **HuggingFace**

[FBGEMM FP8](https://huggingface.co/docs/transformers/main/quantization/fbgemm_fp8)

> With FBGEMM FP8 quantization method, you can quantize your model in FP8 (W8A8):

> * the weights will be quantized in 8bit (FP8) per channel
> * the activation will be quantized in 8bit (FP8) per token

> It relies on the [FBGEMM](https://github.com/pytorch/FBGEMM) library which provides efficient low-precision general matrix multiplication for small batch sizes and support for accuracy-loss minimizing techniques such as row-wise quantization and outlier-aware quantization.

## 2.4 **vLLM**

[Announcing Llama 3.1 Support in vLLM](https://blog.vllm.ai/2024/07/23/llama31.html)

> Currently, vLLM supports the official Meta Llama 3.1 405B FP8 model quantized via **FBGEMM** by leveraging per-channel quantization in the MLP layer. In particular, each channel of the up/gate/down projections are quantized and multiplied by a static scaling factor. Combined with skipping quantization for the first and the last layer, and a static upper bound, this approach has minimal impact on the model’s accuracy.


## 2.5 **TensorRT-LLM**

[Supercharging Llama 3.1 across NVIDIA Platforms](https://developer.nvidia.com/blog/supercharging-llama-3-1-across-nvidia-platforms/)

[Code pointer](https://github.com/NVIDIA/TensorRT-LLM/blame/5fa9436e17c2f9aeace070f49aa645d2577f676b/cpp/tensorrt_llm/common/quantTypeUtils.cuh#L47)

```cpp
// Ref: https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/experimental/gen_ai/src/quantize/quantize.cu#L720
```

> During the TensorRT engine build process, some complex layer fusions cannot be automatically discovered. TensorRT-LLM optimizes these using plugins that are explicitly inserted into the network graph definition at compile time to replace user-defined kernels such as the matrix multiplications from **FBGEMM** for the Llama 3.1 models.
