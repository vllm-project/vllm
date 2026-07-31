# Portable PyTorch Kimi

This package is a text-only Kimi K3 and Kimi-Linear implementation in the same
style as other vLLM models. It accepts `VllmConfig`, flattened token tensors,
engine-managed attention state, and standard vLLM checkpoint loading.

The model reuses vLLM's tensor-parallel linear, embedding, logits,
weight-loading, attention-registration, and inner-state infrastructure. The
K3-specific math stays explicit:

- MLA is a registered vLLM `Attention` layer. It expands its latent projection
  into ordinary K/V before calling the standard attention path.
- KDA performs causal convolution and the recurrent delta update with PyTorch
  tensor operations.
- Standard and Stable LatentMoE route tokens with a direct, readable expert
  loop.
- SITU and attention residuals are written as ordinary PyTorch equations.

There are no custom K3 fused kernels and no model-facing cache objects. MLA KV
cache and KDA recurrent state are allocated and passed by vLLM through their
registered attention layers.

Pipeline parallelism is intentionally unsupported. Tensor parallelism uses
vLLM's standard sharded linear, embedding, and collective layers.
