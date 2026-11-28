# SM89 BF16 gated projection

The dense Qwen3.5 MLP has an opt-in gate/up projection and SiLU-and-multiply fusion
on SM89. Measurements used RTX 4090; other SM89 devices need their own performance
qualification. The measured path targets BF16, hidden size 2560, intermediate size 9216,
and 768–4096 input tokens. Other input sizes use the existing linear and
activation operations, including captured decode graphs.

This path requires unquantized linears, tensor parallel size one, the automatic
linear backend, and no LoRA, weight offloading, or batch-invariant mode. Expert-gated MLPs and
alternate linear or activation implementations keep their existing path.

The kernel reads the original gate/up rows in paired order. It creates no
transformed weight copy and sees weight updates at the original storage address.
The token count is a runtime argument, so changing it does not require a kernel
binary for every prefill size.

Projection, SiLU, and product retain their BF16 cast boundaries. GEMM accumulation
can differ from the vendor implementation; this optimization does not promise
bitwise-identical generated tokens. Model-level quality checks remain necessary.

Enable the path explicitly through the existing custom-op control:

```console
vllm serve MODEL --compilation-config '{"custom_ops": ["all", "+gated_linear_silu"]}'
```

The base mode `all` alone leaves this integration disabled. This avoids changing
defaults based on measurements from a single GPU/model configuration. Keep your
other custom-op overrides when adding this opt-in.
