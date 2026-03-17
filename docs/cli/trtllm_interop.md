# TensorRT-LLM Interoperability

TensorRT-LLM is treated here as an optional NVIDIA interoperability path, not as a replacement for native vLLM CUDA execution.

## Scope

The local CLI surfaces TensorRT-LLM relevance through diagnostics:

```bash
vllm doctor
vllm doctor deepseek-r1:8b
vllm inspect deepseek-r1:8b --json
```

The report focuses on:

- whether the current environment looks TensorRT-LLM-capable
- whether FlashInfer is available
- whether vLLM's existing TensorRT-LLM-related hooks are likely relevant
- whether the requested model family is one where those hooks are meaningful

## What This Is Not

This local-runtime layer does not:

- replace native vLLM CUDA execution with TensorRT-LLM
- force TensorRT-LLM into CPU or Apple flows
- claim universal model support

## Expected Use

Use the diagnostics output to decide when to stay on native vLLM CUDA paths and when deeper TensorRT-LLM-specific work is worth exploring.

Future work can add a more explicit export or staging command, but the first step is inspectability and clear eligibility reporting.
