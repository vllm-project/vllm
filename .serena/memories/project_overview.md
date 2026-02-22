# vLLM Project Overview

## Purpose
vLLM is a fast and easy-to-use library for LLM (Large Language Model) inference and serving, developed originally in the Sky Computing Lab at UC Berkeley and now a community-driven project under the PyTorch Foundation.

## Key Features
- State-of-the-art serving throughput with PagedAttention
- Efficient memory management for attention key and value memory
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Support for various quantizations (GPTQ, AWQ, AutoRound, INT4, INT8, FP8)
- Optimized CUDA kernels with FlashAttention and FlashInfer integration
- Speculative decoding and chunked prefill
- Seamless integration with Hugging Face models
- Multi-modal LLM support
- OpenAI-compatible API server

## Tech Stack
- **Primary Language**: Python (3.10-3.13 supported, 3.12 recommended for development)
- **ML Framework**: PyTorch 2.9.0
- **Build System**: CMake 3.26.1+, setuptools-scm, ninja
- **Compute**: CUDA, ROCm, CPU, TPU, Intel Gaudi, IBM Spyre, Huawei Ascend
- **Testing**: pytest, pytest-asyncio
- **Documentation**: MkDocs with Material theme
- **Code Quality**: pre-commit hooks (ruff, mypy, clang-format, typos, markdownlint, actionlint)

## Architecture
The project follows a modular architecture:
- Core engine logic in `vllm/engine/`
- Model execution in `vllm/model_executor/`
- Attention mechanisms in `vllm/attention/`
- Distributed inference in `vllm/distributed/`
- Multi-modal support in `vllm/multimodal/`
- V1 architecture (major upgrade) in `vllm/v1/`
- CUDA/C++ kernels in `csrc/`
- Entry points (CLI, API server) in `vllm/entrypoints/`

## License
Apache 2.0 with DCO (Developer Certificate of Origin) requirement
