# vllm-project/vllm Overview

This document provides a high-level overview of the vLLM repository structure for new contributors working on the codebase.
Given the rapid development of the codebase, this document focuses on aspects that are less likely to change frequently.

## Core Directories

- `vllm/`: The main Python package for vLLM. This is where the core inference engine, model implementations, and utilities reside. Key subdirectories:
    - `attention/`: Attention mechanism implementations
    - `compilation/`: Manages `torch.compile` integration and custom fusion
    - `distributed/`: Multi-device/node distribution logic
    - `engine/`: Core inference engine (`LLMEngine`, `AsyncLLMEngine`)
    - `entrypoints/`: API interfaces (CLI, OpenAI API, `LLM` class for offline inference)
    - `executor/`: Responsible for executing the model on one device, or it can be a distributed executor that can execute the model on multiple devices
    - `lora/`: Multi-LoRA batching support
    - `model_executor/`: Model definitions (`models/`), weight loading (`model_loader/`), and layer definitions (`layers/`)
    - `multimodal/`: Image/video/audio-language model support
    - `platforms/`: Hardware-specific utilities
    - `worker/`: Hardware-specific execution classes (`ModelRunner`, `Worker`)

- `csrc/`: Contains C++/CUDA source code compiled for performance-critical kernels and operations. Most compilation is specified in `CMakeLists.txt` Important subdirectories:
    - `attention/`: Attention operations
    - `core/`: Core utilities
    - `cpu/`: CPU-specific kernels
    - `mamba/`: Kernels for state-space models
    - `moe/`: Mixture of Experts operations
    - `quantization/`: Quantization kernels
    - `rocm/`: ROCm-specific kernels
    - `torch_bindings.cpp` and `ops.h` are where compiled operations are registered to PyTorch

- `docker/`: Dockerfiles for building vLLM wheels and containers. Used for CI and release.

- `docs/`: Project documentation, including API references, tutorials, and design documents.

- `tests/`: Pytest-based test suite organized by component. Execution of most CI tests are defined in `.buildkite/test-pipeline.yaml`
