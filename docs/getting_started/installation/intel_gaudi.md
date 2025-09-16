# Intel Gaudi (HPU)

vLLM supports Intel Gaudi (Habana Processing Unit) accelerators through the
`vllm-gaudi` plugin that is maintained by the vLLM project.

## Overview

Intel Gaudi accelerators provide an alternative to GPUs for running large
language models. The architecture is designed for efficient deep-learning
training and inference workloads, and exposes an execution backend that vLLM can
target via the Gaudi plugin.

## Requirements

Before installing the plugin, make sure the Intel Gaudi software stack is set
up.

- Intel Gaudi 2 or Gaudi 3 accelerator
- Intel Gaudi software (SynapseAI) **1.21.0 or newer**
- Python 3.10

## Installation

1. Install vLLM (CPU/GPU build). Either install the wheel or build from source
   with an "empty" backend so that the Gaudi plugin can supply the execution
   kernels:

   ```bash
   pip install vllm
   # or build from source
   git clone https://github.com/vllm-project/vllm
   cd vllm
   pip install -r <(sed '/^[torch]/d' requirements/build.txt)
   VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
   cd ..
   ```

2. Install the Gaudi plugin from source:

   ```bash
   git clone https://github.com/vllm-project/vllm-gaudi
   cd vllm-gaudi
   pip install -e .
   ```

## Supported Models

Validated model configurations are documented in the Gaudi plugin repository.
Consult the
[validated models list](https://github.com/vllm-project/vllm-gaudi/blob/main/docs/models/validated_models.md)
for supported models, tensor-parallel configurations, and datatypes.

## Getting Help

For Gaudi-specific issues and questions:

- Review the [vllm-gaudi issue tracker](https://github.com/vllm-project/vllm-gaudi/issues)
- Join the vLLM community chats for Gaudi topics

## See Also

- [vLLM-Gaudi Repository](https://github.com/vllm-project/vllm-gaudi)
- [vLLM Main Documentation](../../README.md)
