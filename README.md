# nm-vllm

## Overview

`nm-vllm` is our supported enterprise distribution of [vLLM](https://github.com/vllm-project/vllm).

## Installation

### PyPI
The [nm-vllm PyPi package](https://pypi.neuralmagic.com/simple/nm-vllm/index.html) includes pre-compiled binaries for CUDA (version 12.1) kernels. For other PyTorch or CUDA versions, please compile the package from source.

Install it using pip:
```bash
pip install nm-vllm --extra-index-url https://pypi.neuralmagic.com/simple
```

To utilize the weight sparsity features, include the optional `sparse` dependencies.
```bash
pip install nm-vllm[sparse] --extra-index-url https://pypi.neuralmagic.com/simple
```

You can also build and install `nm-vllm` from source (this will take ~10 minutes):
```bash
git clone https://github.com/neuralmagic/nm-vllm.git
cd nm-vllm
pip install -e .[sparse] --extra-index-url https://pypi.neuralmagic.com/simple
```

### Docker

The [`nm-vllm` container registry](https://github.com/neuralmagic/nm-vllm/pkgs/container/nm-vllm-openai) includes premade docker images.

Launch the OpenAI-compatible server with:

```bash
MODEL_ID=Qwen/Qwen2-0.5B-Instruct
docker run --gpus all --shm-size 2g ghcr.io/neuralmagic/nm-vllm-openai:latest --model $MODEL_ID
```

## Models

Neural Magic maintains a variety of optimized models on our Hugging Face organization profiles:
- [neuralmagic](https://huggingface.co/neuralmagic)
- [nm-testing](https://huggingface.co/nm-testing)
