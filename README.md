<h1 style="display: flex; align-items: center;" >
     <img width="100" height="100" alt="tool icon" src="https://neuralmagic.com/wp-content/uploads/2024/04/icon_nm_vllm-002-copy.svg" />
      <span>&nbsp;&nbsp;nm-vllm</span>
  </h1>

## Overview

[vLLM](https://github.com/vllm-project/vllm) is a fast and easy-to-use library for LLM inference that Neural Magic regularly contributes to. 

`nm-vllm` is our supported enterprise distribution of vLLM.

## Installation

The [nm-vllm PyPi package](https://pypi.neuralmagic.com/simple/nm-vllm/index.html) includes pre-compiled binaries for CUDA (version 12.1) kernels, streamlining the setup process. For other PyTorch or CUDA versions, please compile the package from source.

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

## Models

Neural Magic maintains a variety of optimized models on our Hugging Face organization profiles:
- [neuralmagic](https://huggingface.co/neuralmagic)
- [nm-testing](https://huggingface.co/nm-testing)
