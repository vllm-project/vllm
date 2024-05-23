<h1 style="display: flex; align-items: center;" >
     <img width="100" height="100" alt="tool icon" src="https://neuralmagic.com/wp-content/uploads/2024/04/icon_nm_vllm-002-copy.svg" />
      <span>&nbsp;&nbsp;nm-vllm</span>
  </h1>

## Overview

[vLLM](https://github.com/vllm-project/vllm) is a fast and easy-to-use library for LLM inference that Neural Magic regularly contributes upstream improvements to. This fork, `nm-vllm` is our opinionated focus on incorporating the latest LLM optimizations like quantization and sparsity for enhanced performance.

## Installation
The [nm-vllm PyPi package](https://pypi.org/project/nm-vllm/) includes pre-compiled binaries for CUDA (version 12.1) kernels, streamlining the setup process. For other PyTorch or CUDA versions, please compile the package from source.

Install it using pip:
```bash
pip install nm-vllm --extra-index-url https://pypi.neuralmagic.com/simple
```

For utilizing weight-sparsity kernels, such as through `sparsity="sparse_w16a16"`, you can extend the installation with the `sparsity` extras:
```bash
pip install nm-vllm[sparse] --extra-index-url https://pypi.neuralmagic.com/simple
```

You can also build and install `nm-vllm` from source (this will take ~10 minutes):
```bash
git clone https://github.com/neuralmagic/nm-vllm.git
cd nm-vllm
pip install -e .
```

## Quickstart

Neural Magic maintains a variety of sparse models on our Hugging Face organization profiles, [neuralmagic](https://huggingface.co/neuralmagic) and [nm-testing](https://huggingface.co/nm-testing).

A collection of ready-to-use SparseGPT and GPTQ models in inference optimized marlin format are [available on Hugging Face](https://huggingface.co/collections/neuralmagic/compressed-llms-for-nm-vllm-65e73e3d51d3200e34b77431)

#### Model Inference with Marlin (4-bit Quantization)

Marlin is an extremely optimized FP16xINT4 matmul kernel aimed at LLM inference that can deliver close to ideal (4x) speedups up to batchsizes of 16-32 tokens.
To use Marlin within nm-vllm, simply pass the Marlin quantized directly to the engine. It will detect the quantization from the model's config.

Here is a demonstraiton with a [4-bit quantized OpenHermes Mistral](https://huggingface.co/neuralmagic/OpenHermes-2.5-Mistral-7B-marlin) model:

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/OpenHermes-2.5-Mistral-7B-marlin"
model = LLM(model_id, max_model_len=4096)
tokenizer = AutoTokenizer.from_pretrained(model_id)
sampling_params = SamplingParams(max_tokens=100, temperature=0.8, top_p=0.95)

messages = [
    {"role": "user", "content": "What is synthetic data in machine learning?"},
]
formatted_prompt =  tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = model.generate(formatted_prompt, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

#### Model Inference with Weight Sparsity

For a quick demonstration, here's how to run a small [50% sparse llama2-110M](https://huggingface.co/nm-testing/llama2.c-stories110M-pruned50) model trained on storytelling:

```python
from vllm import LLM, SamplingParams

model = LLM(
    "neuralmagic/llama2.c-stories110M-pruned50",
    sparsity="sparse_w16a16",   # If left off, model will be loaded as dense
)

sampling_params = SamplingParams(max_tokens=100, temperature=0)
outputs = model.generate("Hello my name is", sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

Here is a more realistic example of running a 50% sparse OpenHermes 2.5 Mistral 7B model finetuned for instruction-following:

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/OpenHermes-2.5-Mistral-7B-pruned50"
model = LLM(model_id, sparsity="sparse_w16a16", max_model_len=4096)
tokenizer = AutoTokenizer.from_pretrained(model_id)
sampling_params = SamplingParams(max_tokens=100, temperature=0.8, top_p=0.95)

messages = [
    {"role": "user", "content": "What is sparsity in deep learning?"},
]
formatted_prompt =  tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = model.generate(formatted_prompt, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

There is also support for semi-structured 2:4 sparsity using the `sparsity="semi_structured_sparse_w16a16"` argument:
```python
from vllm import LLM, SamplingParams

model = LLM("neuralmagic/llama2.c-stories110M-pruned2.4", sparsity="semi_structured_sparse_w16a16")
sampling_params = SamplingParams(max_tokens=100, temperature=0)
outputs = model.generate("Once upon a time, ", sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

#### Integration with OpenAI-Compatible Server

You can also quickly use the same flow with an OpenAI-compatible model server:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model neuralmagic/OpenHermes-2.5-Mistral-7B-pruned50 \
    --sparsity sparse_w16a16 \
    --max-model-len 4096
```

## Quantized Inference Performance

Developed in collaboration with IST-Austria, [GPTQ](https://arxiv.org/abs/2210.17323) is the leading quantization algorithm for LLMs, which enables compressing the model weights from 16 bits to 4 bits with limited impact on accuracy. nm-vllm includes support for the recently-developed Marlin kernels for accelerating GPTQ models. Prior to Marlin, the existing kernels for INT4 inference failed to scale in scenarios with multiple concurrent users.

<p align="center">
   <img alt="Marlin Performance" src="https://github.com/neuralmagic/nm-vllm/assets/3195154/6ac9f5b0-667a-41f3-8e6d-ca51c268bec5" width="60%" />
</p>

## Sparse Inference Performance

Developed in collaboration with IST-Austria, [SparseGPT](https://arxiv.org/abs/2301.00774) and [Sparse Fine-tuning](https://arxiv.org/abs/2310.06927) are the leading algorithms for pruning LLMs, which enables removing at least half of model weights with limited impact on accuracy.

nm-vllm includes support for newly-developed sparse inference kernels, which provides both memory reduction and acceleration of sparse models leveraging sparsity.

<p align="center">
   <img alt="Sparse Memory Compression" src="https://github.com/neuralmagic/nm-vllm/assets/3195154/2fdd2212-3081-4b97-b492-a809ce23fdd3" width="40%" />
   <img alt="Sparse Inference Performance" src="https://github.com/neuralmagic/nm-vllm/assets/3195154/3448e3ee-535f-4c50-ac9b-00645673cc8c" width="40%" />
</p>

