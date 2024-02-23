# Neural Magic vLLM

## About

[vLLM](https://github.com/vllm-project/vllm) is a fast and easy-to-use library for LLM inference and serving that Neural Magic regularly lands upstream improvements to. This fork is our opinionated focus on the latest LLM optimizations, such as quantization and sparsity.

## Installation

`nm-vllm` is a Python library that contained pre-compiled C++ and CUDA (12.1) binaries.

Install it using pip (coming soon):
```bash
pip install nm-vllm
```

You can also build and install `nm-vllm` from source (this will take ~10 minutes):
```bash
git clone https://github.com/neuralmagic/neuralmagic-vllm.git
cd neuralmagic-vllm
pip install -e .
```

In order to use the weight-sparsity kernels, like through `sparsity="sparse_w16a16"`, you must also install `magic_wand`:
```bash
pip install magic_wand
```

## Quickstart

There are many sparse models already pushed up on our HF organization profiles, [neuralmagic](https://huggingface.co/neuralmagic) and [nm-testing](https://huggingface.co/nm-testing). You can find [this collection of SparseGPT models ready for inference](https://huggingface.co/collections/nm-testing/sparsegpt-llms-65ca6def5495933ab05cd439).

Here is a smoke test using a small test `llama2-110M` model train on storytelling:

```python
from vllm import LLM, SamplingParams

model = LLM(
    "nm-testing/llama2.c-stories110M-pruned2.4", 
    sparsity="sparse_w16a16",   # If left off, model will be loaded as dense
)

sampling_params = SamplingParams(max_tokens=100, temperature=0)
outputs = model.generate("Hello my name is", sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

Here is a more realistic example of running a 50% sparse OpenHermes 2.5 Mistral 7B model finetuned for instruction-following:

```python
from vllm import LLM, SamplingParams

model = LLM(
    "nm-testing/OpenHermes-2.5-Mistral-7B-pruned50",
    sparsity="sparse_w16a16",
    max_model_len=1024
)

sampling_params = SamplingParams(max_tokens=100, temperature=0)
outputs = model.generate("Hello my name is", sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
```

You can also quickly use the same flow with an OpenAI-compatible model server:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model nm-testing/OpenHermes-2.5-Mistral-7B-pruned50 \
    --sparsity sparse_w16a16
```
