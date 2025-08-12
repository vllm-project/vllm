# Basic

The `LLM` class provides the primary Python interface for doing offline inference, which is interacting with a model without using a separate model inference server.

## Usage

The first script in this example shows the most basic usage of vLLM. If you are new to Python and vLLM, you should start here.

```bash
python examples/offline_inference/basic/basic.py
```

The rest of the scripts include an [argument parser](https://docs.python.org/3/library/argparse.html), which you can use to pass any arguments that are compatible with [`LLM`](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html). Try running the script with `--help` for a list of all available arguments.

```bash
python examples/offline_inference/basic/classify.py
```

```bash
python examples/offline_inference/basic/embed.py
```

```bash
python examples/offline_inference/basic/score.py
```

The chat and generate scripts also accept the [sampling parameters](https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-parameters): `max_tokens`, `temperature`, `top_p` and `top_k`.

```bash
python examples/offline_inference/basic/chat.py
```

```bash
python examples/offline_inference/basic/generate.py
```

## Features

In the scripts that support passing arguments, you can experiment with the following features.

### Default generation config

The `--generation-config` argument specifies where the generation config will be loaded from when calling `LLM.get_default_sampling_params()`. If set to ‘auto’, the generation config will be loaded from model path. If set to a folder path, the generation config will be loaded from the specified folder path. If it is not provided, vLLM defaults will be used.

> If max_new_tokens is specified in generation config, then it sets a server-wide limit on the number of output tokens for all requests.

Try it yourself with the following argument:

```bash
--generation-config auto
```

### Quantization

#### AQLM

vLLM supports models that are quantized using AQLM.

Try one yourself by passing one of the following models to the `--model` argument:

- `ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf`
- `ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf`
- `ISTA-DASLab/Llama-2-13b-AQLM-2Bit-1x16-hf`
- `ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf`
- `BlackSamorez/TinyLlama-1_1B-Chat-v1_0-AQLM-2Bit-1x16-hf`

> Some of these models are likely to be too large for a single GPU. You can split them across multiple GPUs by setting `--tensor-parallel-size` to the number of required GPUs.

#### GGUF

vLLM supports models that are quantized using GGUF.

Try one yourself by downloading a GUFF quantised model and using the following arguments:

```python
from huggingface_hub import hf_hub_download
repo_id = "bartowski/Phi-3-medium-4k-instruct-GGUF"
filename = "Phi-3-medium-4k-instruct-IQ2_M.gguf"
print(hf_hub_download(repo_id, filename=filename))
```

```bash
--model {local-path-printed-above} --tokenizer microsoft/Phi-3-medium-4k-instruct
```

### CPU offload

The `--cpu-offload-gb` argument can be seen as a virtual way to increase the GPU memory size. For example, if you have one 24 GB GPU and set this to 10, virtually you can think of it as a 34 GB GPU. Then you can load a 13B model with BF16 weight, which requires at least 26GB GPU memory. Note that this requires fast CPU-GPU interconnect, as part of the model is loaded from CPU memory to GPU memory on the fly in each model forward pass.

Try it yourself with the following arguments:

```bash
--model meta-llama/Llama-2-13b-chat-hf --cpu-offload-gb 10
```
