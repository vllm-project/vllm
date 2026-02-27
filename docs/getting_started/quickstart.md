# Quickstart

This guide will help you quickly get started with vLLM to perform:

- [Offline batched inference](#offline-batched-inference)
- [Online serving using OpenAI-compatible server](#openai-compatible-server)

## Prerequisites

- OS: Linux
- Python: 3.10 -- 3.13

## Installation

=== "NVIDIA CUDA"

    If you are using NVIDIA GPUs, you can install vLLM using [pip](https://pypi.org/project/vllm/) directly.

    It's recommended to use [uv](https://docs.astral.sh/uv/), a very fast Python environment manager, to create and manage Python environments. Please follow the [documentation](https://docs.astral.sh/uv/#getting-started) to install `uv`. After installing `uv`, you can create a new Python environment and install vLLM using the following commands:

    ```bash
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    uv pip install vllm --torch-backend=auto
    ```

    `uv` can [automatically select the appropriate PyTorch index at runtime](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection) by inspecting the installed CUDA driver version via `--torch-backend=auto` (or `UV_TORCH_BACKEND=auto`). To select a specific backend (e.g., `cu126`), set `--torch-backend=cu126` (or `UV_TORCH_BACKEND=cu126`).

    Another delightful way is to use `uv run` with `--with [dependency]` option, which allows you to run commands such as `vllm serve` without creating any permanent environment:

    ```bash
    uv run --with vllm vllm --help
    ```

    You can also use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) to create and manage Python environments. You can install `uv` to the conda environment through `pip` if you want to manage it within the environment.

    ```bash
    conda create -n myenv python=3.12 -y
    conda activate myenv
    pip install --upgrade uv
    uv pip install vllm --torch-backend=auto
    ```

=== "AMD ROCm"

    If you are using AMD GPUs, you can install vLLM using `uv`.

    It's recommended to use [uv](https://docs.astral.sh/uv/), as it gives the extra index [higher priority than the default index](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes). `uv` is also a very fast Python environment manager, to create and manage Python environments. Please follow the [documentation](https://docs.astral.sh/uv/#getting-started) to install `uv`. After installing `uv`, you can create a new Python environment and install vLLM using the following commands:

    ```bash
    uv venv --python 3.12 --seed
    source .venv/bin/activate
    uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
    ```

    !!! note
        It currently supports Python 3.12, ROCm 7.0 and `glibc >= 2.35`.

    !!! note    
        Note that, previously, docker images were published using AMD's docker release pipeline and were located `rocm/vllm-dev`. This is being deprecated by using vLLM's docker release pipeline.

=== "Google TPU"

    To run vLLM on Google TPUs, you need to install the `vllm-tpu` package.
    
    ```bash
    uv pip install vllm-tpu
    ```

    !!! note
        For more detailed instructions, including Docker, installing from source, and troubleshooting, please refer to the [vLLM on TPU documentation](https://docs.vllm.ai/projects/tpu/en/latest/).

!!! note
    For more detail and non-CUDA platforms, please refer to the [installation guide](installation/README.md) for specific instructions on how to install vLLM.

## Offline Batched Inference

With vLLM installed, you can start generating texts for list of input prompts (i.e. offline batch inferencing). See the example script: [examples/offline_inference/basic/basic.py](../../examples/offline_inference/basic/basic.py)

The first line of this example imports the classes [LLM][vllm.LLM] and [SamplingParams][vllm.SamplingParams]:

- [LLM][vllm.LLM] is the main class for running offline inference with vLLM engine.
- [SamplingParams][vllm.SamplingParams] specifies the parameters for the sampling process.

```python
from vllm import LLM, SamplingParams
```

The next section defines a list of input prompts and sampling parameters for text generation. The [sampling temperature](https://arxiv.org/html/2402.05201v1) is set to `0.8` and the [nucleus sampling probability](https://en.wikipedia.org/wiki/Top-p_sampling) is set to `0.95`. You can find more information about the sampling parameters [here](../api/README.md#inference-parameters).

!!! important
    By default, vLLM will use sampling parameters recommended by model creator by applying the `generation_config.json` from the Hugging Face model repository if it exists. In most cases, this will provide you with the best results by default if [SamplingParams][vllm.SamplingParams] is not specified.

    However, if vLLM's default sampling parameters are preferred, please set `generation_config="vllm"` when creating the [LLM][vllm.LLM] instance.

```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```

The [LLM][vllm.LLM] class initializes vLLM's engine and the [OPT-125M model](https://arxiv.org/abs/2205.01068) for offline inference. The list of supported models can be found [here](../models/supported_models.md).

```python
llm = LLM(model="facebook/opt-125m")
```

!!! note
    By default, vLLM downloads models from [Hugging Face](https://huggingface.co/). If you would like to use models from [ModelScope](https://www.modelscope.cn), set the environment variable `VLLM_USE_MODELSCOPE` before initializing the engine.

    ```shell
    export VLLM_USE_MODELSCOPE=True
    ```

Now, the fun part! The outputs are generated using `llm.generate`. It adds the input prompts to the vLLM engine's waiting queue and executes the vLLM engine to generate the outputs with high throughput. The outputs are returned as a list of `RequestOutput` objects, which include all of the output tokens.

```python
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

!!! note
    The `llm.generate` method does not automatically apply the model's chat template to the input prompt. Therefore, if you are using an Instruct model or Chat model, you should manually apply the corresponding chat template to ensure the expected behavior. Alternatively, you can use the `llm.chat` method and pass a list of messages which have the same format as those passed to OpenAI's `client.chat.completions`:

    ??? code
    
        ```python
        # Using tokenizer to apply chat template
        from transformers import AutoTokenizer
    
        tokenizer = AutoTokenizer.from_pretrained("/path/to/chat_model")
        messages_list = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]
        texts = tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Generate outputs
        outputs = llm.generate(texts, sampling_params)
        
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
        # Using chat interface.
        outputs = llm.chat(messages_list, sampling_params)
        for idx, output in enumerate(outputs):
            prompt = prompts[idx]
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        ```

## OpenAI-Compatible Server

vLLM can be deployed as a server that implements the OpenAI API protocol. This allows vLLM to be used as a drop-in replacement for applications using OpenAI API.
By default, it starts the server at `http://localhost:8000`. You can specify the address with `--host` and `--port` arguments. The server currently hosts one model at a time and implements endpoints such as [list models](https://platform.openai.com/docs/api-reference/models/list), [create chat completion](https://platform.openai.com/docs/api-reference/chat/completions/create), and [create completion](https://platform.openai.com/docs/api-reference/completions/create) endpoints.

Run the following command to start the vLLM server with the [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) model:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

!!! note
    By default, the server uses a predefined chat template stored in the tokenizer.
    You can learn about overriding it [here](../serving/openai_compatible_server.md#chat-template).
!!! important
    By default, the server applies `generation_config.json` from the huggingface model repository if it exists. This means the default values of certain sampling parameters can be overridden by those recommended by the model creator.

    To disable this behavior, please pass `--generation-config vllm` when launching the server.

This server can be queried in the same format as OpenAI API. For example, to list the models:

```bash
curl http://localhost:8000/v1/models
```

You can pass in the argument `--api-key` or environment variable `VLLM_API_KEY` to enable the server to check for API key in the header.
You can pass multiple keys after `--api-key`, and the server will accept any of the keys passed, this can be useful for key rotation.

### OpenAI Completions API with vLLM

Once your server is started, you can query the model with input prompts:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

Since this server is compatible with OpenAI API, you can use it as a drop-in replacement for any applications using OpenAI API. For example, another way to query the server is via the `openai` Python package:

??? code

    ```python
    from openai import OpenAI

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    completion = client.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        prompt="San Francisco is a",
    )
    print("Completion result:", completion)
    ```

A more detailed client example can be found here: [examples/offline_inference/basic/basic.py](../../examples/offline_inference/basic/basic.py)

### OpenAI Chat Completions API with vLLM

vLLM is designed to also support the OpenAI Chat Completions API. The chat interface is a more dynamic, interactive way to communicate with the model, allowing back-and-forth exchanges that can be stored in the chat history. This is useful for tasks that require context or more detailed explanations.

You can use the [create chat completion](https://platform.openai.com/docs/api-reference/chat/completions/create) endpoint to interact with the model:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'
```

Alternatively, you can use the `openai` Python package:

??? code

    ```python
    from openai import OpenAI
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ],
    )
    print("Chat response:", chat_response)
    ```

## On Attention Backends

Currently, vLLM supports multiple backends for efficient Attention computation across different platforms and accelerator architectures. It automatically selects the most performant backend compatible with your system and model specifications.

If desired, you can also manually set the backend of your choice using the `--attention-backend` CLI argument:

```bash
# For online serving
vllm serve Qwen/Qwen2.5-1.5B-Instruct --attention-backend FLASH_ATTN

# For offline inference
python script.py --attention-backend FLASHINFER
```

Some of the available backend options include:

- On NVIDIA CUDA: `FLASH_ATTN` or `FLASHINFER`.
- On AMD ROCm: `TRITON_ATTN`, `ROCM_ATTN`, `ROCM_AITER_FA`, `ROCM_AITER_UNIFIED_ATTN`, `TRITON_MLA`, `ROCM_AITER_MLA` or `ROCM_AITER_TRITON_MLA`.

!!! warning
    There are no pre-built vllm wheels containing Flash Infer, so you must install it in your environment first. Refer to the [Flash Infer official docs](https://docs.flashinfer.ai/) or see [docker/Dockerfile](../../docker/Dockerfile) for instructions on how to install it.
