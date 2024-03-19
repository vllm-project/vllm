# OpenAI Compatible Server

vLLM provides an HTTP server that implements OpenAI's [Completions](https://platform.openai.com/docs/api-reference/completions) and [Chat](https://platform.openai.com/docs/api-reference/chat) API.

You can start the server using Python, or using [Docker](deploying_with_docker.rst):
```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-hf --dtype float32 --api-key token-abc123
```

To call the server, you can use the official OpenAI Python client library, or any other HTTP client.
```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
  model="meta-llama/Llama-2-7b-hf",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
```

## API Reference
Please see the [OpenAI API Reference](https://platform.openai.com/docs/api-reference) for more information on the API. We support all parameters except:
- Chat: `tools`, and `tool_choice`.
- Completions: `suffix`.

## Extra Parameters
vLLM supports a set of parameters that are not part of the OpenAI API.
In order to use them, you can pass them as extra parameters in the OpenAI client.
Or directly merge them into the JSON payload if you are using HTTP call directly.

```python
completion = client.chat.completions.create(
  model="meta-llama/Llama-2-7b-hf",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}
  ],
  extra_body={
    "guided_choice": ["positive", "negative"]
  }
)
```

### Extra Parameters for Chat API
The following [sampling parameters (click through to see documentation)](../dev/sampling_params.rst) are supported.

```{literalinclude} ../../../vllm/entrypoints/openai/protocol.py
:language: python
:start-after: begin-chat-completion-sampling-params
:end-before: end-chat-completion-sampling-params
```

The following extra parameters are supported:

```{literalinclude} ../../../vllm/entrypoints/openai/protocol.py
:language: python
:start-after: begin-chat-completion-extra-params
:end-before: end-chat-completion-extra-params
```

### Extra Parameters for Completions API
The following [sampling parameters (click through to see documentation)](../dev/sampling_params.rst) are supported.

```{literalinclude} ../../../vllm/entrypoints/openai/protocol.py
:language: python
:start-after: begin-completion-sampling-params
:end-before: end-completion-sampling-params
```

The following extra parameters are supported:

```{literalinclude} ../../../vllm/entrypoints/openai/protocol.py
:language: python
:start-after: begin-completion-extra-params
:end-before: end-completion-extra-params
```

## Chat Template

In order for the language model to support chat protocol, vLLM requires the model to include
a chat template in its tokenizer configuration. The chat template is a Jinja2 template that
specifies how are roles, messages, and other chat-specific tokens are encoded in the input.

An example chat template for `meta-llama/Llama-2-7b-chat-hf` can be found [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/09bd0f49e16738cdfaa6e615203e126038736eb0/tokenizer_config.json#L12)

Some models do not provide a chat template even though they are instruction/chat fine-tuned. For those model,
you can manually specify their chat template in the `--chat-template` parameter with the file path to the chat
template, or the template in string form. Without a chat template, the server will not be able to process chat
and all chat requests will error.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ... \
  --chat-template ./path-to-chat-template.jinja
```

vLLM community provides a set of chat templates for popular models. You can find them in the examples
directory [here](https://github.com/vllm-project/vllm/tree/main/examples/)

## Command line arguments for the server

```{argparse}
:module: vllm.entrypoints.openai.cli_args
:func: make_arg_parser
:prog: vllm-openai-server
```