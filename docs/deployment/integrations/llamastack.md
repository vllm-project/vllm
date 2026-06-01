# Llama Stack

vLLM is also available via [Llama Stack](https://github.com/llamastack/llama-stack).

To install Llama Stack, run

```bash
pip install llama-stack -q
```

## Inference using OpenAI-Compatible API

Then start the Llama Stack server and configure it to point to your vLLM server with the following settings:

```yaml
inference:
  - provider_id: vllm0
    provider_type: remote::vllm
    config:
      url: http://127.0.0.1:8000
```

Please refer to [this guide](https://llama-stack.readthedocs.io/en/latest/providers/inference/remote_vllm.html) for more details on this remote vLLM provider.

## Inference using Embedded vLLM

An [inline provider](https://github.com/llamastack/llama-stack/tree/main/llama_stack/providers/inline/inference)
is also available. This is a sample of configuration using that method:

```yaml
inference:
  - provider_type: vllm
    config:
      model: Llama3.1-8B-Instruct
      tensor_parallel_size: 4
```
