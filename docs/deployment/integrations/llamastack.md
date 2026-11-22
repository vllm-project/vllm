# OGX (formerly Llama Stack)

vLLM is also available via [OGX](https://github.com/ogx-ai/ogx), the project previously named Llama Stack.

To install OGX, run

```bash
uv pip install ogx
```

## Inference using OpenAI-Compatible API

Then start the OGX server and configure it to point to your vLLM server with the following settings:

```yaml
inference:
  - provider_id: vllm0
    provider_type: remote::vllm
    config:
      base_url: http://127.0.0.1:8000/v1
```

Please refer to [this guide](https://ogx-ai.github.io/docs/providers/inference/remote_vllm) for more details on this remote vLLM provider.
