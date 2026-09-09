# Context Extension

!!! note
    The `--rope-scaling` parameter used in older versions of vLLM is no longer supported. Please use the `--hf-overrides` method with `rope_parameters` instead.
This directory contains examples for extending the context length of models using vLLM.

## Offline Inference Example

The [`context_extension.py`](../../examples/features/context_extension/context_extension_offline.py) script demonstrates how to extend the context length of a Qwen model using the YARN method (rope_parameters) and run a simple chat example.

### Usage

```bash
python examples/features/context_extension/context_extension_offline.py
```

## OpenAI Online Method

You can also use vLLM's OpenAI-compatible API to serve models with extended context length.

### Usage

Run the vLLM server with the following command to extend the context length using YARN:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --hf-overrides '{"rope_parameters": {"factor": 4.0, "original_max_position_embeddings": 32768, "rope_theta": 1000000, "rope_type": "yarn"}}' \
  --max-model-len 131072
```

### Client Example

After starting the server, you can use the OpenAI Python client to interact with it:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"  # Dummy API key, required by the client
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"}
    ],
    max_tokens=128,
    temperature=0.8,
    top_p=0.95
)

print(response.choices[0].message.content)
```

### Key Parameters

The available parameters depend on the `rope_type` you choose. For detailed information about all supported RoPE types and their specific parameters, please refer to the [Hugging Face Transformers RoPE documentation](https://huggingface.co/docs/transformers/main/en/internal/rope_utils#transformers.RopeParameters).

Common parameters include:

- `rope_type`: The type of RoPE implementation (e.g., "yarn", "linear", "dynamic")
- `factor`: The factor by which to extend the context length
- `original_max_position_embeddings`: The original maximum position embeddings of the model

The following parameters are specific to vLLM:

- `max_model_len`: The new maximum sequence length after extension (original * factor).
  Used for KV cache pre‑allocation and request limit at serving time.

## Request-static YaRN

Static YaRN applies the configured factor to every request, including requests
inside the model's native context window. For mRoPE models, vLLM can instead
precompute several immutable YaRN tables and select one when each request is
admitted:

```bash
vllm serve Qwen/Qwen3.8-27B \
  --hf-overrides '{"text_config":{"rope_parameters":{"factor":4.0,"original_max_position_embeddings":262144,"rope_theta":10000000,"rope_type":"yarn","mrope_section":[11,11,10],"mrope_interleaved":true,"partial_rotary_factor":0.25}}}' \
  --request-static-yarn-factors 1 2 4 \
  --max-model-len 1000000
```

The selected factor is the smallest configured factor that covers
`prompt_tokens + max_tokens`. It remains fixed for the request lifetime, so
cached keys never require cross-factor conversion. Prefix-cache block hashes
and LMCache request tags include the complete RoPE profile identity, preventing
cross-profile cache reuse.

Completed request counts are exposed as:

```text
vllm:request_static_yarn_factor_total{factor="1",...}
vllm:request_static_yarn_factor_total{factor="2",...}
vllm:request_static_yarn_factor_total{factor="4",...}
```

The factors must be sorted and unique, and the largest value must equal the
`factor` in `rope_parameters`. Request-static YaRN currently requires mRoPE and
does not support resumable requests.
