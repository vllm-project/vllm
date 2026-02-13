# Llama Stack

vLLM is also available via [Llama Stack](https://github.com/llamastack/llama-stack).

To install Llama Stack, run

```bash
llama stack list-deps starter | xargs -L1 uv pip install
llama stack run starter
```

To force CPU-only on the Llama Stack server:

```bash
CUDA_VISIBLE_DEVICES="" llama stack run starter
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

## File search (Llama Stack integration)

If you want `file_search` to use Llama Stack, start the Llama Stack server and
point vLLM at a handler in your environment:

```bash
export LLAMA_STACK_URL="http://localhost:8321"
export VLLM_GPT_OSS_FILE_SEARCH_HANDLER="tools.llama_stack_file_search_demo:handle"
```

The handler should accept a `dict` of tool arguments (e.g., `query`,
`filters`, `vector_store_ids`) and return an OpenAI-compatible payload:

```json
{"results": [{"file_id": "...", "filename": "...", "score": 0.0, "attributes": {}, "content": [{"type": "text", "text": "..."}]}]}
```

Results are only included in Responses output when the request includes
`include=["file_search_call.results"]`.

Example request (recommended):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
response = client.responses.create(
    model="openai/gpt-oss-20b",
    input="Search for work.",
    tools=[{"type": "file_search", "vector_store_ids": ["<vector_store_id>"]}],
    include=["file_search_call.results"],
    temperature=0,
    top_p=0.1,
)
print(response)
```

Note: avoid instructing the model to emit raw JSON as a normal message.
Rely on the tool call output (`file_search_call.results`) instead.

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
