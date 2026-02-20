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
point vLLM at a handler in your environment. You must first allowlist the
handler module in `vllm/entrypoints/openai/responses/context.py` and restart
the server (an example entry is commented there).

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

### Preparing `vs_demo` for integration tests

The Responses file_search integration test in vLLM expects a vector store ID
named `vs_demo` to exist and contain at least one document in your Llama Stack
instance. The example below uses the OpenAI client against Llama Stack to
create a vector store, ingest a document, and then run a sample Responses call.

```python
import io
import requests
from openai import OpenAI

url = "https://www.paulgraham.com/greatwork.html"
client = OpenAI(base_url="http://localhost:8321/v1/", api_key="none")

# Create vector store
vs = client.vector_stores.create()

response = requests.get(url)
pseudo_file = io.BytesIO(str(response.content).encode("utf-8"))
file_id = client.files.create(
    file=(url, pseudo_file, "text/html"),
    purpose="assistants",
).id
client.vector_stores.files.create(vector_store_id=vs.id, file_id=file_id)

# Automatic tool calling (calls Responses API directly)
resp = client.responses.create(
    model="gpt-4o",
    input="How do you do great work?",
    tools=[{"type": "file_search", "vector_store_ids": [vs.id]}],
    include=["file_search_call.results"],
)

print(resp.output[-1].content[-1].text)
```

If you want the vLLM test to pass without modification, create the vector
store with ID `vs_demo` and ensure it has at least one document.

To run the vLLM file_search integration test against Llama Stack with a
dynamic vector store ID, set the test-only environment variables before
running pytest:

```bash
LLAMA_STACK_URL="http://localhost:8321" \
VLLM_GPT_OSS_FILE_SEARCH_HANDLER="tools.llama_stack_file_search_demo:handle" \
VLLM_RUN_GPT_OSS_FILE_SEARCH_IT=1 \
VLLM_TEST_VECTOR_STORE_ID="<your_vector_store_id>" \
VLLM_TEST_FILE_SEARCH_MAX_RESULTS=3 \
pytest -q tests/entrypoints/openai/responses/test_harmony.py -k file_search_integration
```

### Custom handlers

The `tools.llama_stack_file_search_demo` handler works with Llama Stack, but is
disabled by default because no modules are allowlisted. To use a different file
search system, you can implement a custom handler. Custom handlers must be added
to `_ALLOWED_FILE_SEARCH_HANDLER_MODULES` in the vLLM source code.

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
