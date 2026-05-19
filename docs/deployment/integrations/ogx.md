# OGX

vLLM is also available via [OGX](https://github.com/ogx-ai/ogx).

To install OGX, run

```bash
ogx list-deps starter | xargs -L1 uv pip install
ogx run starter
```

To force CPU-only on the OGX server:

```bash
CUDA_VISIBLE_DEVICES="" ogx run starter
```

## Inference using OpenAI-Compatible API

Then start the OGX server and configure it to point to your vLLM server with the following settings:

```yaml
inference:
  - provider_id: vllm0
    provider_type: remote::vllm
    config:
      url: http://127.0.0.1:8000
```

Please refer to [the OGX docs](https://ogx-ai.github.io/) for more details on this remote vLLM provider.

## File search (OGX integration)

If you want `file_search` to use OGX, install the handler as a vLLM
plugin and start the OGX server.

Register the handler in your `pyproject.toml`:

```toml
[project.entry-points."vllm.file_search_plugins"]
ogx = "vllm.plugins.file_search.ogx_handler:create_handler"
```

Then set the OGX server URL:

```bash
export OGX_URL="http://localhost:8321"
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
named `vs_demo` to exist and contain at least one document in your OGX
instance. The example below uses the OpenAI client against OGX to
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

To run the vLLM file_search integration test against OGX with a
dynamic vector store ID, set the test-only environment variables before
running pytest:

```bash
OGX_URL="http://localhost:8321" \
VLLM_RUN_GPT_OSS_FILE_SEARCH_IT=1 \
VLLM_TEST_VECTOR_STORE_ID="<your_vector_store_id>" \
VLLM_TEST_FILE_SEARCH_MAX_RESULTS=3 \
pytest -q tests/entrypoints/openai/responses/test_harmony.py -k file_search_integration
```

### Custom handlers

The `tools.ogx_file_search_demo` handler works with OGX. To use
a different file search backend, implement a custom handler by subclassing
`vllm.plugins.file_search.FileSearchHandler` and registering it as a plugin
in your `pyproject.toml`:

```toml
[project.entry-points."vllm.file_search_plugins"]
my_handler = "my_package.my_module:create_handler"
```

## Inference using Embedded vLLM

An [inline provider](https://github.com/ogx-ai/ogx/tree/main/src/ogx/providers/inline/inference)
is also available. This is a sample of configuration using that method:

```yaml
inference:
  - provider_type: vllm
    config:
      model: Llama3.1-8B-Instruct
      tensor_parallel_size: 4
```
