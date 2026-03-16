# Pooling API

The Pooling API includes the offline `LLM.encode` API and the online `/pooling` endpoint.

## Offline Inference

The [encode][vllm.LLM.encode] method is available to all pooling models in vLLM.

!!! note
    Please use one of the more specific methods or set the task directly when using `LLM.encode`:

    - For embeddings, use `LLM.embed(...)` or `pooling_task="embed"`.
    - For classification logits, use `LLM.classify(...)` or `pooling_task="classify"`.
    - For similarity scores, use `LLM.score(...)`.
    - For rewards, use `LLM.reward(...)` or `pooling_task="token_classify"`.
    - For token classification, use `pooling_task="token_classify"`.
    - For multi-vector retrieval, use `pooling_task="token_embed"`.
    - For IO Processor Plugins, use `pooling_task="plugin"`.

### Examples

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="embed")

data = output.outputs.data
print(f"Data: {data!r}")
```

## Online Serving

Our Pooling endpoint (`/pooling`) is similar to `LLM.encode`, being applicable to all types of pooling models.

The input format is the same as [Embeddings API](#embeddings-api), but the output data can contain an arbitrary nested list, not just a 1-D list of floats.

Code example: [examples/pooling/pooling/pooling_online.py](../../../examples/pooling/pooling/pooling_online.py)

!!! note
    Please use one of the more specific endpoints or set the task directly when using the [Pooling API](../../serving/openai_compatible_server.md#pooling-api):

    - For embeddings, use [Embeddings API](embed.md) or `"task":"embed"`.
    - For classification logits, use [Classification API](classify.md) or `"task":"classify"`.
    - For similarity scores, use [Score API](score.md).
    - For rewards, use `"task":"token_classify"`.
    - For token classification, use `"task":"token_classify"`.
    - For multi-vector retrieval, use `"task":"token_embed"`.
    - For IO Processor Plugins, use `"task":"plugin"`.

### Examples

```python
# start a supported embeddings model server with `vllm serve`, e.g.
# vllm serve intfloat/e5-small
import requests

host = "localhost"
port = "8000"
model_name = "intfloat/e5-small"

api_url = f"http://{host}:{port}/pooling"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
prompt = {"model": model_name, "input": prompts, "task": "embed"}

response = requests.post(api_url, json=prompt)

for output in response.json()["data"]:
    data = output["data"]
    print(f"Data: {data!r} (size={len(data)})")
```

## More examples
More examples can be found here: [examples/pooling/pooling](../../../examples/pooling/pooling)
