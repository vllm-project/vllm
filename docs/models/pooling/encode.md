# Pooling API

## Offline Encode API

### `LLM.encode`

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

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="embed")

data = output.outputs.data
print(f"Data: {data!r}")
```

## Online Pooling API

[Pooling API](../serving/openai_compatible_server.md#pooling-api) is similar to `LLM.encode`, being applicable to all types of pooling models.

!!! note
    Please use one of the more specific endpoints or set the task directly when using the [Pooling API](../serving/openai_compatible_server.md#pooling-api):

    - For embeddings, use [Embeddings API](../serving/openai_compatible_server.md#embeddings-api) or `"task":"embed"`.
    - For classification logits, use [Classification API](../serving/openai_compatible_server.md#classification-api) or `"task":"classify"`.
    - For similarity scores, use [Score API](../serving/openai_compatible_server.md#score-api).
    - For rewards, use `"task":"token_classify"`.
    - For token classification, use `"task":"token_classify"`.
    - For multi-vector retrieval, use `"task":"token_embed"`.
    - For IO Processor Plugins, use `"task":"plugin"`.