# Token Embedding Usages

## Summary

- Model Usage: Token classification models
- Pooling Tasks: `token_embed`
- Offline APIs:
    - `LLM.encode(..., pooling_task="token_embed")`
- Online APIs:
    - Pooling API (`/pooling`)

The difference between the (sequence) embedding task and the token embedding task is that (sequence) embedding outputs one embedding for each sequence, while token embedding outputs a embedding for each token.

Many embedding models support both (sequence) embedding and token embedding. For further details on (sequence) embedding, please refer to [this page](embed.md).

## Typical Use Cases

### Multi-Vector Retrieval

For implementation examples, see:

Offline: [examples/pooling/token_embed/multi_vector_retrieval_offline.py](../../../examples/pooling/token_embed/multi_vector_retrieval_offline.py)

Online: [examples/pooling/token_embed/multi_vector_retrieval_online.py](../../../examples/pooling/token_embed/multi_vector_retrieval_online.py)

### Late interaction

Similarity scores can be computed using late interaction between two input prompts via the score API. For more information, see [Score API](scoring.md).

### Extract last hidden states

Models of any architecture can be converted into embedding models using `--convert embed`. Token embedding can then be used to extract the last hidden states from these models.

## Supported Models

--8<-- "docs/models/pooling_models/supported_models.inc.md:token-embed-models"

## Offline Inference

### Pooling Parameters

The following [pooling parameters][vllm.PoolingParams] are supported.

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
--8<-- "vllm/pooling_params.py:embed-pooling-params"
```

### `LLM.encode`

The [encode][vllm.LLM.encode] method is available to all pooling models in vLLM.

Set `pooling_task="token_embed"` when using `LLM.encode` for token embedding Models:

```python
from vllm import LLM

llm = LLM(model="answerdotai/answerai-colbert-small-v1", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="token_embed")

data = output.outputs.data
print(f"Data: {data!r}")
```

### `LLM.score`

The [score][vllm.LLM.score] method outputs similarity scores between sentence pairs.

All models that support token embedding task also support using the score API to compute similarity scores by calculating the late interaction of two input prompts.

```python
from vllm import LLM

llm = LLM(model="answerdotai/answerai-colbert-small-v1", runner="pooling")
(output,) = llm.score(
    "What is the capital of France?",
    "The capital of Brazil is Brasilia.",
)

score = output.outputs.score
print(f"Score: {score}")
```

## Online Serving

Please refer to the [pooling API](README.md#pooling-api) and use `"task":"token_embed"`.

## More examples

More examples can be found here: [examples/pooling/token_embed](../../../examples/pooling/token_embed)

## Features

Token embedding features should be consistent with (sequence) embedding. For more information, see [this page](embed.md#features).
