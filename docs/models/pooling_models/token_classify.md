# Token Classification Usages

## Summary

- Model Usage: token classification
- Pooling Tasks: `token_classify`
- Offline APIs:
    - `LLM.encode(..., pooling_task="token_classify")`
- Online APIs:
    - Pooling API (`/pooling`)

The key distinction between (sequence) classification and token classification lies in their output granularity: (sequence) classification produces a single result for an entire input sequence, whereas token classification yields a result for each individual token within the sequence.

Many classification models support both (sequence) classification and token classification. For further details on (sequence) classification, please refer to [this page](classify.md).

## Typical Use Cases

### Named Entity Recognition (NER)

For implementation examples, see:

Offline: [examples/pooling/token_classify/ner_offline.py](../../../examples/pooling/token_classify/ner_offline.py)

Online: [examples/pooling/token_classify/ner_online.py](../../../examples/pooling/token_classify/ner_online.py)

### Sparse retrieval (lexical matching)

The BAAI/bge-m3 model leverages token classification for sparse retrieval. For more information, see [this page](specific_models.md#baaibge-m3).

## Supported Models

--8<-- "docs/models/pooling_models/supported_models.inc.md:token-classify-models"

### As Reward Models

Using token classification models as reward models. For details on reward models, see [Reward Models](reward.md).

--8<-- "docs/models/pooling_models/supported_models.inc.md:token-reward-models"

## Offline Inference

### Pooling Parameters

The following [pooling parameters][vllm.PoolingParams] are supported.

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
--8<-- "vllm/pooling_params.py:classify-pooling-params"
```

### `LLM.encode`

The [encode][vllm.LLM.encode] method is available to all pooling models in vLLM.

Set `pooling_task="token_classify"` when using `LLM.encode` for token classification Models:

```python
from vllm import LLM

llm = LLM(model="boltuix/NeuroBERT-NER", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="token_classify")

data = output.outputs.data
print(f"Data: {data!r}")
```

## Online Serving

Please refer to the [pooling API](README.md#pooling-api) and use `"task":"token_classify"`.

## More examples

More examples can be found here: [examples/pooling/token_classify](../../../examples/pooling/token_classify)

## Features

Token classification features should be consistent with (sequence) classification. For more information, see [this page](classify.md#features).
