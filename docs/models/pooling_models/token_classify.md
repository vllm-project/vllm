# Token Classification Task

## Summary

- Model Types: Token classification models
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

## Advanced Use Cases

### As Reward Models

Using token classification models as reward models. For details on reward models, see [Reward Models](reward.md).

## Supported Models

| Architecture | Models | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | --------------------------- | --------------------------------------- |
| `BertForTokenClassification` | bert-based | `boltuix/NeuroBERT-NER` (see note), etc. | | |
| `ErnieForTokenClassification` | BERT-like Chinese ERNIE | `gyr66/Ernie-3.0-base-chinese-finetuned-ner` | | |
| `ModernBertForTokenClassification` | ModernBERT-based | `disham993/electrical-ner-ModernBERT-base` | | |
| `Qwen3ForTokenClassification`<sup>C</sup> | Qwen3-based | `bd2lcco/Qwen3-0.6B-finetuned` | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

<sup>C</sup> Automatically converted into a classification model via `--convert classify`. ([details](./README.md#model-conversion))  
\* Feature support is the same as that of the original model.

If your model is not in the above list, we will try to automatically convert the model using
[as_seq_cls_model][vllm.model_executor.models.adapters.as_seq_cls_model]. By default, the class probabilities are extracted from the softmaxed hidden state corresponding to the last token.

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
