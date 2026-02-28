# Classification

These models primarily support the `classify` task.

## Supported Models

| Architecture | Models | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `JambaForSequenceClassification` | Jamba | `ai21labs/Jamba-tiny-reward-dev`, etc. | ✅︎ | ✅︎ |
| `GPT2ForSequenceClassification` | GPT2 | `nie3e/sentiment-polish-gpt2-small` | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

<sup>C</sup> Automatically converted into a classification model via `--convert classify`. ([details](./pooling_models.md#model-conversion))  
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

### `LLM.classify`

The [classify][vllm.LLM.classify] method outputs a probability vector for each prompt.
It is primarily designed for classification models.

```python
from vllm import LLM

llm = LLM(model="jason9693/Qwen2.5-1.5B-apeach", runner="pooling")
(output,) = llm.classify("Hello, my name is")

probs = output.outputs.probs
print(f"Class Probabilities: {probs!r} (size={len(probs)})")
```

A code example can be found here: [examples/offline_inference/basic/classify.py](../../examples/offline_inference/basic/classify.py)

### `LLM.encode`

The [encode][vllm.LLM.encode] method is available to all pooling models in vLLM.

!!! note
    Please use one of the more specific methods or set the task directly when using `LLM.encode`:

    - For classification logits, use `LLM.classify(...)` or `pooling_task="classify"`.

```python
from vllm import LLM

llm = LLM(model="jason9693/Qwen2.5-1.5B-apeach", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="classify")

data = output.outputs.data
print(f"Data: {data!r}")
```

## Online Serving

Classification API is similar to `LLM.classify` and is applicable to sequence classification models.

### Completion Parameters

The following Classification API parameters are supported:

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:completion-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-params"
    ```

The following extra parameters are supported:

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:completion-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-extra-params"
    ```

### Chat Parameters

For chat-like input (i.e. if `messages` is passed), the following parameters are supported:

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:chat-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-params"
    ```

these extra parameters are supported instead:

??? code

    ```python
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:pooling-common-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:chat-extra-params"
    --8<-- "vllm/entrypoints/pooling/base/protocol.py:classify-extra-params"
    ```

### Example Requests

!!! note
    More examples can be found in [examples/pooling/classify/](../../../examples/pooling/classify/)

Code example: [examples/pooling/classify/classification_online.py](../../../examples/pooling/classify/classification_online.py)

You can classify multiple texts by passing an array of strings:

```bash
curl -v "http://127.0.0.1:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason9693/Qwen2.5-1.5B-apeach",
    "input": [
      "Loved the new café—coffee was great.",
      "This update broke everything. Frustrating."
    ]
  }'
```

??? console "Response"

    ```json
    {
      "id": "classify-7c87cac407b749a6935d8c7ce2a8fba2",
      "object": "list",
      "created": 1745383065,
      "model": "jason9693/Qwen2.5-1.5B-apeach",
      "data": [
        {
          "index": 0,
          "label": "Default",
          "probs": [
            0.565970778465271,
            0.4340292513370514
          ],
          "num_classes": 2
        },
        {
          "index": 1,
          "label": "Spoiled",
          "probs": [
            0.26448777318000793,
            0.7355121970176697
          ],
          "num_classes": 2
        }
      ],
      "usage": {
        "prompt_tokens": 20,
        "total_tokens": 20,
        "completion_tokens": 0,
        "prompt_tokens_details": null
      }
    }
    ```

You can also pass a string directly to the `input` field:

```bash
curl -v "http://127.0.0.1:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jason9693/Qwen2.5-1.5B-apeach",
    "input": "Loved the new café—coffee was great."
  }'
```

??? console "Response"

    ```json
    {
      "id": "classify-9bf17f2847b046c7b2d5495f4b4f9682",
      "object": "list",
      "created": 1745383213,
      "model": "jason9693/Qwen2.5-1.5B-apeach",
      "data": [
        {
          "index": 0,
          "label": "Default",
          "probs": [
            0.565970778465271,
            0.4340292513370514
          ],
          "num_classes": 2
        }
      ],
      "usage": {
        "prompt_tokens": 10,
        "total_tokens": 10,
        "completion_tokens": 0,
        "prompt_tokens_details": null
      }
    }
    ```

### Remove softmax from PoolingParams

We have already removed softmax and activation from PoolingParams in v0.15. Instead, use `use_activation`, since we allow `classify` and `token_classify` to use any activation function.




