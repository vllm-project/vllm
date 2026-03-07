# Reward Models

## Supported Models

- Sequence Classification

| Architecture | Models | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `Qwen3ForSequenceClassification` | InternLM2-based | `Skywork/Skywork-Reward-V2-Qwen3-0.6B`, etc. | ✅︎ | ✅︎ |

- Token Classification

These models primarily support the [`LLM.reward`](./pooling_models.md#llmreward) API.

| Architecture | Models | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `InternLM2ForRewardModel` | InternLM2-based | `internlm/internlm2-1_8b-reward`, `internlm/internlm2-7b-reward`, etc. | ✅︎ | ✅︎ |
| `LlamaForCausalLM` | Llama-based | `peiyi9979/math-shepherd-mistral-7b-prm`, etc. | ✅︎ | ✅︎ |
| `Qwen2ForRewardModel` | Qwen2-based | `Qwen/Qwen2.5-Math-RM-72B`, etc. | ✅︎ | ✅︎ |
| `Qwen2ForProcessRewardModel` | Qwen2-based | `Qwen/Qwen2.5-Math-PRM-7B`, etc. | ✅︎ | ✅︎ |

!!! important
    For process-supervised reward models such as `peiyi9979/math-shepherd-mistral-7b-prm`, the pooling config should be set explicitly,
    e.g.: `--pooler-config '{"pooling_type": "STEP", "step_tag_id": 123, "returned_token_ids": [456, 789]}'`.

## Offline Inference

### `LLM.reward`

The [reward][vllm.LLM.reward] method is available to all reward models in vLLM.

```python
from vllm import LLM

llm = LLM(model="internlm/internlm2-1_8b-reward", runner="pooling", trust_remote_code=True)
(output,) = llm.reward("Hello, my name is")

data = output.outputs.data
print(f"Data: {data!r}")
```

A code example can be found here: [examples/offline_inference/basic/reward.py](../../examples/offline_inference/basic/reward.py)


### `LLM.encode`

The [encode][vllm.LLM.encode] method is available to all pooling models in vLLM.

!!! note
    Please use one of the more specific methods or set the task directly when using `LLM.encode`:

    - For rewards, use `LLM.reward(...)` or `pooling_task="token_classify"`.

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="embed")

data = output.outputs.data
print(f"Data: {data!r}")
```