# Pooling Models

vLLM also supports pooling models, such as embedding, classification, and reward models.

In vLLM, pooling models implement the [VllmModelForPooling][vllm.model_executor.models.VllmModelForPooling] interface.
These models use a [Pooler][vllm.model_executor.layers.pooler.Pooler] to extract the final hidden states of the input
before returning them.

!!! note
    We currently support pooling models primarily for convenience. This is not guaranteed to provide any performance improvements over using Hugging Face Transformers or Sentence Transformers directly.

    We plan to optimize pooling models in vLLM. Please comment on <https://github.com/vllm-project/vllm/issues/21796> if you have any suggestions!

## Configuration

### Model Runner

Run a model in pooling mode via the option `--runner pooling`.

!!! tip
    There is no need to set this option in the vast majority of cases as vLLM can automatically
    detect the appropriate model runner via `--runner auto`.

### Model Conversion

vLLM can adapt models for various pooling tasks via the option `--convert <type>`.

If `--runner pooling` has been set (manually or automatically) but the model does not implement the
[VllmModelForPooling][vllm.model_executor.models.VllmModelForPooling] interface,
vLLM will attempt to automatically convert the model according to the architecture names
shown in the table below.

| Architecture                                    | `--convert` | Supported pooling tasks               |
| ----------------------------------------------- | ----------- | ------------------------------------- |
| `*ForTextEncoding`, `*EmbeddingModel`, `*Model` | `embed`     | `token_embed`, `embed`                |
| `*ForRewardModeling`, `*RewardModel`            | `embed`     | `token_embed`, `embed`                |
| `*For*Classification`, `*ClassificationModel`   | `classify`  | `token_classify`, `classify`, `score` |

!!! tip
    You can explicitly set `--convert <type>` to specify how to convert the model.

### Pooling Tasks

Each pooling model in vLLM supports one or more of these tasks according to
[Pooler.get_supported_tasks][vllm.model_executor.layers.pooler.Pooler.get_supported_tasks],
enabling the corresponding APIs:

| Task             | APIs                                                                          |
| ---------------- | ----------------------------------------------------------------------------- |
| `embed`          | `LLM.embed(...)`, `LLM.score(...)`\*, `LLM.encode(..., pooling_task="embed")` |
| `classify`       | `LLM.classify(...)`, `LLM.encode(..., pooling_task="classify")`               |
| `score`          | `LLM.score(...)`                                                              |
| `token_classify` | `LLM.reward(...)`, `LLM.encode(..., pooling_task="token_classify")`           |
| `token_embed`    | `LLM.encode(..., pooling_task="token_embed")`                                 |
| `plugin`         | `LLM.encode(..., pooling_task="plugin")`                                      |

\* The `LLM.score(...)` API falls back to `embed` task if the model does not support `score` task.

### Pooler Configuration

#### Predefined models

If the [Pooler][vllm.model_executor.layers.pooler.Pooler] defined by the model accepts `pooler_config`,
you can override some of its attributes via the `--pooler-config` option.

#### Converted models

If the model has been converted via `--convert` (see above),
the pooler assigned to each task has the following attributes by default:

| Task       | Pooling Type | Normalization | Softmax |
| ---------- | ------------ | ------------- | ------- |
| `embed`    | `LAST`       | ✅︎            | ❌      |
| `classify` | `LAST`       | ❌            | ✅︎      |

When loading [Sentence Transformers](https://huggingface.co/sentence-transformers) models,
its Sentence Transformers configuration file (`modules.json`) takes priority over the model's defaults.

You can further customize this via the `--pooler-config` option,
which takes priority over both the model's and Sentence Transformers' defaults.

## Deprecated Features

### Encode task

We have split the `encode` task into two more specific token-wise tasks: `token_embed` and `token_classify`:

- `token_embed` is the same as `embed`, using normalization as the activation.
- `token_classify` is the same as `classify`, by default using softmax as the activation.

Pooling models now default support all pooling, you can use it without any settings.

- Extracting hidden states prefers using `token_embed` task.
- Reward models prefers using `token_classify` task.
