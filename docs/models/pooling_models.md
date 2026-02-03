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
|-------------------------------------------------|-------------|---------------------------------------|
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
|------------------|-------------------------------------------------------------------------------|
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
|------------|--------------|---------------|---------|
| `embed`    | `LAST`       | ✅︎            | ❌      |
| `classify` | `LAST`       | ❌            | ✅︎      |

When loading [Sentence Transformers](https://huggingface.co/sentence-transformers) models,
its Sentence Transformers configuration file (`modules.json`) takes priority over the model's defaults.

You can further customize this via the `--pooler-config` option,
which takes priority over both the model's and Sentence Transformers' defaults.

## Offline Inference

The [LLM][vllm.LLM] class provides various methods for offline inference.
See [configuration](../api/README.md#configuration) for a list of options when initializing the model.

### `LLM.embed`

The [embed][vllm.LLM.embed] method outputs an embedding vector for each prompt.
It is primarily designed for embedding models.

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.embed("Hello, my name is")

embeds = output.outputs.embedding
print(f"Embeddings: {embeds!r} (size={len(embeds)})")
```

A code example can be found here: [examples/offline_inference/basic/embed.py](../../examples/offline_inference/basic/embed.py)

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

### `LLM.score`

The [score][vllm.LLM.score] method outputs similarity scores between sentence pairs.
It is designed for embedding models and cross-encoder models. Embedding models use cosine similarity, and [cross-encoder models](https://www.sbert.net/examples/applications/cross-encoder/README.html) serve as rerankers between candidate query-document pairs in RAG systems.

!!! note
    vLLM can only perform the model inference component (e.g. embedding, reranking) of RAG.
    To handle RAG at a higher level, you should use integration frameworks such as [LangChain](https://github.com/langchain-ai/langchain).

```python
from vllm import LLM

llm = LLM(model="BAAI/bge-reranker-v2-m3", runner="pooling")
(output,) = llm.score(
    "What is the capital of France?",
    "The capital of Brazil is Brasilia.",
)

score = output.outputs.score
print(f"Score: {score}")
```

A code example can be found here: [examples/offline_inference/basic/score.py](../../examples/offline_inference/basic/score.py)

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

## Online Serving

Our [OpenAI-Compatible Server](../serving/openai_compatible_server.md) provides endpoints that correspond to the offline APIs:

- [Embeddings API](../serving/openai_compatible_server.md#embeddings-api) is similar to `LLM.embed`, accepting both text and [multi-modal inputs](../features/multimodal_inputs.md) for embedding models.
- [Classification API](../serving/openai_compatible_server.md#classification-api) is similar to `LLM.classify` and is applicable to sequence classification models.
- [Score API](../serving/openai_compatible_server.md#score-api) is similar to `LLM.score` for cross-encoder models.
- [Pooling API](../serving/openai_compatible_server.md#pooling-api) is similar to `LLM.encode`, being applicable to all types of pooling models.

!!! note
    Please use one of the more specific endpoints or set the task directly when using the [Pooling API](../serving/openai_compatible_server.md#pooling-api):

    - For embeddings, use [Embeddings API](../serving/openai_compatible_server.md#embeddings-api) or `"task":"embed"`.
    - For classification logits, use [Classification API](../serving/openai_compatible_server.md#classification-api) or `"task":"classify"`.
    - For similarity scores, use [Score API](../serving/openai_compatible_server.md#score-api).
    - For rewards, use `"task":"token_classify"`.
    - For token classification, use `"task":"token_classify"`.
    - For multi-vector retrieval, use `"task":"token_embed"`.
    - For IO Processor Plugins, use `"task":"plugin"`.

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

## Matryoshka Embeddings

[Matryoshka Embeddings](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html#matryoshka-embeddings) or [Matryoshka Representation Learning (MRL)](https://arxiv.org/abs/2205.13147) is a technique used in training embedding models. It allows users to trade off between performance and cost.

!!! warning
    Not all embedding models are trained using Matryoshka Representation Learning. To avoid misuse of the `dimensions` parameter, vLLM returns an error for requests that attempt to change the output dimension of models that do not support Matryoshka Embeddings.

    For example, setting `dimensions` parameter while using the `BAAI/bge-m3` model will result in the following error.

    ```json
    {"object":"error","message":"Model \"BAAI/bge-m3\" does not support matryoshka representation, changing output dimensions will lead to poor results.","type":"BadRequestError","param":null,"code":400}
    ```

### Manually enable Matryoshka Embeddings

There is currently no official interface for specifying support for Matryoshka Embeddings. In vLLM, if `is_matryoshka` is `True` in `config.json`, you can change the output dimension to arbitrary values. Use `matryoshka_dimensions` to control the allowed output dimensions.

For models that support Matryoshka Embeddings but are not recognized by vLLM, manually override the config using `hf_overrides={"is_matryoshka": True}` or `hf_overrides={"matryoshka_dimensions": [<allowed output dimensions>]}` (offline), or `--hf-overrides '{"is_matryoshka": true}'` or `--hf-overrides '{"matryoshka_dimensions": [<allowed output dimensions>]}'` (online).

Here is an example to serve a model with Matryoshka Embeddings enabled.

```bash
vllm serve Snowflake/snowflake-arctic-embed-m-v1.5 --hf-overrides '{"matryoshka_dimensions":[256]}'
```

### Offline Inference

You can change the output dimensions of embedding models that support Matryoshka Embeddings by using the dimensions parameter in [PoolingParams][vllm.PoolingParams].

```python
from vllm import LLM, PoolingParams

llm = LLM(
    model="jinaai/jina-embeddings-v3",
    runner="pooling",
    trust_remote_code=True,
)
outputs = llm.embed(
    ["Follow the white rabbit."],
    pooling_params=PoolingParams(dimensions=32),
)
print(outputs[0].outputs)
```

A code example can be found here: [examples/pooling/embed/embed_matryoshka_fy_offline.py](../../examples/pooling/embed/embed_matryoshka_fy_offline.py)

### Online Inference

Use the following command to start the vLLM server.

```bash
vllm serve jinaai/jina-embeddings-v3 --trust-remote-code
```

You can change the output dimensions of embedding models that support Matryoshka Embeddings by using the dimensions parameter.

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Follow the white rabbit.",
    "model": "jinaai/jina-embeddings-v3",
    "encoding_format": "float",
    "dimensions": 32
  }'
```

Expected output:

```json
{"id":"embd-5c21fc9a5c9d4384a1b021daccaf9f64","object":"list","created":1745476417,"model":"jinaai/jina-embeddings-v3","data":[{"index":0,"object":"embedding","embedding":[-0.3828125,-0.1357421875,0.03759765625,0.125,0.21875,0.09521484375,-0.003662109375,0.1591796875,-0.130859375,-0.0869140625,-0.1982421875,0.1689453125,-0.220703125,0.1728515625,-0.2275390625,-0.0712890625,-0.162109375,-0.283203125,-0.055419921875,-0.0693359375,0.031982421875,-0.04052734375,-0.2734375,0.1826171875,-0.091796875,0.220703125,0.37890625,-0.0888671875,-0.12890625,-0.021484375,-0.0091552734375,0.23046875]}],"usage":{"prompt_tokens":8,"total_tokens":8,"completion_tokens":0,"prompt_tokens_details":null}}
```

An OpenAI client example can be found here: [examples/pooling/embed/openai_embedding_matryoshka_fy_client.py](../../examples/pooling/embed/openai_embedding_matryoshka_fy_client.py)

## Specific models

### BAAI/bge-m3

The `BAAI/bge-m3` model comes with extra weights for sparse and colbert embeddings but unfortunately in its `config.json`
the architecture is declared as `XLMRobertaModel`, which makes `vLLM` load it as a vanilla ROBERTA model without the
extra weights. To load the full model weights, override its architecture like this:

```shell
vllm serve BAAI/bge-m3 --hf-overrides '{"architectures": ["BgeM3EmbeddingModel"]}'
```

Then you obtain the sparse embeddings like this:

```shell
curl -s http://localhost:8000/pooling -H "Content-Type: application/json" -d '{
     "model": "BAAI/bge-m3",
     "task": "token_classify",
     "input": ["What is BGE M3?", "Defination of BM25"]
}'
```

Due to limitations in the output schema, the output consists of a list of
token scores for each token for each input. This means that you'll have to call
`/tokenize` as well to be able to pair tokens with scores.
Refer to the tests in  `tests/models/language/pooling/test_bge_m3.py` to see how
to do that.

You can obtain the colbert embeddings like this:

```shell
curl -s http://localhost:8000/pooling -H "Content-Type: application/json" -d '{
     "model": "BAAI/bge-m3",
     "task": "token_embed",
     "input": ["What is BGE M3?", "Defination of BM25"]
}'
```

## Deprecated Features

### Encode task

We have split the `encode` task into two more specific token-wise tasks: `token_embed` and `token_classify`:

- `token_embed` is the same as `embed`, using normalization as the activation.
- `token_classify` is the same as `classify`, by default using softmax as the activation.

Pooling models now default support all pooling, you can use it without any settings.

- Extracting hidden states prefers using `token_embed` task.
- Reward models prefers using `token_classify` task.
