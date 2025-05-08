(pooling-models)=

# Pooling Models

vLLM also supports pooling models, including embedding, reranking and reward models.

In vLLM, pooling models implement the {class}`~vllm.model_executor.models.VllmModelForPooling` interface.
These models use a {class}`~vllm.model_executor.layers.Pooler` to extract the final hidden states of the input
before returning them.

:::{note}
We currently support pooling models primarily as a matter of convenience.
As shown in the [Compatibility Matrix](#compatibility-matrix), most vLLM features are not applicable to
pooling models as they only work on the generation or decode stage, so performance may not improve as much.
:::

For pooling models, we support the following `--task` options.
The selected option sets the default pooler used to extract the final hidden states:

:::{list-table}
:widths: 50 25 25 25
:header-rows: 1

- * Task
  * Pooling Type
  * Normalization
  * Softmax
- * Embedding (`embed`)
  * `LAST`
  * ✅︎
  * ❌
- * Classification (`classify`)
  * `LAST`
  * ❌
  * ✅︎
- * Sentence Pair Scoring (`score`)
  * \*
  * \*
  * \*
- * Reward Modeling (`reward`)
  * `ALL`
  * ❌
  * ❌
:::

\*The default pooler is always defined by the model.

:::{note}
If the model's implementation in vLLM defines its own pooler, the default pooler is set to that instead of the one specified in this table.
:::

When loading [Sentence Transformers](https://huggingface.co/sentence-transformers) models,
we attempt to override the default pooler based on its Sentence Transformers configuration file (`modules.json`).

:::{tip}
You can customize the model's pooling method via the `--override-pooler-config` option,
which takes priority over both the model's and Sentence Transformers's defaults.
:::

## Offline Inference

The {class}`~vllm.LLM` class provides various methods for offline inference.
See <project:#configuration> for a list of options when initializing the model.

### `LLM.encode`

The {class}`~vllm.LLM.encode` method is available to all pooling models in vLLM.
It returns the extracted hidden states directly, which is useful for reward models.

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-Math-RM-72B", task="reward")
(output,) = llm.encode("Hello, my name is")

data = output.outputs.data
print(f"Data: {data!r}")
```

### `LLM.embed`

The {class}`~vllm.LLM.embed` method outputs an embedding vector for each prompt.
It is primarily designed for embedding models.

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-mistral-7b-instruct", task="embed")
(output,) = llm.embed("Hello, my name is")

embeds = output.outputs.embedding
print(f"Embeddings: {embeds!r} (size={len(embeds)})")
```

A code example can be found here: <gh-file:examples/offline_inference/basic/embed.py>

### `LLM.classify`

The {class}`~vllm.LLM.classify` method outputs a probability vector for each prompt.
It is primarily designed for classification models.

```python
from vllm import LLM

llm = LLM(model="jason9693/Qwen2.5-1.5B-apeach", task="classify")
(output,) = llm.classify("Hello, my name is")

probs = output.outputs.probs
print(f"Class Probabilities: {probs!r} (size={len(probs)})")
```

A code example can be found here: <gh-file:examples/offline_inference/basic/classify.py>

### `LLM.score`

The {class}`~vllm.LLM.score` method outputs similarity scores between sentence pairs.
It is designed for embedding models and cross encoder models. Embedding models use cosine similarity, and [cross-encoder models](https://www.sbert.net/examples/applications/cross-encoder/README.html) serve as rerankers between candidate query-document pairs in RAG systems.

:::{note}
vLLM can only perform the model inference component (e.g. embedding, reranking) of RAG.
To handle RAG at a higher level, you should use integration frameworks such as [LangChain](https://github.com/langchain-ai/langchain).
:::

```python
from vllm import LLM

llm = LLM(model="BAAI/bge-reranker-v2-m3", task="score")
(output,) = llm.score("What is the capital of France?",
                      "The capital of Brazil is Brasilia.")

score = output.outputs.score
print(f"Score: {score}")
```

A code example can be found here: <gh-file:examples/offline_inference/basic/score.py>

## Online Serving

Our [OpenAI-Compatible Server](#openai-compatible-server) provides endpoints that correspond to the offline APIs:

- [Pooling API](#pooling-api) is similar to `LLM.encode`, being applicable to all types of pooling models.
- [Embeddings API](#embeddings-api) is similar to `LLM.embed`, accepting both text and [multi-modal inputs](#multimodal-inputs) for embedding models.
- [Score API](#score-api) is similar to `LLM.score` for cross-encoder models.

## Matryoshka Embeddings

[Matryoshka Embeddings](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html#matryoshka-embeddings) or [Matryoshka Representation Learning (MRL)](https://arxiv.org/abs/2205.13147) is a technique used in training embedding models. It allows user to trade off between performance and cost.

:::{warning}
Not all embedding models are trained using Matryoshka Representation Learning. To avoid misuse of the `dimensions` parameter, vLLM returns an error for requests that attempt to change the output dimension of models that do not support Matryoshka Embeddings.

For example, setting `dimensions` parameter while using the `BAAI/bge-m3` model will result in the following error.

```json
{"object":"error","message":"Model \"BAAI/bge-m3\" does not support matryoshka representation, changing output dimensions will lead to poor results.","type":"BadRequestError","param":null,"code":400}
```

:::

### Manually enable Matryoshka Embeddings

There is currently no official interface for specifying support for Matryoshka Embeddings. In vLLM, if `is_matryoshka` is `True` in `config.json,` it is allowed to change the output to arbitrary dimensions. Using `matryoshka_dimensions` can control the allowed output dimensions.

For models that support Matryoshka Embeddings but not recognized by vLLM, please manually override the config using `hf_overrides={"is_matryoshka": True}`, `hf_overrides={"matryoshka_dimensions": [<allowed output dimensions>]}` (offline) or `--hf_overrides '{"is_matryoshka": true}'`,  `--hf_overrides '{"matryoshka_dimensions": [<allowed output dimensions>]}'`(online).

Here is an example to serve a model with Matryoshka Embeddings enabled.

```text
vllm serve Snowflake/snowflake-arctic-embed-m-v1.5 --hf_overrides '{"matryoshka_dimensions":[256]}'
```

### Offline Inference

You can change the output dimensions of embedding models that support Matryoshka Embeddings by using the dimensions parameter in {class}`~vllm.PoolingParams`.

```python
from vllm import LLM, PoolingParams

model = LLM(model="jinaai/jina-embeddings-v3", 
            task="embed", 
            trust_remote_code=True)
outputs = model.embed(["Follow the white rabbit."], 
                      pooling_params=PoolingParams(dimensions=32))
print(outputs[0].outputs)
```

A code example can be found here: <gh-file:examples/offline_inference/embed_matryoshka_fy.py>

### Online Inference

Use the following command to start vllm server.

```text
vllm serve jinaai/jina-embeddings-v3 --trust-remote-code
```

You can change the output dimensions of embedding models that support Matryoshka Embeddings by using the dimensions parameter.

```text
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

A openai client example can be found here: <gh-file:examples/online_serving/openai_embedding_matryoshka_fy.py>
