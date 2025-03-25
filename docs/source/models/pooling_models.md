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
See [Engine Arguments](#engine-args) for a list of options when initializing the model.

### `LLM.encode`

The {class}`~vllm.LLM.encode` method is available to all pooling models in vLLM.
It returns the extracted hidden states directly, which is useful for reward models.

```python
llm = LLM(model="Qwen/Qwen2.5-Math-RM-72B", task="reward")
(output,) = llm.encode("Hello, my name is")

data = output.outputs.data
print(f"Data: {data!r}")
```

### `LLM.embed`

The {class}`~vllm.LLM.embed` method outputs an embedding vector for each prompt.
It is primarily designed for embedding models.

```python
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
