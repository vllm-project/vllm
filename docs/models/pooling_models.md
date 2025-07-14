# Pooling Models

vLLM also supports pooling models, including embedding, reranking and reward models.

In vLLM, pooling models implement the [VllmModelForPooling][vllm.model_executor.models.VllmModelForPooling] interface.
These models use a [Pooler][vllm.model_executor.layers.Pooler] to extract the final hidden states of the input
before returning them.

!!! note
    We currently support pooling models primarily as a matter of convenience.
    As shown in the [Compatibility Matrix](../features/compatibility_matrix.md), most vLLM features are not applicable to
    pooling models as they only work on the generation or decode stage, so performance may not improve as much.

For pooling models, we support the following `--task` options.
The selected option sets the default pooler used to extract the final hidden states:

| Task                            | Pooling Type   | Normalization   | Softmax   |
|---------------------------------|----------------|-----------------|-----------|
| Embedding (`embed`)             | `LAST`         | ✅︎              | ❌         |
| Classification (`classify`)     | `LAST`         | ❌               | ✅︎        |
| Sentence Pair Scoring (`score`) | \*             | \*              | \*        |

\*The default pooler is always defined by the model.

!!! note
    If the model's implementation in vLLM defines its own pooler, the default pooler is set to that instead of the one specified in this table.

When loading [Sentence Transformers](https://huggingface.co/sentence-transformers) models,
we attempt to override the default pooler based on its Sentence Transformers configuration file (`modules.json`).

!!! tip
    You can customize the model's pooling method via the `--override-pooler-config` option,
    which takes priority over both the model's and Sentence Transformers's defaults.

## Chunked Processing for Long Text

vLLM supports **chunked processing** for embedding models to handle text inputs that exceed the model's maximum token length. This feature automatically splits long text into manageable chunks, processes them separately, and aggregates the results.

### Supported Models

Chunked processing is supported for the following embedding models:

- `intfloat/multilingual-e5-large` (Recommended pool type: `MEAN`)
- `jinaai/jina-embeddings-v3` (Recommended pool type: `MEAN`)  
- `jinaai/jina-embeddings-v4-vllm-retrieval` (Recommended pool type: `MEAN`)
- `Qwen/Qwen3-Embedding-4B` (Recommended pool type: `MEAN`)

Other embedding models can be extended to support this feature by ensuring proper pooling type compatibility.

### How Chunked Processing Works

1. **Flexible Input Validation**: Configure `max_embed_len` to accept inputs longer than `max_model_len` without environment variables
2. **Smart Chunking**: Text is split based on `max_position_embeddings` to maintain semantic integrity  
3. **Parallel Processing**: Each chunk is processed independently through the model
4. **Intelligent Aggregation**: Results are combined using weighted averaging based on chunk token counts
5. **Consistent Output**: Final embeddings maintain the same dimensionality as standard processing

### Configuration

Enable chunked processing and configure maximum embedding input length:

```bash
vllm serve intfloat/multilingual-e5-large \
  --task embed \
  --override-pooler-config '{"pooling_type": "MEAN", "normalize": true, "enable_chunked_processing": true, "max_embed_len": 3072000}' \
  --trust-remote-code
```

#### Configuration Parameters

- `enable_chunked_processing`: Enable chunked processing for long inputs (default: `false`)
- `max_embed_len`: Maximum input length allowed for embedding generation (default: `null`)
  - When set, allows inputs longer than `max_model_len` without requiring `VLLM_ALLOW_LONG_MAX_MODEL_LEN`
  - Inputs exceeding `max_embed_len` are rejected with clear error messages
  - Chunking is triggered when inputs exceed `max_position_embeddings`

### Aggregation Algorithm

The chunked processing uses a FastChat-inspired weighted averaging algorithm:

```python
# Weighted average: sum(embedding_i * token_count_i) / total_tokens
weighted_sum = sum(embeddings[i] * weights[i] for i in range(num_chunks))
final_embedding = weighted_sum / sum(weights)
```

This ensures that longer chunks contribute proportionally more to the final representation.

### Performance Characteristics

| Aspect | Short Text (≤ max_position_embeddings) | Long Text (> max_position_embeddings) |
|--------|----------------------------------------|---------------------------------------|
| **Processing Time** | Standard | Increased (multiple inference calls) |
| **Memory Usage** | Standard | Reduced (chunks processed separately) |
| **Quality** | Standard | Maintains semantic representation |
| **Compatibility** | Full | Full (backward compatible) |
| **Input Validation** | Standard max_model_len check | Extended max_embed_len check |

#### Extreme Long Text Support

With the enhanced `max_embed_len` configuration (up to 3M+ tokens), you can process:
- **Complete Documents**: Research papers, legal contracts, technical manuals
- **Large Codebases**: Entire repositories and documentation
- **Books and Literature**: Full chapters or small books
- **Multi-document Analysis**: Combined content for comprehensive understanding

### Example Usage

#### Basic Configuration

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="http://localhost:31090/v1"
)

# This will automatically use chunked processing for very long text
# max_embed_len=3072000 allows inputs up to 3M+ tokens
response = client.embeddings.create(
    input="Very long text that exceeds the model's position embeddings..." * 5000,
    model="multilingual-e5-large"
)

print(f"Embedding dimension: {len(response.data[0].embedding)}")
```

#### Alternative Model Configurations

```bash
# For Jina embeddings v3 (optimized for performance)
vllm serve jinaai/jina-embeddings-v3 \
  --task embed \
  --override-pooler-config '{"pooling_type": "MEAN", "normalize": true, "enable_chunked_processing": true, "max_embed_len": 1048576}' \
  --trust-remote-code

# For Jina embeddings v4 (latest retrieval model)  
vllm serve jinaai/jina-embeddings-v4-vllm-retrieval \
  --task embed \
  --override-pooler-config '{"pooling_type": "MEAN", "normalize": true, "enable_chunked_processing": true, "max_embed_len": 2097152}' \
  --trust-remote-code

# For Qwen3 Embedding (large-scale multilingual)
vllm serve Qwen/Qwen3-Embedding-4B \
  --task embed \
  --override-pooler-config '{"pooling_type": "MEAN", "normalize": true, "enable_chunked_processing": true, "max_embed_len": 1572864}' \
  --trust-remote-code
```

### Logging and Monitoring

When chunked processing is active, you'll see informative log messages:

```
INFO: Input length 100000 exceeds max_position_embeddings 512, will use chunked processing
INFO: Split input of 100000 tokens into 196 chunks (max_chunk_size: 512)
```

### Limitations

- **Increased Latency**: Processing multiple chunks takes longer than single-chunk processing
- **Model Support**: Currently limited to specific embedding models
- **Context Boundaries**: Chunking may split related content, though weighted averaging helps preserve overall semantics

## Offline Inference

The [LLM][vllm.LLM] class provides various methods for offline inference.
See [configuration][configuration] for a list of options when initializing the model.

### `LLM.encode`

The [encode][vllm.LLM.encode] method is available to all pooling models in vLLM.
It returns the extracted hidden states directly, which is useful for reward models.

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-Math-RM-72B", task="reward")
(output,) = llm.encode("Hello, my name is")

data = output.outputs.data
print(f"Data: {data!r}")
```

### `LLM.embed`

The [embed][vllm.LLM.embed] method outputs an embedding vector for each prompt.
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

The [classify][vllm.LLM.classify] method outputs a probability vector for each prompt.
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

The [score][vllm.LLM.score] method outputs similarity scores between sentence pairs.
It is designed for embedding models and cross encoder models. Embedding models use cosine similarity, and [cross-encoder models](https://www.sbert.net/examples/applications/cross-encoder/README.html) serve as rerankers between candidate query-document pairs in RAG systems.

!!! note
    vLLM can only perform the model inference component (e.g. embedding, reranking) of RAG.
    To handle RAG at a higher level, you should use integration frameworks such as [LangChain](https://github.com/langchain-ai/langchain).

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

Our [OpenAI-Compatible Server](../serving/openai_compatible_server.md) provides endpoints that correspond to the offline APIs:

- [Pooling API][pooling-api] is similar to `LLM.encode`, being applicable to all types of pooling models.
- [Embeddings API][embeddings-api] is similar to `LLM.embed`, accepting both text and [multi-modal inputs](../features/multimodal_inputs.md) for embedding models.
- [Classification API][classification-api] is similar to `LLM.classify` and is applicable to sequence classification models.
- [Score API][score-api] is similar to `LLM.score` for cross-encoder models.

## Matryoshka Embeddings

[Matryoshka Embeddings](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html#matryoshka-embeddings) or [Matryoshka Representation Learning (MRL)](https://arxiv.org/abs/2205.13147) is a technique used in training embedding models. It allows user to trade off between performance and cost.

!!! warning
    Not all embedding models are trained using Matryoshka Representation Learning. To avoid misuse of the `dimensions` parameter, vLLM returns an error for requests that attempt to change the output dimension of models that do not support Matryoshka Embeddings.

    For example, setting `dimensions` parameter while using the `BAAI/bge-m3` model will result in the following error.

    ```json
    {"object":"error","message":"Model \"BAAI/bge-m3\" does not support matryoshka representation, changing output dimensions will lead to poor results.","type":"BadRequestError","param":null,"code":400}
    ```

### Manually enable Matryoshka Embeddings

There is currently no official interface for specifying support for Matryoshka Embeddings. In vLLM, if `is_matryoshka` is `True` in `config.json,` it is allowed to change the output to arbitrary dimensions. Using `matryoshka_dimensions` can control the allowed output dimensions.

For models that support Matryoshka Embeddings but not recognized by vLLM, please manually override the config using `hf_overrides={"is_matryoshka": True}`, `hf_overrides={"matryoshka_dimensions": [<allowed output dimensions>]}` (offline) or `--hf_overrides '{"is_matryoshka": true}'`,  `--hf_overrides '{"matryoshka_dimensions": [<allowed output dimensions>]}'`(online).

Here is an example to serve a model with Matryoshka Embeddings enabled.

```text
vllm serve Snowflake/snowflake-arctic-embed-m-v1.5 --hf_overrides '{"matryoshka_dimensions":[256]}'
```

### Offline Inference

You can change the output dimensions of embedding models that support Matryoshka Embeddings by using the dimensions parameter in [PoolingParams][vllm.PoolingParams].

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
curl http://127.0.0.1:31090/v1/embeddings \
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
