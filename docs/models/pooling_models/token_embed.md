# Token Embedding Usages

## Summary

- Model Usage: Token classification models
- Pooling Tasks: `token_embed`
- Offline APIs:
    - `LLM.embed(...)` with the `token_embed` pooling task
    - `LLM.encode(..., pooling_task="token_embed")`
- Online APIs:
    - Pooling API (`/pooling`)
    - Embedding API (`/v1/embeddings`), with optional [late chunking](#online-late-chunking)

The difference between the (sequence) embedding task and the token embedding task is that (sequence) embedding outputs one embedding for each sequence, while token embedding outputs an embedding for each token.

With chunked prefill, preempted requests recompute their prompt without duplicating token embedding vectors. Ordinary scheduling pauses retain the accumulated vectors.

Many embedding models support both (sequence) embedding and token embedding. For further details on (sequence) embedding, please refer to [this page](embed.md).

!!! note

    Pooling multitask support has been removed since v0.21. When the default pooling task (embed) is not 
    what you want, you need to manually specify it via `PoolerConfig(task="token_embed")` offline or
    `--pooler-config.task token_embed` online.

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

--8<-- [start:supported-token-embed-models]

### Text-only Models

| Architecture | Models | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `ColBERTLfm2Model` | LFM2 | `LiquidAI/LFM2-ColBERT-350M` | | |
| `ColBERTModernBertModel` | ModernBERT | `lightonai/GTE-ModernColBERT-v1` | | |
| `ColBERTJinaRobertaModel` | Jina XLM-RoBERTa | `jinaai/jina-colbert-v2` | | |
| `HF_ColBERT` | BERT | `answerdotai/answerai-colbert-small-v1`, `colbert-ir/colbertv2.0` | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

### Multimodal Models

!!! note
    For more information about multimodal models inputs, see [this page](../supported_models.md#list-of-multimodal-language-models).

| Architecture | Models | Inputs | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----- | ----------------- | ------------------------------ | ------------------------------------------ |
| `ColModernVBertForRetrieval` | ColModernVBERT | T / I | `ModernVBERT/colmodernvbert-merged` | | |
| `ColUltraVBertForRetrieval` | ColUltraVBERT | T / I | `athrael-soju/ColUltraVBERT` | | |
| `ColPaliForRetrieval` | ColPali | T / I | `vidore/colpali-v1.3-hf` | | |
| `ColQwen3` | Qwen3-VL | T / I | `TomoroAI/tomoro-colqwen3-embed-4b`, `TomoroAI/tomoro-colqwen3-embed-8b` | | |
| `ColQwen3_5` | ColQwen3.5 | T + I + V | `athrael-soju/colqwen3.5-4.5B-v3`, `vultr/VultronRetrieverPrime-Qwen3.5-8B` | | |
| `OpsColQwen3Model` | Qwen3-VL | T / I | `OpenSearch-AI/Ops-Colqwen3-4B`, `OpenSearch-AI/Ops-Colqwen3-8B` | | |
| `Qwen3VLNemotronEmbedModel` | Qwen3-VL | T / I | `nvidia/nemotron-colembed-vl-4b-v2`, `nvidia/nemotron-colembed-vl-8b-v2` | ✅︎ | ✅︎ |
| `*ForConditionalGeneration`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | \* | N/A | \* | \* |

<sup>C</sup> Automatically converted into an embedding model via `--convert embed`. ([details](./README.md#model-conversion))  
\* Feature support is the same as that of the original model.

If your model is not in the above list, we will try to automatically convert the model using [as_embedding_model][vllm.model_executor.models.adapters.as_embedding_model].

### Special models

| Architecture | Models | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `JinaForRanking` | Qwen3-based | `jinaai/jina-reranker-v3` | | |

jina-reranker-v3 is a listwise document reranker model with a novel `last but not late interaction` architecture. More information can be found at: [examples/pooling/token_embed/jina_reranker_v3_offline.py](../../../examples/pooling/token_embed/jina_reranker_v3_offline.py)

--8<-- [end:supported-token-embed-models]

## Offline Inference

### Pooling Parameters

The following [pooling parameters][vllm.PoolingParams] are supported.

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
--8<-- "vllm/pooling_params.py:embed-pooling-params"
--8<-- "vllm/pooling_params.py:token-embed-pooling-params"
```

### `LLM.encode`

The [encode][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.encode] method is available to all pooling models in vLLM.

Set `pooling_task="token_embed"` when using `LLM.encode` for token embedding models:

```python
from vllm import LLM

llm = LLM(model="answerdotai/answerai-colbert-small-v1", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="token_embed")

data = output.outputs.data
print(f"Data: {data!r}")
```

### `LLM.embed`

The [embed][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.embed] method uses the model's configured pooling task. For `token_embed`, it returns a `PoolingRequestOutput` per input, with a token-vector matrix in `outputs.data`. Ordinary `embed` models keep returning `EmbeddingRequestOutput` with one vector in `outputs.embedding`.

```python
from vllm import LLM

llm = LLM(model="answerdotai/answerai-colbert-small-v1", runner="pooling")
(output,) = llm.embed("Hello, my name is")

data = output.outputs.data
print(f"Data: {data!r}")
```

### Fixed-length late chunking

Late chunking encodes the full input before averaging contextual token representations within each chunk. The chunk means pass through the model's embedding head, including its projection and optional normalization.

With an instance configured for `token_embed`, enable late chunking per request through `LLM.embed` for plain-text input:

```python
from vllm import PoolingParams
from vllm.pooling_params import LateChunkingParams

outputs = llm.embed(
    [document],
    pooling_params=PoolingParams(
        late_chunking_params=LateChunkingParams(chunk_size=256)
    ),
)
vectors = outputs[0].outputs.data  # [num_chunks, embedding_dimension]
chunks = outputs[0].late_chunking.chunks
```

To use `LLM.encode` instead, pass the same pooling parameters and set `pooling_task="token_embed"`.

For `N` model-input tokens and chunk size `C`, the output contains `ceil(N / C)` vectors. The last chunk uses its actual length. Special tokens and instruction prefixes count toward both chunk boundaries and means. Omitting `late_chunking_params` preserves ordinary token embeddings.

Each vector row corresponds to one entry in `chunks`. Its `token_range` and `char_range` are zero-based, half-open ranges. Character ranges index the submitted Python string; use `document[start:end]` to retrieve the source text. A chunk containing only special tokens has `char_range=None`.

Use nonempty text within the model's context window and a fast tokenizer that provides aligned source offsets. Input truncation, explicit padding and renderer text normalization are not supported. The custom document/query outputs of `JinaForRanking` are also unsupported. Late chunking does not extend the model's context window or reduce its attention computation.

The [offline example](../../../examples/pooling/token_embed/late_chunking_offline.py) loads an E5 model and prints the vector shape, token ranges and source excerpts for two documents:

```bash
python examples/pooling/token_embed/late_chunking_offline.py --chunk-size 16
```

### `LLM.score`

The [score][vllm.entrypoints.pooling.offline.PoolingOfflineMixin.score] method outputs similarity scores between sentence pairs.

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

### Embedding API

On a server configured for `token_embed`, `/v1/embeddings` returns token vectors by default and chunk vectors when `late_chunking_params` is provided.

#### Online late chunking

Start a token embedding server, for example:

```bash
vllm serve intfloat/multilingual-e5-small \
    --runner pooling --pooler-config.task token_embed --max-model-len 512
```

Set `late_chunking_params.chunk_size` to a positive integer less than `2**63`:

```bash
curl http://localhost:8000/v1/embeddings \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "intfloat/multilingual-e5-small",
      "input": "passage: Berlin has many museums. They display art from around the world.",
      "late_chunking_params": {"chunk_size": 16},
      "encoding_format": "float"
    }'
```

Each response item contains `embedding`, a `[num_chunks, embedding_dimension]` matrix, and `late_chunking`, with these fields:

| Field | Meaning |
| --- | --- |
| `strategy` | `"fixed_tokens"` |
| `chunk_size` | Requested number of model-input tokens per chunk |
| `input_tokens` | Number of tokens in this document, including special tokens |
| `chunks` | One entry per vector row, containing `token_range` and `char_range` |

Ranges have the same semantics as the offline API. JSON represents them as two-element arrays; `char_range` is `null` for a special-token-only chunk. Character indices count Unicode code points, so clients using UTF-16 string indices must convert them before slicing.

Use a string or a nonempty list of strings as `input`. A batch shares one chunk size, returns one item per document in input order, and counts input tokens once in `usage`. Values of `chunk_size` larger than the input token count produce one chunk.

Omit `late_chunking_params` or set it to `null` to return ordinary token vectors through the same endpoint. Each item then contains a token-vector matrix in `embedding` and omits `late_chunking`. This mode also accepts token IDs. With late chunking enabled, input must be plain text and must not request truncation or padding, because source ranges refer to the complete submitted text. Only `encoding_format="float"` is supported; chat messages and `dimensions` are unsupported by this endpoint.

The matrix-valued `embedding` and `late_chunking` fields are a vLLM extension, not the standard OpenAI embedding response. Use an HTTP client that handles this format. Ordinary `embed` servers keep their existing response format; they do not enable late chunking and may ignore an unrecognized `late_chunking_params` field. Cohere `/v2/embed`, the batch API, and `/invocations` are not extended with late chunking.

#### Retrieve source text from chunk vectors

The [retrieval example](../../../examples/pooling/token_embed/late_chunking_retrieval_online.py) uses the server above to:

1. Encode documents with E5's `passage:` prefix and request contextual chunks.
2. Build an in-memory index containing vectors, document IDs, chunk indices and source ranges.
3. Encode a question with the `query:` prefix, using one chunk covering the whole query.
4. Rank stored vectors by cosine similarity and print the matching source excerpts.

```bash
python examples/pooling/token_embed/late_chunking_retrieval_online.py \
    --chunk-size 16 --query "Which city has museums with art collections?"
```

Add `--index-path late_chunking_index.json` to save the index. The example creates no index file by default.

For E5, one chunk covering the complete query produces its mean-pooled sequence embedding. Other models may require a different query head. Keep the model revision, prefixes, projection and normalization consistent between indexing and querying.

Source ranges refer to the exact submitted text, including the prefix. The example excludes special-token-only chunks from retrieval. Fixed token boundaries may split sentences; late chunking does not guarantee better retrieval on every dataset.

### Pooling API

Use the [Pooling API](README.md#pooling-api) with `"task": "token_embed"` to return token vectors. The Pooling API does not support late chunking.

## More examples

More examples can be found here: [examples/pooling/token_embed](../../../examples/pooling/token_embed)

## Supported Features

Token embedding features should be consistent with (sequence) embedding. For more information, see [this page](embed.md#supported-features).
