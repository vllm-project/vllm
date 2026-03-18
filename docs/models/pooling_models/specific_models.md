# Specific Model Examples

## ColBERT Late Interaction Models

[ColBERT](https://arxiv.org/abs/2004.12832) (Contextualized Late Interaction over BERT) is a retrieval model that uses per-token embeddings and MaxSim scoring for document ranking. Unlike single-vector embedding models, ColBERT retains token-level representations and computes relevance scores through late interaction, providing better accuracy while being more efficient than cross-encoders.

vLLM supports ColBERT models with multiple encoder backbones:

| Architecture | Backbone | Example HF Models |
| - | - | - |
| `HF_ColBERT` | BERT | `answerdotai/answerai-colbert-small-v1`, `colbert-ir/colbertv2.0` |
| `ColBERTModernBertModel` | ModernBERT | `lightonai/GTE-ModernColBERT-v1` |
| `ColBERTJinaRobertaModel` | Jina XLM-RoBERTa | `jinaai/jina-colbert-v2` |

**BERT-based ColBERT** models work out of the box:

```shell
vllm serve answerdotai/answerai-colbert-small-v1
```

For **non-BERT backbones**, use `--hf-overrides` to set the correct architecture:

```shell
# ModernBERT backbone
vllm serve lightonai/GTE-ModernColBERT-v1 \
    --hf-overrides '{"architectures": ["ColBERTModernBertModel"]}'

# Jina XLM-RoBERTa backbone
vllm serve jinaai/jina-colbert-v2 \
    --hf-overrides '{"architectures": ["ColBERTJinaRobertaModel"]}' \
    --trust-remote-code
```

Then you can use the rerank API:

```shell
curl -s http://localhost:8000/rerank -H "Content-Type: application/json" -d '{
    "model": "answerdotai/answerai-colbert-small-v1",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language.",
        "Deep learning uses neural networks."
    ]
}'
```

Or the score API:

```shell
curl -s http://localhost:8000/score -H "Content-Type: application/json" -d '{
    "model": "answerdotai/answerai-colbert-small-v1",
    "text_1": "What is machine learning?",
    "text_2": ["Machine learning is a subset of AI.", "The weather is sunny."]
}'
```

You can also get the raw token embeddings using the pooling API with `token_embed` task:

```shell
curl -s http://localhost:8000/pooling -H "Content-Type: application/json" -d '{
    "model": "answerdotai/answerai-colbert-small-v1",
    "input": "What is machine learning?",
    "task": "token_embed"
}'
```

An example can be found here: [examples/pooling/score/colbert_rerank_online.py](../../../examples/pooling/score/colbert_rerank_online.py)

## ColQwen3 Multi-Modal Late Interaction Models

ColQwen3 is based on [ColPali](https://arxiv.org/abs/2407.01449), which extends ColBERT's late interaction approach to **multi-modal** inputs. While ColBERT operates on text-only token embeddings, ColPali/ColQwen3 can embed both **text and images** (e.g. PDF pages, screenshots, diagrams) into per-token L2-normalized vectors and compute relevance via MaxSim scoring. ColQwen3 specifically uses Qwen3-VL as its vision-language backbone.

| Architecture | Backbone | Example HF Models |
| - | - | - |
| `ColQwen3` | Qwen3-VL | `TomoroAI/tomoro-colqwen3-embed-4b`, `TomoroAI/tomoro-colqwen3-embed-8b` |
| `OpsColQwen3Model` | Qwen3-VL | `OpenSearch-AI/Ops-Colqwen3-4B`, `OpenSearch-AI/Ops-Colqwen3-8B` |
| `Qwen3VLNemotronEmbedModel` | Qwen3-VL | `nvidia/nemotron-colembed-vl-4b-v2`, `nvidia/nemotron-colembed-vl-8b-v2` |

Start the server:

```shell
vllm serve TomoroAI/tomoro-colqwen3-embed-4b --max-model-len 4096
```

### Text-only scoring and reranking

Use the `/rerank` API:

```shell
curl -s http://localhost:8000/rerank -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language.",
        "Deep learning uses neural networks."
    ]
}'
```

Or the `/score` API:

```shell
curl -s http://localhost:8000/score -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "text_1": "What is the capital of France?",
    "text_2": ["The capital of France is Paris.", "Python is a programming language."]
}'
```

### Multi-modal scoring and reranking (text query × image documents)

The `/score` and `/rerank` APIs also accept multi-modal inputs directly.
Pass image documents using the `data_1`/`data_2` (for `/score`) or `documents` (for `/rerank`) fields
with a `content` list containing `image_url` and `text` parts — the same format used by the
OpenAI chat completion API:

Score a text query against image documents:

```shell
curl -s http://localhost:8000/score -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "data_1": "Retrieve the city of Beijing",
    "data_2": [
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
                {"type": "text", "text": "Describe the image."}
            ]
        }
    ]
}'
```

Rerank image documents by a text query:

```shell
curl -s http://localhost:8000/rerank -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "query": "Retrieve the city of Beijing",
    "documents": [
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_1>"}},
                {"type": "text", "text": "Describe the image."}
            ]
        },
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_2>"}},
                {"type": "text", "text": "Describe the image."}
            ]
        }
    ],
    "top_n": 2
}'
```

### Raw token embeddings

You can also get the raw token embeddings using the `/pooling` API with `token_embed` task:

```shell
curl -s http://localhost:8000/pooling -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "input": "What is machine learning?",
    "task": "token_embed"
}'
```

For **image inputs** via the pooling API, use the chat-style `messages` field:

```shell
curl -s http://localhost:8000/pooling -H "Content-Type: application/json" -d '{
    "model": "TomoroAI/tomoro-colqwen3-embed-4b",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
                {"type": "text", "text": "Describe the image."}
            ]
        }
    ]
}'
```

### Examples

- Multi-vector retrieval: [examples/pooling/token_embed/colqwen3_token_embed_online.py](../../../examples/pooling/token_embed/colqwen3_token_embed_online.py)
- Reranking (text + multi-modal): [examples/pooling/score/colqwen3_rerank_online.py](../../../examples/pooling/score/colqwen3_rerank_online.py)

### ColQwen3.5 Multi-Modal Late Interaction Models

ColQwen3.5 is based on [ColPali](https://arxiv.org/abs/2407.01449), extending ColBERT's late interaction approach to **multi-modal** inputs. It uses the Qwen3.5 hybrid backbone (linear + full attention) and produces per-token L2-normalized vectors for MaxSim scoring.

| Architecture | Backbone | Example HF Models |
| - | - | - |
| `ColQwen3_5` | Qwen3.5 | `athrael-soju/colqwen3.5-4.5B` |

Start the server:

```shell
vllm serve athrael-soju/colqwen3.5-4.5B --max-model-len 4096
```

Then you can use the rerank endpoint:

```shell
curl -s http://localhost:8000/rerank -H "Content-Type: application/json" -d '{
    "model": "athrael-soju/colqwen3.5-4.5B",
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language.",
        "Deep learning uses neural networks."
    ]
}'
```

Or the score endpoint:

```shell
curl -s http://localhost:8000/score -H "Content-Type: application/json" -d '{
    "model": "athrael-soju/colqwen3.5-4.5B",
    "text_1": "What is the capital of France?",
    "text_2": ["The capital of France is Paris.", "Python is a programming language."]
}'
```

An example can be found here: [examples/pooling/score/colqwen3_5_rerank_online.py](../../../examples/pooling/score/colqwen3_5_rerank_online.py)

## Llama Nemotron Multimodal

### Embedding Model

Llama Nemotron VL Embedding models combine the bidirectional Llama embedding backbone
(from `nvidia/llama-nemotron-embed-1b-v2`) with SigLIP as the vision encoder to produce
single-vector embeddings from text and/or images.

| Architecture | Backbone | Example HF Models |
| - | - | - |
| `LlamaNemotronVLModel` | Bidirectional Llama + SigLIP | `nvidia/llama-nemotron-embed-vl-1b-v2` |

Start the server:

```shell
vllm serve nvidia/llama-nemotron-embed-vl-1b-v2 \
    --trust-remote-code \
    --chat-template examples/pooling/embed/template/nemotron_embed_vl.jinja
```

!!! note
    The chat template bundled with this model's tokenizer is not suitable for
    the embeddings API. Use the provided override template above when serving
    with the `messages`-based (chat-style) embeddings API.

    The override template uses the message `role` to automatically prepend the
    appropriate prefix: set `role` to `"query"` for queries (prepends `query: `)
    or `"document"` for passages (prepends `passage: `). Any other role omits
    the prefix.

Embed text queries:

```shell
curl -s http://localhost:8000/v1/embeddings -H "Content-Type: application/json" -d '{
    "model": "nvidia/llama-nemotron-embed-vl-1b-v2",
    "messages": [
        {
            "role": "query",
            "content": [
                {"type": "text", "text": "What is machine learning?"}
            ]
        }
    ]
}'
```

Embed images via the chat-style `messages` field:

```shell
curl -s http://localhost:8000/v1/embeddings -H "Content-Type: application/json" -d '{
    "model": "nvidia/llama-nemotron-embed-vl-1b-v2",
    "messages": [
        {
            "role": "document",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
                {"type": "text", "text": "Describe the image."}
            ]
        }
    ]
}'
```

### Reranker Model

Llama Nemotron VL reranker models combine the same bidirectional Llama + SigLIP
backbone with a sequence-classification head for cross-encoder scoring and reranking.

| Architecture | Backbone | Example HF Models |
| - | - | - |
| `LlamaNemotronVLForSequenceClassification` | Bidirectional Llama + SigLIP | `nvidia/llama-nemotron-rerank-vl-1b-v2` |

Start the server:

```shell
vllm serve nvidia/llama-nemotron-rerank-vl-1b-v2 \
    --runner pooling \
    --trust-remote-code \
    --chat-template examples/pooling/score/template/nemotron-vl-rerank.jinja
```

!!! note
    The chat template bundled with this checkpoint's tokenizer is not suitable
    for the Score/Rerank APIs. Use the provided override template when serving:
    `examples/pooling/score/template/nemotron-vl-rerank.jinja`.

Score a text query against an image document:

```shell
curl -s http://localhost:8000/score -H "Content-Type: application/json" -d '{
    "model": "nvidia/llama-nemotron-rerank-vl-1b-v2",
    "data_1": "Find diagrams about autonomous robots",
    "data_2": [
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64>"}},
                {"type": "text", "text": "Robotics workflow diagram."}
            ]
        }
    ]
}'
```

Rerank image documents by a text query:

```shell
curl -s http://localhost:8000/rerank -H "Content-Type: application/json" -d '{
    "model": "nvidia/llama-nemotron-rerank-vl-1b-v2",
    "query": "Find diagrams about autonomous robots",
    "documents": [
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_1>"}},
                {"type": "text", "text": "Robotics workflow diagram."}
            ]
        },
        {
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,<BASE64_2>"}},
                {"type": "text", "text": "General skyline photo."}
            ]
        }
    ],
    "top_n": 2
}'
```

## BAAI/bge-m3

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
     "input": ["What is BGE M3?", "Definition of BM25"]
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
     "input": ["What is BGE M3?", "Definition of BM25"]
}'
```
