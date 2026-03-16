# Embedding Models

Embedding models are a class of machine learning models designed to transform unstructured data—such as text, images, or audio—into a structured numerical representation known as an embedding.

This functionality is supported through the `embed` pooling task, the `offline LLM.embed(...)` and `LLM.encode(..., pooling_task="embed")` APIs, as well as the online `/v1/embeddings` endpoint.

The primary distinction between (sequence) embedding and token embedding lies in their output granularity: (sequence) embedding produces a single embedding vector for an entire input sequence, whereas token embedding generates an embedding for each individual token within the sequence.

Many embedding models support both (sequence) embedding and token embedding. For further details on token embedding, please refer to [this page](token_embed.md).

## Typical Use Cases

### Get embedding

The most basic use case of embedding models is get embedding.

### Get similarity scores

Using the score API to compute similarity scores by calculating the cosine similarity of two input prompt's embeddings. See [Score API](score.md) for more information.

## Supported Models

- Text only models

| Architecture | Models | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `BertModel` | BERT-based | `BAAI/bge-base-en-v1.5`, `Snowflake/snowflake-arctic-embed-xs`, etc. | | |
| `BertSpladeSparseEmbeddingModel` | SPLADE | `naver/splade-v3` | | |
| `ErnieModel` | BERT-like Chinese ERNIE | `shibing624/text2vec-base-chinese-sentence` | | |
| `Gemma2Model`<sup>C</sup> | Gemma 2-based | `BAAI/bge-multilingual-gemma2`, etc. | ✅︎ | ✅︎ |
| `Gemma3TextModel`<sup>C</sup> | Gemma 3-based | `google/embeddinggemma-300m`, etc. | ✅︎ | ✅︎ |
| `GritLM` | GritLM | `parasail-ai/GritLM-7B-vllm`. | ✅︎ | ✅︎ |
| `GteModel` | Arctic-Embed-2.0-M | `Snowflake/snowflake-arctic-embed-m-v2.0`. | | |
| `GteNewModel` | mGTE-TRM (see note) | `Alibaba-NLP/gte-multilingual-base`, etc. | | |
| `LlamaBidirectionalModel`<sup>C</sup> | Llama-based with bidirectional attention | `nvidia/llama-nemotron-embed-1b-v2`, etc. | ✅︎ | ✅︎ |
| `LlamaModel`<sup>C</sup>, `LlamaForCausalLM`<sup>C</sup>, `MistralModel`<sup>C</sup>, etc. | Llama-based | `intfloat/e5-mistral-7b-instruct`, etc. | ✅︎ | ✅︎ |
| `ModernBertModel` | ModernBERT-based | `Alibaba-NLP/gte-modernbert-base`, etc. | | |
| `NomicBertModel` | Nomic BERT | `nomic-ai/nomic-embed-text-v1`, `nomic-ai/nomic-embed-text-v2-moe`, `Snowflake/snowflake-arctic-embed-m-long`, etc. | | |
| `Qwen2Model`<sup>C</sup>, `Qwen2ForCausalLM`<sup>C</sup> | Qwen2-based | `ssmits/Qwen2-7B-Instruct-embed-base` (see note), `Alibaba-NLP/gte-Qwen2-7B-instruct` (see note), etc. | ✅︎ | ✅︎ |
| `Qwen3Model`<sup>C</sup>, `Qwen3ForCausalLM`<sup>C</sup> | Qwen3-based | `Qwen/Qwen3-Embedding-0.6B`, etc. | ✅︎ | ✅︎ |
| `RobertaModel`, `RobertaForMaskedLM` | RoBERTa-based | `sentence-transformers/all-roberta-large-v1`, etc. | | |
| `VoyageQwen3BidirectionalEmbedModel`<sup>C</sup> | Voyage Qwen3-based with bidirectional attention | `voyageai/voyage-4-nano`, etc. | ✅︎ | ✅︎ |
| `XLMRobertaModel` | XLMRobertaModel-based | `BAAI/bge-m3` (see note), `intfloat/multilingual-e5-base`, `jinaai/jina-embeddings-v3` (see note), etc. | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

!!! note
    The second-generation GTE model (mGTE-TRM) is named `NewModel`. The name `NewModel` is too generic, you should set `--hf-overrides '{"architectures": ["GteNewModel"]}'` to specify the use of the `GteNewModel` architecture.

!!! note
    `ssmits/Qwen2-7B-Instruct-embed-base` has an improperly defined Sentence Transformers config.
    You need to manually set mean pooling by passing `--pooler-config '{"pooling_type": "MEAN"}'`.

!!! note
    For `Alibaba-NLP/gte-Qwen2-*`, you need to enable `--trust-remote-code` for the correct tokenizer to be loaded.
    See [relevant issue on HF Transformers](https://github.com/huggingface/transformers/issues/34882).

!!! note
    The `BAAI/bge-m3` model comes with extra weights for sparse and colbert embeddings, See [this page](specific_models.md#baaibge-m3) for more information.

!!! note
    `jinaai/jina-embeddings-v3` supports multiple tasks through LoRA, while vllm temporarily only supports text-matching tasks by merging LoRA weights.

- Multimodal Models

!!! note
    For more information about multimodal models inputs, see [this page](../supported_models.md#list-of-multimodal-language-models).

| Architecture | Models | Inputs | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `CLIPModel` | CLIP | T / I | `openai/clip-vit-base-patch32`, `openai/clip-vit-large-patch14`, etc. | | |
| `LlamaNemotronVLModel` | Llama Nemotron Embedding + SigLIP | T + I | `nvidia/llama-nemotron-embed-vl-1b-v2` | | |
| `LlavaNextForConditionalGeneration`<sup>C</sup> | LLaVA-NeXT-based | T / I | `royokong/e5-v` | | ✅︎ |
| `Phi3VForCausalLM`<sup>C</sup> | Phi-3-Vision-based | T + I | `TIGER-Lab/VLM2Vec-Full` | | ✅︎ |
| `Qwen3VLForConditionalGeneration`<sup>C</sup> | Qwen3-VL | T + I + V | `Qwen/Qwen3-VL-Embedding-2B`, etc. | ✅︎ | ✅︎ |
| `SiglipModel` | SigLIP, SigLIP2 | T / I | `google/siglip-base-patch16-224`, `google/siglip2-base-patch16-224` | | |
| `*ForConditionalGeneration`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | \* | N/A | \* | \* |

<sup>C</sup> Automatically converted into an embedding model via `--convert embed`. ([details](./README.md#model-conversion))  
\* Feature support is the same as that of the original model.

If your model is not in the above list, we will try to automatically convert the model using
[as_embedding_model][vllm.model_executor.models.adapters.as_embedding_model]. By default, the embeddings
of the whole prompt are extracted from the normalized hidden state corresponding to the last token.

!!! note
    Although vLLM supports automatically converting models of any architecture into embedding models via --convert embed, to get the best results, you should use pooling models that are specifically trained as such.

## Offline Inference

### Pooling Parameters

The following [pooling parameters][vllm.PoolingParams] are supported.

```python
--8<-- "vllm/pooling_params.py:common-pooling-params"
--8<-- "vllm/pooling_params.py:embed-pooling-params"
```

### `LLM.embed`

The [embed][vllm.LLM.embed] method outputs an embedding vector for each prompt.

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.embed("Hello, my name is")

embeds = output.outputs.embedding
print(f"Embeddings: {embeds!r} (size={len(embeds)})")
```

A code example can be found here: [examples/offline_inference/basic/embed.py](../../examples/offline_inference/basic/embed.py)

### `LLM.encode`

The [encode][vllm.LLM.encode] method is available to all pooling models in vLLM.

Set `pooling_task="embed"` when using `LLM.encode` for embedding Models:

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.encode("Hello, my name is", pooling_task="embed")

data = output.outputs.data
print(f"Data: {data!r}")
```

### `LLM.score`

The [score][vllm.LLM.score] method outputs similarity scores between sentence pairs.

All models that support embedding task also support using the score API to compute similarity scores by calculating the cosine similarity of two input prompt's embeddings.
```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
(output,) = llm.score(
    "What is the capital of France?",
    "The capital of Brazil is Brasilia.",
)

score = output.outputs.score
print(f"Score: {score}")
```
