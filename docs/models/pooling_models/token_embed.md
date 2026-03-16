# Token Embedding Models

Token embedding is supported through the `token_embed` pooling task, the offline `LLM.encode(..., pooling_task="embed")` API, and the online `/pooling` endpoint.

The difference between the (sequence) embedding task and the token embedding task is that (sequence) embedding outputs one embedding for each sequence, while token embedding outputs a embedding for each token.

Many embedding models support both (sequence) embedding and token embedding. For further details on (sequence) embedding, please refer to [this page](embed.md).

## Typical Use Cases

### Multi-Vector Retrieval

For implementation examples, see:

Offline: [examples/pooling/token_embed/multi_vector_retrieval_offline.py](../../../examples/pooling/token_embed/multi_vector_retrieval_offline.py)

Online: [examples/pooling/token_embed/multi_vector_retrieval_online.py](../../../examples/pooling/token_embed/multi_vector_retrieval_online.py)

### Late interaction

Similarity scores can be computed using late interaction between two input prompts via the score API. For more information, see [Score API](score.md).

### Extract last hidden states

Models of any architecture can be converted into embedding models using `--convert embed`. Token embedding can then be used to extract the last hidden states from these models.

## Supported Models

- Text only models

| Architecture | Models | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `ColBERTModernBertModel` | ModernBERT | `lightonai/GTE-ModernColBERT-v1` | | |
| `ColBERTJinaRobertaModel` | Jina XLM-RoBERTa | `jinaai/jina-colbert-v2` | | |
| `HF_ColBERT` | BERT | `answerdotai/answerai-colbert-small-v1`, `colbert-ir/colbertv2.0` | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

- Multimodal Models

!!! note
    For more information about multimodal models inputs, see [this page](../supported_models.md#list-of-multimodal-language-models).

| Architecture | Models | Inputs | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----- | ----------------- | ------------------------------ | ------------------------------------------ |
| `ColModernVBertForRetrieval` | ColModernVBERT | T / I | `ModernVBERT/colmodernvbert-merged` | | |
| `ColPaliForRetrieval` | ColPali | T / I | `vidore/colpali-v1.3-hf` | | |
| `ColQwen3` | Qwen3-VL | T / I | `TomoroAI/tomoro-colqwen3-embed-4b`, `TomoroAI/tomoro-colqwen3-embed-8b` | | |
| `OpsColQwen3Model` | Qwen3-VL | T / I | `OpenSearch-AI/Ops-Colqwen3-4B`, `OpenSearch-AI/Ops-Colqwen3-8B` | | |
| `Qwen3VLNemotronEmbedModel` | Qwen3-VL | T / I | `nvidia/nemotron-colembed-vl-4b-v2`, `nvidia/nemotron-colembed-vl-8b-v2` | ✅︎ | ✅︎ |
| `*ForConditionalGeneration`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | \* | N/A | \* | \* |

<sup>C</sup> Automatically converted into an embedding model via `--convert embed`. ([details](./README.md#model-conversion))  
\* Feature support is the same as that of the original model.

If your model is not in the above list, we will try to automatically convert the model using [as_embedding_model][vllm.model_executor.models.adapters.as_embedding_model].
