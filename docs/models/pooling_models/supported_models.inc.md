<!-- markdownlint-disable MD041 -->
--8<-- [start:classify-models]

### Text-only Models

| Architecture | Models | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `ErnieForSequenceClassification` | BERT-like Chinese ERNIE | `Forrest20231206/ernie-3.0-base-zh-cls` | | |
| `GPT2ForSequenceClassification` | GPT2 | `nie3e/sentiment-polish-gpt2-small` | | |
| `Qwen2ForSequenceClassification`<sup>C</sup> | Qwen2-based | `jason9693/Qwen2.5-1.5B-apeach` | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

### Multimodal Models

!!! note
    For more information about multimodal models inputs, see [this page](../supported_models.md#list-of-multimodal-language-models).

| Architecture | Models | Inputs | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `Qwen2_5_VLForSequenceClassification`<sup>C</sup> | Qwen2_5_VL-based | T + I<sup>E+</sup> + V<sup>E+</sup> | `muziyongshixin/Qwen2.5-VL-7B-for-VideoCls` | | |
| `*ForConditionalGeneration`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | \* | N/A | \* | \* |

<sup>C</sup> Automatically converted into a classification model via `--convert classify`. ([details](./README.md#model-conversion))  
\* Feature support is the same as that of the original model.

If your model is not in the above list, we will try to automatically convert the model using
[as_seq_cls_model][vllm.model_executor.models.adapters.as_seq_cls_model]. By default, the class probabilities are extracted from the softmaxed hidden state corresponding to the last token.

--8<-- [end:classify-models]

--8<-- [start:embed-models]

### Text-only Models

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

### Multimodal Models

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

--8<-- [end:embed-models]

--8<-- [start:sequence-reward-models]

| Architecture | Models | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `JambaForSequenceClassification` | Jamba | `ai21labs/Jamba-tiny-reward-dev`, etc. | ✅︎ | ✅︎ |
| `Qwen3ForSequenceClassification`<sup>C</sup> | Qwen3-based | `Skywork/Skywork-Reward-V2-Qwen3-0.6B`, etc. | ✅︎ | ✅︎ |
| `LlamaForSequenceClassification`<sup>C</sup> | Llama-based | `Skywork/Skywork-Reward-V2-Llama-3.2-1B`, etc. | ✅︎ | ✅︎ |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

<sup>C</sup> Automatically converted into a classification model via `--convert classify`. ([details](./README.md#model-conversion))  

If your model is not in the above list, we will try to automatically convert the model using
[as_seq_cls_model][vllm.model_executor.models.adapters.as_seq_cls_model]. By default, the class probabilities are extracted from the softmaxed hidden state corresponding to the last token.

--8<-- [end:sequence-reward-models]

--8<-- [start:token-reward-models]

| Architecture | Models | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `InternLM2ForRewardModel` | InternLM2-based | `internlm/internlm2-1_8b-reward`, `internlm/internlm2-7b-reward`, etc. | ✅︎ | ✅︎ |
| `Qwen2ForRewardModel` | Qwen2-based | `Qwen/Qwen2.5-Math-RM-72B`, etc. | ✅︎ | ✅︎ |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

<sup>C</sup> Automatically converted into a classification model via `--convert classify`. ([details](./README.md#model-conversion))  

If your model is not in the above list, we will try to automatically convert the model using
[as_seq_cls_model][vllm.model_executor.models.adapters.as_seq_cls_model].

--8<-- [end:token-reward-models]

--8<-- [start:process-reward-models]

| Architecture | Models | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
| `LlamaForCausalLM` | Llama-based | `peiyi9979/math-shepherd-mistral-7b-prm`, etc. | ✅︎ | ✅︎ |
| `Qwen2ForProcessRewardModel` | Qwen2-based | `Qwen/Qwen2.5-Math-PRM-7B`, etc. | ✅︎ | ✅︎ |

!!! important
    For process-supervised reward models such as `peiyi9979/math-shepherd-mistral-7b-prm`, the pooling config should be set explicitly,
    e.g.: `--pooler-config '{"pooling_type": "STEP", "step_tag_id": 123, "returned_token_ids": [456, 789]}'`.

--8<-- [end:process-reward-models]

--8<-- [start:score-models]

#### Text-only Models

| Architecture | Models | Example HF Models | Score template (see note) | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | ------------------------- | --------------------------- | --------------------------------------- |
| `BertForSequenceClassification` | BERT-based | `cross-encoder/ms-marco-MiniLM-L-6-v2`, etc. | N/A | | |
| `GemmaForSequenceClassification` | Gemma-based | `BAAI/bge-reranker-v2-gemma`(see note), etc. | [bge-reranker-v2-gemma.jinja](../../../examples/pooling/score/template/bge-reranker-v2-gemma.jinja) | ✅︎ | ✅︎ |
| `GteNewForSequenceClassification` | mGTE-TRM (see note) | `Alibaba-NLP/gte-multilingual-reranker-base`, etc. | N/A | | |
| `LlamaBidirectionalForSequenceClassification`<sup>C</sup> | Llama-based with bidirectional attention | `nvidia/llama-nemotron-rerank-1b-v2`, etc. | [nemotron-rerank.jinja](../../../examples/pooling/score/template/nemotron-rerank.jinja) | ✅︎ | ✅︎ |
| `Qwen2ForSequenceClassification`<sup>C</sup> | Qwen2-based | `mixedbread-ai/mxbai-rerank-base-v2`(see note), etc. | [mxbai_rerank_v2.jinja](../../../examples/pooling/score/template/mxbai_rerank_v2.jinja) | ✅︎ | ✅︎ |
| `Qwen3ForSequenceClassification`<sup>C</sup> | Qwen3-based | `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`, `Qwen/Qwen3-Reranker-0.6B`(see note), etc. | [qwen3_reranker.jinja](../../../examples/pooling/score/template/qwen3_reranker.jinja) | ✅︎ | ✅︎ |
| `RobertaForSequenceClassification` | RoBERTa-based | `cross-encoder/quora-roberta-base`, etc. | N/A | | |
| `XLMRobertaForSequenceClassification` | XLM-RoBERTa-based | `BAAI/bge-reranker-v2-m3`, etc. | N/A | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | N/A | \* | \* |

<sup>C</sup> Automatically converted into a classification model via `--convert classify`. ([details](./README.md#model-conversion))  
\* Feature support is the same as that of the original model.

!!! note
    Some models require a specific prompt format to work correctly.

    You can find Example HF Models's corresponding score template in [examples/pooling/score/template/](../../../examples/pooling/score/template)

    Examples : [examples/pooling/score/using_template_offline.py](../../../examples/pooling/score/using_template_offline.py) [examples/pooling/score/using_template_online.py](../../../examples/pooling/score/using_template_online.py)

!!! note
    Load the official original `BAAI/bge-reranker-v2-gemma` by using the following command.

    ```bash
    vllm serve BAAI/bge-reranker-v2-gemma --hf_overrides '{"architectures": ["GemmaForSequenceClassification"],"classifier_from_token": ["Yes"],"method": "no_post_processing"}'
    ```

!!! note
    The second-generation GTE model (mGTE-TRM) is named `NewForSequenceClassification`. The name `NewForSequenceClassification` is too generic, you should set `--hf-overrides '{"architectures": ["GteNewForSequenceClassification"]}'` to specify the use of the `GteNewForSequenceClassification` architecture.

!!! note
    Load the official original `mxbai-rerank-v2` by using the following command.

    ```bash
    vllm serve mixedbread-ai/mxbai-rerank-base-v2 --hf_overrides '{"architectures": ["Qwen2ForSequenceClassification"],"classifier_from_token": ["0", "1"], "method": "from_2_way_softmax"}'
    ```

!!! note
    Load the official original `Qwen3 Reranker` by using the following command. More information can be found at: [examples/pooling/score/qwen3_reranker_offline.py](../../../examples/pooling/score/qwen3_reranker_offline.py) [examples/pooling/score/qwen3_reranker_online.py](../../../examples/pooling/score/qwen3_reranker_online.py).

    ```bash
    vllm serve Qwen/Qwen3-Reranker-0.6B --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
    ```

#### Multimodal Models

!!! note
    For more information about multimodal models inputs, see [this page](../supported_models.md#list-of-multimodal-language-models).

| Architecture | Models | Inputs | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ------ | ----------------- | ------------------------------ | ------------------------------------------ |
| `JinaVLForSequenceClassification` | JinaVL-based | T + I<sup>E+</sup> | `jinaai/jina-reranker-m0`, etc. | ✅︎ | ✅︎ |
| `LlamaNemotronVLForSequenceClassification` | Llama Nemotron Reranker + SigLIP | T + I<sup>E+</sup> | `nvidia/llama-nemotron-rerank-vl-1b-v2` | | |
| `Qwen3VLForSequenceClassification` | Qwen3-VL-Reranker | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-Reranker-2B`(see note), etc. | ✅︎ | ✅︎ |

<sup>C</sup> Automatically converted into a classification model via `--convert classify`. ([details](README.md#model-conversion))  
\* Feature support is the same as that of the original model.

!!! note
    Similar to Qwen3-Reranker, you need to use the following `--hf_overrides` to load the official original `Qwen3-VL-Reranker`.

    ```bash
    vllm serve Qwen/Qwen3-VL-Reranker-2B --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
    ```

--8<-- [end:score-models]

--8<-- [start:token-classify-models]

| Architecture | Models | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | --------------------------- | --------------------------------------- |
| `BertForTokenClassification` | bert-based | `boltuix/NeuroBERT-NER` (see note), etc. | | |
| `ErnieForTokenClassification` | BERT-like Chinese ERNIE | `gyr66/Ernie-3.0-base-chinese-finetuned-ner` | | |
| `ModernBertForTokenClassification` | ModernBERT-based | `disham993/electrical-ner-ModernBERT-base` | | |
| `Qwen3ForTokenClassification`<sup>C</sup> | Qwen3-based | `bd2lcco/Qwen3-0.6B-finetuned` | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

<sup>C</sup> Automatically converted into a classification model via `--convert classify`. ([details](./README.md#model-conversion))  
\* Feature support is the same as that of the original model.

If your model is not in the above list, we will try to automatically convert the model using
[as_seq_cls_model][vllm.model_executor.models.adapters.as_seq_cls_model]. By default, the class probabilities are extracted from the softmaxed hidden state corresponding to the last token.

--8<-- [end:token-classify-models]

--8<-- [start:token-embed-models]

### Text-only Models

| Architecture | Models | Example HF Models | [LoRA](../../features/lora.md) | [PP](../../serving/parallelism_scaling.md) |
| ------------ | ------ | ----------------- | -------------------- | ------------------------- |
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
| `ColPaliForRetrieval` | ColPali | T / I | `vidore/colpali-v1.3-hf` | | |
| `ColQwen3` | Qwen3-VL | T / I | `TomoroAI/tomoro-colqwen3-embed-4b`, `TomoroAI/tomoro-colqwen3-embed-8b` | | |
| `ColQwen3_5` | ColQwen3.5 | T + I + V | `athrael-soju/colqwen3.5-4.5B-v3` | | |
| `OpsColQwen3Model` | Qwen3-VL | T / I | `OpenSearch-AI/Ops-Colqwen3-4B`, `OpenSearch-AI/Ops-Colqwen3-8B` | | |
| `Qwen3VLNemotronEmbedModel` | Qwen3-VL | T / I | `nvidia/nemotron-colembed-vl-4b-v2`, `nvidia/nemotron-colembed-vl-8b-v2` | ✅︎ | ✅︎ |
| `*ForConditionalGeneration`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | \* | N/A | \* | \* |

<sup>C</sup> Automatically converted into an embedding model via `--convert embed`. ([details](./README.md#model-conversion))  
\* Feature support is the same as that of the original model.

If your model is not in the above list, we will try to automatically convert the model using [as_embedding_model][vllm.model_executor.models.adapters.as_embedding_model].

--8<-- [end:token-embed-models]
