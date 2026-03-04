# Score API

## Cross-encoder / Reranker

### Supported Models

#### Text-only Language Models

Cross-encoder and reranker models are a subset of classification models that accept two prompts as input.

| Architecture | Models | Example HF Models | Score template (see note) | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|---------------------------|-----------------------------|-----------------------------------------|
| `BertForSequenceClassification` | BERT-based | `cross-encoder/ms-marco-MiniLM-L-6-v2`, etc. | N/A | | |
| `GemmaForSequenceClassification` | Gemma-based | `BAAI/bge-reranker-v2-gemma`(see note), etc. | [bge-reranker-v2-gemma.jinja](../../examples/pooling/score/template/bge-reranker-v2-gemma.jinja) | ✅︎ | ✅︎ |
| `GteNewForSequenceClassification` | mGTE-TRM (see note) | `Alibaba-NLP/gte-multilingual-reranker-base`, etc. | N/A | | |
| `LlamaBidirectionalForSequenceClassification`<sup>C</sup> | Llama-based with bidirectional attention | `nvidia/llama-nemotron-rerank-1b-v2`, etc. | [nemotron-rerank.jinja](../../examples/pooling/score/template/nemotron-rerank.jinja) | ✅︎ | ✅︎ |
| `Qwen2ForSequenceClassification`<sup>C</sup> | Qwen2-based | `mixedbread-ai/mxbai-rerank-base-v2`(see note), etc. | [mxbai_rerank_v2.jinja](../../examples/pooling/score/template/mxbai_rerank_v2.jinja) | ✅︎ | ✅︎ |
| `Qwen3ForSequenceClassification`<sup>C</sup> | Qwen3-based | `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`, `Qwen/Qwen3-Reranker-0.6B`(see note), etc. | [qwen3_reranker.jinja](../../examples/pooling/score/template/qwen3_reranker.jinja) | ✅︎ | ✅︎ |
| `RobertaForSequenceClassification` | RoBERTa-based | `cross-encoder/quora-roberta-base`, etc. | N/A | | |
| `XLMRobertaForSequenceClassification` | XLM-RoBERTa-based | `BAAI/bge-reranker-v2-m3`, etc. | N/A | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | N/A | \* | \* |

<sup>C</sup> Automatically converted into a classification model via `--convert classify`. ([details](./pooling_models.md#model-conversion))  
\* Feature support is the same as that of the original model.

!!! note
    Some models require a specific prompt format to work correctly.

    You can find Example HF Models's corresponding score template in [examples/pooling/score/template/](../../examples/pooling/score/template)

    Examples : [examples/pooling/score/using_template_offline.py](../../examples/pooling/score/using_template_offline.py) [examples/pooling/score/using_template_online.py](../../examples/pooling/score/using_template_online.py)

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
    Load the official original `Qwen3 Reranker` by using the following command. More information can be found at: [examples/pooling/score/qwen3_reranker_offline.py](../../examples/pooling/score/qwen3_reranker_offline.py) [examples/pooling/score/qwen3_reranker_online.py](../../examples/pooling/score/qwen3_reranker_online.py).

    ```bash
    vllm serve Qwen/Qwen3-Reranker-0.6B --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
    ```

### Multimodal Language Models

The following modalities are supported depending on the model:

- **T**ext
- **I**mage
- **V**ideo
- **A**udio

Any combination of modalities joined by `+` are supported.

- e.g.: `T + I` means that the model supports text-only, image-only, and text-with-image inputs.

On the other hand, modalities separated by `/` are mutually exclusive.

- e.g.: `T / I` means that the model supports text-only and image-only inputs, but not text-with-image inputs.

See [this page](../features/multimodal_inputs.md) on how to pass multi-modal inputs to the model.

!!! tip
    For hybrid-only models such as Llama-4, Step3, Mistral-3 and Qwen-3.5, a text-only mode can be enabled by setting all supported multimodal modalities to 0 (`--language-model-only`) so that their multimodal modules will not be loaded to free up more GPU memory for KV cache.

!!! note
    vLLM currently supports adding LoRA adapters to the language backbone for most multimodal models. Additionally, vLLM now experimentally supports adding LoRA to the tower and connector modules for some multimodal models. See [this page](../features/lora.md).

!!! note
    To get the best results, you should use pooling models that are specifically trained as such.

The following table lists those that are tested in vLLM.

| Architecture | Models | Inputs | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|--------|-------------------|----------------------|---------------------------|
| `JinaVLForSequenceClassification` | JinaVL-based | T + I<sup>E+</sup> | `jinaai/jina-reranker-m0`, etc. | ✅︎ | ✅︎ |
| `LlamaNemotronVLForSequenceClassification` | Llama Nemotron Reranker + SigLIP | T + I<sup>E+</sup> | `nvidia/llama-nemotron-rerank-vl-1b-v2` | | |
| `Qwen3VLForSequenceClassification` | Qwen3-VL-Reranker | T + I<sup>E+</sup> + V<sup>E+</sup> | `Qwen/Qwen3-VL-Reranker-2B`(see note), etc. | ✅︎ | ✅︎ |

<sup>C</sup> Automatically converted into a classification model via `--convert classify`. ([details](./pooling_models.md#model-conversion))  
\* Feature support is the same as that of the original model.

!!! note
    Similar to Qwen3-Reranker, you need to use the following `--hf_overrides` to load the official original `Qwen3-VL-Reranker`.

    ```bash
    vllm serve Qwen/Qwen3-VL-Reranker-2B --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}'
    ```


## Bi-encoder / Embedding


## Late Interaction Models

## Offline Inference

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
