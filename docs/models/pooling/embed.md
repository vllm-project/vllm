# Embedding models

## Supported Models

#### Text-only Language Models

| Architecture | Models | Example HF Models | [LoRA](../features/lora.md) | [PP](../serving/parallelism_scaling.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `BertModel`<sup>C</sup> | BERT-based | `BAAI/bge-base-en-v1.5`, `Snowflake/snowflake-arctic-embed-xs`, etc. | | |
| `BertSpladeSparseEmbeddingModel` | SPLADE | `naver/splade-v3` | | |
| `Gemma2Model`<sup>C</sup> | Gemma 2-based | `BAAI/bge-multilingual-gemma2`, etc. | ✅︎ | ✅︎ |
| `Gemma3TextModel`<sup>C</sup> | Gemma 3-based | `google/embeddinggemma-300m`, etc. | ✅︎ | ✅︎ |
| `GritLM` | GritLM | `parasail-ai/GritLM-7B-vllm`. | ✅︎ | ✅︎ |
| `GteModel`<sup>C</sup> | Arctic-Embed-2.0-M | `Snowflake/snowflake-arctic-embed-m-v2.0`. |  |  |
| `GteNewModel`<sup>C</sup> | mGTE-TRM (see note) | `Alibaba-NLP/gte-multilingual-base`, etc. |  |  |
| `ModernBertModel`<sup>C</sup> | ModernBERT-based | `Alibaba-NLP/gte-modernbert-base`, etc. |  |  |
| `NomicBertModel`<sup>C</sup> | Nomic BERT | `nomic-ai/nomic-embed-text-v1`, `nomic-ai/nomic-embed-text-v2-moe`, `Snowflake/snowflake-arctic-embed-m-long`, etc. |  |  |
| `LlamaBidirectionalModel`<sup>C</sup> | Llama-based with bidirectional attention | `nvidia/llama-nemotron-embed-1b-v2`, etc. | ✅︎ | ✅︎ |
| `LlamaModel`<sup>C</sup>, `LlamaForCausalLM`<sup>C</sup>, `MistralModel`<sup>C</sup>, etc. | Llama-based | `intfloat/e5-mistral-7b-instruct`, etc. | ✅︎ | ✅︎ |
| `Qwen2Model`<sup>C</sup>, `Qwen2ForCausalLM`<sup>C</sup> | Qwen2-based | `ssmits/Qwen2-7B-Instruct-embed-base` (see note), `Alibaba-NLP/gte-Qwen2-7B-instruct` (see note), etc. | ✅︎ | ✅︎ |
| `Qwen3Model`<sup>C</sup>, `Qwen3ForCausalLM`<sup>C</sup> | Qwen3-based | `Qwen/Qwen3-Embedding-0.6B`, etc. | ✅︎ | ✅︎ |
| `VoyageQwen3BidirectionalEmbedModel`<sup>C</sup> | Voyage Qwen3-based with bidirectional attention | `voyageai/voyage-4-nano`, etc. | ✅︎ | ✅︎ |
| `RobertaModel`, `RobertaForMaskedLM` | RoBERTa-based | `sentence-transformers/all-roberta-large-v1`, etc. | | |
| `*Model`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | N/A | \* | \* |

<sup>C</sup> Automatically converted into an embedding model via `--convert embed`. ([details](./pooling_models.md#model-conversion))  
\* Feature support is the same as that of the original model.

!!! note
    `ssmits/Qwen2-7B-Instruct-embed-base` has an improperly defined Sentence Transformers config.
    You need to manually set mean pooling by passing `--pooler-config '{"pooling_type": "MEAN"}'`.

!!! note
    For `Alibaba-NLP/gte-Qwen2-*`, you need to enable `--trust-remote-code` for the correct tokenizer to be loaded.
    See [relevant issue on HF Transformers](https://github.com/huggingface/transformers/issues/34882).

!!! note
    `jinaai/jina-embeddings-v3` supports multiple tasks through LoRA, while vllm temporarily only supports text-matching tasks by merging LoRA weights.

!!! note
    The second-generation GTE model (mGTE-TRM) is named `NewModel`. The name `NewModel` is too generic, you should set `--hf-overrides '{"architectures": ["GteNewModel"]}'` to specify the use of the `GteNewModel` architecture.

If your model is not in the above list, we will try to automatically convert the model using
[as_embedding_model][vllm.model_executor.models.adapters.as_embedding_model]. By default, the embeddings
of the whole prompt are extracted from the normalized hidden state corresponding to the last token.

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
| `CLIPModel` | CLIP | T / I | `openai/clip-vit-base-patch32`, `openai/clip-vit-large-patch14`, etc. | | |
| `ColModernVBertForRetrieval` | ColModernVBERT | T / I | `ModernVBERT/colmodernvbert-merged` | | |
| `LlamaNemotronVLModel` | Llama Nemotron Embedding + SigLIP | T + I | `nvidia/llama-nemotron-embed-vl-1b-v2` | | |
| `LlavaNextForConditionalGeneration`<sup>C</sup> | LLaVA-NeXT-based | T / I | `royokong/e5-v` | | ✅︎ |
| `Phi3VForCausalLM`<sup>C</sup> | Phi-3-Vision-based | T + I | `TIGER-Lab/VLM2Vec-Full` | | ✅︎ |
| `Qwen3VLForConditionalGeneration`<sup>C</sup> | Qwen3-VL | T + I + V | `Qwen/Qwen3-VL-Embedding-2B`, etc. | ✅︎ | ✅︎ |
| `SiglipModel` | SigLIP, SigLIP2 | T / I | `google/siglip-base-patch16-224`, `google/siglip2-base-patch16-224` | | |
| `*ForConditionalGeneration`<sup>C</sup>, `*ForCausalLM`<sup>C</sup>, etc. | Generative models | \* | N/A | \* | \* |

<sup>C</sup> Automatically converted into an embedding model via `--convert embed`. ([details](./pooling_models.md#model-conversion))  
\* Feature support is the same as that of the original model.


## Supported Features

### Matryoshka Embeddings

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

## Offline Inference

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

