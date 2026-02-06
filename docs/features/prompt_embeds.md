# Prompt Embedding Inputs

This page teaches you how to pass prompt embedding inputs to vLLM.

## What are prompt embeddings?

The traditional flow of text data for a Large Language Model goes from text to token ids (via a tokenizer) then from token ids to prompt embeddings. For a traditional decoder-only model (such as meta-llama/Llama-3.1-8B-Instruct), this step of converting token ids to prompt embeddings happens via a look-up from a learned embedding matrix, but the model is not limited to processing only the embeddings corresponding to its token vocabulary.

## Offline Inference

To input multi-modal data, follow this schema in [vllm.inputs.EmbedsPrompt][]:

- `prompt_embeds`: A torch tensor representing a sequence of prompt/token embeddings. This has the shape (sequence_length, hidden_size), where sequence length is the number of tokens embeddings and hidden_size is the hidden size (embedding size) of the model.

### Hugging Face Transformers Inputs

You can pass prompt embeddings from Hugging Face Transformers models to the  `'prompt_embeds'` field of the prompt embedding dictionary, as shown in the following examples:

[examples/offline_inference/prompt_embed_inference.py](../../examples/offline_inference/prompt_embed_inference.py)

## Online Serving

Our OpenAI-compatible server accepts prompt embeddings inputs via the [Completions API](https://platform.openai.com/docs/api-reference/completions). Prompt embeddings inputs are added via a new `'prompt_embeds'` key in the JSON package and are enabled by the `--enable-prompt-embeds` flag in `vllm serve`.

When a mixture of `'prompt_embeds'` and `'prompt'` inputs are provided in a single request, the prompt embeds are always returned first.

Prompt embeddings are passed in as base64 encoded torch tensors.

!!! warning
    The vLLM engine may crash if incorrect shape of embeddings is passed.
    Only enable this flag for trusted users!

### Transformers Inputs via OpenAI Client

First, launch the OpenAI-compatible server:

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct --runner generate \
  --max-model-len 4096 --enable-prompt-embeds
```

Then, you can use the OpenAI client as follows:

[examples/online_serving/prompt_embed_inference_with_openai_client.py](../../examples/online_serving/prompt_embed_inference_with_openai_client.py)
