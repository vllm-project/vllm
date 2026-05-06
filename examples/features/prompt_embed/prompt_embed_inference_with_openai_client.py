# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM OpenAI-Compatible Client with Prompt Embeddings.

This script demonstrates how to:
1. Generate prompt embeddings using Hugging Face Transformers.
2. Encode them in base64 format.
3. Send them to a vLLM server for inference via both:
    - OpenAI-compatible Chat Completions API
    - OpenAI-compatible Completions API

Important distinction between the two APIs:

- Chat Completions API: `prompt_embeds` content parts should encode ONLY
  the user-provided content, not a templated conversation. The server
  renders the surrounding chat template around the embedded content at
  request time, the same way it would for a plain text `content` string.
  Embedding a full templated conversation here would double-apply the
  template and likely produce undesirable results.

- Completions API: the server does NOT apply a chat template to
  `prompt_embeds`. The caller is responsible for producing embeddings for
  the full, already-templated prompt (i.e. apply the chat template first, 
  then embed the resulting token IDs). Anything the model would normally
  need (system prompt, role markers, generation prompt, etc.) must already
  be baked into the embedded tokens.

Run the vLLM server first:
vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --runner generate \
  --max-model-len 4096 \
  --enable-prompt-embeds

Run the client:
python examples/features/prompt_embed/prompt_embed_inference_with_openai_client.py

Model: meta-llama/Llama-3.2-1B-Instruct
Note: This model is gated on Hugging Face Hub.
      You must request access to use it:
      https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

Dependencies:
- transformers
- torch
- openai
"""

import transformers
from openai import OpenAI

from vllm.utils.serial_utils import tensor2base64


def run_chat_completion_prompt_embeds(
    client: OpenAI,
    model_name: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    embedding_layer,
    messages: list[dict],
) -> None:
    """Run a Chat Completions API request using prompt_embeds content parts.

    This example embeds ONLY the user-provided content of the final user turn, the
    vLLM server applies the chat template around it at request time.
    """
    user_content = messages[-1]["content"]
    content_token_ids = tokenizer(
        user_content, return_tensors="pt", add_special_tokens=False
    ).input_ids
    content_prompt_embeds = embedding_layer(content_token_ids).squeeze(0)
    encoded_embeds = tensor2base64(content_prompt_embeds)

    api_messages = [
        *messages[:-1],
        {
            "role": messages[-1]["role"],
            "content": [{"type": "prompt_embeds", "data": encoded_embeds}],
        },
    ]

    chat_completion = client.chat.completions.create(
        model=model_name,
        max_tokens=6,
        temperature=0.0,
        messages=api_messages,
    )

    print("-" * 30)
    print("Chat Completions API")
    print(chat_completion.choices[0].message.content)
    print("-" * 30)


def run_completion_prompt_embeds(
    client: OpenAI,
    model_name: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    embedding_layer,
    messages: list[dict],
) -> None:
    """Run a Completions API request using prompt embeddings.

    The Completions endpoint does not apply a chat template,
    so the caller must apply it and embed the full templated prompt.
    """
    templated_token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).input_ids
    templated_prompt_embeds = embedding_layer(templated_token_ids).squeeze(0)
    encoded_embeds = tensor2base64(templated_prompt_embeds)

    completion = client.completions.create(
        model=model_name,
        prompt=None,
        max_tokens=6,
        temperature=0.0,
        # NOTE: The OpenAI client allows passing in extra JSON body via the
        # `extra_body` argument.
        extra_body={"prompt_embeds": encoded_embeds},
    )

    print("-" * 30)
    print("Completions API")
    print(completion.choices[0].text)
    print("-" * 30)


def main() -> None:
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )

    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    transformers_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    embedding_layer = transformers_model.get_input_embeddings()

    messages = [
        {"role": "user", "content": "Please tell me about the capital of France."}
    ]

    # Chat Completions API: embed ONLY the user content. The server wraps
    # the embedding in the chat template when it renders the messages.
    run_chat_completion_prompt_embeds(
        client, model_name, tokenizer, embedding_layer, messages
    )

    # Completions API: embed the FULL templated prompt. The caller must
    # apply the chat template up-front.
    run_completion_prompt_embeds(
        client, model_name, tokenizer, embedding_layer, messages
    )


if __name__ == "__main__":
    main()
