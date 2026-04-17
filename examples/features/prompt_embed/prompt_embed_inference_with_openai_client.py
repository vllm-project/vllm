# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM OpenAI-Compatible Client with Prompt Embeddings.

This script demonstrates how to:
1. Generate prompt embeddings using Hugging Face Transformers.
2. Encode them in base64 format.
3. Send them to a vLLM server for inference via both:
    - OpenAI-compatible Completions API
    - OpenAI-compatible Chat Completions API

Important distinction between the two APIs:

- Completions API: the server does NOT apply a chat template to
  `prompt_embeds`. The caller is responsible for producing embeddings for
  the full, already-templated prompt (i.e. apply the chat template on the
  token IDs first, then embed those IDs). Anything the model would normally
  need (system prompt, role markers, generation prompt, etc.) must already
  be baked into the embedded tokens.

- Chat Completions API: `prompt_embeds` content parts should encode ONLY
  the user-provided content, not a templated conversation. The server
  renders the surrounding chat template around the embedded content at
  request time, the same way it would for a plain text `content` string.
  Embedding a full templated conversation here would double-apply the
  template and likely produce undesirable results.

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


def run_completion_prompt_embeds(
    client: OpenAI,
    model_name: str,
    encoded_embeds: str,
) -> None:
    """Run a Completions API request using prompt embeddings."""
    completion = client.completions.create(
        model=model_name,
        prompt=None,
        max_tokens=5,
        temperature=0.0,
        # NOTE: The OpenAI client allows passing in extra JSON body via the
        # `extra_body` argument.
        extra_body={"prompt_embeds": encoded_embeds},
    )

    print("-" * 30)
    print("Completions API")
    print(completion.choices[0].text)
    print("-" * 30)


def run_chat_completion_prompt_embeds(
    client: OpenAI,
    model_name: str,
    encoded_embeds: str,
) -> None:
    """Run a Chat Completions API request using prompt_embeds content parts.

    `encoded_embeds` here must encode only the user content — the server
    applies the chat template around it at request time.
    """
    chat_completion = client.chat.completions.create(
        model=model_name,
        max_tokens=5,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "prompt_embeds", "data": encoded_embeds},
                    {"type": "text", "text": "Continue:"},
                ],
            }
        ],
    )

    print("-" * 30)
    print("Chat Completions API")
    print(chat_completion.choices[0].message.content)
    print("-" * 30)


def main() -> None:
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )

    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    # Transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    transformers_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    embedding_layer = transformers_model.get_input_embeddings()

    user_content = "Please tell me about the capital of France."

    # Completions API: embed the FULL templated prompt.
    # The Completions endpoint does not apply a chat template to
    # prompt_embeds, so system/role/generation-prompt tokens must already be
    # baked in on the client side.
    # Refer to the HuggingFace repo for the correct format to use.
    chat = [{"role": "user", "content": user_content}]
    templated_token_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).input_ids
    completion_prompt_embeds = embedding_layer(templated_token_ids).squeeze(0)
    completion_encoded_embeds = tensor2base64(completion_prompt_embeds)

    # Chat Completions API: embed ONLY the user content string.
    # The server wraps these embeddings in the chat template when it renders
    # the messages, so the client must not pre-apply the template here.
    content_token_ids = tokenizer(user_content, return_tensors="pt").input_ids
    chat_content_prompt_embeds = embedding_layer(content_token_ids).squeeze(0)
    chat_content_encoded_embeds = tensor2base64(chat_content_prompt_embeds)

    run_completion_prompt_embeds(client, model_name, completion_encoded_embeds)
    run_chat_completion_prompt_embeds(client, model_name, chat_content_encoded_embeds)


if __name__ == "__main__":
    main()
