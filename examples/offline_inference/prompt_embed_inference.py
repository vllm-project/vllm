# SPDX-License-Identifier: Apache-2.0
"""
Demonstrates how to generate prompt embeddings using
Hugging Face Transformers  and use them as input to vLLM
for both single and batch inference.

Model: meta-llama/Llama-3.2-1B-Instruct
Note: This model is gated on Hugging Face Hub.
      You must request access to use it:
      https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

Requirements:
- vLLM
- transformers

Run:
    python examples/offline_inference/prompt_embed_inference.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from vllm import LLM


def init_tokenizer_and_llm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformers_model = AutoModelForCausalLM.from_pretrained(model_name)
    embedding_layer = transformers_model.get_input_embeddings()
    llm = LLM(model=model_name, enable_prompt_embeds=True)
    return tokenizer, embedding_layer, llm


def get_prompt_embeds(
    chat: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    embedding_layer: torch.nn.Module,
):
    token_ids = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt"
    )
    prompt_embeds = embedding_layer(token_ids).squeeze(0)
    return prompt_embeds


def single_prompt_inference(
    llm: LLM, tokenizer: PreTrainedTokenizer, embedding_layer: torch.nn.Module
):
    chat = [{"role": "user", "content": "Please tell me about the capital of France."}]
    prompt_embeds = get_prompt_embeds(chat, tokenizer, embedding_layer)

    outputs = llm.generate(
        {
            "prompt_embeds": prompt_embeds,
        }
    )

    print("\n[Single Inference Output]")
    print("-" * 30)
    for o in outputs:
        print(o.outputs[0].text)
    print("-" * 30)


def batch_prompt_inference(
    llm: LLM, tokenizer: PreTrainedTokenizer, embedding_layer: torch.nn.Module
):
    chats = [
        [{"role": "user", "content": "Please tell me about the capital of France."}],
        [{"role": "user", "content": "When is the day longest during the year?"}],
        [{"role": "user", "content": "Where is bigger, the moon or the sun?"}],
    ]

    prompt_embeds_list = [
        get_prompt_embeds(chat, tokenizer, embedding_layer) for chat in chats
    ]

    outputs = llm.generate([{"prompt_embeds": embeds} for embeds in prompt_embeds_list])

    print("\n[Batch Inference Outputs]")
    print("-" * 30)
    for i, o in enumerate(outputs):
        print(f"Q{i + 1}: {chats[i][0]['content']}")
        print(f"A{i + 1}: {o.outputs[0].text}\n")
    print("-" * 30)


def main():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer, embedding_layer, llm = init_tokenizer_and_llm(model_name)
    single_prompt_inference(llm, tokenizer, embedding_layer)
    batch_prompt_inference(llm, tokenizer, embedding_layer)


if __name__ == "__main__":
    main()
