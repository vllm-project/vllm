# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm import LLM, SamplingParams


def main():
    model_name = "NEMOTRON SUPER PLACEHOLDER"
    # model_name = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"

    # The prompt is designed so that:
    # - The FIRST chunk contains a critical fact (the secret number is 42)
    # - Filler text pads the middle to push total tokens past some threshold
    # - The LAST chunk asks about the secret number
    # If any chunk is misprocessed, the model won't know the answer or it won't know what to respond with.

    # In my experience, failed runs give totally meaningless answers.
    messages = [
        {
            "role": "user",
            "content": (
                "Important: The secret number is 42. "
                "The sky is green in this hypothetical world. "
                "Apples grow on trees in the forest. "
                "Rivers flow through the valleys and mountains. "
                "Birds sing songs in the early morning light. "
                "The weather today is sunny with clear skies ahead. "
                "Flowers bloom in the garden during spring season. "
                "Now answer with ONLY the number and nothing else: "
                "What is the secret number?"
            ),
        }
    ]

    small_message = [
        {
            "role": "user",
            "content": "The secret beta value is 64. What is the secret beta?",
        }
    ]

    # Set so large that both prefills will be classified as decodes in a mixed batch
    chunk_size = 256
    num_draft_tokens = 100

    # --- Tokenize to show token count before running inference ---
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    token_ids = tokenizer.encode(chat_text)
    print(f"Prompt token count: {len(token_ids)}")
    print(f"max_num_batched_tokens: {chunk_size}")
    print(f"MTP num_speculative_tokens (K): {num_draft_tokens}")
    print(
        f"reorder_batch_threshold will be: 1 + {num_draft_tokens} = {1 + num_draft_tokens}"
    )
    print(
        f"So any prefill chunk with <= {1 + num_draft_tokens} tokens will be misclassified as decode"
    )
    print(
        f"Number of chunks needed: ~{(len(token_ids) + chunk_size - 1) // chunk_size}"
    )
    print()

    first_k_tokens = tokenizer.decode(token_ids[:chunk_size])
    print(f"First {chunk_size} tokens: {first_k_tokens!r}")

    last_k_tokens = tokenizer.decode(token_ids[-chunk_size:])
    print(f"Last {chunk_size} tokens: {last_k_tokens!r}")

    # --- Create the LLM with MTP speculative decoding ---
    llm = LLM(
        model=model_name,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": num_draft_tokens,
        },
        max_num_batched_tokens=chunk_size,
        max_model_len=512,
        enforce_eager=True,  # simpler for debugging
        gpu_memory_utilization=0.90,
        tensor_parallel_size=2,
        trust_remote_code=True,
        attention_config={
            "backend": "flash_attn",
        },
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,  # short output for easy debugging
    )

    # First small message gets prefilled first, under normal conditions since the
    # batch is not yet mixed.

    # Then the second prefill arrives as a mixed batch, but is shorter than num_speculative_tokens,
    # so it gets misclassified as a decode and processed with the wrong state management logic,
    # causing the critical fact from the first chunk to be lost and the model to generate nonsense.
    outputs = llm.chat([small_message, messages], sampling_params, use_tqdm=True)

    for output in outputs:
        prompt_text = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt (last 80 chars): ...{prompt_text[-80:]}")
        print(f"Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
