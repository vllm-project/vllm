# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bidirectional attention spans with prefix caching.

Serves a prompt whose state is split into fixed-size bidirectional spans, then
the same state grown by one step, and prints how many prompt tokens of each
were served from the prefix cache: the grown prompt reuses every span that ends
before the new tokens.

A real deployment uses a checkpoint trained with spans, whose config declares
`bidirectional_span_layer_types` and `bidirectional_span_max_length`. This
script declares them on a stock Gemma 4 through `hf_overrides` to show the
mechanics, so its answers are not meaningful.

Run:
python examples/features/bidirectional_spans/bidirectional_spans_offline.py
"""

from vllm import LLM, SamplingParams, TokensPrompt

SPAN_LENGTH = 256


def chunked(num_tokens: int, size: int) -> list[tuple[int, int]]:
    return [(s, min(s + size, num_tokens)) for s in range(0, num_tokens, size)]


def main():
    llm = LLM(
        model="google/gemma-4-E2B-it",
        hf_overrides={
            "text_config": {
                "bidirectional_span_layer_types": ["full_attention"],
                "bidirectional_span_max_length": SPAN_LENGTH,
            }
        },
        limit_mm_per_prompt={"image": 0, "audio": 0},
        max_model_len=4096,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    question = tokenizer.encode(
        "\n\nQuestion: How many files did the agent read?\nAnswer:",
        add_special_tokens=False,
    )
    state = " ".join(
        f"Step {i}: the agent read file_{i}.txt and found {7 * i} rows."
        for i in range(60)
    )
    params = SamplingParams(max_tokens=8, temperature=0.0)
    for text in (state, state + " Step 60: the agent wrote a summary."):
        state_ids = tokenizer.encode(text)
        prompt = TokensPrompt(
            prompt_token_ids=state_ids + question,
            bidirectional_spans=chunked(len(state_ids), SPAN_LENGTH),
        )
        output = llm.generate(prompt, params)[0]
        print(
            f"{len(state_ids)} state tokens, {output.num_cached_tokens} prompt "
            f"tokens from cache: {output.outputs[0].text!r}"
        )


if __name__ == "__main__":
    main()
