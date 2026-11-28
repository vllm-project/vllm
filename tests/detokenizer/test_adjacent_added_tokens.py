# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generated text must match the full decode of the generated token ids when
spaces between special tokens are turned off."""

from vllm import SamplingParams

MODEL = "google/gemma-3-1b-it"


def test_adjacent_non_special_added_tokens(vllm_runner):
    # Gemma stores runs of spaces and newlines as non-special added tokens, so
    # indented code puts several of them next to each other.
    prompt = "Write a Python class with two methods. Only code, no explanation.\n"
    params = SamplingParams(
        temperature=0,
        max_tokens=128,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )
    with vllm_runner(MODEL, max_model_len=512, enforce_eager=True) as vllm_model:
        llm = vllm_model.llm
        output = llm.generate([prompt], params)[0].outputs[0]
        tokenizer = llm.get_tokenizer()

    token_ids = list(output.token_ids)
    if output.finish_reason == "stop":
        # The stop token is in token_ids but not in the text.
        token_ids = token_ids[:-1]
    added = {
        tid for tid, tok in tokenizer.added_tokens_decoder.items() if not tok.special
    }
    assert any(a in added and b in added for a, b in zip(token_ids, token_ids[1:]))
    assert output.text == tokenizer.decode(token_ids, skip_special_tokens=False)
