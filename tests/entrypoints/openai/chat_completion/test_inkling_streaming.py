# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tests.utils import RemoteOpenAIServer
from vllm.tokenizers import get_tokenizer


def test_inkling_bare_text_streams_before_finish():
    # The tiny model's vocabulary covers Inkling's framing token IDs.
    model = "vllm-project/tiny-random-Gemma2ForCausalLM"
    tokenizer = "thinkingmachines/Inkling"
    (token_id,) = get_tokenizer(tokenizer).encode("x", add_special_tokens=False)
    args = [
        "--tokenizer",
        tokenizer,
        "--tokenizer-mode",
        "inkling",
        "--reasoning-parser",
        "inkling",
        "--tool-call-parser",
        "inkling",
        "--enable-auto-tool-choice",
        "--max-model-len",
        "128",
        "--enforce-eager",
    ]

    with RemoteOpenAIServer(model, args) as server, server.get_client() as client:
        for truncate_prompt_tokens in (None, -1):
            stream = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                reasoning_effort="none",
                stream=True,
                temperature=0,
                max_tokens=32,
                extra_body={
                    "allowed_token_ids": [token_id],
                    "truncate_prompt_tokens": truncate_prompt_tokens,
                },
            )
            pieces = [
                choice.delta.content
                for chunk in stream
                for choice in chunk.choices
                if choice.delta.content
            ]
            assert "".join(pieces) == "x" * 32
            if truncate_prompt_tokens is None:
                assert len(pieces) > 1, pieces
            else:
                assert pieces == ["x" * 32]
