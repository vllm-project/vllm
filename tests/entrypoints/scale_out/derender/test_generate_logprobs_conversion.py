# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the generate -> OpenAI logprob conversion in derender.

These exercise `_resolve_logprobs` directly with a stub tokenizer, so they
cover the byte-fallback correction path without needing a model.
"""

from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateLogProb,
    GenerateLogProbs,
    GenerateLogProbsContent,
)
from vllm.renderers.online_derenderer import _resolve_logprobs

# A two-byte character split across two byte-fallback tokens: neither decodes
# to anything printable on its own, but the pair decodes to "é".
FIRST_BYTE = 10
SECOND_BYTE = 11
PLAIN = 12

VOCAB = {FIRST_BYTE: "<0xC3>", SECOND_BYTE: "<0xA9>", PLAIN: "ok"}


class _StubTokenizer:
    """Decodes byte-fallback tokens only when both halves are present."""

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str | None]:
        return [VOCAB.get(i) for i in ids]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self.decode([k for t in tokens for k, v in VOCAB.items() if v == t])

    def decode(self, ids: list[int], **kwargs) -> str:
        out = ""
        i = 0
        while i < len(ids):
            if ids[i] == FIRST_BYTE and i + 1 < len(ids) and ids[i + 1] == SECOND_BYTE:
                out += "é"
                i += 2
            elif ids[i] in (FIRST_BYTE, SECOND_BYTE):
                out += "�"
                i += 1
            else:
                out += VOCAB.get(ids[i], "")
                i += 1
        return out


def test_token_ids_are_decoded_with_bytes_and_order_preserved():
    logprobs = GenerateLogProbs(
        content=[
            GenerateLogProbsContent(
                token_id=PLAIN,
                logprob=-0.5,
                rank=1,
                top_logprobs=[
                    GenerateLogProb(token_id=PLAIN, logprob=-0.5, rank=1),
                    GenerateLogProb(token_id=FIRST_BYTE, logprob=-2.0, rank=2),
                ],
            )
        ]
    )

    resolved = _resolve_logprobs(logprobs, _StubTokenizer())

    assert resolved.content is not None
    entry = resolved.content[0]
    assert entry.token == "ok"
    assert entry.logprob == -0.5
    assert entry.bytes == list(b"ok")
    # Rank order survives the conversion (the OpenAI shape has no rank field).
    assert [t.logprob for t in entry.top_logprobs] == [-0.5, -2.0]
    assert entry.top_logprobs[0].token == "ok"


def test_byte_fallback_second_half_uses_preceding_token_as_context():
    """U+FFFD correction needs the preceding sampled ids, which the integer
    shape carries directly (no placeholder parsing)."""
    logprobs = GenerateLogProbs(
        content=[
            GenerateLogProbsContent(token_id=FIRST_BYTE, logprob=-0.1),
            GenerateLogProbsContent(token_id=SECOND_BYTE, logprob=-0.2),
        ]
    )

    resolved = _resolve_logprobs(logprobs, _StubTokenizer())

    assert resolved.content is not None
    # The first half has no preceding context to repair it with, so it stays
    # empty (existing `_correct_decoded_token` behaviour). The second resolves
    # to the complete character once the first is used as context.
    assert resolved.content[0].token == ""
    assert resolved.content[0].bytes == []
    assert resolved.content[1].token == "é"
    assert resolved.content[1].bytes == list("é".encode())


def test_content_none_is_not_an_error():
    resolved = _resolve_logprobs(GenerateLogProbs(), _StubTokenizer())
    assert resolved.content is None
