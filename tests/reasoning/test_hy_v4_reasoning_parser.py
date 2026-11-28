# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.reasoning.hy_v4_reasoning_parser import HYV4ReasoningExtractor

# The extractor works on plain data, so a synthetic vocab is enough; no
# checkpoint download or trust_remote_code is needed.
VOCAB = {"<think>": 1, "</think>": 2, "abc": 3, "def": 4}


def stream(
    extractor: HYV4ReasoningExtractor, deltas: list[tuple[str, list[int]]]
) -> tuple[str, str]:
    """Feed (delta_text, delta_token_ids) pairs and join what comes back."""
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    previous_text = ""
    previous_ids: list[int] = []
    for delta_text, delta_ids in deltas:
        current_text = previous_text + delta_text
        current_ids = previous_ids + delta_ids
        delta = extractor.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_ids,
            current_ids,
            delta_ids,
        )
        if delta is not None:
            if delta["reasoning"]:
                reasoning_parts.append(delta["reasoning"])
            if delta["content"]:
                content_parts.append(delta["content"])
        previous_text, previous_ids = current_text, current_ids
    return "".join(reasoning_parts), "".join(content_parts)


# A streaming delta can carry <think> together with following tokens (for
# example under speculative decoding). The marker must be stripped the same
# way the non-streaming path strips it.
MULTI_TOKEN_DELTAS = [
    pytest.param(
        [("<think>abc", [1, 3]), ("</think>", [2]), ("def", [4])],
        id="start_marker_with_reasoning",
    ),
    pytest.param(
        [("<think>abc</think>def", [1, 3, 2, 4])],
        id="start_and_end_marker_in_one_delta",
    ),
    pytest.param(
        # Start token id present, but its text was skipped by detokenization.
        [("abc</think>def", [1, 3, 2, 4])],
        id="start_marker_id_without_text",
    ),
]


@pytest.mark.parametrize("deltas", MULTI_TOKEN_DELTAS)
def test_streaming_strips_start_marker_from_multi_token_delta(
    deltas: list[tuple[str, list[int]]],
):
    extractor = HYV4ReasoningExtractor(VOCAB, token_suffix="", thinking=True)

    full_text = "".join(text for text, _ in deltas)
    assert extractor.extract_reasoning(full_text) == ("abc", "def")

    assert stream(extractor, deltas) == ("abc", "def")
