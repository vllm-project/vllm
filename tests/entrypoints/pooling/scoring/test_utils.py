# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest

from vllm.entrypoints.pooling.scoring.typing import ScoreInput
from vllm.entrypoints.pooling.scoring.utils import (
    truncate_text_to_tokens,
    validate_score_input,
)
from vllm.exceptions import VLLMValidationError


@pytest.mark.parametrize(
    ("data_1", "data_2", "is_multimodal_model", "architecture", "message"),
    [
        pytest.param(
            {"content": []},
            "document",
            False,
            "TestModel",
            "MultiModalParam is not supported for TestModel",
            id="unsupported-multimodal-input",
        ),
        pytest.param(
            ["query 1", "query 2"],
            ["document"],
            False,
            "TestModel",
            "Input lengths must be either 1:1, 1:N or N:N",
            id="incompatible-input-lengths",
        ),
        pytest.param(
            [],
            ["document"],
            False,
            "TestModel",
            "At least one text element must be given",
            id="empty-first-input",
        ),
        pytest.param(
            ["query"],
            [],
            False,
            "TestModel",
            "At least one text_pair element must be given",
            id="empty-second-input",
        ),
    ],
)
def test_validate_score_input_rejects_invalid_inputs(
    data_1: ScoreInput | list[ScoreInput],
    data_2: ScoreInput | list[ScoreInput],
    is_multimodal_model: bool,
    architecture: str,
    message: str,
):
    with pytest.raises(VLLMValidationError) as exc_info:
        validate_score_input(
            data_1,
            data_2,
            is_multimodal_model=is_multimodal_model,
            architecture=architecture,
        )

    assert str(exc_info.value) == message
    assert exc_info.value.parameter is None
    assert exc_info.value.value is None


class _OvershootTokenizer:
    """Tokenizer stub whose token-N char slice retokenizes above the limit."""

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del add_special_tokens, kwargs
        # Full string: 4 tokens with overlapping spans (mirrors CJK/BPE cases).
        if text == "abcd":
            result: dict[str, Any] = {"input_ids": [1, 2, 3, 4]}
            if return_offsets_mapping:
                result["offset_mapping"] = [(0, 1), (1, 2), (2, 3), (2, 4)]
            return result
        # Naive slice text[:3] == "abc" would still look like 4 tokens.
        if text == "abc":
            return {"input_ids": [1, 2, 3, 4]}
        if text == "ab":
            return {"input_ids": [1, 2]}
        if text == "a":
            return {"input_ids": [1]}
        if text == "":
            return {"input_ids": []}
        # Prefixes of length 1..2 stay within a 3-token budget.
        n = min(len(text), 3)
        return {"input_ids": list(range(1, n + 1))}


def test_truncate_text_to_tokens_shrinks_when_char_slice_overshoots():
    tokenizer = _OvershootTokenizer()
    truncated = truncate_text_to_tokens("abcd", tokenizer, max_tokens=3)
    assert truncated == "ab"
    assert len(tokenizer(truncated, add_special_tokens=False)["input_ids"]) <= 3


def test_truncate_text_to_tokens_keeps_safe_char_slice():
    class _SafeTokenizer:
        def __call__(
            self,
            text: str,
            add_special_tokens: bool = True,
            return_offsets_mapping: bool = False,
            **kwargs: Any,
        ) -> dict[str, Any]:
            del add_special_tokens, kwargs
            ids = list(range(len(text.split())))
            result: dict[str, Any] = {"input_ids": ids}
            if return_offsets_mapping:
                offsets = []
                pos = 0
                for i, word in enumerate(text.split()):
                    if i:
                        pos += 1  # space
                    start = pos
                    pos = start + len(word)
                    offsets.append((start, pos))
                result["offset_mapping"] = offsets
            return result

    tokenizer = _SafeTokenizer()
    text = "one two three four"
    truncated = truncate_text_to_tokens(text, tokenizer, max_tokens=3)
    assert truncated == "one two three"
    assert len(tokenizer(truncated, add_special_tokens=False)["input_ids"]) == 3


@pytest.mark.parametrize(
    ("text", "max_tokens"),
    [
        ("中 文 测试", 3),
        ("😵\u200d💫x", 2),
        ("one two three four", 3),
    ],
)
def test_truncate_text_to_tokens_no_overshoot_real_tokenizer(
    text: str, max_tokens: int
):
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            local_files_only=True,
            trust_remote_code=True,
        )
    except OSError:
        pytest.skip("Qwen/Qwen2.5-0.5B tokenizer not cached")

    truncated = truncate_text_to_tokens(text, tokenizer, max_tokens)
    n_tokens = len(tokenizer(truncated, add_special_tokens=False)["input_ids"])
    assert n_tokens <= max_tokens
