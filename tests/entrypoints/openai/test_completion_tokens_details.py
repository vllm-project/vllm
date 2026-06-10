# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for completion_tokens_details (reasoning tokens) in UsageInfo."""

from vllm.entrypoints.openai.engine.protocol import (
    CompletionTokenUsageInfo,
    PromptTokenUsageInfo,
    UsageInfo,
)


class TestCompletionTokenUsageInfo:
    def test_default(self):
        info = CompletionTokenUsageInfo()
        assert info.reasoning_tokens == 0

    def test_with_reasoning_tokens(self):
        info = CompletionTokenUsageInfo(reasoning_tokens=42)
        assert info.reasoning_tokens == 42

    def test_serialization(self):
        info = CompletionTokenUsageInfo(reasoning_tokens=7)
        d = info.model_dump()
        assert d["reasoning_tokens"] == 7


class TestUsageInfoWithCompletionTokensDetails:
    def test_completion_tokens_details_none_by_default(self):
        u = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        assert u.completion_tokens_details is None
        assert u.prompt_tokens_details is None

    def test_with_completion_tokens_details(self):
        u = UsageInfo(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=CompletionTokenUsageInfo(reasoning_tokens=2),
        )
        assert u.completion_tokens_details is not None
        assert u.completion_tokens_details.reasoning_tokens == 2

    def test_with_both_details(self):
        u = UsageInfo(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=3),
            completion_tokens_details=CompletionTokenUsageInfo(reasoning_tokens=2),
        )
        d = u.model_dump()
        assert d["prompt_tokens_details"]["cached_tokens"] == 3
        assert d["completion_tokens_details"]["reasoning_tokens"] == 2

    def test_json_excludes_none_details(self):
        u = UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        j = u.model_dump_json(exclude_none=True)
        assert "completion_tokens_details" not in j
        assert "prompt_tokens_details" not in j

    def test_json_roundtrip_with_details(self):
        u = UsageInfo(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=CompletionTokenUsageInfo(reasoning_tokens=4),
        )
        j = u.model_dump_json()
        u2 = UsageInfo.model_validate_json(j)
        assert u2.completion_tokens_details is not None
        assert u2.completion_tokens_details.reasoning_tokens == 4
