# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for JinaRankingIOProcessor online request building."""

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.scoring.io_processor import JinaRankingIOProcessor
from vllm.entrypoints.pooling.scoring.protocol import RerankRequest
from vllm.entrypoints.pooling.scoring.typing import ScoringData

pytestmark = pytest.mark.skip_global_cleanup


def test_online_forwards_truncate_prompt_tokens_to_proxy(monkeypatch):
    """The proxy request handed to the base factory must carry
    truncate_prompt_tokens/truncation_side from the real request.

    JinaRankingIOProcessor swaps ctx.request for a proxy
    PoolingCompletionRequest before delegating to the base factory, which
    reads truncation off ctx.request. Dropping the fields on the proxy
    silently disables truncate_prompt_tokens for Jina rerank/score.
    """
    proc = JinaRankingIOProcessor.__new__(JinaRankingIOProcessor)
    proc.valid_inputs_online = MagicMock(
        return_value=ScoringData(data_1=["query"], data_2=["doc"])
    )
    proc._get_token_limits = MagicMock(return_value=(0, 0))
    proc.ensure_str = MagicMock(side_effect=lambda data: list(data))
    proc.format_docs_prompts_func = MagicMock(return_value="formatted prompt")

    captured: dict[str, object] = {}

    def _spy_base(self, ctx):
        captured["truncate_prompt_tokens"] = ctx.request.truncate_prompt_tokens
        captured["truncation_side"] = ctx.request.truncation_side
        return []

    monkeypatch.setattr(PoolingIOProcessor, "get_request_factory_online", _spy_base)

    request = RerankRequest(
        model="m",
        query="query",
        documents=["doc"],
        truncate_prompt_tokens=512,
        truncation_side="left",
    )
    ctx = MagicMock()
    ctx.request = request
    ctx.prompt_extras = None

    proc.get_request_factory_online(ctx)

    assert captured["truncate_prompt_tokens"] == 512
    assert captured["truncation_side"] == "left"
    # The real request is restored after delegating.
    assert ctx.request is request
