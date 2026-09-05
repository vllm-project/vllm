# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for JinaRankingIOProcessor request building and rendering."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm import PoolingParams, TextPrompt, TokensPrompt
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.scoring.io_processor import JinaRankingIOProcessor
from vllm.entrypoints.pooling.scoring.protocol import RerankRequest
from vllm.entrypoints.pooling.scoring.typing import ScoringData
from vllm.entrypoints.pooling.typing import EncodeCMPLRenderParams
from vllm.exceptions import VLLMValidationError
from vllm.inputs import tokens_input
from vllm.renderers import TokenizeParams

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


@pytest.mark.parametrize("truncation_side", ["left", "right"])
@pytest.mark.parametrize("truncate_prompt_tokens", [5, 6, None])
def test_render_rejects_truncation_only_when_ranking_markers_are_lost(
    truncation_side, truncate_prompt_tokens
):
    """Dropping a document/query marker must not produce mislabeled scores."""
    proc = JinaRankingIOProcessor.__new__(JinaRankingIOProcessor)
    proc.model_config = SimpleNamespace(is_encoder_decoder=False)
    markers = {"<|embed_token|>": 151670, "<|rerank_token|>": 151671}
    proc.tokenizer = MagicMock()
    proc.tokenizer.convert_tokens_to_ids.side_effect = markers.__getitem__
    token_ids = [7, 151670, 8, 151670, 9, 151671, 10]

    def render_cmpl(*, tok_params, **kwargs):
        prompt = TokensPrompt(prompt_token_ids=token_ids.copy())
        tok_params.apply_post_tokenization(None, prompt)
        return [tokens_input(prompt_token_ids=prompt["prompt_token_ids"])]

    proc.renderer = MagicMock()
    proc.renderer.render_cmpl.side_effect = render_cmpl
    prompt = proc.format_docs_prompts_func(
        query="query <|rerank_token|>",
        docs=["first <|embed_token|>", "second"],
    )
    render_params = EncodeCMPLRenderParams(
        prompts=TextPrompt(prompt=prompt),
        tok_params=TokenizeParams(
            max_total_tokens=100,
            truncate_prompt_tokens=truncate_prompt_tokens,
            truncation_side=truncation_side,
        ),
        prompt_extras=None,
        skip_mm_cache=False,
        params=PoolingParams(task="token_embed"),
        lora_requests=None,
        priorities=0,
    )

    if truncate_prompt_tokens == 5:
        with pytest.raises(VLLMValidationError, match="removed query or document"):
            proc.render(render_params)
    else:
        result = proc.render(render_params)
        expected = token_ids
        if truncate_prompt_tokens is not None:
            expected = token_ids[-6:] if truncation_side == "left" else token_ids[:6]
        assert result["prompts"]["prompt_token_ids"] == expected
