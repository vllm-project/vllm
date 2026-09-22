# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for JinaRankingIOProcessor request building."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from vllm import PoolingParams
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.scoring.io_processor import JinaRankingIOProcessor
from vllm.entrypoints.pooling.scoring.protocol import RerankRequest
from vllm.entrypoints.pooling.scoring.typing import ScoringData
from vllm.entrypoints.pooling.typing import OfflineScoringInputsContext
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


@pytest.fixture
def offline_processor_and_context():
    backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    proc = JinaRankingIOProcessor.__new__(JinaRankingIOProcessor)
    proc.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    proc.model_config = SimpleNamespace(max_model_len=1024, is_encoder_decoder=False)
    proc.renderer = SimpleNamespace(
        default_cmpl_tok_params=TokenizeParams(max_total_tokens=1024)
    )
    ctx = OfflineScoringInputsContext(
        pooling_task="token_embed",
        scoring_data=ScoringData(
            data_1=["query alpha beta", "query gamma delta"],
            data_2=["document one two", "document three four"],
        ),
        pooling_params=PoolingParams(
            extra_kwargs={
                "chat_template_kwargs": {"instruction": "Keep this instruction"}
            }
        ),
        tokenization_kwargs=None,
        chat_template=None,
        lora_request=None,
        priorities=None,
    )
    return proc, ctx


@pytest.mark.parametrize(
    ("query_limit", "doc_limit", "n_queries"),
    [(None, None, 1), (0, 0, 2), (1, None, 1), (None, 2, 1), (1, 2, 1), (1, 2, 2)],
    ids=["unset", "zero", "query", "doc", "both-1-to-n", "both-n-to-n"],
)
def test_offline_truncates_text_before_formatting(
    offline_processor_and_context, query_limit, doc_limit, n_queries
):
    proc, ctx = offline_processor_and_context
    queries = ctx.scoring_data.data_1[:n_queries]
    docs = ctx.scoring_data.data_2
    ctx.scoring_data = ScoringData(data_1=queries, data_2=docs)
    ctx.pooling_params.extra_kwargs.update(
        {
            name: limit
            for name, limit in (
                ("max_tokens_per_query", query_limit),
                ("max_tokens_per_doc", doc_limit),
            )
            if limit is not None
        }
    )

    factory, num_requests = proc.get_request_factory_offline(ctx)
    requests = list(factory())
    assert len(requests) == num_requests == n_queries
    for i, request in enumerate(requests):
        prompt = request["prompts"]["prompt"]
        query = " ".join(queries[i].split()[: query_limit or None])
        assert f"<query>\n{query}<|rerank_token|>\n</query>" in prompt
        prompt_docs = docs if n_queries == 1 else [docs[i]]
        for j, doc in enumerate(prompt_docs):
            doc = " ".join(doc.split()[: doc_limit or None])
            assert f'<passage id="{j}">\n{doc}<|embed_token|>\n</passage>' in prompt
        assert prompt.count("<|embed_token|>") == len(prompt_docs)
        assert prompt.count("<|rerank_token|>") == 1
        assert "<instruct>\nKeep this instruction\n</instruct>" in prompt


@pytest.mark.parametrize("name", ["max_tokens_per_query", "max_tokens_per_doc"])
@pytest.mark.parametrize("limit", [-1, 1024])
def test_offline_rejects_invalid_token_limits(
    offline_processor_and_context, name, limit
):
    proc, ctx = offline_processor_and_context
    ctx.pooling_params.extra_kwargs[name] = limit
    with pytest.raises(ValueError, match=name):
        proc.get_request_factory_offline(ctx)
