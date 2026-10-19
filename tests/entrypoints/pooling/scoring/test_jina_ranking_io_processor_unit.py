# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for JinaRankingIOProcessor request building and rendering."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import TokenizersBackend

from vllm import PoolingParams
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.embed.io_processor import JinaRankingTokenEmbedIOProcessor
from vllm.entrypoints.pooling.pooling.protocol import PoolingCompletionRequest
from vllm.entrypoints.pooling.scoring.io_processor import JinaRankingIOProcessor
from vllm.entrypoints.pooling.scoring.protocol import (
    RerankRequest,
    ScoreQueriesDocumentsRequest,
)
from vllm.entrypoints.pooling.scoring.typing import ScoringData
from vllm.entrypoints.pooling.typing import (
    OfflineEncodeInputsContext,
    OfflineScoringInputsContext,
    PoolingServeContext,
)
from vllm.exceptions import VLLMValidationError
from vllm.renderers import TokenizeParams
from vllm.renderers.chat_utils import ChatTemplateConfig
from vllm.renderers.hf import HfRenderer
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.tokenizers.hf import get_cached_tokenizer

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
    monkeypatch.setattr(
        proc,
        "valid_inputs_online",
        MagicMock(return_value=ScoringData(data_1=["query"], data_2=["doc"])),
    )
    monkeypatch.setattr(proc, "_get_token_limits", MagicMock(return_value=(0, 0)))
    monkeypatch.setattr(
        proc, "ensure_str", MagicMock(side_effect=lambda data: list(data))
    )
    monkeypatch.setattr(
        proc, "format_docs_prompts_func", MagicMock(return_value="formatted prompt")
    )

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
    proc.tokenizer = TokenizersBackend(tokenizer_object=backend)
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


@pytest.fixture
def ranking_renderer():
    backend = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "<|embed_token|>": 1, "<|rerank_token|>": 2},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = TokenizersBackend(tokenizer_object=backend)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|embed_token|>", "<|rerank_token|>"]}
    )
    model_config = SimpleNamespace(
        architecture="JinaForRanking",
        max_model_len=1024,
        is_encoder_decoder=False,
        is_multimodal_model=False,
        supports_multimodal_inputs=False,
        encoder_config={},
        hf_config=SimpleNamespace(),
        renderer_num_workers=1,
    )
    config = SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(_api_process_rank=0),
    )
    renderer = HfRenderer(config, get_cached_tokenizer(tokenizer))
    yield renderer
    renderer.shutdown()


@pytest.mark.parametrize("api", ["score", "rerank", "pooling", "encode"])
@pytest.mark.parametrize("truncation_side", ["left", "right"])
@pytest.mark.parametrize("truncation", ["lost", "preserved", None])
def test_rejects_truncation_only_when_ranking_markers_are_lost(
    ranking_renderer, api, truncation_side, truncation
):
    """Every Jina entrypoint must retain all query/document vectors."""
    renderer = ranking_renderer
    processor_cls = (
        JinaRankingIOProcessor
        if api in ("score", "rerank")
        else JinaRankingTokenEmbedIOProcessor
    )
    proc = processor_cls(
        renderer.config, renderer, ChatTemplateConfig(chat_template=None)
    )
    query = "query <|rerank_token|>"
    docs = ["first <|embed_token|>", "second"]
    prompt = proc.format_docs_prompts_func(query=query, docs=docs)
    token_ids = renderer.get_tokenizer().encode(prompt)
    if truncation is None:
        limit = None
    elif truncation_side == "left":
        limit = len(token_ids) - token_ids.index(1) - (truncation == "lost")
    else:
        limit = token_ids.index(2) + (truncation == "preserved")

    if api == "encode":
        ctx = OfflineEncodeInputsContext(
            pooling_task="token_embed",
            prompts=docs + [query],
            pooling_params=PoolingParams(task="token_embed"),
            tokenization_kwargs={
                "truncate_prompt_tokens": limit,
                "truncation_side": truncation_side,
            },
            lora_request=None,
            priorities=None,
        )
        factory, _ = proc.get_request_factory_offline(ctx)
        (render_params,) = list(factory())
    else:
        if api == "score":
            request = ScoreQueriesDocumentsRequest(queries=query, documents=docs)
        elif api == "rerank":
            request = RerankRequest(query=query, documents=docs)
        else:
            request = PoolingCompletionRequest(input=docs + [query])
        request.truncate_prompt_tokens = limit
        request.truncation_side = truncation_side
        ctx = PoolingServeContext(
            request=request,
            model_name="jina",
            request_id="test",
            pooling_params=PoolingParams(task="token_embed"),
            lora_request=None,
            priorities=None,
            prompt_extras=None,
        )
        (render_params,) = proc.get_request_factory_online(ctx)

    if truncation == "lost":
        with pytest.raises(
            VLLMValidationError, match="removed query or document"
        ) as exc_info:
            proc.render(render_params)
        assert exc_info.value.parameter == "truncate_prompt_tokens"
    else:
        result = proc.render(render_params)
        expected = token_ids
        if limit is not None:
            expected = (
                token_ids[-limit:] if truncation_side == "left" else token_ids[:limit]
            )
        prompt_token_ids = extract_prompt_components(
            proc.model_config, result["prompts"]
        ).token_ids
        assert prompt_token_ids == expected
