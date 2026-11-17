# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for JinaRankingIOProcessor request building."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import TokenizersBackend

from vllm import PoolingParams
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.scoring.io_processor import JinaRankingIOProcessor
from vllm.entrypoints.pooling.scoring.types import ScoringData
from vllm.entrypoints.pooling.types import OfflineScoringInputsContext
from vllm.exceptions import VLLMValidationError
from vllm.renderers import TokenizeParams

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def offline_processor_and_context():
    backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    proc = JinaRankingIOProcessor.__new__(JinaRankingIOProcessor)
    proc.tokenizer = TokenizersBackend(tokenizer_object=backend)
    proc.model_config = SimpleNamespace(
        max_model_len=1024, is_encoder_decoder=False, multimodal_config=None
    )
    proc.renderer = SimpleNamespace(
        default_cmpl_tok_params=TokenizeParams(max_total_tokens=1024),
        tokenizer=proc.tokenizer,
    )
    proc.chat_template = None
    proc.template_kwargs = None
    proc.tool_dicts = None
    cast(Any, proc).chat_template_content_format = "auto"
    ctx = OfflineScoringInputsContext(
        pooling_task="token_embed",
        scoring_data=ScoringData(
            data_1=["query alpha beta", "query gamma delta"],
            data_2=["document one two", "document three four"],
        ),
        pooling_params=PoolingParams(extra_kwargs={}),
        tokenization_kwargs=None,
        chat_template=None,
        chat_template_kwargs={"instruction": "Keep this instruction"},
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
def capture_encode_ctx(monkeypatch):
    """Capture the `OfflineEncodeInputsContext` handed to the base processor."""
    captured: dict[str, Any] = {}
    original = PoolingIOProcessor.get_request_factory_offline

    def wrapper(self, ctx):
        captured["chat_template_kwargs"] = dict(ctx.chat_template_kwargs or {})
        return original(self, ctx)

    monkeypatch.setattr(PoolingIOProcessor, "get_request_factory_offline", wrapper)
    return captured


def test_offline_explicit_kwargs_win_over_top_level_flag(
    offline_processor_and_context, capture_encode_ctx
):
    """A top-level `True` must not overwrite an explicit `False` in kwargs."""
    proc, ctx = offline_processor_and_context
    ctx.chat_template_kwargs = {"add_generation_prompt": False}
    ctx.add_generation_prompt = True

    proc.get_request_factory_offline(ctx)

    assert capture_encode_ctx["chat_template_kwargs"]["add_generation_prompt"] is False


def test_offline_top_level_flag_fills_missing_kwarg(
    offline_processor_and_context, capture_encode_ctx
):
    """A top-level `True` fills in the key when kwargs omit it."""
    proc, ctx = offline_processor_and_context
    ctx.chat_template_kwargs = {"instruction": "Keep this instruction"}
    ctx.add_generation_prompt = True

    proc.get_request_factory_offline(ctx)

    merged = capture_encode_ctx["chat_template_kwargs"]
    assert merged["add_generation_prompt"] is True
    assert merged["instruction"] == "Keep this instruction"


def test_offline_rejects_mutually_exclusive_flags_across_layers(
    offline_processor_and_context,
):
    """A top-level flag must conflict with the same flag set in kwargs.

    `OfflineInputsContext.__post_init__` only checks the two top-level
    fields, so the cross-layer conflict has to be caught by the merge step.
    """
    proc, ctx = offline_processor_and_context
    ctx.chat_template_kwargs = {"continue_final_message": True}
    ctx.add_generation_prompt = True

    with pytest.raises(VLLMValidationError, match="Cannot set both"):
        proc.get_request_factory_offline(ctx)
