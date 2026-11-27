# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for JinaRankingIOProcessor request building."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import TokenizersBackend

from vllm import PoolingParams
from vllm.entrypoints.pooling.scoring.io_processor import JinaRankingIOProcessor
from vllm.entrypoints.pooling.scoring.types import ScoringData
from vllm.entrypoints.pooling.types import OfflineScoringInputsContext
from vllm.renderers import TokenizeParams

pytestmark = pytest.mark.skip_global_cleanup


def test_prompt_formatting_preserves_text_around_reserved_markers():
    clean_prompt = JinaRankingIOProcessor.format_docs_prompts_func(
        query="¿Qué cafés hay?",
        docs=["Café São Paulo", "喫茶店"],
        instruction="Prioriza locales céntricos.",
    )
    prompt = JinaRankingIOProcessor.format_docs_prompts_func(
        query="¿Qué <|rerank_token|>cafés hay?",
        docs=["Café <|embed_token|>São Paulo", "喫茶店<|rerank_token|>"],
        instruction="Prioriza locales <|embed_token|>céntricos.",
    )

    assert prompt == clean_prompt


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
    )
    return proc, ctx


def _make_jina_context(
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> OfflineScoringInputsContext:
    return OfflineScoringInputsContext(
        pooling_task="token_embed",
        scoring_data=ScoringData(
            data_1=["query alpha beta"],
            data_2=["document one two"],
        ),
        pooling_params=PoolingParams(extra_kwargs=extra_kwargs),
        chat_template_kwargs=chat_template_kwargs,
    )


def _call_single_prompt(
    proc: JinaRankingIOProcessor, ctx: OfflineScoringInputsContext
) -> str:
    factory, num_requests = proc.get_request_factory_offline(ctx)
    assert num_requests == 1
    request = cast(Any, next(iter(factory())))
    return cast(str, request["prompts"]["prompt"])


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
    base_extra = dict(ctx.pooling_params.extra_kwargs or {})
    base_extra.update(
        {
            name: limit
            for name, limit in (
                ("max_tokens_per_query", query_limit),
                ("max_tokens_per_doc", doc_limit),
            )
            if limit is not None
        }
    )
    ctx.pooling_params = PoolingParams(extra_kwargs=base_extra)

    factory, num_requests = proc.get_request_factory_offline(ctx)
    requests = list(factory())
    assert len(requests) == num_requests == n_queries
    for i, raw_request in enumerate(requests):
        request = cast(Any, raw_request)
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
    extra = dict(ctx.pooling_params.extra_kwargs or {})
    extra[name] = limit
    ctx.pooling_params = PoolingParams(extra_kwargs=extra)
    with pytest.raises(ValueError, match=name):
        proc.get_request_factory_offline(ctx)


# =============================================================================
# JinaRankingIOProcessor instruction resolution tests
# =============================================================================


class TestJinaRankingInstructionResolution:
    """Instruction resolution order for offline Jina ranking:
    `ctx.chat_template_kwargs["instruction"]`
    > `pooling_params.extra_kwargs["chat_template_kwargs"]["instruction"]`
    > `pooling_params.extra_kwargs["instruction"]`."""

    @pytest.mark.parametrize(
        ("ctx_kwargs", "extra_kwargs", "expected_instruction"),
        [
            # Direct `ctx.chat_template_kwargs` works when no nested/legacy key is set.
            (
                {"instruction": "From ctx kwargs"},
                None,
                "From ctx kwargs",
            ),
            # Nested `extra_kwargs["chat_template_kwargs"]` works when `ctx.chat_template_kwargs` is unset. # noqa: E501
            (
                None,
                {"chat_template_kwargs": {"instruction": "From nested extra_kwargs"}},
                "From nested extra_kwargs",
            ),
            # Legacy top-level `extra_kwargs["instruction"]` still works as fallback.
            (
                None,
                {"instruction": "From top-level extra_kwargs"},
                "From top-level extra_kwargs",
            ),
            # `ctx.chat_template_kwargs` beats both nested and top-level `extra_kwargs`.
            (
                {"instruction": "Winner ctx"},
                {
                    "chat_template_kwargs": {"instruction": "Loser nested"},
                    "instruction": "Loser top-level",
                },
                "Winner ctx",
            ),
            # Nested `extra_kwargs["chat_template_kwargs"]` beats top-level `extra_kwargs["instruction"]`. # noqa: E501
            (
                None,
                {
                    "chat_template_kwargs": {"instruction": "Winner nested"},
                    "instruction": "Loser top-level",
                },
                "Winner nested",
            ),
        ],
    )
    def test_instruction_sources_and_precedence(
        self,
        offline_processor_and_context,
        ctx_kwargs: dict[str, Any] | None,
        extra_kwargs: dict[str, Any] | None,
        expected_instruction: str,
    ):
        proc, _ = offline_processor_and_context
        ctx = _make_jina_context(
            chat_template_kwargs=ctx_kwargs,
            extra_kwargs=extra_kwargs,
        )

        prompt = _call_single_prompt(proc, ctx)

        assert f"<instruct>\n{expected_instruction}\n</instruct>" in prompt

    @pytest.mark.parametrize(
        ("ctx_kwargs", "extra_kwargs"),
        [
            (None, None),
            ({"instruction": None}, None),
            (None, {"chat_template_kwargs": {"instruction": None}}),
            (None, {"instruction": None}),
            ({"instruction": ""}, None),
        ],
    )
    def test_missing_or_empty_instruction_omits_instruct_block(
        self,
        offline_processor_and_context,
        ctx_kwargs: dict[str, Any] | None,
        extra_kwargs: dict[str, Any] | None,
    ):
        proc, _ = offline_processor_and_context
        ctx = _make_jina_context(
            chat_template_kwargs=ctx_kwargs,
            extra_kwargs=extra_kwargs,
        )

        prompt = _call_single_prompt(proc, ctx)

        assert "<instruct>" not in prompt
        assert "</instruct>" not in prompt

    def test_none_instruction_in_ctx_kwargs_falls_back_to_extra_kwargs(
        self, offline_processor_and_context
    ):
        proc, _ = offline_processor_and_context
        ctx = _make_jina_context(
            chat_template_kwargs={"instruction": None},
            extra_kwargs={"instruction": "Fallback instruction"},
        )

        prompt = _call_single_prompt(proc, ctx)

        assert "<instruct>\nFallback instruction\n</instruct>" in prompt
