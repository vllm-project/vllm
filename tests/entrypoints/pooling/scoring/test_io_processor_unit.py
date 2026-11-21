# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ScoringIOProcessor helpers."""

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    Mistral3Config,
    Qwen3Config,
    Qwen3VLConfig,
    TokenizersBackend,
)

from vllm import PoolingParams, TokensPrompt
from vllm.config import ModelConfig, VllmConfig
from vllm.connections import MediaRequestDownloadSizeExceededError
from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.entrypoints.pooling.base.protocol import ChatRequestOptionsMixin
from vllm.entrypoints.pooling.scoring.io_processor import (
    BiEncoderIOProcessor,
    CrossEncoderIOProcessor,
    JinaRankingIOProcessor,
    LateInteractionIOProcessor,
)
from vllm.entrypoints.pooling.scoring.types import ScoringData
from vllm.entrypoints.pooling.scoring.utils import compress_token_type_ids
from vllm.entrypoints.pooling.types import (
    EncodeCMPLRenderParams,
    OfflineScoringInputsContext,
    PoolingEngineInput,
    PoolingServeContext,
    ScoringRenderParams,
)
from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.jina_vl import JinaVLForSequenceClassification
from vllm.multimodal.utils import encode_image_url
from vllm.outputs import PoolingOutput, PoolingRequestOutput, RequestError
from vllm.renderers import ChatParams, TokenizeParams
from vllm.renderers.chat_utils import (
    AsyncMultiModalItemTracker,
    ChatTemplateConfig,
    MultiModalItemTracker,
)
from vllm.renderers.chat_utils.item_tracker import (
    MediaRequestBudgets,
    use_media_request_budgets,
)
from vllm.renderers.hf import HfRenderer
from vllm.tokenizers import TokenizerLike

pytestmark = pytest.mark.skip_global_cleanup


@dataclass
class _DummyTokenizer:
    truncation_side: str = "left"
    # Outside the range of the prompt ids below, so a test can tell a pad
    # token apart from a real one.
    pad_token_id: int = 99999
    pad_token_type_id: int = 0


@pytest.fixture
def scoring_tokenizer():
    vocab = {
        "[UNK]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "a": 3,
        "bc": 4,
        "##b": 5,
        "##c": 6,
    }
    backend = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )

    return TokenizersBackend(
        tokenizer_object=backend,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


@pytest.fixture
def llm_reranker_processor(tmp_path, scoring_tokenizer):
    Qwen3Config(
        architectures=["Qwen3ForSequenceClassification"], use_sep_token=False
    ).save_pretrained(tmp_path)
    scoring_tokenizer.save_pretrained(tmp_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            runner="pooling",
            max_model_len=64,
            enforce_eager=True,
        )
    )
    renderer = HfRenderer(config, TokenizersBackend.from_pretrained(tmp_path))
    try:
        yield CrossEncoderIOProcessor(config, renderer, ChatTemplateConfig())
    finally:
        renderer.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("budget_attr", "limit_env", "limit_setting", "limit_bytes", "error_type"),
    [
        ("decoded_budget", "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", "8", 8, ValueError),
        (
            "download_budget",
            "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB",
            "1",
            1 << 20,
            MediaRequestDownloadSizeExceededError,
        ),
    ],
)
async def test_pooling_workers_share_only_their_request_media_budget(
    llm_reranker_processor,
    monkeypatch,
    budget_attr,
    limit_env,
    limit_setting,
    limit_bytes,
    error_type,
):
    monkeypatch.setenv(limit_env, limit_setting)

    class AccountingProcessor(PoolingIOProcessor):
        def render(self, render_params):
            tracker = MultiModalItemTracker(self.model_config)
            getattr(tracker, budget_attr).charge(limit_bytes)
            return super().render(render_params)

    processor = AccountingProcessor(
        llm_reranker_processor.vllm_config,
        llm_reranker_processor.renderer,
        ChatTemplateConfig(),
    )
    params = EncodeCMPLRenderParams(
        prompts=TokensPrompt(prompt_token_ids=[3]),
        tok_params=TokenizeParams(max_total_tokens=64),
        prompt_extras=None,
        skip_mm_cache=False,
        params=PoolingParams(),
        lora_requests=None,
        priorities=0,
    )

    budget = MediaRequestBudgets()
    with use_media_request_budgets(budget):
        tracker = AsyncMultiModalItemTracker(processor.model_config)
        getattr(tracker, budget_attr).charge(1)

    rejected: PoolingEngineInput | BaseException
    accepted: PoolingEngineInput | BaseException
    rejected, accepted = await asyncio.gather(
        processor.render_async(params, budget),
        processor.render_async(params, MediaRequestBudgets()),
        return_exceptions=True,
    )
    assert isinstance(rejected, error_type)
    assert limit_env in str(rejected)
    assert not isinstance(accepted, BaseException)
    prompt = accepted["prompts"]
    assert prompt["type"] == "token"
    assert prompt["prompt_token_ids"] == [3]
    assert getattr(MultiModalItemTracker(processor.model_config), budget_attr).used == 0


def test_llm_reranker_tokenization_is_independent_of_nonbinding_doc_limit(
    llm_reranker_processor: CrossEncoderIOProcessor,
):
    tok_params = TokenizeParams(max_total_tokens=32).with_kwargs(
        add_special_tokens=True
    )
    chat_params = ChatParams()

    _, uncapped = llm_reranker_processor._get_score_prompt(
        "a", "bc", tok_params, chat_params
    )
    _, nonbinding = llm_reranker_processor._get_score_prompt(
        "a", "bc", tok_params, chat_params, max_tokens_per_doc=10
    )

    expected = [1, 3, 5, 6, 2]
    assert uncapped["prompt_token_ids"] == expected
    assert nonbinding["prompt_token_ids"] == expected


@pytest.mark.parametrize(
    "default_template,request_template,expected",
    [
        ("{{ messages[0].content }} {{ messages[1].content }}", None, "a bc"),
        (
            "{{ messages[0].content }} {{ messages[1].content }}",
            "{{ messages[1].content }} {{ messages[0].content }}",
            "bc a",
        ),
        (None, "{{ messages[1].content }} {{ messages[0].content }}", "bc a"),
        (None, None, "abc"),
    ],
)
def test_offline_scoring_uses_default_template_unless_overridden(
    llm_reranker_processor, default_template, request_template, expected
):
    processor = llm_reranker_processor
    processor.chat_template = default_template
    processor.renderer = SimpleNamespace(
        default_cmpl_tok_params=TokenizeParams(max_total_tokens=32),
        default_chat_tok_params=TokenizeParams(max_total_tokens=32),
    )
    ctx = OfflineScoringInputsContext(
        pooling_task="classify",
        scoring_data=ScoringData(data_1=["a"], data_2=["bc"]),
        chat_template=request_template,
        pooling_params=PoolingParams(),
    )
    factory, _ = processor.get_request_factory_offline(ctx)
    (request,) = factory()
    text, prompt = processor._get_score_prompt(
        request["data_1"],
        request["data_2"],
        request["tok_params"],
        request["chat_params"],
    )

    assert text == expected
    assert prompt["prompt_token_ids"] == processor.tokenizer.encode(expected)


@pytest.mark.parametrize(
    "truncate_prompt_tokens,max_total_tokens,truncation_side",
    [(4, 32, None), (-1, 8, None), (4, 32, "right")],
)
def test_truncation_keeps_jinavl_score_token(
    llm_reranker_processor, truncate_prompt_tokens, max_total_tokens, truncation_side
):
    """JinaVL scores the token it appends to the prompt, so binding
    truncation must keep it within the requested length. An explicit
    truncation side skips tokenizer truncation and cuts in
    apply_post_tokenization instead."""
    processor = llm_reranker_processor
    processor.supports_score_template = True
    processor.model = JinaVLForSequenceClassification
    processor.renderer = SimpleNamespace(
        process_for_engine=lambda prompt, _, **kw: prompt
    )

    engine_input = processor.render(
        ScoringRenderParams(
            data_1="a",
            data_2="bc bc bc bc",
            chat_params=ChatParams(chat_template_content_format="string"),
            max_tokens_per_query=0,
            max_tokens_per_doc=0,
            tok_params=TokenizeParams(
                max_total_tokens=max_total_tokens,
                truncate_prompt_tokens=truncate_prompt_tokens,
                truncation_side=truncation_side,
            ),
            prompt_extras=None,
            skip_mm_cache=False,
            params=PoolingParams(task="classify"),
            lora_requests=None,
            priorities=0,
        )
    )

    prompt_token_ids = engine_input["prompts"]["prompt_token_ids"]
    expected_len = (
        truncate_prompt_tokens if truncate_prompt_tokens > 0 else max_total_tokens
    )
    assert len(prompt_token_ids) == expected_len
    assert prompt_token_ids[-1] == 100


def test_token_type_ids_stay_aligned_with_a_truncated_padded_prompt():
    """The cross-encoder segment boundary must survive truncate + pad.

    `token_type_ids` are parallel to `prompt_token_ids` and are reduced to the
    document segment's start and end by `compress_token_type_ids`. If the two
    arrays are truncated and padded in different orders they no longer
    describe the same positions, and the model is told the query segment is
    empty.
    """
    tokenizer = cast(TokenizerLike, _DummyTokenizer())
    num_query, num_doc = 20, 30
    prompt = TokensPrompt(prompt_token_ids=list(range(num_query + num_doc)))
    token_type_ids = [0] * num_query + [1] * num_doc

    tok_params = TokenizeParams(
        max_total_tokens=100,
        pad_prompt_tokens=-1,
        truncate_prompt_tokens=40,
        truncation_side="left",
    )

    processed_prompt = tok_params.apply_post_tokenization(
        tokenizer,
        prompt,
    )
    assert "prompt_token_ids" in processed_prompt
    token_type_ids = CrossEncoderIOProcessor._apply_post_tokenization_to_token_type_ids(
        tokenizer, tok_params, token_type_ids
    )

    prompt_token_ids = processed_prompt["prompt_token_ids"]
    assert len(token_type_ids) == len(prompt_token_ids)

    # Keeping the last 40 tokens drops the first 10 query tokens, so 10 query
    # tokens survive and the document starts at index 10.
    doc_start, doc_end = compress_token_type_ids(token_type_ids)
    assert (doc_start, doc_end) == (10, 40)
    assert prompt_token_ids[:doc_start] == list(range(10, num_query))
    assert prompt_token_ids[doc_start:doc_end] == list(range(num_query, 50))
    assert token_type_ids[doc_end:] == [_DummyTokenizer.pad_token_type_id] * 60


def test_token_type_padding_uses_the_tokenizer_pad_type():
    tokenizer = _DummyTokenizer()
    tok_params = TokenizeParams(
        max_total_tokens=8,
        pad_prompt_tokens=8,
        truncate_prompt_tokens=None,
        truncation_side="right",
    )

    token_type_ids = CrossEncoderIOProcessor._apply_post_tokenization_to_token_type_ids(
        tokenizer, tok_params, [0, 0, 0, 1, 1]
    )

    assert token_type_ids == [0, 0, 0, 1, 1, 0, 0, 0]
    assert compress_token_type_ids(token_type_ids) == (3, 5)


@pytest.mark.parametrize("token_type_ids", [[0, 1, 0, 1], [0, 2]])
def test_compress_token_type_ids_rejects_invalid_sequences(token_type_ids):
    with pytest.raises(ValueError, match="optional trailing zeros"):
        compress_token_type_ids(token_type_ids)


def _pooling_output(
    request_id: str,
    data: torch.Tensor,
    error: RequestError | None = None,
) -> PoolingRequestOutput:
    return PoolingRequestOutput(
        request_id=request_id,
        outputs=PoolingOutput(data),
        prompt_token_ids=[1],
        num_cached_tokens=0,
        finished=True,
        error=error,
    )


@pytest.mark.parametrize(
    "processor_type", [BiEncoderIOProcessor, LateInteractionIOProcessor]
)
@pytest.mark.parametrize("failed_index", [0, 1])
def test_paired_scoring_preserves_request_error(processor_type, failed_index):
    processor = processor_type.__new__(processor_type)
    processor.pad_token_id = None
    error = RequestError(
        code="multimodal_cache_miss",
        message="Multi-modal processor cache miss.",
        retryable=True,
    )
    outputs = [
        _pooling_output("query", torch.tensor([1.0, 2.0])),
        _pooling_output("document", torch.tensor([3.0, 4.0])),
    ]
    outputs[failed_index] = _pooling_output(
        outputs[failed_index].request_id,
        torch.empty(0),
        error,
    )

    (result,) = processor._post_process(outputs, n_queries=1)

    assert result.request_id == "query_document"
    assert result.error is error
    assert result.outputs is outputs[failed_index].outputs


def test_jina_scoring_preserves_request_error_before_indexing_output():
    processor = JinaRankingIOProcessor.__new__(JinaRankingIOProcessor)
    error = RequestError(
        code="multimodal_cache_miss",
        message="Multi-modal processor cache miss.",
        retryable=True,
    )
    failed = _pooling_output("request", torch.empty(0), error)

    assert processor._post_process([failed], n_queries=1) == [failed]


_BLOCK_TEMPLATE = """
{%- for message in messages -%}
  {{- message.role + ':' -}}
  {%- for part in message.content -%}
    {%- if part.type == 'text' -%}
      {{- part.text | replace('[IMG]', '[literal IMG]') -}}
    {%- elif part.type == 'image' -%}
      {{- '[IMG]' -}}
    {%- endif -%}
  {%- endfor -%}
{%- endfor -%}
"""


@pytest.fixture
def multimodal_reranker_processor(tmp_path, request, scoring_tokenizer):
    model_type = getattr(request, "param", "mistral3")
    config_cls = Mistral3Config if model_type == "mistral3" else Qwen3VLConfig
    architecture = (
        f"{config_cls.__name__.removesuffix('Config')}ForConditionalGeneration"
    )
    config_cls(architectures=[architecture], use_sep_token=False).save_pretrained(
        tmp_path
    )
    scoring_tokenizer.chat_template = _BLOCK_TEMPLATE
    scoring_tokenizer.save_pretrained(tmp_path)
    for filename, data in {
        "config_sentence_transformers.json": {"model_type": "CrossEncoder"},
        "modules.json": [
            {"type": "sentence_transformers.models.Transformer", "path": ""}
        ],
        "sentence_bert_config.json": {
            "transformer_task": "sequence-classification",
            "modality_config": {"message": {"format": "flat"}},
        },
    }.items():
        (tmp_path / filename).write_text(json.dumps(data))
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            runner="pooling",
            convert="classify",
            max_model_len=64,
            enforce_eager=True,
        )
    )
    renderer = HfRenderer(config, TokenizersBackend.from_pretrained(tmp_path))
    try:
        yield CrossEncoderIOProcessor(
            config,
            renderer,
            ChatTemplateConfig(chat_template_content_format="openai"),
        )
    finally:
        renderer.shutdown()


def _multimodal_tok_params() -> TokenizeParams:
    return TokenizeParams(max_total_tokens=64)


@pytest.mark.parametrize(
    "multimodal_reranker_processor", ["mistral3", "qwen3_vl"], indirect=True
)
def test_scoring_template_keeps_images_distinct_from_literal_placeholder_text(
    multimodal_reranker_processor,
):
    """Image identity must survive models with and without text placeholders."""
    processor = multimodal_reranker_processor
    document = [
        {"type": "text", "text": "literal [IMG]"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_url(Image.new("RGB", (2, 2), "red"))},
            "uuid": "document-image",
        },
    ]
    chat_params = ChatParams(
        chat_template=_BLOCK_TEMPLATE,
        chat_template_content_format="openai",
    )
    prompt, inputs = processor._get_score_prompt(
        "a", document, _multimodal_tok_params(), chat_params
    )
    expected = "query:adocument:literal [literal IMG][IMG]"
    assert prompt == expected
    assert inputs["prompt_token_ids"] == processor.tokenizer.encode(expected)
    assert inputs["multi_modal_data"]["image"][0].getpixel((0, 0)) == (255, 0, 0)
    assert inputs["multi_modal_uuids"] == {"image": ["document-image"]}


@pytest.mark.parametrize(
    ("doc_limit", "expected_document"),
    [(1, "a[IMG]"), (3, "a bc[IMG]bc")],
)
def test_scoring_text_limits_share_budget_across_blocks_and_keep_images(
    multimodal_reranker_processor,
    doc_limit,
    expected_document,
):
    processor = multimodal_reranker_processor
    last_text = {"type": "text", "text": "bc a"}
    document = [
        {"type": "text", "text": "a bc"},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_url(Image.new("RGB", (2, 2)))},
        },
        last_text,
    ]
    chat_params = ChatParams(
        chat_template=_BLOCK_TEMPLATE,
        chat_template_content_format="openai",
    )
    prompt, inputs = processor._get_score_prompt(
        "a bc",
        document,
        _multimodal_tok_params(),
        chat_params,
        max_tokens_per_query=1,
        max_tokens_per_doc=doc_limit,
    )
    assert prompt == "query:adocument:" + expected_document
    assert len(inputs["multi_modal_data"]["image"]) == 1
    assert last_text["text"] == "bc a"


def test_scoring_text_limit_is_independent_of_adjacent_block_boundaries(
    multimodal_reranker_processor,
):
    # This tokenizer splits "abc" into three tokens, but "a" and "bc" into one each.
    chat_params = ChatParams(
        chat_template=_BLOCK_TEMPLATE,
        chat_template_content_format="openai",
    )
    for document in (
        "abc",
        [{"type": "text", "text": "a"}, {"type": "text", "text": "bc"}],
    ):
        prompt, _ = multimodal_reranker_processor._get_score_prompt(
            "a",
            document,
            _multimodal_tok_params(),
            chat_params,
            max_tokens_per_doc=2,
        )
        assert prompt == "query:adocument:ab"


@pytest.mark.parametrize("multimodal_reranker_processor", ["qwen3_vl"], indirect=True)
def test_string_scoring_templates_keep_model_placeholders(
    multimodal_reranker_processor,
):
    processor = multimodal_reranker_processor
    chat_params = ChatParams(
        chat_template="{{ messages[0].content + messages[1].content }}",
        chat_template_content_format="string",
    )
    prompt, _ = processor._get_score_prompt(
        "a",
        [
            {
                "type": "image_url",
                "image_url": {"url": encode_image_url(Image.new("RGB", (2, 2)))},
            }
        ],
        _multimodal_tok_params(),
        chat_params,
    )
    assert prompt == "a<|vision_start|><|image_pad|><|vision_end|>"


def test_scoring_detects_structured_content_with_saved_template(
    multimodal_reranker_processor,
):
    processor = multimodal_reranker_processor
    chat_params = processor._prepare_chat_params(
        ChatParams(chat_template=processor.saved_chat_template)
    )
    assert chat_params.chat_template_content_format == "openai"

    prompt, _ = processor._get_score_prompt(
        "a", "bc", _multimodal_tok_params(), chat_params
    )
    assert prompt == "query:adocument:bc"


# =============================================================================
# Saved Sentence Transformers chat template kwargs tests
# =============================================================================


class _DummyOnlineScoreRequest(ChatRequestOptionsMixin, BaseModel):
    model_config = ConfigDict(extra="allow")

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("model", "dummy-model")
        kwargs.setdefault(
            "messages",
            [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "bc"},
            ],
        )
        kwargs.setdefault("skip_mm_cache", False)
        super().__init__(**kwargs)

    def build_tok_params(
        self, model_config: Any, default_tok_params: TokenizeParams
    ) -> TokenizeParams:
        return default_tok_params


def _make_processor(
    base_processor: CrossEncoderIOProcessor,
    *,
    saved_template: str | None = "{{ messages[0].content }} {{ messages[1].content }}",
    saved_kwargs: dict[str, Any] | None = None,
    processor_template: str | None = None,
) -> CrossEncoderIOProcessor:
    base_processor.chat_template = processor_template
    base_processor.saved_chat_template = saved_template
    base_processor.saved_chat_template_kwargs = (
        dict(saved_kwargs) if saved_kwargs is not None else {}
    )
    base_processor.trust_request_chat_template = True
    base_processor.renderer = SimpleNamespace(
        default_cmpl_tok_params=TokenizeParams(max_total_tokens=32),
        default_chat_tok_params=TokenizeParams(max_total_tokens=32),
    )
    return base_processor


def _call_online_chat_params(
    processor: CrossEncoderIOProcessor,
    request: _DummyOnlineScoreRequest,
) -> ChatParams:
    serve_ctx = SimpleNamespace(
        request=request,
        pooling_params=PoolingParams(),
        lora_request=None,
        priorities=0,
        prompt_extras=None,
    )
    with patch.object(
        processor,
        "valid_inputs_online",
        return_value=ScoringData(data_1=["a"], data_2=["bc"]),
    ):
        render_params = processor.get_request_factory_online(
            cast(PoolingServeContext, serve_ctx)
        )
    assert len(render_params) == 1
    return cast(ScoringRenderParams, render_params[0])["chat_params"]


def _call_offline_chat_params(
    processor: CrossEncoderIOProcessor,
    ctx: OfflineScoringInputsContext,
) -> ChatParams:
    factory, count = processor.get_request_factory_offline(ctx)
    assert count == 1
    (render_params,) = factory()
    return cast(ScoringRenderParams, render_params)["chat_params"]


class TestSavedChatTemplateKwargsDefaults:
    """Saved ST kwargs override default False flags when no higher-precedence
    override is supplied, while unset flags stay False."""

    def test_offline_saved_true_overrides_default_false(
        self, llm_reranker_processor: CrossEncoderIOProcessor
    ):
        processor = _make_processor(
            llm_reranker_processor,
            saved_kwargs={"add_generation_prompt": True},
        )
        ctx = OfflineScoringInputsContext(
            pooling_task="classify",
            scoring_data=ScoringData(data_1=["a"], data_2=["bc"]),
            pooling_params=PoolingParams(),
        )

        chat_params = _call_offline_chat_params(processor, ctx)

        assert chat_params.chat_template_kwargs == {
            "add_generation_prompt": True,
            "continue_final_message": False,
        }

    def test_online_saved_true_overrides_default_false(
        self, llm_reranker_processor: CrossEncoderIOProcessor
    ):
        processor = _make_processor(
            llm_reranker_processor,
            saved_kwargs={"add_generation_prompt": True},
        )
        request = _DummyOnlineScoreRequest()

        chat_params = _call_online_chat_params(processor, request)

        assert chat_params.chat_template_kwargs == {
            "add_generation_prompt": True,
            "continue_final_message": False,
        }


class TestSavedChatTemplateKwargsPrecedence:
    """Precedence order: explicit flags > chat_template_kwargs > saved ST kwargs
    > default False flags."""

    @pytest.mark.parametrize(
        ("req_kwargs", "explicit_flag", "expected_flag"),
        [
            ({"add_generation_prompt": False}, None, False),
            ({"add_generation_prompt": True}, False, False),
            (None, None, True),
        ],
    )
    def test_offline_overrides_saved_kwargs(
        self,
        llm_reranker_processor: CrossEncoderIOProcessor,
        req_kwargs: dict[str, Any] | None,
        explicit_flag: bool | None,
        expected_flag: bool,
    ):
        processor = _make_processor(
            llm_reranker_processor,
            saved_kwargs={"add_generation_prompt": True, "saved_only": "kept"},
        )
        ctx = OfflineScoringInputsContext(
            pooling_task="classify",
            scoring_data=ScoringData(data_1=["a"], data_2=["bc"]),
            pooling_params=PoolingParams(),
            chat_template_kwargs=req_kwargs,
            add_generation_prompt=explicit_flag,
        )

        chat_params = _call_offline_chat_params(processor, ctx)

        assert (
            chat_params.chat_template_kwargs["add_generation_prompt"] is expected_flag
        )
        assert chat_params.chat_template_kwargs["saved_only"] == "kept"

    @pytest.mark.parametrize(
        ("req_kwargs", "explicit_flag", "expected_flag"),
        [
            ({"add_generation_prompt": False}, None, False),
            ({"add_generation_prompt": True}, False, False),
            (None, None, True),
        ],
    )
    def test_online_overrides_saved_kwargs(
        self,
        llm_reranker_processor: CrossEncoderIOProcessor,
        req_kwargs: dict[str, Any] | None,
        explicit_flag: bool | None,
        expected_flag: bool,
    ):
        processor = _make_processor(
            llm_reranker_processor,
            saved_kwargs={"add_generation_prompt": True, "saved_only": "kept"},
        )
        request = _DummyOnlineScoreRequest(
            chat_template_kwargs=req_kwargs,
            add_generation_prompt=explicit_flag,
        )

        chat_params = _call_online_chat_params(processor, request)

        assert (
            chat_params.chat_template_kwargs["add_generation_prompt"] is expected_flag
        )
        assert chat_params.chat_template_kwargs["saved_only"] == "kept"


class TestSavedChatTemplateReplacementAndValidation:
    """Replacement templates bypass saved ST kwargs; conflicting True flags
    raise an error."""

    def test_offline_replacement_template_skips_saved_kwargs(
        self, llm_reranker_processor: CrossEncoderIOProcessor
    ):
        processor = _make_processor(
            llm_reranker_processor,
            saved_kwargs={"add_generation_prompt": True},
        )
        ctx = OfflineScoringInputsContext(
            pooling_task="classify",
            scoring_data=ScoringData(data_1=["a"], data_2=["bc"]),
            pooling_params=PoolingParams(),
            chat_template="{{ messages[1].content }} {{ messages[0].content }}",
        )

        chat_params = _call_offline_chat_params(processor, ctx)

        assert chat_params.chat_template_kwargs == {
            "add_generation_prompt": False,
            "continue_final_message": False,
        }

    def test_online_replacement_template_skips_saved_kwargs(
        self, llm_reranker_processor: CrossEncoderIOProcessor
    ):
        processor = _make_processor(
            llm_reranker_processor,
            saved_kwargs={"add_generation_prompt": True},
        )
        request = _DummyOnlineScoreRequest(
            chat_template="{{ messages[1].content }} {{ messages[0].content }}"
        )

        chat_params = _call_online_chat_params(processor, request)

        assert chat_params.chat_template_kwargs == {
            "add_generation_prompt": False,
            "continue_final_message": False,
        }

    def test_conflicting_true_flags_raise_error(
        self, llm_reranker_processor: CrossEncoderIOProcessor
    ):
        processor = _make_processor(
            llm_reranker_processor,
            saved_kwargs={"add_generation_prompt": True},
        )
        ctx = OfflineScoringInputsContext(
            pooling_task="classify",
            scoring_data=ScoringData(data_1=["a"], data_2=["bc"]),
            pooling_params=PoolingParams(),
            continue_final_message=True,
        )

        with pytest.raises(VLLMValidationError):
            _call_offline_chat_params(processor, ctx)


# =============================================================================
# CrossEncoder tok_params selection tests (template vs completion vs chat input)
# =============================================================================


class TestCrossEncoderTokParamsSelection:
    """Plain text pairs with an active chat template should use
    `default_chat_tok_params` (avoiding double BOS on BOS-adding tokenizers),
    while template-less plain text pairs use `default_cmpl_tok_params`."""

    @staticmethod
    def _configure_distinct_tok_params(
        processor: CrossEncoderIOProcessor,
    ) -> tuple[TokenizeParams, TokenizeParams]:
        cmpl_tok = TokenizeParams(max_total_tokens=32).with_kwargs(
            add_special_tokens=True
        )
        chat_tok = TokenizeParams(max_total_tokens=32).with_kwargs(
            add_special_tokens=False
        )
        processor.renderer = SimpleNamespace(
            default_cmpl_tok_params=cmpl_tok,
            default_chat_tok_params=chat_tok,
        )
        return cmpl_tok, chat_tok

    @pytest.mark.parametrize(
        ("data_1", "data_2", "chat_template", "expect_chat_tok"),
        [
            # Plain text + active chat template -> chat_tok_params
            (
                ["a"],
                ["bc", "a"],
                "{{ messages[0].content }} {{ messages[1].content }}",
                True,
            ),
            # Plain text + no chat template -> cmpl_tok_params
            (["a"], ["bc", "a"], None, False),
            # Chat messages list + no explicit template -> chat_tok_params
            (
                [[{"role": "user", "content": "a"}]],
                [
                    [{"role": "user", "content": "bc"}],
                    [{"role": "user", "content": "a"}],
                ],
                None,
                True,
            ),
            # Chat messages list + active template -> chat_tok_params
            (
                [[{"role": "user", "content": "a"}]],
                [
                    [{"role": "user", "content": "bc"}],
                    [{"role": "user", "content": "a"}],
                ],
                "{{ messages[0].content }} {{ messages[1].content }}",
                True,
            ),
        ],
    )
    def test_offline_and_online_tok_params_selection(
        self,
        llm_reranker_processor: CrossEncoderIOProcessor,
        data_1: Any,
        data_2: Any,
        chat_template: str | None,
        expect_chat_tok: bool,
    ):
        processor = llm_reranker_processor
        processor.chat_template = None
        processor.saved_chat_template = None
        processor.saved_chat_template_kwargs = {}
        processor.trust_request_chat_template = True
        cmpl_tok, chat_tok = self._configure_distinct_tok_params(processor)
        expected_tok = chat_tok if expect_chat_tok else cmpl_tok

        # 1) Offline path
        offline_ctx = OfflineScoringInputsContext(
            pooling_task="classify",
            scoring_data=ScoringData(data_1=data_1, data_2=data_2),
            pooling_params=PoolingParams(),
            chat_template=chat_template,
        )
        factory, count = processor.get_request_factory_offline(offline_ctx)
        assert count == len(data_2)
        offline_items = [cast(ScoringRenderParams, p) for p in factory()]
        assert len(offline_items) == len(data_2)
        assert all(item["tok_params"] is expected_tok for item in offline_items)

        # 2) Online path
        request = _DummyOnlineScoreRequest(chat_template=chat_template)
        serve_ctx = SimpleNamespace(
            request=request,
            pooling_params=PoolingParams(),
            lora_request=None,
            priorities=0,
            prompt_extras=None,
        )
        with patch.object(
            processor,
            "valid_inputs_online",
            return_value=ScoringData(data_1=data_1, data_2=data_2),
        ):
            online_items = [
                cast(ScoringRenderParams, p)
                for p in processor.get_request_factory_online(
                    cast(PoolingServeContext, serve_ctx)
                )
            ]
        assert len(online_items) == len(data_2)
        assert all(item["tok_params"] is expected_tok for item in online_items)
