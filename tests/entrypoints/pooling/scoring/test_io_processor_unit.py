# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ScoringIOProcessor helpers."""

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    Mistral3Config,
    Qwen3Config,
    Qwen3VLConfig,
    TokenizersBackend,
)

from vllm import PoolingParams, TokensPrompt
from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.pooling.scoring.io_processor import (
    BiEncoderIOProcessor,
    CrossEncoderIOProcessor,
    JinaRankingIOProcessor,
    LateInteractionIOProcessor,
    _apply_post_tokenization_to_token_type_ids,
)
from vllm.entrypoints.pooling.scoring.types import ScoringData
from vllm.entrypoints.pooling.scoring.utils import compress_token_type_ids
from vllm.entrypoints.pooling.types import (
    OfflineScoringInputsContext,
    ScoringRenderParams,
)
from vllm.model_executor.models.jina_vl import JinaVLForSequenceClassification
from vllm.multimodal.utils import encode_image_url
from vllm.outputs import PoolingOutput, PoolingRequestOutput, RequestError
from vllm.renderers import TokenizeParams
from vllm.renderers.chat_utils import ChatTemplateConfig
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


def test_llm_reranker_tokenization_is_independent_of_nonbinding_doc_limit(
    llm_reranker_processor: CrossEncoderIOProcessor,
):
    encode_kwargs = {"add_special_tokens": True}

    _, uncapped = llm_reranker_processor.get_score_prompt("a", "bc", encode_kwargs)
    _, nonbinding = llm_reranker_processor.get_score_prompt(
        "a", "bc", encode_kwargs, max_tokens_per_doc=10
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
        default_cmpl_tok_params=TokenizeParams(max_total_tokens=32)
    )
    ctx = OfflineScoringInputsContext(
        pooling_task="classify",
        tokenization_kwargs=None,
        lora_request=None,
        priorities=None,
        scoring_data=ScoringData(data_1=["a"], data_2=["bc"]),
        chat_template=request_template,
        pooling_params=PoolingParams(),
    )
    factory, _ = processor.get_request_factory_offline(ctx)
    (request,) = factory()
    text, prompt = processor.get_score_prompt(
        request["data_1"],
        request["data_2"],
        request["tok_params"].get_encode_kwargs(),
        chat_template=request["chat_template"],
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
    processor.renderer = SimpleNamespace(process_for_engine=lambda prompt, _: prompt)

    engine_input = processor.render(
        ScoringRenderParams(
            data_1="a",
            data_2="bc bc bc bc",
            chat_template=None,
            chat_template_content_format="auto",
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
    token_type_ids = _apply_post_tokenization_to_token_type_ids(
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

    token_type_ids = _apply_post_tokenization_to_token_type_ids(
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
        yield CrossEncoderIOProcessor(config, renderer, ChatTemplateConfig())
    finally:
        renderer.shutdown()


@pytest.mark.parametrize(
    ("text", "with_image", "multimodal_reranker_processor"),
    [
        pytest.param(None, True, "mistral3", id="image-only"),
        pytest.param("literal [IMG]", True, "mistral3", id="mixed-no-placeholder"),
        pytest.param("literal [IMG]", True, "qwen3_vl", id="mixed-placeholder"),
        pytest.param("literal [IMG]", False, "mistral3", id="literal-text"),
        pytest.param(None, False, "mistral3", id="empty-content"),
    ],
    indirect=["multimodal_reranker_processor"],
)
def test_scoring_template_preserves_image_blocks_and_literal_text(
    multimodal_reranker_processor, text, with_image
):
    """Image identity must not depend on model placeholders or literal user text."""
    processor = multimodal_reranker_processor
    document = [] if text is None else [{"type": "text", "text": text}]
    if with_image:
        document.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image_url(Image.new("RGB", (2, 2), "red"))},
                "uuid": "document-image",
            }
        )
    prompt, inputs = processor.get_score_prompt(
        "a", document, {}, chat_template=_BLOCK_TEMPLATE
    )
    expected = "query:adocument:" + (text or "").replace("[IMG]", "[literal IMG]")
    expected += "[IMG]" if with_image else ""
    assert prompt == expected
    assert inputs["prompt_token_ids"] == processor.tokenizer.encode(expected)
    if with_image:
        assert inputs["multi_modal_data"]["image"][0].getpixel((0, 0)) == (255, 0, 0)
        assert inputs["multi_modal_uuids"] == {"image": ["document-image"]}
    else:
        assert "multi_modal_data" not in inputs


@pytest.mark.parametrize(
    ("doc_limit", "expected_document"),
    [(1, "a[IMG]"), (3, "a bc[IMG]bc"), (10, "a bc[IMG]bc a")],
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
    prompt, inputs = processor.get_score_prompt(
        "a bc",
        document,
        {},
        chat_template=_BLOCK_TEMPLATE,
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
    for document in (
        "abc",
        [{"type": "text", "text": "abc"}],
        [{"type": "text", "text": "a"}, {"type": "text", "text": "bc"}],
    ):
        prompt, _ = multimodal_reranker_processor.get_score_prompt(
            "a", document, {}, chat_template=_BLOCK_TEMPLATE, max_tokens_per_doc=2
        )
        assert prompt == "query:adocument:ab"


@pytest.mark.parametrize("multimodal_reranker_processor", ["qwen3_vl"], indirect=True)
@pytest.mark.parametrize("content_format", ["auto", "string"])
def test_string_scoring_templates_keep_model_placeholders(
    multimodal_reranker_processor, content_format
):
    processor = multimodal_reranker_processor
    prompt, _ = processor.get_score_prompt(
        "a",
        [
            {
                "type": "image_url",
                "image_url": {"url": encode_image_url(Image.new("RGB", (2, 2)))},
            }
        ],
        {},
        chat_template="{{ messages[0].content + messages[1].content }}",
        chat_template_content_format=content_format,
    )
    assert prompt == "a<|vision_start|><|image_pad|><|vision_end|>"


@pytest.mark.parametrize("content_format", ["auto", "openai"])
@pytest.mark.parametrize("saved_template", [False, True])
def test_scoring_template_formats_plain_strings_as_text_blocks(
    multimodal_reranker_processor, content_format, saved_template
):
    processor = multimodal_reranker_processor
    prompt, _ = processor.get_score_prompt(
        "a",
        "bc",
        {},
        chat_template=None if saved_template else _BLOCK_TEMPLATE,
        chat_template_content_format=content_format,
    )
    assert prompt == "query:adocument:bc"


def test_scoring_content_format_override_for_indirect_template(
    multimodal_reranker_processor,
):
    # The detector cannot infer structured content from a filter alone.
    prompt, _ = multimodal_reranker_processor.get_score_prompt(
        "a",
        "bc",
        {},
        chat_template="{{ messages[1].content | tojson }}",
        chat_template_content_format="openai",
    )
    assert json.loads(prompt) == [{"type": "text", "text": "bc"}]
