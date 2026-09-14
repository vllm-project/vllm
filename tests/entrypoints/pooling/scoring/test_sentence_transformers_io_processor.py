# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from transformers import BertConfig, BertModel, BertTokenizer

from vllm import PoolingParams
from vllm.entrypoints.pooling.scoring import utils as scoring_utils
from vllm.entrypoints.pooling.scoring.io_processor import (
    CrossEncoderIOProcessor,
    _validate_sentence_transformers_tokenizer,
)
from vllm.entrypoints.pooling.scoring.protocol import ScoreQueriesDocumentsRequest
from vllm.entrypoints.pooling.scoring.typing import ScoringData
from vllm.entrypoints.pooling.typing import (
    OfflineScoringInputsContext,
    PoolingServeContext,
)
from vllm.exceptions import VLLMValidationError
from vllm.renderers import TokenizeParams

_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['role'] }}:"
    "{% for item in message['content'] %}"
    "{% if item['type'] == 'image' %}[IMG]"
    "{% elif item['type'] == 'text' %}{{ item['text'] }}{% endif %}"
    "{% endfor %};{% endfor %}"
)


class _MultiModalParser:
    def __init__(self, tracker):
        self.model_config = tracker.model_config
        self.tracker = tracker

    def parse_image(self, image_url, uuid=None):
        self.tracker.images.append(image_url)


class _MultiModalTracker:
    def __init__(self, model_config):
        self.model_config = model_config
        self.images = []

    def create_parser(self, **_kwargs):
        return _MultiModalParser(self)

    def resolve_items(self):
        mm_data = {"image": self.images} if self.images else None
        return mm_data, None


@pytest.fixture
def tokenizer():
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "query": 5,
        "document": 6,
        "extra": 7,
        "words": 8,
        ":": 9,
        ";": 10,
        "0": 11,
        "1": 12,
        "2": 13,
        "3": 14,
        "4": 15,
    }
    return BertTokenizer(vocab=vocab, do_lower_case=False)


@pytest.fixture
def processor(tokenizer):
    processor = CrossEncoderIOProcessor.__new__(CrossEncoderIOProcessor)
    processor.model_config = SimpleNamespace(
        enable_prompt_embeds=False, max_model_len=16, encoder_config={}
    )
    processor.tokenizer = tokenizer
    processor.supports_score_template = False
    processor.model = None
    processor.use_sep_token = True
    processor.sentence_transformers_config = SimpleNamespace(uses_message_format=True)
    processor.renderer = SimpleNamespace(
        default_cmpl_tok_params=TokenizeParams(max_total_tokens=16),
        process_for_engine=lambda prompt, arrival_time: prompt,
    )
    processor.is_multimodal_model = False
    processor.architecture = "BertModel"
    return processor


@pytest.fixture
def st_transformer(tmp_path, tokenizer):
    pytest.importorskip("sentence_transformers", minversion="5.7.0")
    from sentence_transformers.sentence_transformer.modules import Transformer

    BertModel(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=16,
        )
    ).save_pretrained(tmp_path)
    tokenizer.model_max_length = 16
    tokenizer.chat_template = _CHAT_TEMPLATE + "[SEP]"
    tokenizer.save_pretrained(tmp_path)
    return Transformer(
        str(tmp_path),
        modality_config={
            "message": {
                "method": "forward",
                "method_output_name": "last_hidden_state",
                "format": "structured",
            },
        },
        module_output_name="token_embeddings",
    )


def _render_pair(processor, pair, tokenization_kwargs=None, chat_template_kwargs=None):
    context = OfflineScoringInputsContext(
        pooling_task="classify",
        scoring_data=ScoringData(data_1=[pair[0]], data_2=[pair[1]]),
        pooling_params=PoolingParams(
            extra_kwargs={"chat_template_kwargs": chat_template_kwargs}
        ),
        tokenization_kwargs=tokenization_kwargs,
        lora_request=None,
        chat_template=processor.tokenizer.chat_template,
        priorities=None,
    )
    factory, count = processor.get_request_factory_offline(context)
    assert count == 1
    return processor.render(next(factory()))


@pytest.mark.parametrize(
    ("data_2", "expected_document_content"),
    [
        (
            [{"type": "image_url", "image_url": {"url": "image.png"}}],
            [{"type": "image", "image": "image.png"}],
        ),
        (
            [
                {"type": "image_url", "image_url": {"url": "image.png"}},
                {"type": "text", "text": "document"},
            ],
            [
                {"type": "image", "image": "image.png"},
                {"type": "text", "text": "document"},
            ],
        ),
    ],
)
def test_structured_cross_encoder_matches_saved_pair_template(
    monkeypatch,
    processor,
    tokenizer,
    data_2,
    expected_document_content,
):
    monkeypatch.setattr(
        scoring_utils,
        "MultiModalItemTracker",
        _MultiModalTracker,
    )
    expected_messages = [
        {
            "role": "query",
            "content": [{"type": "text", "text": "query"}],
        },
        {
            "role": "document",
            "content": expected_document_content,
        },
    ]

    full_prompt, engine_prompt = processor.get_score_prompt(
        data_1="query",
        data_2=data_2,
        encode_kwargs={"add_special_tokens": True},
        chat_template=_CHAT_TEMPLATE,
    )

    expected_prompt = tokenizer.apply_chat_template(
        expected_messages,
        chat_template=_CHAT_TEMPLATE,
        tokenize=False,
    )
    expected_token_ids = tokenizer.apply_chat_template(
        expected_messages,
        chat_template=_CHAT_TEMPLATE,
        tokenize=True,
        return_dict=False,
    )
    assert full_prompt == expected_prompt
    assert "[IMG]" in full_prompt
    assert engine_prompt["prompt_token_ids"] == expected_token_ids
    assert (
        engine_prompt["prompt_token_ids"]
        != tokenizer(full_prompt, add_special_tokens=True)["input_ids"]
    )
    assert engine_prompt["multi_modal_data"] == {"image": ["image.png"]}


def test_structured_cross_encoder_truncates_text_parts(
    monkeypatch,
    processor,
):
    monkeypatch.setattr(
        scoring_utils,
        "MultiModalItemTracker",
        _MultiModalTracker,
    )

    full_prompt, _ = processor.get_score_prompt(
        data_1="query extra",
        data_2=[
            {"type": "image_url", "image_url": {"url": "image.png"}},
            {"type": "text", "text": "document extra words"},
        ],
        encode_kwargs={},
        chat_template=_CHAT_TEMPLATE,
        max_tokens_per_query=1,
        max_tokens_per_doc=1,
    )

    assert full_prompt == "query:query;document:[IMG]document;"


def test_text_pair_uses_saved_message_template(
    monkeypatch,
    processor,
    tokenizer,
):
    monkeypatch.setattr(
        scoring_utils,
        "MultiModalItemTracker",
        _MultiModalTracker,
    )
    expected_messages = [
        {"role": "query", "content": [{"type": "text", "text": "query"}]},
        {
            "role": "document",
            "content": [{"type": "text", "text": "document"}],
        },
    ]

    full_prompt, engine_prompt = processor.get_score_prompt(
        data_1="query",
        data_2="document",
        encode_kwargs={"add_special_tokens": True},
        chat_template=_CHAT_TEMPLATE,
    )

    expected_token_ids = tokenizer.apply_chat_template(
        expected_messages,
        chat_template=_CHAT_TEMPLATE,
        tokenize=True,
        return_dict=False,
    )
    assert full_prompt == "query:query;document:document;"
    assert engine_prompt["prompt_token_ids"] == expected_token_ids


def test_effective_left_padding_is_rejected():
    config = SimpleNamespace(pooler_config={"seq_pooling_type": "CLS"})

    with pytest.raises(ValueError, match="CLS pooling.*left-padded"):
        _validate_sentence_transformers_tokenizer(
            SimpleNamespace(padding_side="left"),
            config,
        )


def test_explicit_template_preserves_special_token_setting(processor, tokenizer):
    processor.sentence_transformers_config = None
    template = "{{ messages[0]['content'] }} {{ messages[1]['content'] }}"
    full_prompt, engine_prompt = processor.get_score_prompt(
        data_1="query",
        data_2="document",
        encode_kwargs={"add_special_tokens": True},
        chat_template=template,
    )

    expected = tokenizer(full_prompt, add_special_tokens=True)
    assert engine_prompt["prompt_token_ids"] == expected["input_ids"]


@pytest.mark.parametrize("document_tokens", [3, 4, 5])
@pytest.mark.parametrize("truncation_side", ["left", "right"])
def test_structured_truncation_matches_sentence_transformers(
    monkeypatch, processor, st_transformer, document_tokens, truncation_side
):
    """Both LAST and MEAN readouts must see the same boundary-length inputs."""
    monkeypatch.setattr(scoring_utils, "MultiModalItemTracker", _MultiModalTracker)
    pair = ("query", " ".join(["document"] * document_tokens))
    st_transformer.tokenizer.truncation_side = truncation_side
    expected = st_transformer.preprocess(
        [pair], processing_kwargs={"text": {"max_length": 12}}
    )

    result = _render_pair(
        processor,
        pair,
        {"truncate_prompt_tokens": 12, "truncation_side": truncation_side},
    )

    assert result["prompts"]["prompt_token_ids"] == expected["input_ids"][0].tolist()
    assert result["params"].extra_kwargs["compressed_token_type_ids"] == len(
        expected["input_ids"][0]
    )


def test_suffix_restore_preserves_padding(monkeypatch, processor, st_transformer):
    monkeypatch.setattr(scoring_utils, "MultiModalItemTracker", _MultiModalTracker)
    pair = ("query", " ".join(["document"] * 10))
    expected = st_transformer.preprocess(
        [pair], processing_kwargs={"text": {"max_length": 12}}
    )["input_ids"][0].tolist()

    result = _render_pair(
        processor, pair, {"truncate_prompt_tokens": 12, "pad_prompt_tokens": 16}
    )

    assert result["prompts"]["prompt_token_ids"] == expected + [0] * 4
    assert result["params"].extra_kwargs["compressed_token_type_ids"] == 16


def test_explicit_suffix_restore_opt_out(monkeypatch, processor, st_transformer):
    monkeypatch.setattr(scoring_utils, "MultiModalItemTracker", _MultiModalTracker)
    pair = ("query", " ".join(["document"] * 10))
    chat_kwargs = {"restore_suffix": False}
    expected = st_transformer.preprocess(
        [pair],
        processing_kwargs={"text": {"max_length": 12}, "chat_template": chat_kwargs},
    )["input_ids"][0].tolist()

    result = _render_pair(processor, pair, {"truncate_prompt_tokens": 12}, chat_kwargs)

    assert result["prompts"]["prompt_token_ids"] == expected
    assert expected[-1] == processor.tokenizer.convert_tokens_to_ids("document")
    assert chat_kwargs == {"restore_suffix": False}


def test_media_rows_do_not_restore_text_suffix(monkeypatch, processor, st_transformer):
    monkeypatch.setattr(scoring_utils, "MultiModalItemTracker", _MultiModalTracker)
    text = " ".join(["document"] * 10)
    messages = [
        {"role": "query", "content": [{"type": "text", "text": "query"}]},
        {
            "role": "document",
            "content": [
                {"type": "image", "image": "image.png"},
                {"type": "text", "text": text},
            ],
        },
    ]
    expected = st_transformer._process_chat_messages(
        [messages],
        modality_kwargs={"text": {"truncation": True, "max_length": 12}},
        common_kwargs={"return_tensors": "pt"},
    )["input_ids"][0].tolist()
    pair = (
        "query",
        [
            {"type": "image_url", "image_url": {"url": "image.png"}},
            {"type": "text", "text": text},
        ],
    )

    result = _render_pair(processor, pair, {"truncate_prompt_tokens": 12})

    assert result["prompts"]["prompt_token_ids"] == expected
    assert result["prompts"]["multi_modal_data"] == {"image": ["image.png"]}


@pytest.mark.parametrize(
    "template",
    [
        "{% for message in messages %}{% for part in message.content %}"
        "{{ part.text }} {% endfor %}{% endfor %}",
        "{% for message in messages %}{% for part in message.content %}"
        "{% if part.text == '0' %}{{ raise_exception('No zero filler') }}{% endif %}"
        "{{ part.text }} {% endfor %}{% endfor %}[SEP]",
    ],
    ids=["no-fixed-suffix", "suffix-probe-fails"],
)
def test_unavailable_suffix_matches_sentence_transformers(
    monkeypatch, processor, st_transformer, template
):
    monkeypatch.setattr(scoring_utils, "MultiModalItemTracker", _MultiModalTracker)
    processor.tokenizer.chat_template = st_transformer.tokenizer.chat_template = (
        template
    )
    pair = ("query", " ".join(["document"] * 16))
    expected = st_transformer.preprocess(
        [pair], processing_kwargs={"text": {"max_length": 12}}
    )["input_ids"][0].tolist()

    result = _render_pair(processor, pair, {"truncate_prompt_tokens": 12})

    assert result["prompts"]["prompt_token_ids"] == expected


@pytest.mark.parametrize(
    "tokenization_kwargs",
    [
        {"truncation": "do_not_truncate"},
        {"truncation": False},
        {"truncate_prompt_tokens": None},
    ],
)
def test_explicit_offline_truncation_opt_out(
    monkeypatch, processor, st_transformer, tokenization_kwargs
):
    monkeypatch.setattr(scoring_utils, "MultiModalItemTracker", _MultiModalTracker)
    pair = ("query", " ".join(["document"] * 21))

    with pytest.raises(VLLMValidationError, match="maximum context length"):
        _render_pair(processor, pair, tokenization_kwargs)

    result = _render_pair(processor, pair)
    assert len(result["prompts"]["prompt_token_ids"]) == 16


@pytest.mark.parametrize("explicit_opt_out", [False, True])
def test_online_default_preserves_explicit_truncation_opt_out(
    monkeypatch, processor, st_transformer, explicit_opt_out
):
    monkeypatch.setattr(scoring_utils, "MultiModalItemTracker", _MultiModalTracker)
    kwargs = {"truncate_prompt_tokens": None} if explicit_opt_out else {}
    request = ScoreQueriesDocumentsRequest(
        queries="query", documents=" ".join(["document"] * 21), **kwargs
    )
    processor.chat_template = processor.tokenizer.chat_template
    context = PoolingServeContext(
        request=request,
        model_name="local-model",
        request_id="test",
        pooling_params=PoolingParams(),
        lora_request=None,
        priorities=None,
        prompt_extras=None,
    )
    render_params = processor.get_request_factory_online(context)[0]

    if explicit_opt_out:
        with pytest.raises(VLLMValidationError, match="maximum context length"):
            processor.render(render_params)
    else:
        result = processor.render(render_params)
        assert len(result["prompts"]["prompt_token_ids"]) == 16
