# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from transformers import BertTokenizer

from vllm.entrypoints.pooling.scoring import utils as scoring_utils
from vllm.entrypoints.pooling.scoring.io_processor import (
    CrossEncoderIOProcessor,
    _validate_sentence_transformers_tokenizer,
)

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
    }
    return BertTokenizer(vocab=vocab, do_lower_case=False)


@pytest.fixture
def processor(tokenizer):
    processor = CrossEncoderIOProcessor.__new__(CrossEncoderIOProcessor)
    processor.model_config = SimpleNamespace(enable_prompt_embeds=False)
    processor.tokenizer = tokenizer
    processor.supports_score_template = False
    processor.model = None
    processor.use_sep_token = True
    processor.sentence_transformers_config = SimpleNamespace(uses_message_format=True)
    return processor


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
