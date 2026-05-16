# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.pooling.base.io_processor import PoolingIOProcessor
from vllm.pooling_params import PoolingParams


class FakeTokenizer:
    mask_token_id = 250001
    eos_token_id = 2
    sep_token_id = 2
    pad_token_id = 1
    unk_token_id = 3

    def convert_tokens_to_ids(self, token: str):
        return {
            "[QueryMarker]": 250002,
            "[DocumentMarker]": 250003,
        }.get(token, self.unk_token_id)


class FakeRenderer:
    tokenizer = FakeTokenizer()


class FakeModelConfig:
    architecture = "ColBERTJinaRobertaModel"
    max_model_len = 4


def make_processor():
    processor = object.__new__(PoolingIOProcessor)
    processor.name = "token_embed"
    processor.renderer = FakeRenderer()
    processor.model_config = FakeModelConfig()
    return processor


def test_colbert_query_input_type_inserts_marker_and_expands_to_32():
    processor = make_processor()

    token_ids = processor._colbert_token_ids([0, 100, 101, 2], "query")

    assert token_ids[:5] == [0, 250002, 100, 101, 2]
    assert len(token_ids) == 32
    assert token_ids[5:] == [250001] * 27


def test_colbert_query_input_type_truncates_before_marker_insert():
    processor = make_processor()

    token_ids = processor._colbert_token_ids([0, *range(100, 150), 2], "query")

    assert len(token_ids) == 32
    assert token_ids[0] == 0
    assert token_ids[1] == 250002
    assert token_ids[-1] == 2


def test_colbert_document_input_type_inserts_marker_without_query_expansion():
    processor = make_processor()

    token_ids = processor._colbert_token_ids([0, 100, 101, 2], "document")

    assert token_ids == [0, 250003, 100, 2]


def test_colbert_input_type_requires_token_embed_offline():
    processor = make_processor()
    processor.name = "embed"

    ctx = type(
        "FakeContext",
        (),
        {
            "prompts": "test",
            "pooling_params": PoolingParams(task="embed"),
            "tokenization_kwargs": {"input_type": "query"},
        },
    )()

    try:
        processor.pre_process_offline(ctx)
    except ValueError as e:
        assert str(e) == "input_type is only supported with task 'token_embed'."
    else:
        raise AssertionError("Expected ValueError")


def test_colbert_input_type_rejects_non_colbert_models():
    processor = make_processor()
    processor.model_config.architecture = "BertModel"

    try:
        processor._apply_colbert_input_type(
            [{"prompt_token_ids": [0, 100, 2]}], "query"
        )
    except ValueError as e:
        assert str(e) == "input_type is only supported for ColBERT models."
    else:
        raise AssertionError("Expected ValueError")
