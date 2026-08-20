# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.exceptions import VLLMValidationError
from vllm.inputs import embeds_input, mm_input, tokens_input
from vllm.multimodal.inputs import MultiModalKwargsItems, PlaceholderRange
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.xdrope import validate_xdrope_input


def make_input_processor(
    vocab_size: int, *, tokenizer_vocab_size: int | None = None
) -> InputProcessor:
    processor = object.__new__(InputProcessor)
    processor.model_config = SimpleNamespace(get_vocab_size=lambda: vocab_size)
    tokenizer = (
        None
        if tokenizer_vocab_size is None
        else SimpleNamespace(max_token_id=tokenizer_vocab_size - 1)
    )
    processor.renderer = SimpleNamespace(tokenizer=tokenizer)
    processor._validate_prompt_len = Mock()
    processor.mm_encoder_cache_size = 100
    return processor


@pytest.mark.parametrize("token_ids", [[0], [0, 9]])
def test_pretokenized_prompt_ids_accept_valid_non_negative_ids(token_ids: list[int]):
    processor = make_input_processor(vocab_size=10, tokenizer_vocab_size=10)

    processor._validate_model_input(tokens_input(token_ids), prompt_type="decoder")


@pytest.mark.parametrize("token_ids", [[-1], [1, -1], [-1034240]])
def test_pretokenized_prompt_ids_reject_negative_ids_before_model_execution(
    token_ids: list[int],
):
    processor = make_input_processor(vocab_size=10, tokenizer_vocab_size=10)

    with pytest.raises(VLLMValidationError, match="out of vocabulary"):
        processor._validate_model_input(tokens_input(token_ids), prompt_type="decoder")


def test_tokens_only_prompt_rejects_model_out_of_vocab_id():
    processor = make_input_processor(vocab_size=10)

    with pytest.raises(VLLMValidationError, match="out of vocabulary"):
        processor._validate_model_input(tokens_input([10]), prompt_type="decoder")


def test_ordinary_prompt_rejects_tokenizer_only_id():
    processor = make_input_processor(vocab_size=10, tokenizer_vocab_size=20)

    with pytest.raises(VLLMValidationError, match="out of vocabulary"):
        processor._validate_model_input(tokens_input([10]), prompt_type="decoder")


def test_prompt_accepts_model_only_extra_token():
    processor = make_input_processor(vocab_size=20, tokenizer_vocab_size=10)

    processor._validate_model_input(tokens_input([19]), prompt_type="decoder")


def test_multimodal_prompt_accepts_replaced_tokenizer_only_placeholder():
    processor = make_input_processor(vocab_size=10, tokenizer_vocab_size=20)
    prompt = mm_input(
        [1, 15, 15, 2],
        MultiModalKwargsItems({}),
        {},
        {"image": [PlaceholderRange(offset=1, length=2)]},
    )

    processor._validate_model_input(prompt, prompt_type="decoder")


def test_multimodal_prompt_rejects_unreplaced_tokenizer_only_id():
    processor = make_input_processor(vocab_size=10, tokenizer_vocab_size=20)
    prompt = mm_input(
        [15, 1, 1],
        MultiModalKwargsItems({}),
        {},
        {"image": [PlaceholderRange(offset=1, length=2)]},
    )

    with pytest.raises(VLLMValidationError, match="out of vocabulary"):
        processor._validate_model_input(prompt, prompt_type="decoder")


def test_multimodal_prompt_uses_partial_embedding_replacement_mask():
    processor = make_input_processor(vocab_size=10, tokenizer_vocab_size=20)
    prompt = mm_input(
        [1, 15, 15],
        MultiModalKwargsItems({}),
        {},
        {
            "image": [
                PlaceholderRange(
                    offset=1,
                    length=2,
                    is_embed=torch.tensor([False, True]),
                )
            ]
        },
    )

    with pytest.raises(VLLMValidationError, match="out of vocabulary"):
        processor._validate_model_input(prompt, prompt_type="decoder")


def test_mixed_embeds_accepts_out_of_vocab_id_only_at_embed_position():
    processor = make_input_processor(vocab_size=10)
    prompt = embeds_input(
        torch.randn(2, 4),
        prompt_token_ids=[15, 1],
        is_token_ids=[False, True],
    )

    processor._validate_model_input(prompt, prompt_type="decoder")


def test_mixed_embeds_rejects_out_of_vocab_real_token_id():
    processor = make_input_processor(vocab_size=10)
    prompt = embeds_input(
        torch.randn(2, 4),
        prompt_token_ids=[1, 15],
        is_token_ids=[False, True],
    )

    with pytest.raises(VLLMValidationError, match="out of vocabulary"):
        processor._validate_model_input(prompt, prompt_type="decoder")


def make_xdrope_feature(grid_thw: list[int]):
    item = {"image_grid_thw": SimpleNamespace(data=torch.tensor(grid_thw))}
    return SimpleNamespace(data=item)


def make_xdrope_config():
    return SimpleNamespace(
        uses_xdrope_dim=4,
        hf_config=SimpleNamespace(
            image_start_token_id=9,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        ),
    )


def test_xdrope_rejects_marker_without_geometry():
    with pytest.raises(VLLMValidationError, match="image marker"):
        validate_xdrope_input(make_xdrope_config(), [9, 1, 1], [])


def test_xdrope_ignores_marker_value_at_prompt_embed_position():
    validate_xdrope_input(
        make_xdrope_config(),
        [9],
        [],
        prompt_is_token_ids=[False],
    )


def test_xdrope_rejects_geometry_without_marker():
    with pytest.raises(VLLMValidationError, match="image marker"):
        validate_xdrope_input(
            make_xdrope_config(),
            [1, 1, 1],
            [make_xdrope_feature([1, 2, 2])],
        )


def test_xdrope_defers_unresolved_frontend_cache_feature():
    validate_xdrope_input(
        make_xdrope_config(),
        [9, 1, 1],
        [SimpleNamespace(data=None, modality="image")],
        allow_unresolved_features=True,
    )


def test_xdrope_rejects_unresolved_engine_cache_feature():
    with pytest.raises(VLLMValidationError, match="image marker"):
        validate_xdrope_input(
            make_xdrope_config(),
            [9, 1, 1],
            [SimpleNamespace(data=None, modality="image")],
        )


def test_xdrope_rejects_geometry_span_past_prompt():
    with pytest.raises(VLLMValidationError, match="extends past"):
        validate_xdrope_input(
            make_xdrope_config(),
            [9, 1, 1],
            [make_xdrope_feature([1, 4, 4])],
        )


def test_xdrope_accepts_consistent_marker_geometry():
    validate_xdrope_input(
        make_xdrope_config(),
        [9, 1, 1, 1, 1, 1],
        [make_xdrope_feature([1, 2, 2])],
    )
