# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, TokenizersBackend

from vllm import SamplingParams
from vllm.config import DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.inputs import embeds_input, mm_input, tokens_input
from vllm.multimodal.inputs import MultiModalKwargsItems, PlaceholderRange
from vllm.v1.engine.input_processor import InputProcessor


@pytest.fixture
def input_processor_factory(tmp_path):
    processors: list[InputProcessor] = []

    def make(vocab_size=10, tokenizer_vocab_size=20, skip_tokenizer_init=False):
        model_path = tmp_path / str(len(processors))
        LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
            architectures=["LlamaForCausalLM"],
        ).save_pretrained(model_path)
        tokenizer = TokenizersBackend(
            tokenizer_object=Tokenizer(
                WordLevel(
                    {str(i): i for i in range(tokenizer_vocab_size)}, unk_token="0"
                )
            ),
            unk_token="0",
        )
        tokenizer.save_pretrained(model_path)
        config = VllmConfig(
            model_config=ModelConfig(
                model=str(model_path),
                dtype="float32",
                max_model_len=32,
                enforce_eager=True,
                skip_tokenizer_init=skip_tokenizer_init,
            ),
            device_config=DeviceConfig("cpu"),
            scheduler_config=SchedulerConfig(
                is_encoder_decoder=False, max_model_len=32
            ),
        )
        processor = InputProcessor(config)
        processors.append(processor)
        return processor

    yield make
    for processor in processors:
        processor.renderer.shutdown()


@pytest.mark.parametrize("skip_tokenizer_init", [False, True])
@pytest.mark.parametrize("token_id", [-1, 10, 19])
def test_prompt_rejects_ids_outside_model_vocabulary(
    input_processor_factory, skip_tokenizer_init, token_id
):
    """Tokenizer membership does not make an ID a valid embedding index."""
    processor = input_processor_factory(skip_tokenizer_init=skip_tokenizer_init)
    with pytest.raises(VLLMValidationError, match="out of vocabulary"):
        processor.process_inputs(
            "invalid",
            tokens_input([token_id]),
            SamplingParams(max_tokens=1),
            supported_tasks=("generate",),
        )
    request = processor.process_inputs(
        "valid",
        tokens_input([9]),
        SamplingParams(max_tokens=1),
        supported_tasks=("generate",),
    )
    assert request.prompt_token_ids == [9]


def test_prompt_accepts_model_only_extra_token(input_processor_factory):
    processor = input_processor_factory(vocab_size=20, tokenizer_vocab_size=10)
    request = processor.process_inputs(
        "valid",
        tokens_input([19]),
        SamplingParams(max_tokens=1),
        supported_tasks=("generate",),
    )
    assert request.prompt_token_ids == [19]


@pytest.mark.parametrize("token_ids", [[15, 1], [1, 15]])
def test_mixed_embeds_rejects_invalid_ids_before_embedding_lookup(
    input_processor_factory, token_ids
):
    """Mixed-mode IDs reach embedding lookup before prompt embeddings are applied."""
    processor = input_processor_factory()
    prompt = embeds_input(
        torch.zeros(2, 16), prompt_token_ids=token_ids, is_token_ids=[False, True]
    )
    with pytest.raises(VLLMValidationError, match="out of vocabulary"):
        processor.process_inputs(
            "invalid",
            prompt,
            SamplingParams(max_tokens=1),
            supported_tasks=("generate",),
        )


@pytest.mark.parametrize(
    ("is_embed", "expected"),
    [(None, [False, True, True]), (torch.tensor([False, True]), [False, False, True])],
)
def test_multimodal_replacement_mask_excludes_text_positions(is_embed, expected):
    prompt = mm_input(
        [1, 15, 15],
        MultiModalKwargsItems({}),
        {},
        {"image": [PlaceholderRange(offset=1, length=2, is_embed=is_embed)]},
    )
    assert InputProcessor._get_embedding_replacement_positions(prompt, 3) == expected
