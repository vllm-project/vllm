# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from types import SimpleNamespace

import pytest

from vllm.model_executor.models.blip2 import Blip2ForConditionalGeneration
from vllm.multimodal.inputs import MultiModalKwargsItem


@pytest.fixture
def blip2_token_counts():
    # Match the 32 query tokens and 16x16 vision patches plus CLS in BLIP-2 OPT.
    model = SimpleNamespace(
        config=SimpleNamespace(num_query_tokens=32),
        _vision_tokens_per_image=257,
    )
    return partial(Blip2ForConditionalGeneration.get_mm_lora_token_counts, model)


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(5000, (156 * 257, 4992)), (64, (514, 64)), (0, (0, 0))],
)
def test_lora_sizing_counts_only_whole_images(blip2_token_counts, budget, expected):
    assert (
        blip2_token_counts(modality="image", mm_kwargs=None, num_mm_embeds=budget)
        == expected
    )


def test_lora_counts_for_image_input(blip2_token_counts):
    assert blip2_token_counts(
        modality="image", mm_kwargs=MultiModalKwargsItem.dummy(), num_mm_embeds=32
    ) == (257, 32)
