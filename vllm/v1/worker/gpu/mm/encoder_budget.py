# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

from vllm.config import ModelConfig
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.registry import MultiModalRegistry

EncoderProfileInputFactory = Callable[
    [str, int], list[tuple[str, MultiModalKwargsItem]]
]


def get_dummy_encoder_profile_inputs(
    model_config: ModelConfig,
    mm_registry: MultiModalRegistry,
    modality: str,
    max_items_per_batch: int,
) -> list[tuple[str, MultiModalKwargsItem]]:
    dummy_mm_inputs = mm_registry.get_dummy_mm_inputs(
        model_config,
        mm_counts={modality: 1},
    )
    dummy_mm_item = dummy_mm_inputs["mm_kwargs"][modality][0]
    assert dummy_mm_item is not None, "Dummy item should be generated"

    return [(modality, dummy_mm_item)] * max_items_per_batch
