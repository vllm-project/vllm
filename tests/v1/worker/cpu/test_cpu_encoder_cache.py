# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from dataclasses import replace

import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalSharedField,
    PlaceholderRange,
)
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.cpu.encoder_cache import EncoderCache
from vllm.v1.worker.cpu.encoder_runner import EncoderRunner

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)

HIDDEN_SIZE = 4


def _embeds_item(embeds: torch.Tensor) -> MultiModalKwargsItem:
    return MultiModalKwargsItem(
        {
            "embedding": MultiModalFieldElem(
                data=embeds, field=MultiModalSharedField(batch_size=1)
            )
        }
    )


def _feature(identifier: str, offset: int, length: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=identifier,
        mm_position=PlaceholderRange(offset=offset, length=length),
    )


def test_consumed_input_updates_are_request_local_and_replayable() -> None:
    pixels = torch.ones(8, HIDDEN_SIZE)
    feature = replace(_feature("same", 0, 8), data=_embeds_item(pixels))
    later = replace(feature, mm_position=PlaceholderRange(offset=16, length=8))
    source = [feature, later]
    cache = EncoderCache()
    cache.add_request("a", source)
    cache.add_request("b", source)
    cache.encoder_outputs["same"] = torch.ones(8, HIDDEN_SIZE)

    cache.update_encoder_inputs("a", {0: None})

    assert cache.mm_features["a"][0].data is None
    assert cache.mm_features["a"][0].mm_position is feature.mm_position
    assert cache.mm_features["a"][1].data is feature.data
    assert cache.mm_features["b"][0].data is feature.data
    assert source[0].data is not None
    assert "same" in cache.encoder_outputs

    cache.free_encoder_cache("same")
    cache.update_encoder_inputs("a", {0: source[0].data})
    runner = EncoderRunner(
        model=torch.nn.Module(),
        max_num_tokens=32,
        hidden_size=HIDDEN_SIZE,
        encoder_cache=cache,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    hashes, inputs = runner.prepare_mm_inputs({"a": [0]})
    assert hashes == ["same"]
    assert inputs == [(feature.modality, feature.data)]


def test_consumed_input_releases_storage_before_request_finishes() -> None:
    cache = EncoderCache()
    pixels = torch.ones(8, HIDDEN_SIZE)
    ref = weakref.ref(pixels)
    cache.add_request("a", [replace(_feature("h", 0, 8), data=_embeds_item(pixels))])
    del pixels

    assert ref() is not None
    cache.update_encoder_inputs("a", {0: None})
    assert ref() is None
    assert "a" in cache.mm_features
