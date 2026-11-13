# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Processing tests for Nemotron Parse 3.5."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import BatchFeature

from vllm.model_executor.models.nemotron_parse3_5 import (
    NemotronParseProcessingInfo,
    RadioMBartPrefixMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import BaseProcessingInfo

from ...utils import build_model_context


def _noise_image(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("mm_device_do_normalize", [None, True])
def test_causal_prefix_preserves_host_normalized_packed_images(
    mm_device_do_normalize: bool | None,
):
    """Packed Parse 3.5 inputs must not inherit legacy device normalization."""
    ctx = build_model_context(
        "nvidia/Nemotron-3.5-Parse",
        limit_mm_per_prompt={"image": 1},
        model_config_kwargs={"mm_device_do_normalize": mm_device_do_normalize},
    )
    assert not ctx.model_config.multimodal_config.mm_device_do_normalize
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    image = _noise_image(310, 470)
    expected = processor.info.get_hf_processor().image_processor(
        [image], return_tensors="pt"
    )
    mm_items = processor.info.parse_mm_data({"image": [image]})
    actual = processor([1], mm_items=mm_items)["mm_kwargs"].get_data()

    pixel_values = actual["pixel_values"]
    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.is_floating_point()
    assert pixel_values.shape[-1] == 3 * 16**2
    for name in ("pixel_values", "image_sizes", "content_sizes"):
        value = actual[name]
        assert isinstance(value, torch.Tensor)
        torch.testing.assert_close(
            value, expected[name].to(value.dtype), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("navit_packing", False),
        ("navit_coordinate_frame", "canvas"),
        ("navit_padding_mode", "center"),
        ("do_normalize", False),
    ],
)
def test_causal_prefix_rejects_incompatible_processor_settings(
    monkeypatch: pytest.MonkeyPatch, setting: str, value: object
) -> None:
    image_processor = SimpleNamespace(
        navit_packing=True,
        navit_coordinate_frame="content_normalized",
        navit_padding_mode="top_left_white",
        do_normalize=True,
    )
    setattr(image_processor, setting, value)
    monkeypatch.setattr(
        BaseProcessingInfo,
        "get_hf_processor",
        lambda self, **kwargs: SimpleNamespace(image_processor=image_processor),
    )
    info = NemotronParseProcessingInfo.__new__(NemotronParseProcessingInfo)
    monkeypatch.setattr(
        info,
        "get_hf_config",
        lambda: SimpleNamespace(navit_packing=True),
    )

    with pytest.raises(ValueError, match=f"requires {setting}="):
        info.get_hf_processor()


def test_causal_prefix_fields_config_handles_empty_cached_hf_inputs() -> None:
    processor = RadioMBartPrefixMultiModalProcessor.__new__(
        RadioMBartPrefixMultiModalProcessor
    )
    processor.info = SimpleNamespace(
        get_hf_config=lambda: SimpleNamespace(encoder=SimpleNamespace(patch_size=16))
    )

    fields = processor._get_mm_fields_config(BatchFeature(), {})

    assert getattr(fields["pixel_values"].field, "slices") == []  # noqa: B009
