# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item

from ...utils import build_model_context


@pytest.mark.parametrize("max_model_len", [4096, 16384, 65536])
def test_dynamic_image_budget_matches_profiled_image(max_model_len: int):
    """The image budget must match the profiled dummy image, which the tiler
    caps at `max_num_patches`."""
    ctx = build_model_context(
        "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
        model_config_kwargs={"max_model_len": max_model_len},
        limit_mm_per_prompt={"image": 1, "video": 0},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    budget = get_mm_max_toks_per_item(processor, {"image": 1}, None)
    dummy = processor.get_dummy_mm_inputs({"image": 1})
    num_embeds = sum(p.get_num_embeds() for p in dummy["mm_placeholders"]["image"])

    assert budget["image"] == num_embeds
