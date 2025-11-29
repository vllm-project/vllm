# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["google/gemma-3-4b-it"])
def test_get_image_size_with_most_features(
    image_assets: ImageTestAssets, model_id: str
):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs={"do_pan_and_scan": True},
        limit_mm_per_prompt={"image": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    hf_processor_mm_kwargs: dict[str, object] = {}
    hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)

    max_image_size = processor.info.get_image_size_with_most_features()
    max_tokens = processor.info.get_num_image_tokens(
        image_width=max_image_size.width,
        image_height=max_image_size.height,
        processor=hf_processor,
    )

    prompt = "<start_of_image>"
    image_seq_length = hf_processor.image_seq_length

    for asset in image_assets:
        mm_data = {"image": [asset.pil_image]}
        processed_inputs = processor.apply(prompt, mm_data, hf_processor_mm_kwargs)
        mm_kwargs_data = processed_inputs["mm_kwargs"].get_data()
        num_patches_tensor = mm_kwargs_data["num_patches"]
        tokens = int(num_patches_tensor.item()) * image_seq_length
        assert tokens <= max_tokens
