# SPDX-License-Identifier: Apache-2.0

import pytest
from PIL import Image

from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import _ImageAssets
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["MiniMaxAI/MiniMax-VL-01"])
# yapf: enable
@pytest.mark.parametrize("num_imgs", [1, 2])
def test_processor_override(
    image_assets: _ImageAssets,
    model_id: str,
    num_imgs: int,
):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    prompt = "<image>" * num_imgs
    image = Image.new("RGB", size=(364, 364))
    mm_data = {"image": [image] * num_imgs}

    processed_inputs = processor.apply(prompt, mm_data, {})
    image_placeholders = processed_inputs["mm_placeholders"]["image"]

    assert len(image_placeholders) == num_imgs
    assert processed_inputs["prompt_token_ids"] == processed_inputs[
        "input_ids"][0]
