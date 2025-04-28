# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm.multimodal import MULTIMODAL_REGISTRY
from PIL import Image
from ....conftest import _ImageAssets
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["MiniMaxAI/MiniMax-VL-01"])
# yapf: enable
@pytest.mark.parametrize("num_imgs", [1, 2])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_override(
    image_assets: _ImageAssets,
    model_id: str,
    mm_processor_kwargs: dict[str, object],
    num_imgs: int,
    kwargs_on_init: bool,
):
    """Ensure MiniMaxVL01MultiModalProcessor handles min/max pixels properly."""
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    tokenizer = processor.info.get_tokenizer()
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs

    # Build the image str / prompt based on the number of images we pass
    prompt = "<image>" * num_imgs
    image = Image.new("RGB", size=(334, 334))
    mm_data = {"image": [image] * num_imgs}

    processed_inputs = processor.apply(prompt, mm_data, hf_processor_mm_kwargs)

    image_placeholders = processed_inputs["mm_placeholders"]["image"]

    assert len(image_placeholders) == num_imgs
