# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from PIL import Image

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import get_mm_max_toks_per_item
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs

from ...utils import build_model_context


def _image_embeds(outputs) -> int:
    return sum(p.get_num_embeds() for p in outputs["mm_placeholders"]["image"])


@pytest.mark.parametrize(
    "mm_processor_kwargs",
    [None, {"max_pixels": 153664}, {"image_max_pixels": 1003520}],
)
def test_image_budget_covers_non_square_images(mm_processor_kwargs: dict | None):
    """The image budget must match the profiled image and cover non-square
    photos, under the pixel limits the processor actually applies."""
    ctx = build_model_context(
        "baidu/ERNIE-4.5-VL-28B-A3B-PT",
        mm_processor_kwargs=mm_processor_kwargs,
        limit_mm_per_prompt={"image": 1, "video": 0},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    budget = get_mm_max_toks_per_item(processor, {"image": 1}, None)["image"]
    dummy = processor.get_dummy_mm_inputs({"image": 1})
    assert budget == _image_embeds(dummy)

    prompt = processor.info.get_tokenizer().encode("<|image@placeholder|>")
    for width, height in [(6000, 4000), (4032, 3024)]:
        inputs = ProcessorInputs(
            prompt=prompt,
            mm_data_items=processor.info.parse_mm_data(
                {"image": Image.new("RGB", (width, height))}
            ),
        )
        real = processor.apply(inputs, TimingContext(enabled=False))
        assert _image_embeds(real) <= budget
