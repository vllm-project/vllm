# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.conftest import ImageTestAssets
from tests.model_executor.layers.test_mm_input_norm import (
    _DEVICE_TYPE,
    requires_accelerator,
    requires_vllm_config,
)
from tests.models.utils import GenerateModelInfo, build_model_context
from vllm.model_executor.layers.fusion.mm_input_norm import FusedMMInputNorm
from vllm.multimodal import MULTIMODAL_REGISTRY

from .ppl_utils import vqa_ppl_test

MODELS = [
    GenerateModelInfo("Qwen/Qwen2-VL-2B-Instruct", hf_ppl=41081356.0),
    GenerateModelInfo("Qwen/Qwen2.5-VL-3B-Instruct", hf_ppl=18330016.0),
]


mm_processor_kwargs = {
    "min_pixels": 28 * 28,
    "max_pixels": 1280 * 28 * 28,
}


@pytest.mark.parametrize("model_info", MODELS)
@pytest.mark.parametrize("mm_device_do_normalize", [True, False])
def test_ppl(
    hf_runner, vllm_runner, model_info: GenerateModelInfo, mm_device_do_normalize: bool
):
    vqa_ppl_test(
        hf_runner,
        vllm_runner,
        model_info,
        vllm_extra_kwargs={"mm_device_do_normalize": mm_device_do_normalize},
        mm_processor_kwargs=mm_processor_kwargs,
    )


# ===========================================================================
# End-to-end: processor integration with mm_device_do_normalize
# This test is relatively slow and requires a GPU, so it has been moved here
# from tests/model_executor/layers/test_mm_input_norm.py.
# ===========================================================================
@requires_vllm_config
@requires_accelerator
class TestMMDeviceDoNormalize:
    """Processor-level integration: FusedMMInputNorm must reproduce the
    on-CPU processor normalization exactly."""

    @pytest.mark.parametrize(
        "model_id",
        ["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"],
    )
    @pytest.mark.parametrize("num_imgs", [1, 2])
    def test_mm_device_do_normalize(
        self,
        image_assets: ImageTestAssets,
        model_id: str,
        num_imgs: int,
    ):
        """Ensure that enable mm_device_do_normalize yields the correct result."""
        ctx = build_model_context(
            model_id,
            limit_mm_per_prompt={"image": num_imgs},
        )
        ctx.model_config.multimodal_config.mm_device_do_normalize = False
        processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

        # Build the image str / prompt based on the number of images we pass
        prompt = "<|vision_start|><|image_pad|><|vision_end|>" * num_imgs
        mm_data = {"image": [image_assets[0].pil_image] * num_imgs}

        processed_inputs_with_normalize = processor(
            prompt,
            mm_items=processor.info.parse_mm_data(mm_data),
        )
        pixel_values_with_normalize = processed_inputs_with_normalize[
            "mm_kwargs"
        ].get_data()["pixel_values"]
        dtype = pixel_values_with_normalize.dtype

        processed_inputs_without_normalize = processor(
            prompt,
            mm_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={"do_normalize": False, "do_rescale": False},
        )
        pixel_values_without_normalize = processed_inputs_without_normalize[
            "mm_kwargs"
        ].get_data()["pixel_values"]

        ctx.model_config.multimodal_config.mm_device_do_normalize = True
        input_norm = FusedMMInputNorm.from_model_config(ctx.model_config).to(
            _DEVICE_TYPE
        )

        pixel_values_do_input_norm = input_norm(
            pixel_values_without_normalize.to(dtype).to(_DEVICE_TYPE), dtype
        )

        torch.testing.assert_close(
            pixel_values_with_normalize.to(_DEVICE_TYPE), pixel_values_do_input_norm
        )
