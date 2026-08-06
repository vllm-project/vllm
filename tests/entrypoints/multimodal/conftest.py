# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from typing import Any

import pytest

# Test different image extensions (JPG/PNG) and formats (gray/RGB/RGBA)
TEST_IMAGE_ASSETS = [
    "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    "Grayscale_8bits_palette_sample_image.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/Grayscale_8bits_palette_sample_image.png",
    "1280px-Venn_diagram_rgb.svg.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/1280px-Venn_diagram_rgb.svg.png",
    "RGBA_comp.png",  # "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/RGBA_comp.png",
]


@pytest.fixture
def multimodal_llm_factory(vllm_runner_factory) -> Callable[..., Any]:
    def make_llm(*args: Any, **kwargs: Any) -> Any:
        runner_kwargs = kwargs.copy()
        tokenizer_name = runner_kwargs.pop("tokenizer", None)

        if args:
            model_name, *runner_args = args
        else:
            model_name = runner_kwargs.pop("model")
            runner_args = []

        runner = vllm_runner_factory(
            model_name,
            *runner_args,
            tokenizer_name=tokenizer_name,
            **runner_kwargs,
        )
        return runner.llm

    return make_llm
