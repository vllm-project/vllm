# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise centered MRoPE, YaRN and image DP through a real Ling VL engine."""

import pytest
import regex as re
import torch
from PIL import Image, ImageDraw

from vllm import SamplingParams
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.utils import encode_image_url

from ....utils import multi_gpu_only

MODEL = "inclusionAI/Ling-3.0-flash-VL"


def _image_then_text_positions(model):
    merge = model.config.vision_config.spatial_merge_size
    data = MultiModalKwargsItems.from_hf_inputs(
        {"image_grid_thw": torch.tensor([[1, merge, 3 * merge]])},
        {"image_grid_thw": MultiModalFieldConfig.batched("image")},
    )
    feature = MultiModalFeatureSpec(
        data=data["image"][0],
        modality="image",
        identifier="three-horizontal-patches",
        mm_position=PlaceholderRange(offset=0, length=3),
    )
    positions, delta = model.get_mrope_input_positions([0] * 4, [feature])
    return positions.tolist(), delta


@multi_gpu_only(num_gpus=4)
@pytest.mark.usefixtures("enable_pickle")
@pytest.mark.parametrize("encoder_mode", ["weights", "data"])
def test_bailing_yarn_image_order_and_centered_positions(vllm_runner, encoder_mode):
    """Use real weights; cover idle encoder ranks and uneven five-image batches."""
    with vllm_runner(
        MODEL,
        dtype="bfloat16",
        tensor_parallel_size=4,
        mm_encoder_tp_mode=encoder_mode,
        max_model_len=8192,
        max_num_seqs=1,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 5},
        mm_processor_kwargs={"max_pixels": 1048576},
        hf_overrides={
            "text_config": {
                "max_position_embeddings": 262144,
                "rope_parameters": {
                    "rope_type": "yarn",
                    "factor": 2.0,
                    "rope_theta": 6000000,
                    "original_max_position_embeddings": 131072,
                },
            }
        },
    ) as runner:
        expected_positions = ([[0, 0, 0, 1], [0, 0, 0, 1], [-1, 0, 1, 1]], -2)
        for actual in runner.apply_model(_image_then_text_positions):
            assert actual == expected_positions

        sizes = [(640, 640), (64, 64), (320, 160), (160, 640), (96, 96)]
        images = []
        expected_colors: list[str] = []
        for i, (width, height) in enumerate(sizes):
            colors = ("red", "blue") if i % 2 == 0 else ("green", "yellow")
            image = Image.new("RGB", (width, height), colors[0])
            ImageDraw.Draw(image).rectangle(
                (width // 2, 0, width, height), fill=colors[1]
            )
            images.append(
                {"type": "image_url", "image_url": {"url": encode_image_url(image)}}
            )
            expected_colors.extend(colors)

        llm = runner.get_llm()
        sampling_params = SamplingParams(temperature=0, seed=0, max_tokens=128)
        for count in (1, 5):
            content = images[:count] + [
                {
                    "type": "text",
                    "text": "For each image in order, name the left and right colors. "
                    "Return only one pair per line.",
                }
            ]
            result = llm.chat(
                [{"role": "user", "content": content}],
                sampling_params,
                chat_template_kwargs={"enable_thinking": False},
                use_tqdm=False,
            )[0].outputs[0]
            assert result.finish_reason == "stop"
            assert (
                re.findall(r"\b(red|blue|green|yellow)\b", result.text.lower())
                == (expected_colors[: 2 * count])
            )
