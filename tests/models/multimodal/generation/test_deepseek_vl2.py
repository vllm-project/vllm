# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from PIL import Image

from vllm import SamplingParams

from ....conftest import VllmRunner


@pytest.mark.core_model
@pytest.mark.parametrize("image_counts", [(1, 3), (3, 1)])
def test_deepseek_vl2_encoder_cache_cropping(
    vllm_runner: type[VllmRunner], image_counts: tuple[int, int]
):
    images = [Image.new("RGB", (768, 768), color) for color in ("red", "green", "blue")]
    with vllm_runner(
        "deepseek-ai/deepseek-vl2-tiny",
        hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]},
        max_model_len=4096,
        max_num_seqs=1,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 3},
    ) as runner:
        for count in image_counts:
            prompt = "<|User|>: " + "<image>\n" * count + "Describe.\n\n<|Assistant|>:"
            outputs = runner.llm.generate(
                {"prompt": prompt, "multi_modal_data": {"image": images[:count]}},
                SamplingParams(temperature=0, max_tokens=1),
            )
            output = outputs[0].outputs[0]
            assert output.finish_reason == "length"
            assert len(output.token_ids) == 1
