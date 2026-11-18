# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial

import pytest
from PIL import Image

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams

from ...registry import HF_EXAMPLE_MODELS
from ...utils import dummy_hf_overrides


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_engine_applies_max_pixels(vllm_runner) -> None:
    arch = "OpenCUAForConditionalGeneration"
    model_info = HF_EXAMPLE_MODELS.get_hf_info(arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(
        on_fail="skip", check_max_version=False, check_version_reason="vllm"
    )

    with vllm_runner(
        model_info.default,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        load_format="dummy",
        enforce_eager=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_pixels": 512**2},
        hf_overrides=partial(dummy_hf_overrides, model_arch=arch),
    ) as runner:
        llm = runner.llm
        assert not llm.model_config.multimodal_config.mm_device_do_normalize
        image_token_id = llm.get_tokenizer().convert_tokens_to_ids(
            "<|media_placeholder|>"
        )
        outputs = llm.generate(
            {
                "prompt": "<|media_begin|>image<|media_content|>"
                "<|media_placeholder|><|media_end|>Describe.",
                "multi_modal_data": {"image": Image.new("RGB", (1920, 1080))},
            },
            SamplingParams(temperature=0, max_tokens=1),
        )

    assert outputs[0].prompt_token_ids.count(image_token_id) == 312
