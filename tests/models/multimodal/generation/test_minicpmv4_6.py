# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams

from ...registry import HF_EXAMPLE_MODELS


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_engine_admits_video_of_largest_frames(vllm_runner) -> None:
    arch = "MiniCPMV4_6ForConditionalGeneration"
    model_info = HF_EXAMPLE_MODELS.get_hf_info(arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(
        on_fail="skip", check_max_version=False, check_version_reason="vllm"
    )
    num_frames = 16
    video = (
        np.zeros((num_frames, 1112, 1265, 3), dtype=np.uint8),
        {
            "fps": 2.0,
            "duration": num_frames / 2.0,
            "total_num_frames": num_frames,
            "frames_indices": list(range(num_frames)),
            "video_backend": "opencv",
            "do_sample_frames": False,
        },
    )

    with vllm_runner(
        model_info.default,
        enforce_eager=True,
        max_model_len=16384,
        enable_chunked_prefill=True,
        max_num_batched_tokens=8192,
        limit_mm_per_prompt={"image": 1, "video": 1},
    ) as runner:
        llm = runner.llm
        video_token_id = llm.get_tokenizer().convert_tokens_to_ids("<|video_pad|>")
        outputs = llm.generate(
            {
                "prompt": "<|im_start|>user\n<|video_pad|>\nDescribe.<|im_end|>\n"
                "<|im_start|>assistant\n",
                "multi_modal_data": {"video": video},
            },
            SamplingParams(temperature=0, max_tokens=1),
        )

    assert outputs[0].prompt_token_ids.count(video_token_id) == num_frames * 720
