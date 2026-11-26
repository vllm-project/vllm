# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audio encoder CUDA graphs keep the outputs of the eager audio tower."""

import pytest

from vllm.assets.audio import AudioAsset
from vllm.platforms import current_platform

from ...utils import check_logprobs_close

MODEL = "Qwen/Qwen3-ASR-0.6B"
PROMPT = (
    "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
    "<|im_start|>assistant\n"
)


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA graphs")
@pytest.mark.parametrize("mm_encoder_attn_backend", ["FLASH_ATTN", "TRITON_ATTN"])
def test_qwen3_asr_audio_encoder_cudagraph(vllm_runner, mm_encoder_attn_backend):
    audios = [
        [AudioAsset(name).audio_and_sample_rate]
        for name in ("mary_had_lamb", "winning_call")
    ]
    prompts = [PROMPT] * len(audios)
    outputs = {}
    for cudagraph_mm_encoder in (False, True):
        with vllm_runner(
            MODEL,
            dtype="bfloat16",
            max_model_len=4096,
            max_num_seqs=2,
            limit_mm_per_prompt={"audio": 1},
            mm_encoder_attn_backend=mm_encoder_attn_backend,
            compilation_config={"cudagraph_mm_encoder": cudagraph_mm_encoder},
        ) as vllm_model:
            outputs[cudagraph_mm_encoder] = vllm_model.generate_greedy_logprobs(
                prompts, max_tokens=64, num_logprobs=5, audios=audios
            )

    check_logprobs_close(
        outputs_0_lst=outputs[False],
        outputs_1_lst=outputs[True],
        name_0="eager_encoder",
        name_1="cudagraph_encoder",
    )
