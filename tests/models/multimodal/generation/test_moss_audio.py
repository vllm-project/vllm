# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.assets.audio import AudioAsset
from vllm.model_executor.models.moss_audio import MOSS_AUDIO_PLACEHOLDER

from ...registry import HF_EXAMPLE_MODELS
from ...utils import check_logprobs_close

MODELS = [
    "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
    "OpenMOSS-Team/MOSS-Audio-4B-Thinking",
    "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
    "OpenMOSS-Team/MOSS-Audio-8B-Thinking",
]


@pytest.mark.core_model
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [16])
@pytest.mark.parametrize("num_logprobs", [10])
def test_moss_audio_hf_vllm_accuracy(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    prompts = [
        f"{MOSS_AUDIO_PLACEHOLDER}\nTranscribe or summarize this audio.",
        f"{MOSS_AUDIO_PLACEHOLDER}\nBriefly describe what is happening in this audio.",
    ]
    audios = [
        [AudioAsset("mary_had_lamb").audio_and_sample_rate[0]],
        [AudioAsset("winning_call").audio_and_sample_rate[0]],
    ]

    with vllm_runner(
        model,
        dtype=dtype,
        enforce_eager=True,
        max_model_len=1024,
        limit_mm_per_prompt={"audio": 1},
        trust_remote_code=True,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            prompts,
            max_tokens,
            num_logprobs=num_logprobs,
            audios=audios,
        )

    with hf_runner(model, dtype=dtype, trust_remote_code=True) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            prompts,
            max_tokens,
            num_logprobs=num_logprobs,
            audios=audios,
        )

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
