# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.assets.audio import AudioAsset
from vllm.model_executor.models.moss_audio import MOSS_AUDIO_PLACEHOLDER
from vllm.platforms import current_platform

from ...registry import HF_EXAMPLE_MODELS
from ...utils import check_logprobs_close

MODELS = [
    "OpenMOSS-Team/MOSS-Audio-4B-Instruct",
    "OpenMOSS-Team/MOSS-Audio-4B-Thinking",
    "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
    "OpenMOSS-Team/MOSS-Audio-8B-Thinking",
]

PARALLEL_SMOKE_CASES = [
    pytest.param({"tensor_parallel_size": 2}, id="tp2"),
    pytest.param({"pipeline_parallel_size": 2}, id="pp2"),
    pytest.param(
        {"tensor_parallel_size": 2, "pipeline_parallel_size": 2},
        id="tp2_pp2",
    ),
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


@pytest.mark.core_model
@pytest.mark.parametrize("parallel_kwargs", PARALLEL_SMOKE_CASES)
def test_moss_audio_parallel_smoke(vllm_runner, parallel_kwargs) -> None:
    model = "OpenMOSS-Team/MOSS-Audio-4B-Instruct"
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    required_gpus = parallel_kwargs.get(
        "tensor_parallel_size", 1
    ) * parallel_kwargs.get("pipeline_parallel_size", 1)
    if current_platform.device_count() < required_gpus:
        # TP/PP integration smoke runs on local or multi-GPU CI only.
        pytest.skip(f"Requires at least {required_gpus} GPUs")

    prompts = [f"{MOSS_AUDIO_PLACEHOLDER}\nBriefly describe this audio."]
    audios = [[AudioAsset("mary_had_lamb").audio_and_sample_rate[0]]]

    with vllm_runner(
        model,
        dtype="half",
        enforce_eager=True,
        max_model_len=1024,
        limit_mm_per_prompt={"audio": 1},
        trust_remote_code=True,
        **parallel_kwargs,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            prompts,
            max_tokens=4,
            audios=audios,
        )

    assert len(outputs) == 1
    assert len(outputs[0][1]) > 0
