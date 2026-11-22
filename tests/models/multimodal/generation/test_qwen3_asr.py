# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-ASR's packed audio attention must preserve clip boundaries."""

import math

import pytest

from vllm.assets.audio import AudioAsset
from vllm.platforms import current_platform

from ....conftest import VllmRunner
from ...utils import check_logprobs_close


@pytest.mark.core_model
@pytest.mark.parametrize(
    "backend",
    [
        "TRITON_ATTN",
        pytest.param(
            "FLASHINFER",
            marks=pytest.mark.skipif(
                not current_platform.is_cuda(), reason="FlashInfer requires CUDA"
            ),
        ),
    ],
)
def test_qwen3_asr_encoder_backends(vllm_runner: type[VllmRunner], backend: str):
    audio, sample_rate = AudioAsset("mary_had_lamb").audio_and_sample_rate
    audios = [(audio, sample_rate), (audio[: int(3 * sample_rate)], sample_rate)]
    prompt = (
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    outputs = []
    for encoder_backend in ("TORCH_SDPA", backend):
        with vllm_runner(
            "Qwen/Qwen3-ASR-0.6B",
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=2048,
            max_num_seqs=2,
            gpu_memory_utilization=0.05,
            kv_cache_memory_bytes=512 * 1024**2,
            attention_config={"backend": "TRITON_ATTN"},
            mm_encoder_attn_backend=encoder_backend,
            limit_mm_per_prompt={"audio": 1},
            enable_prefix_caching=False,
        ) as model:
            result = model.generate_greedy_logprobs(
                [prompt] * len(audios),
                max_tokens=32,
                num_logprobs=5,
                audios=audios,
            )
        assert len(result) == len(audios)
        for tokens, text, logprobs in result:
            assert tokens and text and logprobs
            assert all(
                math.isfinite(value.logprob)
                for position in logprobs
                for value in position.values()
            )
        outputs.append(result)
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="TORCH_SDPA",
        name_1=backend,
    )
