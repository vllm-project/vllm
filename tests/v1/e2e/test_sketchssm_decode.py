# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level check: SketchSSM decode stays close to the default decode."""

import pytest

import vllm.envs as envs

from ...models.utils import check_logprobs_close
from ...utils import large_gpu_mark

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
CALIBRATION = "SketchSSM/Nemotron-3-Nano-4B-BF16"
PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
]


@pytest.fixture(autouse=True)
def _use_v2_model_runner(monkeypatch):
    # SketchSSM requires Model Runner V2.
    with monkeypatch.context() as patch:
        patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
        envs.disable_envs_cache()
        yield
    envs.disable_envs_cache()


@large_gpu_mark(min_gb=40)
@pytest.mark.parametrize("mamba_cache_mode", ["none", "align"])
def test_sketchssm_decode_close_to_baseline(vllm_runner, mamba_cache_mode):
    # 32 tokens cover two W=16 windows, so both flushes and sketch reads run.
    # Compare logprobs, not greedy ids: the sketch read is approximate and can
    # flip a near-tie.
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=mamba_cache_mode != "none",
        mamba_cache_mode=mamba_cache_mode,
        enable_chunked_prefill=True,
        mamba_ssm_cache_dtype="float32",
    )
    with vllm_runner(MODEL, **common) as llm:
        baseline = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)
    with vllm_runner(
        MODEL, sketchssm=CALIBRATION, sketchssm_mean_rank=8, **common
    ) as llm:
        sketch = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)

    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=sketch,
        name_0="baseline",
        name_1="sketchssm",
    )
