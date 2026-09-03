# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end check of the fused Triton iHC ops inside the HY V4 model.

The public checkpoint is ~800B parameters, so this runs a one-layer dummy-weight
model (same hidden size / hc_mult as ``tencent/Hy4-preview``) through the full
engine, CUDA graphs included, and compares greedy logprobs of the fused path
against the eager ``forward_native`` path.
"""

from functools import partial

import pytest

from vllm.models.hy_v4.nvidia import hc as hc_mod
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

from ..utils import check_logprobs_close, dummy_hf_overrides

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="iHC kernels require CUDA and Triton",
)

MODEL = "tencent/Hy4-preview"
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Hyper-connections replace the residual stream with",
    "In 2026, the most popular inference engine",
]


def _greedy_logprobs(vllm_runner, monkeypatch: pytest.MonkeyPatch, fused: bool):
    calls = 0
    real_ihc_pre = hc_mod.ihc_pre

    def counting_pre(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_ihc_pre(*args, **kwargs)

    with monkeypatch.context() as m:
        # Run the engine in-process so the patches below reach the model.
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        if fused:
            m.setattr(hc_mod, "ihc_pre", counting_pre)
        else:
            m.setattr(hc_mod, "_triton_ihc_supported", lambda hc_mult: False)
        with vllm_runner(
            MODEL,
            trust_remote_code=True,
            load_format="dummy",
            hf_overrides=partial(dummy_hf_overrides, model_arch="HYV4ForCausalLM"),
            max_model_len=256,
            max_num_seqs=4,
            block_size=64,  # sparse MLA (FlashMLA) kernel block size
            gpu_memory_utilization=0.6,
            enforce_eager=False,
        ) as vllm_model:
            outputs = vllm_model.generate_greedy_logprobs(
                PROMPTS, max_tokens=8, num_logprobs=5
            )
    if fused:
        assert calls > 0, "fused iHC op was not dispatched"
    return outputs


def test_ihc_fused_matches_native_e2e(vllm_runner, monkeypatch: pytest.MonkeyPatch):
    native = _greedy_logprobs(vllm_runner, monkeypatch, fused=False)
    fused = _greedy_logprobs(vllm_runner, monkeypatch, fused=True)
    check_logprobs_close(
        outputs_0_lst=native,
        outputs_1_lst=fused,
        name_0="native",
        name_1="triton",
    )
