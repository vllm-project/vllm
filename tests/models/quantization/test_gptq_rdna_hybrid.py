# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Symmetric GPTQ model on the RDNA hybrid W4A16 kernel vs. the Triton kernel."""

import pytest

from vllm.model_executor.kernels.linear.mixed_precision import (
    RDNAHybridW4A16LinearKernel,
    TritonW4A16LinearKernel,
)
from vllm.platforms import current_platform

from ..utils import check_logprobs_close

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.platforms.rocm import on_gfx1x, on_gfx1100  # noqa: E402

MODEL = "JunHowie/Qwen3-0.6B-GPTQ-Int4"
MAX_MODEL_LEN = 1024


def _kernel_name(model) -> str:
    return type(model.model.layers[0].self_attn.qkv_proj.quant_method.kernel).__name__


def _run(vllm_runner, example_prompts, max_tokens, num_logprobs, expected_kernel):
    with vllm_runner(
        MODEL,
        quantization="gptq",
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
    ) as model:
        assert model.apply_model(_kernel_name) == [expected_kernel.__name__]
        return model.generate_greedy_logprobs(example_prompts, max_tokens, num_logprobs)


@pytest.mark.skipif(
    not on_gfx1x() or on_gfx1100(), reason="Needs the RDNA hybrid kernel (gfx11/gfx12)"
)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_gptq_rdna_hybrid_matches_triton(
    vllm_runner,
    example_prompts,
    monkeypatch,
    enable_pickle,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    hybrid_outputs = _run(
        vllm_runner,
        example_prompts,
        max_tokens,
        num_logprobs,
        RDNAHybridW4A16LinearKernel,
    )

    monkeypatch.setenv("VLLM_DISABLED_KERNELS", RDNAHybridW4A16LinearKernel.__name__)
    triton_outputs = _run(
        vllm_runner, example_prompts, max_tokens, num_logprobs, TritonW4A16LinearKernel
    )

    check_logprobs_close(
        outputs_0_lst=triton_outputs,
        outputs_1_lst=hybrid_outputs,
        name_0="triton",
        name_1="rdna_hybrid",
    )
