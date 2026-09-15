# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.warmup.qwen_triton_warmup as qwen_triton_warmup_module
from vllm.model_executor.warmup.qwen_triton_warmup import (
    _FLA_POST_CONV_WARMUP_LENGTHS,
    _QwenGDNWarmupConfig,
    _warm_causal_conv1d_fwd_kernel,
    _warm_fused_post_conv_kernel,
    _warm_gated_rms_norm_kernel,
)
from vllm.platforms import current_platform


def _cuda_gdn_config() -> _QwenGDNWarmupConfig:
    h, hv, k, v = 2, 2, 16, 16
    conv_kernel_size = 4
    conv_dim = 2 * h * k + hv * v
    device = torch.device("cuda")
    conv_state = torch.empty(
        (8, conv_dim, conv_kernel_size - 1),
        dtype=torch.bfloat16,
        device=device,
    )
    return _QwenGDNWarmupConfig(
        h=h,
        hv=hv,
        k=k,
        v=v,
        conv_kernel_size=conv_kernel_size,
        conv_state=conv_state,
        conv_dtype=conv_state.dtype,
        norm_weight_dtype=torch.bfloat16,
        norm_before_gate=True,
        norm_activation="silu",
        a_log=torch.zeros(hv, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(hv, dtype=torch.float32, device=device),
        state_stride_token=hv * v * k,
        state_dtype=torch.float32,
    )


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="CUDA is required")
def test_qwen_gdn_prefill_warmup_kernels_compile_on_gpu() -> None:
    config = _cuda_gdn_config()
    device = torch.device("cuda")
    _warm_gated_rms_norm_kernel(
        device, config, max_num_tokens=16, x_dtype=config.conv_dtype
    )
    _warm_causal_conv1d_fwd_kernel(device, config)
    _warm_fused_post_conv_kernel(device, config)
    assert _FLA_POST_CONV_WARMUP_LENGTHS == (1, 2, 16)
    torch.accelerator.synchronize(device)


def _warmup_runner() -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        compilation_config=SimpleNamespace(static_forward_context={}),
        max_num_tokens=1,
        is_pooling_model=False,
    )


def _warmup_model_config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type=model_type),
        hf_config=SimpleNamespace(model_type=model_type),
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize(
    ("model_type", "dispatched"),
    [
        ("qwen4_exp", True),
        ("qwen4_exp_text", True),
        ("qwen3_next", True),
        # The drafter never reaches this warmup: the worker passes its own
        # model_config, which is the target's.
        ("qwen4_exp_mtp", False),
        ("llama", False),
    ],
)
def test_qwen_triton_warmup_dispatch_gate(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    dispatched: bool,
) -> None:
    """Which model types reach the Qwen Triton warmup at all.

    Stops at the gate: the kernels themselves need a GPU and are covered by the
    CUDA test above.
    """
    reached: list[object] = []

    def _record(static_forward_context: object) -> None:
        reached.append(static_forward_context)
        return None

    monkeypatch.setattr(qwen_triton_warmup_module, "_qwen_gdn_warmup_config", _record)

    qwen_triton_warmup_module.qwen_triton_warmup(
        _warmup_runner(), _warmup_model_config(model_type)
    )

    assert bool(reached) is dispatched
