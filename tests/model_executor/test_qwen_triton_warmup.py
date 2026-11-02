# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import vllm.third_party.flash_linear_attention.ops.fused_gdn_decode_post_conv_mtp as mtp_ops  # noqa: E501
from vllm.model_executor.warmup.qwen_triton_warmup import (
    _FLA_POST_CONV_WARMUP_LENGTHS,
    _QwenGDNWarmupConfig,
    _warm_causal_conv1d_fwd_kernel,
    _warm_fused_gdn_decode_post_conv_mtp_kernel,
    _warm_fused_post_conv_kernel,
    _warm_gated_rms_norm_kernel,
)
from vllm.platforms import current_platform
from vllm.utils.platform_utils import num_compute_units


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


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only Triton kernel")
@pytest.mark.parametrize("max_num_reqs", [1, 1024])
def test_qwen_gdn_fused_mtp_warmup_covers_reachable_tile_sizes(
    max_num_reqs: int,
) -> None:
    """The MTP decode warmup compiles every V tile size the wrapper can pick
    for up to max_num_reqs requests."""
    k = v = 128
    config = replace(
        _cuda_gdn_config(),
        k=k,
        v=v,
        state_stride_token=2 * v * k,
        fused_spec_decode=True,
    )
    device = torch.device("cuda")
    kernel = mtp_ops.fused_gdn_decode_post_conv_mtp_triton
    with patch.object(
        mtp_ops, "fused_gdn_decode_post_conv_mtp_triton", wraps=kernel
    ) as wrapped:
        _warm_fused_gdn_decode_post_conv_mtp_kernel(
            device,
            config,
            torch.bfloat16,
            num_spec_tokens=3,
            max_num_reqs=max_num_reqs,
        )
    torch.accelerator.synchronize(device)

    num_cus = num_compute_units(device.index)
    tiles = set()
    for call in wrapped.call_args_list:
        num_reqs = call.kwargs["cu_seqlens"].numel() - 1
        assert num_reqs <= max_num_reqs
        heads = num_reqs * config.hv
        tiles.add(32 if heads < num_cus else 64 if heads <= 4 * num_cus else v)
    assert tiles == ({32} if max_num_reqs == 1 else {32, 64, v})
