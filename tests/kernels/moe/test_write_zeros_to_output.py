# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Functional unit tests for the ``write_zeros_to_output`` Triton kernel.

Source: ``vllm/model_executor/layers/fused_moe/fused_moe.py``

``write_zeros_to_output`` is a ``@triton.jit`` device function called from
``fused_moe_kernel`` / ``fused_moe_kernel_gptq_awq`` when ``off_experts == -1``
(expert not assigned to the current EP rank). It writes zeros into the
corresponding tile of the output buffer.

Because it consumes Triton tensor expressions (``offs_token``, ``token_mask``),
it cannot be called from Python directly — we wrap it in a thin launcher
kernel and compare against a pure-PyTorch reference.

Run with::

    pytest tests/kernels/moe/test_write_zeros_to_output.py
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.fused_moe import write_zeros_to_output
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
    pytest.skip(
        "write_zeros_to_output requires a Triton-capable accelerator "
        "(CUDA/ROCm/XPU)",
        allow_module_level=True,
    )


@triton.jit
def _write_zeros_launcher(
    c_ptr,
    stride_cm,
    stride_cn,
    sorted_token_ids_ptr,
    num_valid_tokens,
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id.to(tl.int64)).to(
        tl.int64
    )
    token_mask = offs_token < num_valid_tokens

    write_zeros_to_output(
        c_ptr,
        stride_cm,
        stride_cn,
        pid_n,
        N,
        offs_token,
        token_mask,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        tl.float32,
    )


def _ref_write_zeros(
    output: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_valid_tokens: int,
) -> torch.Tensor:
    valid_ids = sorted_token_ids[sorted_token_ids < num_valid_tokens]
    output[valid_ids] = 0.0
    return output


def _launch(
    output: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_valid_tokens: int,
    block_m: int,
    block_n: int,
) -> None:
    m, n = output.shape
    assert m % block_m == 0, (
        f"M={m} must be divisible by BLOCK_SIZE_M={block_m}; otherwise the "
        f"grid m // block_m degenerates and the kernel never runs."
    )
    grid = (m // block_m, triton.cdiv(n, block_n))
    _write_zeros_launcher[grid](
        output,
        output.stride(0),
        output.stride(1),
        sorted_token_ids,
        num_valid_tokens,
        n,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
    )


@pytest.fixture
def device() -> str:
    return current_platform.device_type


# Only (m, block_m) pairs where block_m <= m are valid (otherwise the
# Triton grid m // block_m degenerates to 0 and the kernel never runs).
@pytest.mark.parametrize(
    "m, block_m",
    [(16, 16), (64, 16), (64, 64), (256, 16), (256, 64)],
)
@pytest.mark.parametrize("n", [128, 512, 1024])
@pytest.mark.parametrize("block_n", [64, 128])
def test_all_tokens_valid(device, m, n, block_m, block_n):
    """All tokens valid -> the whole output must be zeroed."""
    output = torch.ones((m, n), device=device, dtype=torch.float32)
    sorted_token_ids = torch.arange(m, device=device, dtype=torch.int64)
    ref = output.clone()

    _launch(output, sorted_token_ids, m, block_m, block_n)
    _ref_write_zeros(ref, sorted_token_ids, m)

    torch.testing.assert_close(output, ref, atol=0, rtol=0)


@pytest.mark.parametrize("m", [16, 64, 256])
@pytest.mark.parametrize("n", [128, 512, 1024])
def test_partial_tokens_valid(device, m, n):
    """Half of tokens valid -> matching rows zeroed, the rest untouched."""
    block_m, block_n = 16, 64
    num_valid = m // 2

    sorted_token_ids = torch.arange(m, device=device, dtype=torch.int64)

    output = torch.full((m, n), 42.0, device=device, dtype=torch.float32)
    ref = output.clone()

    _launch(output, sorted_token_ids, num_valid, block_m, block_n)
    _ref_write_zeros(ref, sorted_token_ids, num_valid)

    torch.testing.assert_close(output, ref, atol=0, rtol=0)


@pytest.mark.parametrize("m", [16, 64, 256])
@pytest.mark.parametrize("n", [128, 512, 1024])
def test_no_tokens_valid(device, m, n):
    """No tokens valid -> output must remain unchanged."""
    block_m, block_n = 16, 64
    sorted_token_ids = torch.arange(m, device=device, dtype=torch.int64)

    output = torch.full((m, n), 42.0, device=device, dtype=torch.float32)
    original = output.clone()

    _launch(output, sorted_token_ids, 0, block_m, block_n)

    torch.testing.assert_close(output, original, atol=0, rtol=0)


@pytest.mark.parametrize("m", [64, 256])
@pytest.mark.parametrize("n", [512, 1024])
def test_shuffled_token_ids(device, m, n):
    """Interleaved valid/invalid token ids exercise the scatter pattern."""
    block_m, block_n = 16, 64
    num_valid = m // 2

    sorted_token_ids = torch.randperm(m, device=device, dtype=torch.int64)

    output = torch.full((m, n), 7.0, device=device, dtype=torch.float32)
    ref = output.clone()

    _launch(output, sorted_token_ids, num_valid, block_m, block_n)
    _ref_write_zeros(ref, sorted_token_ids, num_valid)

    torch.testing.assert_close(output, ref, atol=0, rtol=0)


def test_n_not_multiple_of_block_n(device):
    """N not divisible by BLOCK_SIZE_N exercises the column-bound clamp."""
    m, n = 64, 130  # 130 % 64 != 0
    block_m, block_n = 16, 64
    num_valid = m

    sorted_token_ids = torch.arange(m, device=device, dtype=torch.int64)
    output = torch.full((m, n), 3.0, device=device, dtype=torch.float32)
    ref = output.clone()

    _launch(output, sorted_token_ids, num_valid, block_m, block_n)
    _ref_write_zeros(ref, sorted_token_ids, num_valid)

    torch.testing.assert_close(output, ref, atol=0, rtol=0)
