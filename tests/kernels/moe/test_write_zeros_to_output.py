# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the write_zeros_to_output MoE Triton kernel.

Run `pytest tests/kernels/moe/test_write_zeros_to_output.py`.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.fused_moe import write_zeros_to_output
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import set_random_seed

if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
    pytest.skip(
        "write_zeros_to_output requires a CUDA-alike or XPU device",
        allow_module_level=True,
    )

DEVICE = current_platform.device_type

DTYPE = torch.bfloat16

TL_DTYPE = tl.bfloat16


@triton.jit
def _write_zeros_launcher(
    c_ptr,
    stride_cm,
    stride_cn,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_valid_tokens,
    N,
    EM,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMPUTE_TYPE: tl.constexpr,
):
    # A device function needs a kernel wrapper. Keep the grouped 1-D pid mapping
    # and the gate below: a plain 2-D grid would stop matching the call site.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
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
            COMPUTE_TYPE,
        )


def _launch(
    output,
    sorted_token_ids,
    expert_ids,
    num_valid_tokens,
    block_m,
    block_n,
    group_size_m,
):
    em = sorted_token_ids.numel()
    # A harness constraint, not the kernel's.
    assert em % block_m == 0, f"em={em} must be a multiple of block_m={block_m}"
    num_pid_m = em // block_m
    # expert_ids is loaded unmasked, so a short tensor would read out of bounds.
    assert expert_ids.numel() == num_pid_m, (
        f"expert_ids has {expert_ids.numel()} entries, grid needs {num_pid_m}"
    )
    grid = (num_pid_m * triton.cdiv(output.size(1), block_n),)
    _write_zeros_launcher[grid](
        output,
        output.stride(0),
        output.stride(1),
        sorted_token_ids,
        expert_ids,
        num_valid_tokens,
        output.size(1),
        em,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        GROUP_SIZE_M=group_size_m,
        # Zeros convert exactly to any float dtype, so this cannot affect the result.
        COMPUTE_TYPE=TL_DTYPE,
    )


def _ref_write_zeros(output, sorted_token_ids, expert_ids, num_valid_tokens, block_m):
    gated = (expert_ids == -1).repeat_interleave(block_m)
    gated &= sorted_token_ids < num_valid_tokens
    output[sorted_token_ids[gated].long()] = 0


def _expert_ids(num_pid_m, absent):
    ids = torch.zeros(num_pid_m, device=DEVICE, dtype=torch.int32)
    ids[absent] = -1
    return ids


def _all_absent(num_pid_m):
    return _expert_ids(num_pid_m, slice(None))


def _padded_sorted_ids(num_valid, em, seed=0):
    """Valid ids scattered over em slots, the rest holding the padding sentinel."""
    set_random_seed(seed)
    ids = torch.full((em,), num_valid, device=DEVICE, dtype=torch.int32)
    slots = torch.randperm(em, device=DEVICE)[:num_valid]
    ids[slots] = torch.randperm(num_valid, device=DEVICE, dtype=torch.int32)
    return ids


# GROUP_SIZE_M values are the ones the production heuristics pick (1, 8, 16, 32).
# A value above num_pid_m exercises the tail group, where group_size_m clamps.
@pytest.mark.parametrize("group_size_m", [1, 32])
@pytest.mark.parametrize(
    "m, n, block_m, block_n", [(16, 128, 16, 128), (256, 1024, 64, 64)]
)
@pytest.mark.parametrize("valid_frac", [1.0, 0.5, 0.0], ids=["all", "half", "none"])
@torch.inference_mode()
def test_token_mask_selects_valid_rows(
    m, n, block_m, block_n, valid_frac, group_size_m
):
    """Only rows whose id is below num_valid_tokens may be zeroed."""
    num_valid = int(m * valid_frac)
    sorted_token_ids = torch.arange(m, device=DEVICE, dtype=torch.int32)
    expert_ids = _all_absent(m // block_m)
    output = torch.full((m, n), 42.0, device=DEVICE, dtype=DTYPE)
    ref = output.clone()

    _launch(
        output, sorted_token_ids, expert_ids, num_valid, block_m, block_n, group_size_m
    )
    _ref_write_zeros(ref, sorted_token_ids, expert_ids, num_valid, block_m)

    torch.testing.assert_close(output, ref, atol=0, rtol=0)


# With block_m=16 these em values give 2, 8 and 12 pid_m blocks, so GROUP_SIZE_M=8
# covers a clamped tail, a single full group, and a full group plus a partial tail.
@pytest.mark.parametrize("group_size_m", [1, 8])
@pytest.mark.parametrize("num_valid, em", [(24, 32), (100, 128), (150, 192)])
@torch.inference_mode()
def test_padding_sentinel_ids(num_valid, em, group_size_m):
    """Padded ids interleaved with valid ones must be masked out."""
    n = 512
    block_m, block_n = 16, 64

    sorted_token_ids = _padded_sorted_ids(num_valid, em)
    expert_ids = _all_absent(em // block_m)
    output = torch.full((em, n), 7.0, device=DEVICE, dtype=DTYPE)
    ref = output.clone()

    _launch(
        output, sorted_token_ids, expert_ids, num_valid, block_m, block_n, group_size_m
    )
    _ref_write_zeros(ref, sorted_token_ids, expert_ids, num_valid, block_m)

    torch.testing.assert_close(output, ref, atol=0, rtol=0)


@pytest.mark.parametrize("n", [130, 200])
@torch.inference_mode()
def test_n_not_multiple_of_block_n(n):
    """A partial tail block must not write past column N."""
    m = 64
    block_m, block_n = 16, 64

    # Spare columns beyond N catch an overrun even with every row zeroed.
    buf = torch.full((m, n + block_n), 3.0, device=DEVICE, dtype=DTYPE)
    output = buf[:, :n]
    sorted_token_ids = torch.arange(m, device=DEVICE, dtype=torch.int32)
    expert_ids = _all_absent(m // block_m)
    ref_buf = buf.clone()

    _launch(output, sorted_token_ids, expert_ids, m, block_m, block_n, 1)
    _ref_write_zeros(ref_buf[:, :n], sorted_token_ids, expert_ids, m, block_m)

    torch.testing.assert_close(buf, ref_buf, atol=0, rtol=0)


@torch.inference_mode()
def test_non_contiguous_output():
    """Both strides are kernel arguments, so a strided output view must work."""
    m, n = 64, 128
    block_m, block_n = 16, 64

    buf = torch.full((n, m * 2), 5.0, device=DEVICE, dtype=DTYPE)
    output = buf[:, ::2].t()
    assert output.stride(0) != 1 and output.stride(1) != 1
    sorted_token_ids = torch.arange(m, device=DEVICE, dtype=torch.int32)
    expert_ids = _all_absent(m // block_m)
    ref_buf = buf.clone()

    _launch(output, sorted_token_ids, expert_ids, m, block_m, block_n, 1)
    _ref_write_zeros(ref_buf[:, ::2].t(), sorted_token_ids, expert_ids, m, block_m)

    torch.testing.assert_close(buf, ref_buf, atol=0, rtol=0)


@pytest.mark.parametrize(
    "absent",
    [slice(None), slice(0, 0), slice(None, None, 2)],
    ids=["all", "none", "alt"],
)
@torch.inference_mode()
def test_only_blocks_without_expert_are_zeroed(absent):
    """Only blocks whose expert id is -1 may be zeroed.

    Ids are scattered and partly padding, so the gate has to combine with the
    token mask.
    """
    em, n = 64, 128
    block_m, block_n = 16, 64
    num_valid = 40

    sorted_token_ids = _padded_sorted_ids(num_valid, em)
    expert_ids = _expert_ids(em // block_m, absent)
    output = torch.full((em, n), 11.0, device=DEVICE, dtype=DTYPE)
    ref = output.clone()

    _launch(output, sorted_token_ids, expert_ids, num_valid, block_m, block_n, 1)
    _ref_write_zeros(ref, sorted_token_ids, expert_ids, num_valid, block_m)

    torch.testing.assert_close(output, ref, atol=0, rtol=0)
