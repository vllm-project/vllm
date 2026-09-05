# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The PLE short-conv kernels must honour the stride of ``state_indices``.

With speculative decoding configured the Mamba block table carries
``1 + num_speculative_blocks`` columns per request, and the base Mamba metadata
builder hands the prefill state slots to the layer as the strided column view
``block_table[:, 0]``. A unit-stride load resolved request ``r`` to
``block_table[0, r]`` -- request 0's speculative checkpoint blocks -- so every
request after the first in a prefill-only step wrote its conv state into the
wrong slot and decoded from an unwritten one.
"""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.ple import ple_conv
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Triton kernel")
@pytest.mark.parametrize("num_columns", [2, 4])
def test_prefill_state_store_honours_state_index_stride(num_columns: int):
    torch.manual_seed(0)
    device = "cuda"
    num_reqs, tokens_per_req, channels, kernel = 3, 12, 64, 4
    dilation, state_len = 1, (kernel - 1) * 1
    # Block 0 is the null slot; rows r*num_columns+1 are the primary slots, the
    # other columns stand in for speculative checkpoint blocks and must never
    # receive a prefill state.
    num_blocks = num_reqs * num_columns + 1
    table = torch.arange(1, num_blocks, device=device, dtype=torch.int32).view(
        num_reqs, num_columns
    )
    strided = table[:, 0]
    assert strided.stride(0) == num_columns
    contiguous = strided.contiguous()

    x = torch.randn(
        num_reqs * tokens_per_req, channels, device=device, dtype=torch.bfloat16
    )
    w = torch.randn(channels, kernel, device=device, dtype=torch.bfloat16)
    qsl = torch.arange(
        0,
        (num_reqs + 1) * tokens_per_req,
        tokens_per_req,
        device=device,
        dtype=torch.int32,
    )
    has_init = torch.zeros(num_reqs, device=device, dtype=torch.bool)

    def run(state_indices: torch.Tensor):
        state = torch.zeros(
            num_blocks, channels, state_len, device=device, dtype=torch.bfloat16
        )
        residual = torch.zeros_like(x)
        ple_conv(
            inputs=x,
            residual=residual,
            conv_state=state,
            conv_weights=w,
            state_indices=state_indices,
            mode="prefill",
            dilation=dilation,
            query_start_loc=qsl,
            has_initial_states=has_init,
        )
        return state, residual

    state_ref, out_ref = run(contiguous)
    state_got, out_got = run(strided)
    torch.testing.assert_close(out_got, out_ref)
    torch.testing.assert_close(state_got, state_ref)
    # Every primary slot written, every checkpoint slot untouched.
    for r in range(num_reqs):
        assert state_ref[table[r, 0]].abs().sum() > 0
        for c in range(1, num_columns):
            assert state_ref[table[r, c]].abs().sum() == 0


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Triton kernel")
@pytest.mark.parametrize("num_columns", [2, 4])
def test_decode_state_read_honours_state_index_stride(num_columns: int):
    """Decode reads the previous state through the same lookup; a strided index
    vector must select the same slots as its contiguous copy."""
    torch.manual_seed(1)
    device = "cuda"
    num_reqs, channels, kernel = 3, 64, 4
    dilation, state_len = 1, (kernel - 1) * 1
    num_blocks = num_reqs * num_columns + 1
    table = torch.arange(1, num_blocks, device=device, dtype=torch.int32).view(
        num_reqs, num_columns
    )
    strided = table[:, 0]
    assert strided.stride(0) == num_columns
    contiguous = strided.contiguous()

    x = torch.randn(num_reqs, channels, device=device, dtype=torch.bfloat16)
    w = torch.randn(channels, kernel, device=device, dtype=torch.bfloat16)
    has_init = torch.ones(num_reqs, device=device, dtype=torch.bool)
    # Distinct non-zero history in every slot, so a wrong slot changes the output.
    state_init = torch.randn(
        num_blocks, channels, state_len, device=device, dtype=torch.bfloat16
    )

    def run(state_indices: torch.Tensor):
        state = state_init.clone()
        residual = torch.zeros_like(x)
        ple_conv(
            inputs=x,
            residual=residual,
            conv_state=state,
            conv_weights=w,
            state_indices=state_indices,
            mode="decode",
            dilation=dilation,
            has_initial_states=has_init,
        )
        return state, residual

    state_ref, out_ref = run(contiguous)
    state_got, out_got = run(strided)
    torch.testing.assert_close(out_got, out_ref)
    torch.testing.assert_close(state_got, state_ref)
