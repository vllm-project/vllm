# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm.platforms import current_platform
from vllm.v1.attention.ops import vit_attn_wrappers


def _noncontiguous_bshd(device: torch.device | None = None) -> torch.Tensor:
    return (
        torch.arange(120, dtype=torch.float32, device=device)
        .reshape(1, 3, 5, 8)
        .transpose(1, 2)
    )


def test_registered_attention_wrapper_fake_layouts():
    with FakeTensorMode():
        q = _noncontiguous_bshd()
        cu_seqlens = torch.tensor([0, 5], dtype=torch.int32)
        max_seqlen = torch.tensor(5, dtype=torch.int32)
        flashinfer_cu_seqlens = torch.tensor([0, 5, 0, 5], dtype=torch.int32)
        sequence_lengths = torch.tensor([5], dtype=torch.int32)

        outputs = (
            torch.ops.vllm.flash_attn_maxseqlen_wrapper(
                q, q, q, 1, False, None, None, cu_seqlens, max_seqlen
            ),
            torch.ops.vllm.triton_attn_wrapper(
                q, q, q, 1, None, cu_seqlens, max_seqlen
            ),
            torch.ops.vllm.triton_attn_wrapper(
                q, q, q, 1, None, cu_seqlens, max_seqlen, sequence_lengths
            ),
            torch.ops.vllm.torch_sdpa_wrapper(q, q, q, None, None, False),
            torch.ops.vllm.flashinfer_wrapper(
                q,
                q,
                q,
                1.0,
                torch.empty(1, dtype=torch.uint8),
                flashinfer_cu_seqlens,
                max_seqlen,
                sequence_lengths,
                None,
                None,
                None,
                None,
            ),
        )

        assert all(output.is_contiguous() for output in outputs)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("lengths", [[5, 13, 0], [0, 1, 7, 0]])
def test_triton_precomputed_lengths_match_recomputed(lengths: list[int]):
    torch.manual_seed(0)
    q, k, v = torch.randn(
        3, 1, sum(lengths), 2, 64, device="cuda", dtype=torch.bfloat16
    ).unbind()
    sequence_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    cu_seqlens = torch.cat([sequence_lengths.new_zeros(1), sequence_lengths]).cumsum(
        0, dtype=torch.int32
    )
    max_seqlen = torch.tensor(max(lengths), device="cpu", dtype=torch.int32)
    args = (q, k, v, 1, None, cu_seqlens, max_seqlen)
    expected = vit_attn_wrappers.vit_triton_attn_wrapper(*args)
    actual = vit_attn_wrappers.vit_triton_attn_wrapper(
        *args, sequence_lengths=sequence_lengths
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_aiter_flash_attn_omits_out_and_returns_contiguous_output(
    monkeypatch: pytest.MonkeyPatch,
):
    aiter = ModuleType("aiter")

    def flash_attn_varlen_func(q, k, v, **kwargs):
        assert "out" not in kwargs
        return q

    aiter.__dict__["flash_attn_varlen_func"] = flash_attn_varlen_func
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setattr(
        vit_attn_wrappers, "current_platform", SimpleNamespace(is_rocm=lambda: True)
    )
    q = _noncontiguous_bshd()
    assert not q.is_contiguous()

    output = vit_attn_wrappers.flash_attn_maxseqlen_wrapper(q, q, q, 1, True, None)

    assert torch.equal(output, q)
    assert output.is_contiguous()


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Requires GPU",
)
def test_sdpa_output_layout_without_cu_seqlens():
    q = _noncontiguous_bshd(torch.device(current_platform.device_type))

    output = vit_attn_wrappers.vit_torch_sdpa_wrapper(q, q, q)

    assert output.is_contiguous()
