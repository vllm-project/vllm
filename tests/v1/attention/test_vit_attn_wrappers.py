# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="Requires GPU")
def test_sdpa_output_layout_without_cu_seqlens():
    q = _noncontiguous_bshd(torch.device("cuda"))

    output = vit_attn_wrappers.vit_torch_sdpa_wrapper(q, q, q)

    assert output.is_contiguous()
