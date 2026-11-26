# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY V4 BF16 sparse MLA: decode rows on FlashInfer's trtllm-gen kernel match
FlashMLA's sparse kernel, alone and ahead of prefill rows in a mixed batch."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

if not (current_platform.is_device_capability_family(100) and has_flashinfer()):
    pytest.skip("needs SM100 and FlashInfer", allow_module_level=True)

from vllm.models.hy_v4.nvidia.flashmla_sparse import (  # noqa: E402
    HYV4FlashMLASparseImpl,
)

HEADS, DIM, NUM_KV, TOPK = 8, 576, 1 << 16, 2048


def _impl(sinks: torch.Tensor | None) -> HYV4FlashMLASparseImpl:
    # Only the attributes the BF16 kernels read.
    impl = object.__new__(HYV4FlashMLASparseImpl)
    impl.sinks, impl._sinks_full = sinks, None
    impl.softmax_scale = 0.07
    impl.kv_lora_rank, impl.qk_nope_head_dim, impl.qk_rope_head_dim = 512, 192, 64
    impl.prefill_padding = 64
    impl._decode_rows, impl._decode_page_rows = 0, 64
    return impl


@pytest.mark.parametrize("num_decode,num_prefill", [(1, 0), (16, 0), (4, 37)])
@pytest.mark.parametrize("with_sink", [True, False])
def test_flashinfer_decode_matches_flashmla(num_decode, num_prefill, with_sink):
    torch.manual_seed(0)
    num_tokens = num_decode + num_prefill
    kv = (torch.randn(NUM_KV, DIM, device="cuda") * 0.5).bfloat16()
    q = (torch.randn(num_tokens, HEADS, DIM, device="cuda") * 0.5).bfloat16()
    indices = torch.randint(
        0, NUM_KV, (num_tokens, TOPK), device="cuda", dtype=torch.int32
    )
    lengths = torch.randint(
        1, TOPK + 1, (num_tokens,), device="cuda", dtype=torch.int32
    )
    lengths[0] = TOPK
    # Entries past a row's length are -1, as the index conversion leaves them.
    indices.masked_fill_(torch.arange(TOPK, device="cuda") >= lengths[:, None], -1)
    sinks = torch.randn(HEADS, device="cuda") if with_sink else None
    impl = _impl(sinks)

    ref, _ = impl._bf16_flash_mla_kernel(q, kv, indices, lengths, HEADS)
    impl._decode_rows = num_decode
    out, lse = impl._bf16_flash_mla_kernel(q, kv, indices, lengths, HEADS)

    assert lse is None and out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=2**-7, atol=2e-3)
