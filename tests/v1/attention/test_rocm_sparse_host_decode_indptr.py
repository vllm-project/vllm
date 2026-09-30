# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The host-side pure-decode indptr matches the device seqlen kernel."""

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    generate_sparse_seqlen_triton,
)


def _on_rocm_gpu() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_rocm() and torch.cuda.is_available()


pytestmark = pytest.mark.skipif(not _on_rocm_gpu(), reason="ROCm sparse MLA only")


@pytest.mark.parametrize("topk", [2048, 64])
def test_host_decode_indptr_matches_device_seqlen_kernel(topk):
    # Uneven decode batch: one token per request, contexts on both sides of topk.
    seq_lens = torch.tensor(
        [1, 5, 63, 64, 65, 2047, 2048, 2049, 131072], dtype=torch.int32
    )
    n = seq_lens.numel()
    device_seqlen = generate_sparse_seqlen_triton(
        torch.ones(n, dtype=torch.int32, device="cuda"),
        seq_lens.cuda(),
        torch.arange(n + 1, dtype=torch.int32, device="cuda"),
        topk,
        n,
        1,
    )
    expected = torch.cumsum(device_seqlen, 0).cpu().to(torch.int64)
    host = np.cumsum(np.minimum(seq_lens.numpy(), topk))
    torch.testing.assert_close(torch.from_numpy(host).to(torch.int64), expected)
