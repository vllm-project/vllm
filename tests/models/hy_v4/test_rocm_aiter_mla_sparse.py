# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 ROCm draft rows: AITER sparse MLA reads the first `paged_kv_indptr`
delta entries of each row and maps -1 to slot 0, so the DFly draft's rows must
be front-packed and counted before every layer reads them.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("HYV4 ROCm sparse MLA", allow_module_level=True)

from vllm.models.hy_v4.amd import aiter_mla_sparse
from vllm.models.hy_v4.amd.base_model import HYV4DecoderLayer
from vllm.models.hy_v4.amd.base_model import HYV4ForCausalLM as AMDHYV4ForCausalLM
from vllm.models.hy_v4.amd.vl_model import HYV4VLForConditionalGeneration
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseImpl,
)

DEVICE = "cuda"


def _assert_front_packed(rows: torch.Tensor, counts: torch.Tensor) -> None:
    columns = torch.arange(rows.shape[1], device=rows.device)
    head = columns < counts[:, None].long()
    assert torch.all((rows >= 0) == head)


def _stable_pack_reference(rows: torch.Tensor) -> torch.Tensor:
    packed = torch.full_like(rows, -1)
    for i, row in enumerate(rows):
        live = row[row >= 0]
        packed[i, : live.numel()] = live
    return packed


@pytest.mark.parametrize("width", [2176, 300])
def test_pack_live_indices_is_stable_compaction(width):
    torch.manual_seed(0)
    storage = torch.randint(0, 1 << 20, (9, width + 32), device=DEVICE).int()
    storage.masked_fill_(torch.rand_like(storage, dtype=torch.float) < 0.6, -1)
    storage[3] = -1
    storage[5] = torch.arange(width + 32, device=DEVICE)
    original = storage.clone()
    rows = storage[:, :width]

    counts = aiter_mla_sparse.pack_live_indices_(rows)

    expected = _stable_pack_reference(original[:, :width])
    torch.testing.assert_close(rows, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        counts, (original[:, :width] >= 0).sum(-1).int(), rtol=0, atol=0
    )
    torch.testing.assert_close(storage[:, width:], original[:, width:], rtol=0, atol=0)


def test_packed_impl_rebuilds_indptr_from_live_counts(monkeypatch):
    monkeypatch.setattr(
        ROCMAiterMLASparseImpl, "forward_mqa", lambda *args: "forwarded"
    )
    torch.manual_seed(0)
    buffer = torch.randint(0, 4096, (8, 256), device=DEVICE).int()
    buffer.masked_fill_(torch.rand_like(buffer, dtype=torch.float) < 0.5, -1)
    buffer[6:] = 7
    original = buffer.clone()
    num_tokens = 5
    indptr = torch.full((10,), -3, dtype=torch.int32, device=DEVICE)
    indptr[0] = 0
    impl = object.__new__(aiter_mla_sparse.HYV4AiterMLASparseImpl)
    impl.topk_indices_buffer = buffer
    q = torch.empty(num_tokens, 16, 576, device=DEVICE)

    result = impl.forward_mqa(q, None, SimpleNamespace(paged_kv_indptr=indptr), None)

    assert result == "forwarded"
    counts = (original[:num_tokens] >= 0).sum(-1).int()
    expected = torch.zeros(10, dtype=torch.int32, device=DEVICE)
    expected[1 : num_tokens + 1] = torch.cumsum(counts, 0)
    expected[num_tokens + 1 :] = expected[num_tokens]
    torch.testing.assert_close(indptr, expected, rtol=0, atol=0)
    _assert_front_packed(buffer[:num_tokens], counts)
    torch.testing.assert_close(buffer[num_tokens:], original[num_tokens:])


def test_vl_language_model_uses_amd_attention():
    """The VL text model must share the AMD attention class DFly reuses."""
    from vllm.models.hy_v4.amd.attention import HYV4MLAAttention

    assert HYV4VLForConditionalGeneration.language_model_cls is AMDHYV4ForCausalLM
    assert HYV4DecoderLayer.attention_cls is HYV4MLAAttention
