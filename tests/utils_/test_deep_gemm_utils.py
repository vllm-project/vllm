# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.utils import deep_gemm


def test_get_paged_mqa_logits_metadata_makes_context_lens_contiguous(monkeypatch):
    def fake_impl(context_lens: torch.Tensor, block_size: int, num_sms: int):
        assert context_lens.is_contiguous()
        assert block_size == 64
        assert num_sms == 132
        return context_lens

    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm, "_get_paged_mqa_logits_metadata_impl", fake_impl)

    base = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    non_contiguous = base[:, 0]
    assert not non_contiguous.is_contiguous()

    out = deep_gemm.get_paged_mqa_logits_metadata(non_contiguous, 64, 132)
    assert out.is_contiguous()
    torch.testing.assert_close(out, torch.tensor([10, 20], dtype=torch.int32))
