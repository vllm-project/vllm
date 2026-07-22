# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform

requires_flashinfer_mla = pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="FlashInfer MLA requires compute capability 10 or above.",
)


@requires_flashinfer_mla
def test_flashinfer_mla_masks_empty_dcp_shards():
    from vllm.v1.attention.backends.mla.flashinfer_mla import (
        _flatten_and_mask_empty_kv_rows,
    )

    output = torch.ones(2, 2, 4, 8, dtype=torch.bfloat16)
    output[0].fill_(float("nan"))
    lse = torch.ones(4, 4, dtype=torch.float32)
    seq_lens = torch.tensor([0, 1], dtype=torch.int32)

    output, lse = _flatten_and_mask_empty_kv_rows(
        output, lse, seq_lens, mask_empty_shards=True
    )

    torch.testing.assert_close(output[:2], torch.zeros_like(output[:2]))
    torch.testing.assert_close(output[2:], torch.ones_like(output[2:]))
    assert lse is not None
    assert torch.isneginf(lse[:2]).all()
    torch.testing.assert_close(lse[2:], torch.ones_like(lse[2:]))


@requires_flashinfer_mla
def test_flashinfer_mla_skips_masking_with_direct_a2a():
    """Direct A2A only needs the output flattened."""
    from vllm.v1.attention.backends.mla.flashinfer_mla import (
        _flatten_and_mask_empty_kv_rows,
    )

    output = torch.ones(2, 2, 4, 8, dtype=torch.bfloat16)
    lse = torch.ones(4, 4, dtype=torch.float32)
    seq_lens = torch.tensor([0, 1], dtype=torch.int32)

    output, lse = _flatten_and_mask_empty_kv_rows(
        output, lse, seq_lens, mask_empty_shards=False
    )

    assert output.shape == (4, 4, 8)
    torch.testing.assert_close(output, torch.ones_like(output))
    assert lse is not None and lse.shape == (4, 4)
    torch.testing.assert_close(lse, torch.ones_like(lse))


@requires_flashinfer_mla
def test_flashinfer_mla_selects_backend_for_gathered_heads():
    from vllm.v1.attention.backends.mla.flashinfer_mla import (
        _select_mla_decode_backend,
    )

    assert _select_mla_decode_backend(6) is None
    assert _select_mla_decode_backend(24) == "cute-dsl"


@requires_flashinfer_mla
def test_flashinfer_mla_forward_uses_gathered_head_count(monkeypatch):
    import vllm.v1.attention.backends.mla.flashinfer_mla as flashinfer_mla

    impl = MagicMock()
    impl.bmm1_scale = 1.0
    impl.bmm2_scale = 1.0
    impl.need_to_return_lse_for_decode = True
    impl.num_heads = 6
    impl.qk_nope_head_dim = 128
    impl.kv_lora_rank = 512
    impl.qk_rope_head_dim = 64
    impl.kv_cache_dtype = "auto"

    attn_metadata = MagicMock()
    attn_metadata.causal = True
    attn_metadata.num_decode_tokens = 2
    attn_metadata.num_decodes = 2
    attn_metadata.max_seq_len = 1
    attn_metadata.decode.block_table = torch.zeros(2, 1, dtype=torch.int32)
    attn_metadata.decode.seq_lens = torch.ones(2, dtype=torch.int32)

    query = torch.ones(2, 24, 576, dtype=torch.bfloat16)
    kv_cache = torch.ones(1, 128, 576, dtype=torch.bfloat16)
    kernel = MagicMock(
        return_value=(
            torch.ones(2, 1, 24, 512, dtype=torch.bfloat16),
            torch.ones(2, 24, dtype=torch.float32),
        )
    )
    monkeypatch.setattr(flashinfer_mla, "_get_workspace_buffer", MagicMock())
    monkeypatch.setattr(flashinfer_mla, "trtllm_batch_decode_with_kv_cache_mla", kernel)

    layer = MagicMock(dcp_combine_masks_empty_shards=False)
    output, lse = flashinfer_mla.FlashInferMLAImpl.forward_mqa(
        impl, query, kv_cache, attn_metadata, layer
    )

    assert kernel.call_args.kwargs["backend"] == "cute-dsl"
    assert output.shape == (2, 24, 512)
    assert lse is not None and lse.shape == (2, 24)
