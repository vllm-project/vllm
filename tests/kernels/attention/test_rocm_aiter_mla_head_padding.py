# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Head padding and kernel selection for the ROCm AITER MLA backend.

The asm persistent decode requires a 16-aligned head count, so unaligned
counts through 128 are tile-padded to the next multiple of 16 and sliced back
off the output. Small divisor counts (1/2/4/8) preserve their existing
repeat-interleave path and may keep the Gluon kernel on gfx950.
"""

import math
import types

import pytest
import torch
import torch.nn.functional as F

from vllm._aiter_ops import is_aiter_found
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla import rocm_aiter_mla
from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
    AiterMLADecodeMetadata,
    AiterMLAHelper,
    AiterMLAImpl,
    AiterMLAMetadata,
)

NUM_HEADS = 12
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
CONTEXT_LEN = 4096
SCALE = 1.0 / math.sqrt(QK_HEAD_DIM)

# Small non-divisor counts use tile-and-slice; divisors of 16 keep
# repeat_interleave. Both pad to exactly 16 and round-trip.
NON_DIVISOR_HEADS = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
DIVISOR_HEADS = [1, 2, 4, 8]


@pytest.fixture(autouse=True)
def _disable_native_h24(monkeypatch):
    """Exercise the padding fallback unless a test explicitly enables H24."""
    monkeypatch.setattr(
        rocm_aiter_mla, "_aiter_mla_native_h24_supported", lambda: False
    )


def _rocm_aiter_available() -> bool:
    return current_platform.is_rocm() and is_aiter_found() and torch.cuda.is_available()


def _on_gfx950() -> bool:
    if not (current_platform.is_rocm() and is_aiter_found()):
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def _expected_tile_pad(q: torch.Tensor, num_heads: int, m: int = 16) -> torch.Tensor:
    # get_mla_padded_q tiles the heads and slices to m, i.e. head i of the
    # padded tensor is head (i % num_heads) of the input.
    idx = [i % num_heads for i in range(m)]
    return q[:, idx, :]


def _make_h12_decode_metadata(device: torch.device) -> AiterMLAMetadata:
    page_indices = torch.arange(CONTEXT_LEN, dtype=torch.int32, device=device)
    decode = AiterMLADecodeMetadata(
        block_table=page_indices.view(1, -1),
        seq_lens=torch.tensor([CONTEXT_LEN], dtype=torch.int32, device=device),
        dcp_tot_seq_lens=None,
        paged_kv_indptr=torch.tensor(
            [0, CONTEXT_LEN], dtype=torch.int32, device=device
        ),
        paged_kv_indices=page_indices,
        paged_kv_last_page_len=torch.ones(1, dtype=torch.int32, device=device),
        qo_indptr=torch.tensor([0, 1], dtype=torch.int32, device=device),
        attn_out_dtype=torch.bfloat16,
        max_qo_len=1,
    )
    return AiterMLAMetadata(
        num_reqs=1,
        max_query_len=1,
        max_seq_len=CONTEXT_LEN,
        num_actual_tokens=1,
        query_start_loc=decode.qo_indptr,
        slot_mapping=torch.tensor([CONTEXT_LEN - 1], dtype=torch.int64, device=device),
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=0,
        decode=decode,
    )


def test_h12_query_is_tile_padded_to_h16():
    q = torch.arange(2 * 12 * 4, dtype=torch.bfloat16).view(2, 12, 4)

    padded_q = AiterMLAHelper.get_mla_padded_q(12, q)

    assert padded_q.shape == (2, 16, 4)
    assert padded_q.is_contiguous()
    # The real heads are untouched...
    torch.testing.assert_close(padded_q[:, :12], q)
    # ...and the 4 padding heads are the tiled wrap-around (heads 0..3), not
    # zeros: MLA attention is per-head independent so duplicate query heads are
    # harmless and get sliced back off the output.
    torch.testing.assert_close(padded_q[:, 12:], q[:, :4])


def test_h6_tp16_query_is_padded_to_h16():
    # TP16 puts 6 heads/rank. The old append-only padding produced cat(6, 6) =
    # 12 heads and broke the asm kernel; tile-and-slice must reach exactly 16.
    q = torch.arange(2 * 6 * 4, dtype=torch.bfloat16).view(2, 6, 4)

    padded_q = AiterMLAHelper.get_mla_padded_q(6, q)

    assert padded_q.shape == (2, 16, 4)
    assert padded_q.is_contiguous()
    torch.testing.assert_close(padded_q, _expected_tile_pad(q, 6))


def test_h12_output_discards_padding_heads():
    o = torch.arange(2 * 16 * 4, dtype=torch.bfloat16).view(2, 16, 4)

    unpadded_o = AiterMLAHelper.get_mla_unpadded_o(12, o)

    assert unpadded_o.shape == (2, 12, 4)
    torch.testing.assert_close(unpadded_o, o[:, :12])


def test_h24_query_is_tile_padded_to_h32():
    q = torch.arange(2 * 24 * 4, dtype=torch.float32).view(2, 24, 4)

    padded_q = AiterMLAHelper.get_mla_padded_q(24, q)

    assert padded_q.shape == (2, 32, 4)
    assert padded_q.is_contiguous()
    torch.testing.assert_close(padded_q[:, :24], q)
    torch.testing.assert_close(padded_q[:, 24:], q[:, :8])


def test_h24_output_discards_h32_padding_heads():
    o = torch.arange(2 * 32 * 4, dtype=torch.float32).view(2, 32, 4)

    unpadded_o = AiterMLAHelper.get_mla_unpadded_o(24, o)

    assert unpadded_o.shape == (2, 24, 4)
    torch.testing.assert_close(unpadded_o, o[:, :24])


def test_h24_reducer_without_metadata_still_pads_to_h32(monkeypatch):
    monkeypatch.setattr(
        rocm_aiter_mla, "_aiter_mla_native_h24_reducer_supported", lambda: True
    )
    monkeypatch.setattr(
        rocm_aiter_mla, "_aiter_mla_native_h24_metadata_supported", lambda: False
    )
    monkeypatch.setattr(
        rocm_aiter_mla,
        "_aiter_mla_native_h24_supported",
        lambda: (
            rocm_aiter_mla._aiter_mla_native_h24_reducer_supported()
            and rocm_aiter_mla._aiter_mla_native_h24_metadata_supported()
        ),
    )
    q = torch.arange(2 * 24 * 4, dtype=torch.float32).view(2, 24, 4)

    assert AiterMLAHelper.get_actual_mla_num_heads(24) == 32
    padded_q = AiterMLAHelper.get_mla_padded_q(24, q)
    assert padded_q.shape == (2, 32, 4)
    torch.testing.assert_close(padded_q[:, :24], q)
    torch.testing.assert_close(padded_q[:, 24:], q[:, :8])


def test_native_h24_requires_reducer_and_metadata(monkeypatch):
    monkeypatch.setattr(
        rocm_aiter_mla, "_aiter_mla_native_h24_reducer_supported", lambda: True
    )
    monkeypatch.setattr(
        rocm_aiter_mla, "_aiter_mla_native_h24_metadata_supported", lambda: True
    )
    monkeypatch.setattr(
        rocm_aiter_mla,
        "_aiter_mla_native_h24_supported",
        lambda: (
            rocm_aiter_mla._aiter_mla_native_h24_reducer_supported()
            and rocm_aiter_mla._aiter_mla_native_h24_metadata_supported()
        ),
    )
    q = torch.arange(2 * 24 * 4, dtype=torch.float32).view(2, 24, 4)

    assert AiterMLAHelper.get_actual_mla_num_heads(24) == 24
    assert AiterMLAHelper.get_mla_padded_q(24, q) is q
    assert AiterMLAHelper.get_mla_unpadded_o(24, q) is q


def test_existing_divisor_head_mapping_is_unchanged():
    q = torch.arange(2 * 8 * 4, dtype=torch.bfloat16).view(2, 8, 4)

    padded_q = AiterMLAHelper.get_mla_padded_q(8, q)
    unpadded_o = AiterMLAHelper.get_mla_unpadded_o(8, padded_q)

    # Divisor counts still use repeat_interleave / strided unpad, unchanged.
    torch.testing.assert_close(padded_q, q.repeat_interleave(2, dim=1))
    torch.testing.assert_close(unpadded_o, q)


@pytest.mark.parametrize("num_heads", [17, 24, 31])
def test_unaligned_head_counts_round_trip_through_h32(num_heads: int):
    q = torch.arange(2 * num_heads * 4, dtype=torch.float32).view(2, num_heads, 4)

    padded_q = AiterMLAHelper.get_mla_padded_q(num_heads, q)
    unpadded_o = AiterMLAHelper.get_mla_unpadded_o(num_heads, padded_q)

    assert padded_q.shape == (2, 32, 4)
    assert padded_q.is_contiguous()
    torch.testing.assert_close(unpadded_o, q)


@pytest.mark.parametrize("num_heads", NON_DIVISOR_HEADS + DIVISOR_HEADS)
def test_all_small_head_counts_pad_to_16_and_round_trip(num_heads: int):
    q = torch.arange(2 * num_heads * 4, dtype=torch.float32).view(2, num_heads, 4)

    padded_q = AiterMLAHelper.get_mla_padded_q(num_heads, q)
    unpadded_o = AiterMLAHelper.get_mla_unpadded_o(num_heads, padded_q)

    assert padded_q.shape == (2, 16, 4)
    assert padded_q.is_contiguous()
    # The real heads survive the pad -> unpad round trip exactly.
    torch.testing.assert_close(unpadded_o, q)


def test_aligned_h32_is_zero_copy():
    q = torch.arange(2 * 32 * 4, dtype=torch.float32).view(2, 32, 4)
    assert AiterMLAHelper.get_mla_padded_q(32, q) is q
    assert AiterMLAHelper.get_mla_unpadded_o(32, q) is q


def test_is_valid_num_heads():
    for n in range(1, 129):
        assert AiterMLAHelper.is_valid_num_heads(n)
    assert AiterMLAHelper.is_valid_num_heads(24)
    assert AiterMLAHelper.is_valid_num_heads(127)
    # Aligned counts remain valid above the range where padding is supported.
    assert AiterMLAHelper.is_valid_num_heads(144)
    assert not AiterMLAHelper.is_valid_num_heads(0)
    assert not AiterMLAHelper.is_valid_num_heads(129)


def test_nondivisor_and_multitoken_never_use_gluon():
    # Non-divisor decode always takes the asm path (12 heads/rank at TP8).
    assert not AiterMLAHelper.use_gluon_decode(12, 1, "auto")
    assert not AiterMLAHelper.use_gluon_decode(6, 1, "auto")
    # >=16 heads never use Gluon, including unaligned counts padded for asm.
    assert not AiterMLAHelper.use_gluon_decode(16, 1, "auto")
    assert not AiterMLAHelper.use_gluon_decode(24, 1, "auto")
    # Multi-token (verify / qlen>1) is never the single-token Gluon decode.
    assert not AiterMLAHelper.use_gluon_decode(8, 4, "auto")
    assert not AiterMLAHelper.use_gluon_decode(12, 4, "auto")


def test_divisor_gluon_selection_follows_arch():
    # Divisor head counts keep Gluon only where the kernel exists (gfx950).
    # On gfx942 / non-ROCm they must route to the asm persistent decode.
    on_gfx950 = _on_gfx950()
    assert AiterMLAHelper.use_gluon_decode(8, 1, "auto") is on_gfx950
    assert AiterMLAHelper.use_gluon_decode(4, 1, "auto") is on_gfx950
    assert AiterMLAHelper.use_gluon_decode(1, 1, "auto") is on_gfx950


def test_asm_padding_env_default_is_auto(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_AITER_MLA_ASM_PADDING", raising=False)
    import vllm.envs as envs

    assert envs.VLLM_ROCM_AITER_MLA_ASM_PADDING == "auto"


def test_asm_padding_env_force_asm_disables_gluon(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_ASM_PADDING", "asm")
    # Forcing the asm path: no small-head count uses Gluon on any arch.
    for num_heads in (1, 2, 4, 8, 12):
        assert not AiterMLAHelper.use_gluon_decode(num_heads, 1, "auto")


def test_asm_padding_env_force_gluon_follows_arch(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_ASM_PADDING", "gluon")
    on_gfx950 = _on_gfx950()
    # Forcing Gluon: any 1..15 single-token decode uses it where a build exists
    # (gfx950), including non-divisor counts like 12; gfx942/non-ROCm still
    # falls back to the asm path.
    for num_heads in (1, 2, 4, 8, 12):
        assert AiterMLAHelper.use_gluon_decode(num_heads, 1, "auto") is on_gfx950


def test_asm_padding_env_auto_matches_arch_gate(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_ASM_PADDING", "auto")
    on_gfx950 = _on_gfx950()
    # auto: divisor counts keep Gluon on gfx950, non-divisor counts take asm.
    assert AiterMLAHelper.use_gluon_decode(8, 1, "auto") is on_gfx950
    assert not AiterMLAHelper.use_gluon_decode(12, 1, "auto")


@pytest.mark.skipif(
    not _rocm_aiter_available(),
    reason="12-head AITER MLA asm persistent decode needs ROCm + AITER",
)
@torch.inference_mode()
def test_h12_aiter_mla_decode_matches_reference():
    """The 12-head AITER decode output must match an attention reference.

    12 is a non-divisor of 16 so use_gluon_decode is False on every arch; the
    query is tile-padded to 16 heads and served by the asm persistent decode
    (the CDNA3 path this PR enables, and unchanged on CDNA4).
    """
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    q = torch.randn(1, NUM_HEADS, QK_HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv_cache = torch.randn(
        CONTEXT_LEN, 1, QK_HEAD_DIM, dtype=torch.bfloat16, device=device
    )

    impl = object.__new__(AiterMLAImpl)
    impl.num_heads = NUM_HEADS
    impl.kv_lora_rank = KV_LORA_RANK
    impl.scale = SCALE
    impl.kv_cache_dtype = "auto"

    one = torch.ones(1, dtype=torch.float32, device=device)
    layer = types.SimpleNamespace(_q_scale=one, _k_scale=one)
    metadata = _make_h12_decode_metadata(device)

    out, _ = impl.forward_mqa(q, kv_cache, metadata, layer)

    assert out.shape[1] == NUM_HEADS  # padding heads sliced back off

    key = kv_cache[:, 0].float().unsqueeze(0)
    value = key[..., :KV_LORA_RANK]
    out_ref = F.scaled_dot_product_attention(
        q[0].float().unsqueeze(1),
        key,
        value,
        scale=SCALE,
        enable_gqa=True,
    ).squeeze(1)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out.float(),
        out_ref.unsqueeze(0),
        atol=1e-2,
        rtol=1e-2,
    )
