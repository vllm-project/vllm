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
    impl.dcp_world_size = 1
    impl.kv_lora_rank = KV_LORA_RANK
    impl.qk_rope_head_dim = QK_ROPE_HEAD_DIM
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


# ---------------------------------------------------------------------------
# Native-shape padding (VLLM_ROCM_AITER_MLA_PAD_TO_NATIVE_SHAPE).
#
# AITER's ``natively_supported`` is a disjunction over
# (arch, q fp8?, kv fp8?, num_heads, max_seqlen_qo), and a native verdict from
# the metadata planner still has to be backed by a shipped asm kernel, so these
# tests drive the real probes with verbatim copies of what AITER v0.1.21.post1
# ships. That keeps them runnable off ROCm and makes them fail if the literal
# matching in _aiter_mla_shape_is_native drifts.
# ---------------------------------------------------------------------------

# csrc/kernels/mla/metadata/v1_2_device.cuh, the ``natively_supported``
# initializer only -- the AITER_CHECK allowlist below it in the real file is
# deliberately excluded, since matching against it is the mistake the probe's
# slicing exists to prevent.
_AITER_0_1_21_PREDICATE = """
    const bool natively_supported =
        (num_heads == 16) ||
        ((arch_id == "gfx942" || arch_id == "gfx950") && (num_heads == 64) &&
         q_is_fp8 && kv_is_fp8 && (max_seqlen_qo == 1)) ||
        ((arch_id == "gfx950") && !q_is_fp8 && !kv_is_fp8) ||
        ((arch_id == "gfx942") && (num_heads == 128) && q_is_fp8 && kv_is_fp8) ||
        ((arch_id == "gfx950") && q_is_fp8 && kv_is_fp8 &&
         ((num_heads == 32) || (num_heads == 64) || (num_heads == 128))) ||
        ((arch_id == "gfx950") && q_is_fp8 && kv_is_fp8 && (num_heads == 96) &&
         (max_seqlen_qo <= 6)) ||
        hk_mtp_experimental
"""

# csrc/kernels/mla/reduce.cu MLA_REDUCE_ROUTER, HEAD_DIM 512 instantiations.
_AITER_REDUCER_HEADS = frozenset({8, 16, 24, 32, 48, 64, 80, 96, 112, 128})

# hsa/gfx942/mla/mla_asm.csv, the fp8/fp8 decode rows, as
# (qType, kvType, Gqa, ps, qSeqLen, lse). Note gqa=128 ships at lse=0 only
# while gqa=64 ships both -- that asymmetry is the whole point of the probe.
_AITER_GFX942_KERNELS = frozenset(
    ("fp8", "fp8", *row)
    for row in (
        (16, 1, 1, 0),
        (16, 1, 2, 0),
        (16, 1, 4, 0),
        (16, 0, 1, 0),
        (16, 0, 2, 0),
        (16, 0, 4, 0),
        (64, 1, 1, 0),
        (64, 1, 1, 1),
        (128, 1, 0, 0),
        (128, 0, 0, 0),
        (8, 0, 1, 0),
        (1, 1, 0, 0),
    )
)


@pytest.fixture
def aiter_0_1_21(monkeypatch):
    """Point the native-shape probes at a known AITER revision."""
    stripped = "".join(_AITER_0_1_21_PREDICATE.split())
    monkeypatch.setattr(
        rocm_aiter_mla, "_aiter_mla_native_predicate_source", lambda: stripped
    )
    monkeypatch.setattr(
        rocm_aiter_mla, "_aiter_mla_reducer_head_counts", lambda: _AITER_REDUCER_HEADS
    )
    monkeypatch.setattr(
        rocm_aiter_mla,
        "_aiter_mla_asm_decode_kernels",
        lambda arch: _AITER_GFX942_KERNELS if arch == "gfx942" else frozenset(),
    )


def _resolve(
    num_heads, *, gfx942=False, gfx950=True, q_fp8, kv_fp8, max_qo_lens, needs_lse=False
):
    return AiterMLAHelper.resolve_padded_mla_num_heads(
        num_heads,
        gfx942=gfx942,
        gfx950=gfx950,
        q_fp8=q_fp8,
        kv_fp8=kv_fp8,
        max_qo_lens=max_qo_lens,
        needs_lse=needs_lse,
    )


def test_head_padding_env_default_is_off(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_AITER_MLA_PAD_TO_NATIVE_SHAPE", raising=False)
    import vllm.envs as envs

    assert envs.VLLM_ROCM_AITER_MLA_PAD_TO_NATIVE_SHAPE == "off"


@pytest.mark.parametrize("num_heads", list(range(1, 257)))
def test_off_is_bit_identical_to_the_unpadded_rule(monkeypatch, num_heads):
    """The default must not move a single head count.

    This is the whole safety argument for the knob, so assert it over the
    entire reachable range rather than at a handful of sample points. H24 is
    disabled by the autouse fixture, so the expected rule is a plain ceil.
    """
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_PAD_TO_NATIVE_SHAPE", "off")
    expected = -(-num_heads // 16) * 16
    assert AiterMLAHelper.get_actual_mla_num_heads(num_heads) == expected
    for gfx942, gfx950 in ((True, False), (False, True), (False, False)):
        for fp8 in (True, False):
            assert (
                _resolve(
                    num_heads,
                    gfx942=gfx942,
                    gfx950=gfx950,
                    q_fp8=fp8,
                    kv_fp8=fp8,
                    max_qo_lens=(1, 2),
                )
                == expected
            )


# (gfx942, gfx950, q_fp8, kv_fp8, max_qo_lens, needs_lse, num_heads, expected)
#
# The interesting rows, per slice:
#   gfx950 fp8 qo=1   -- 48 -> 64 is the one reachable win (Kimi-K3 TP8+DCP4);
#                        96 stays put, AITER v0.1.21 having made it native, and
#                        padding it to 128 measured a 6.7% regression;
#                        80 -> 96 rather than 128, 96 being both native and
#                        cheaper (231.7 us vs 247.2 us measured).
#   gfx950 fp8 qo<=8  -- 96 loses its qo <= 6 clause at some reachable qlen, so
#                        the run-level constant has to clear it: 80 and 96 go
#                        to 128.
#   gfx950 bf16       -- the blanket bf16 clause makes everything native, so
#                        every row is the identity. This is the default
#                        --kv-cache-dtype auto configuration; a pad here would
#                        be pure loss.
#   gfx942 fp8 qo=1   -- 32 -> 64 (the fold a fixed {16,32,64,128} tile table
#                        misses, 32 not being native on gfx942).
#   gfx942 fp8 + LSE  -- a DCP rank. gqa=128 ships lse=0 only, so 80/96/112 may
#                        not pad at all; gqa=64 ships both, so 32/48 still can.
#   gfx942 fp8 qo<=4  -- 64 loses its qo == 1 clause, so 48 must NOT go to 64
#                        (that would swap a 3x fold for a 4x one) and 32's only
#                        target is 128, declined by the 2x cost cap.
#   gfx942 bf16       -- only 16 is native; padding could only raise the fold
#                        factor, so everything is the identity.
#   neither arch      -- no clause may fire on an arch AITER does not name.
NATIVE_PAD_CASES = [
    (False, True, True, True, (1,), False, 16, 16),
    (False, True, True, True, (1,), False, 32, 32),
    (False, True, True, True, (1,), False, 48, 64),
    (False, True, True, True, (1,), False, 64, 64),
    (False, True, True, True, (1,), False, 80, 96),
    (False, True, True, True, (1,), False, 96, 96),
    (False, True, True, True, (1,), False, 112, 128),
    (False, True, True, True, (1,), False, 120, 128),
    (False, True, True, True, (1,), False, 128, 128),
    (False, True, True, True, (1,), False, 144, 144),
    (False, True, True, True, (1,), False, 256, 256),
    (False, True, True, True, (1,), True, 48, 64),
    (False, True, True, True, range(1, 5), True, 48, 64),
    (False, True, True, True, range(1, 5), True, 96, 96),
    (False, True, True, True, range(1, 9), True, 80, 128),
    (False, True, True, True, range(1, 9), True, 96, 128),
    (False, True, True, True, range(1, 9), True, 128, 128),
    (False, True, False, False, (1,), False, 48, 48),
    (False, True, False, False, (1,), False, 80, 80),
    (False, True, False, False, (1,), False, 96, 96),
    (False, True, False, False, (1,), False, 112, 112),
    (False, True, False, False, range(1, 5), False, 48, 48),
    (False, True, True, False, (1,), False, 48, 48),
    (True, False, True, True, (1,), False, 16, 16),
    (True, False, True, True, (1,), False, 32, 64),
    (True, False, True, True, (1,), False, 48, 64),
    (True, False, True, True, (1,), False, 64, 64),
    (True, False, True, True, (1,), False, 80, 128),
    (True, False, True, True, (1,), False, 96, 128),
    (True, False, True, True, (1,), False, 112, 128),
    (True, False, True, True, (1,), False, 128, 128),
    (True, False, True, True, (1,), True, 32, 64),
    (True, False, True, True, (1,), True, 48, 64),
    (True, False, True, True, (1,), True, 80, 80),
    (True, False, True, True, (1,), True, 96, 96),
    (True, False, True, True, (1,), True, 112, 112),
    (True, False, True, True, range(1, 5), False, 32, 32),
    (True, False, True, True, range(1, 5), False, 48, 48),
    (True, False, True, True, range(1, 5), False, 64, 128),
    (True, False, True, True, range(1, 5), False, 128, 128),
    (True, False, False, False, (1,), False, 48, 48),
    (True, False, False, False, (1,), False, 80, 80),
    (True, False, False, False, (1,), False, 96, 96),
    (True, False, False, False, (1,), False, 112, 112),
    (False, False, True, True, (1,), False, 32, 32),
    (False, False, True, True, (1,), False, 48, 48),
    (False, False, True, True, (1,), False, 96, 96),
]


@pytest.mark.parametrize(
    "gfx942,gfx950,q_fp8,kv_fp8,max_qo_lens,needs_lse,num_heads,expected",
    NATIVE_PAD_CASES,
)
def test_auto_pads_to_a_natively_supported_shape(
    monkeypatch,
    aiter_0_1_21,
    gfx942,
    gfx950,
    q_fp8,
    kv_fp8,
    max_qo_lens,
    needs_lse,
    num_heads,
    expected,
):
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_PAD_TO_NATIVE_SHAPE", "auto")
    assert (
        _resolve(
            num_heads,
            gfx942=gfx942,
            gfx950=gfx950,
            q_fp8=q_fp8,
            kv_fp8=kv_fp8,
            max_qo_lens=max_qo_lens,
            needs_lse=needs_lse,
        )
        == expected
    )


def test_auto_never_pads_96_on_a_native_96_aiter(monkeypatch, aiter_0_1_21):
    """Regression guard for the AITER 0.1.19 -> 0.1.21 flip.

    A static tile table padded 96 -> 128 here, which was a 6x win on 0.1.19 and
    is a measured 6.7% loss on 0.1.21. Nothing but the probe stops that from
    silently coming back on the next bump.
    """
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_PAD_TO_NATIVE_SHAPE", "auto")
    assert _resolve(96, q_fp8=True, kv_fp8=True, max_qo_lens=(1,)) == 96


def test_auto_declines_a_target_without_an_lse_kernel(monkeypatch, aiter_0_1_21):
    """gfx942 ships gqa=128 at lse=0 only, so a DCP rank cannot land there.

    The metadata planner calls 128 native on gfx942 regardless, so a mirror
    that stopped at the planner would pad here and take out the first decode
    with "cannot find suitable kernel". gqa=64 ships both flags, so the same
    rank may still pad 48 -> 64.
    """
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_PAD_TO_NATIVE_SHAPE", "auto")
    kw = dict(gfx942=True, gfx950=False, q_fp8=True, kv_fp8=True, max_qo_lens=(1,))
    assert _resolve(112, needs_lse=False, **kw) == 128
    assert _resolve(112, needs_lse=True, **kw) == 112
    assert _resolve(48, needs_lse=True, **kw) == 64


def test_auto_requires_every_reachable_qlen_to_be_native(monkeypatch, aiter_0_1_21):
    """One run-level constant must not fold harder at some reachable qlen.

    AITER's fold factor is num_heads/16 with no qlen term, so a target that is
    native only at qlen 1 multiplies the KV traffic of every verify pass. On
    gfx942 with fp8, 64 heads is native at qlen 1 alone: padding 32 -> 64 wins
    a decode-only run and loses an MTP run.
    """
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_PAD_TO_NATIVE_SHAPE", "auto")
    kw = dict(gfx942=True, gfx950=False, q_fp8=True, kv_fp8=True)
    assert _resolve(32, max_qo_lens=(1,), **kw) == 64
    assert _resolve(32, max_qo_lens=(1, 2), **kw) == 32


def test_force_ignores_the_cost_cap(monkeypatch, aiter_0_1_21):
    # gfx942 + fp8 + 32 heads over qlens 1..4: the only target native at every
    # one of them is 128, a 4x head inflation against a 2x fold. "auto"
    # declines it; "force" takes it.
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_PAD_TO_NATIVE_SHAPE", "auto")
    kw = dict(gfx942=True, gfx950=False, q_fp8=True, kv_fp8=True, max_qo_lens=(1, 4))
    assert _resolve(32, **kw) == 32
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_PAD_TO_NATIVE_SHAPE", "force")
    assert _resolve(32, **kw) == 128


@pytest.mark.parametrize("mode", ["off", "auto", "force"])
@pytest.mark.parametrize("num_heads", list(range(1, 257)))
def test_resolved_count_is_always_a_legal_launch_shape(
    monkeypatch, aiter_0_1_21, mode, num_heads
):
    """Never below the real count, always 16-aligned, never above AITER's cap.

    The 16-alignment is a correctness guard rather than an optimization: AITER
    folds only multiples of 16 and asserts on anything else, so no policy may
    return an unaligned value. (Native H24 is the one sanctioned exception and
    the autouse fixture disables it here.) Padding may only grow the count,
    since the extra lanes are sliced back off the output.
    """
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_PAD_TO_NATIVE_SHAPE", mode)
    unpadded = -(-num_heads // 16) * 16
    for gfx942, gfx950 in ((True, False), (False, True), (False, False)):
        for fp8 in (True, False):
            m = _resolve(
                num_heads,
                gfx942=gfx942,
                gfx950=gfx950,
                q_fp8=fp8,
                kv_fp8=fp8,
                max_qo_lens=(1, 4),
            )
            assert m >= unpadded
            assert m % 16 == 0
            assert m <= max(unpadded, AiterMLAHelper._AITER_MAX_PADDED_MLA_HEADS)


@pytest.mark.parametrize("num_heads", [48, 80, 112])
def test_pad_unpad_round_trip_at_the_resolved_count(
    monkeypatch, aiter_0_1_21, num_heads
):
    """Padding to a native shape is exactly reversible for both o and lse."""
    monkeypatch.setenv("VLLM_ROCM_AITER_MLA_PAD_TO_NATIVE_SHAPE", "auto")
    padded_heads = _resolve(num_heads, q_fp8=True, kv_fp8=True, max_qo_lens=(1,))
    assert padded_heads > num_heads

    tokens = 5
    q = torch.randn(tokens, num_heads, QK_HEAD_DIM)
    q_padded = AiterMLAHelper.get_mla_padded_q(num_heads, q, padded_heads)
    assert q_padded.shape == (tokens, padded_heads, QK_HEAD_DIM)
    torch.testing.assert_close(q_padded[:, :num_heads, :], q)

    o = torch.randn(tokens, padded_heads, KV_LORA_RANK)
    torch.testing.assert_close(
        AiterMLAHelper.get_mla_unpadded_o(num_heads, o, padded_heads),
        o[:, :num_heads, :],
    )

    lse = torch.randn(tokens, padded_heads)
    torch.testing.assert_close(
        AiterMLAHelper.get_mla_unpadded_lse(num_heads, lse, padded_heads),
        lse[:, :num_heads],
    )


def test_unpad_uses_the_target_it_is_given():
    """A resolved count threaded to one side only silently returns junk.

    Pad and unpad each branch on ``m % num_heads == 0`` independently, so this
    documents why the impl carries ``padded_num_heads`` on the metadata instead
    of recomputing it.
    """
    tokens, num_heads = 3, 32
    padded_heads = 64  # divides 32, so unpad takes a stride-2 slice
    o = torch.arange(tokens * padded_heads * 4, dtype=torch.float32).reshape(
        tokens, padded_heads, 4
    )
    torch.testing.assert_close(
        AiterMLAHelper.get_mla_unpadded_o(num_heads, o, padded_heads), o[:, ::2, :]
    )
    # Without the target the helper falls back to the unpadded rule (32 == 32)
    # and hands back the padded tensor whole.
    assert AiterMLAHelper.get_mla_unpadded_o(num_heads, o).shape[1] == padded_heads


@pytest.mark.skipif(
    not (current_platform.is_rocm() and is_aiter_found()),
    reason="reads the installed AITER JIT sources",
)
def test_probe_matches_the_installed_aiter():
    """Fail when AITER moves under the mirror, rather than going quietly inert.

    Every clause is admitted by a literal match, so a re-spaced or reworded
    predicate makes them all fall through, the resolver returns the unpadded
    count, and the feature becomes a silent no-op for someone who opted in.
    Assert the literals themselves, not just the outcome.
    """
    src = rocm_aiter_mla._aiter_mla_native_predicate_source()
    assert src.startswith(rocm_aiter_mla._AITER_NATIVE_BLOCK_START)
    # The AITER_CHECK allowlist sits immediately below the initializer and
    # repeats most of its text; the slice must stop before it.
    assert "AITER_CHECK" not in src
    for literal in (
        '(arch_id=="gfx950")&&!q_is_fp8&&!kv_is_fp8',
        "((num_heads==32)||(num_heads==64)||(num_heads==128))",
        "(num_heads==96)&&(max_seqlen_qo<=6)",
        '(arch_id=="gfx942")&&(num_heads==128)',
        '(arch_id=="gfx942"||arch_id=="gfx950")&&(num_heads==64)&&q_is_fp8',
    ):
        assert literal in src, f"AITER predicate no longer contains {literal!r}"
    assert rocm_aiter_mla._aiter_mla_reducer_head_counts() == _AITER_REDUCER_HEADS
    from vllm.platforms.rocm import on_gfx942

    if on_gfx942():
        installed = rocm_aiter_mla._aiter_mla_asm_decode_kernels("gfx942")
        assert ("fp8", "fp8", 64, 1, 1, 1) in installed
        assert ("fp8", "fp8", 128, 1, 0, 1) not in installed
