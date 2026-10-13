# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Head padding and kernel selection for the ROCm AITER MLA backend.

The asm persistent decode requires a 16-aligned head count, so unaligned
counts through 128 are tile-padded to the next multiple of 16 and sliced back
off the output. Small divisor counts (1/2/4/8) preserve their existing
repeat-interleave path and may keep the Gluon kernel on gfx950.
"""

import math
import sys
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


# --------------------------------------------------------------------------
# VLLM_ROCM_AITER_MLA_PAD_TO_NATIVE_SHAPE: padding to a native AITER head count
# --------------------------------------------------------------------------

# (qlen, return_lse, causal) the plain asm decode is launched with: DCP decodes
# only single tokens there and needs the LSE; non-DCP decodes every qlen up to K.
DCP_SHAPES = [(1, True, True)]


def _non_dcp_shapes(max_qlen: int) -> list[tuple[int, bool, bool]]:
    return [(q, False, True) for q in range(1, max_qlen + 1)]


def _gfx950_fp8(n: int, q: int, lse: bool, causal: bool) -> tuple[bool, bool]:
    """(native, supported) modelled on gfx950 with an fp8 KV cache."""
    return n in (16, 32, 64, 128) or (n == 96 and q <= 6), True


def _gfx942_fp8(n: int, q: int, lse: bool, causal: bool) -> tuple[bool, bool]:
    """(native, supported) modelled on gfx942: few LSE kernels."""
    return (
        n in (16, 128) or (n == 64 and q == 1),
        not lse or (n == 64 and q == 1),
    )


@pytest.fixture
def shape_query(monkeypatch):
    """Install a fake aiter.get_mla_decode_shape_support following ``rules``."""

    def install(rules, arch: str = "gfx950") -> list[tuple[int, int, bool, bool]]:
        calls: list[tuple[int, int, bool, bool]] = []

        def get_mla_decode_shape_support(
            num_heads, max_seqlen_qo, q_dtype, kv_dtype, *, return_lse, causal, **kw
        ):
            calls.append((num_heads, max_seqlen_qo, return_lse, causal))
            native, supported = rules(num_heads, max_seqlen_qo, return_lse, causal)
            return types.SimpleNamespace(
                arch=arch,
                native=native,
                supported=supported,
                reasons=() if supported else (f"no {num_heads}-head kernel",),
            )

        monkeypatch.setitem(
            sys.modules,
            "aiter",
            types.SimpleNamespace(
                get_mla_decode_shape_support=get_mla_decode_shape_support
            ),
        )
        return calls

    return install


def _resolve(num_heads: int, shapes, force: bool = False) -> int:
    return AiterMLAHelper.resolve_padded_mla_num_heads(
        num_heads, shapes, torch.float8_e4m3fn, torch.float8_e4m3fn, 512, force
    )


def test_pad_to_native_shape_env_default_is_off(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_AITER_MLA_PAD_TO_NATIVE_SHAPE", raising=False)
    import vllm.envs as envs

    assert envs.VLLM_ROCM_AITER_MLA_PAD_TO_NATIVE_SHAPE == "off"


def test_resolver_requires_the_aiter_shape_query(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiter", types.SimpleNamespace())
    with pytest.raises(RuntimeError, match=r"ROCm/aiter#6107"):
        _resolve(48, DCP_SHAPES)


@pytest.mark.parametrize("num_heads", [1, 8, 12, 16])
def test_resolver_keeps_16_heads_where_16_has_a_kernel(shape_query, num_heads):
    calls = shape_query(_gfx950_fp8)
    assert _resolve(num_heads, DCP_SHAPES) == 16
    assert _resolve(num_heads, _non_dcp_shapes(8), force=True) == 16
    assert {call[0] for call in calls} == {16}


@pytest.mark.parametrize("num_heads", [1, 8, 12, 16])
@pytest.mark.parametrize("force", [False, True])
def test_resolver_pads_16_heads_that_have_no_kernel(shape_query, num_heads, force):
    shape_query(_gfx942_fp8, arch="gfx942")
    # gfx942 has no 16-head fp8 LSE kernel; 64 at qlen 1 is the only one, four
    # times the aligned count.
    assert _resolve(num_heads, DCP_SHAPES, force) == 64
    assert _resolve(num_heads, _non_dcp_shapes(4), force) == 16


@pytest.mark.parametrize(
    "num_heads,expected",
    [(24, 32), (32, 32), (48, 64), (64, 64), (80, 96), (96, 96), (112, 128)]
    + [(120, 128), (128, 128), (144, 144)],
)
def test_resolver_pads_dcp_decode_to_the_smallest_native_count(
    shape_query, num_heads, expected
):
    calls = shape_query(_gfx950_fp8)
    assert _resolve(num_heads, DCP_SHAPES) == expected
    assert {call[1:] for call in calls} == {(1, True, True)}


def test_resolver_needs_a_target_native_at_every_shape(shape_query):
    shape_query(_gfx950_fp8)
    # 96 is native only up to qlen 6, so a K=8 run would fold at qlen 7 and 8.
    assert _resolve(96, _non_dcp_shapes(8)) == 128
    assert _resolve(96, _non_dcp_shapes(4)) == 96


def test_resolver_caps_the_pad_unless_forced(shape_query):
    shape_query(_gfx942_fp8, arch="gfx942")
    # The only target native at qlen 1..4 is 128, four times the aligned count.
    assert _resolve(32, _non_dcp_shapes(4)) == 32
    assert _resolve(32, _non_dcp_shapes(4), force=True) == 128


def test_resolver_ignores_the_cap_when_the_aligned_count_has_no_kernel(shape_query):
    shape_query(_gfx942_fp8, arch="gfx942")
    assert _resolve(32, DCP_SHAPES) == 64

    shape_query(lambda n, *args: (n == 128, n == 128))
    assert _resolve(32, DCP_SHAPES) == 128


def test_resolver_raises_when_nothing_runs_the_shape(shape_query):
    shape_query(_gfx942_fp8, arch="gfx942")
    # 112 has no LSE kernel and neither does 128, the only count above it.
    with pytest.raises(ValueError, match="no 112-head kernel"):
        _resolve(112, DCP_SHAPES)


def test_resolver_keeps_the_aligned_count_when_support_is_unknown(shape_query):
    shape_query(lambda *args: (True, None))
    assert _resolve(48, DCP_SHAPES) == 48
    assert _resolve(48, DCP_SHAPES, force=True) == 48


@pytest.mark.parametrize(
    "rules,shapes",
    [(_gfx950_fp8, _non_dcp_shapes(8)), (_gfx942_fp8, _non_dcp_shapes(4))],
    ids=["gfx950", "gfx942"],
)
@pytest.mark.parametrize("force", [False, True])
def test_resolver_returns_a_legal_launch_count(shape_query, rules, shapes, force):
    shape_query(rules)
    for num_heads in range(1, 257):
        aligned = AiterMLAHelper.get_actual_mla_num_heads(num_heads)
        heads = _resolve(num_heads, shapes, force)
        assert heads == aligned or (heads % 16 == 0 and aligned < heads <= 128)
        if not force:
            assert heads <= 2 * aligned
        q = torch.arange(num_heads, dtype=torch.float32).view(1, num_heads, 1)
        padded = AiterMLAHelper.get_mla_padded_q(num_heads, q, heads)
        assert padded.shape == (1, heads, 1)
        torch.testing.assert_close(
            AiterMLAHelper.get_mla_unpadded_o(num_heads, padded, heads), q
        )


@pytest.mark.parametrize("num_heads,target", [(48, 64), (80, 96), (112, 128)])
def test_padded_output_and_lse_round_trip(num_heads: int, target: int):
    q = torch.arange(2 * num_heads * 4, dtype=torch.float32).view(2, num_heads, 4)

    padded_q = AiterMLAHelper.get_mla_padded_q(num_heads, q, target)
    lse = padded_q[..., 0]

    assert padded_q.shape == (2, target, 4)
    assert padded_q.is_contiguous()
    torch.testing.assert_close(
        AiterMLAHelper.get_mla_unpadded_o(num_heads, padded_q, target), q
    )
    torch.testing.assert_close(
        AiterMLAHelper.get_mla_unpadded_lse(num_heads, lse, target), q[..., 0]
    )


def test_divisor_pad_to_a_native_count_unpads_strided():
    q = torch.arange(2 * 32 * 4, dtype=torch.float32).view(2, 32, 4)

    padded_q = AiterMLAHelper.get_mla_padded_q(32, q, 64)

    torch.testing.assert_close(padded_q, q.repeat_interleave(2, dim=1))
    torch.testing.assert_close(AiterMLAHelper.get_mla_unpadded_o(32, padded_q, 64), q)


def _aiter_has_shape_query() -> bool:
    if not (_on_gfx950() and torch.accelerator.is_available()):
        return False
    import aiter

    return hasattr(aiter, "get_mla_decode_shape_support")


@pytest.mark.skipif(
    not _aiter_has_shape_query(),
    reason="needs gfx950 and AITER with get_mla_decode_shape_support",
)
@pytest.mark.parametrize(
    "dtype,shapes",
    [
        ("fp8", DCP_SHAPES),
        ("fp8", _non_dcp_shapes(4)),
        ("fp8", _non_dcp_shapes(8)),
        ("bf16", DCP_SHAPES),
    ],
    ids=["fp8-dcp", "fp8-k4", "fp8-k8", "bf16-dcp"],
)
@pytest.mark.parametrize("num_heads", [12, 16, 48, 80, 96, 112, 128])
def test_resolver_on_installed_aiter(dtype, shapes, num_heads):
    """The resolver takes the smallest count the real query reports native.

    Derived from the query rather than a copy of AITER's kernel table, so a
    release that adds native counts moves the expectation with it.
    """
    import aiter
    from aiter import dtypes

    kv_dtype = dtypes.fp8 if dtype == "fp8" else dtypes.bf16

    def support(heads: int) -> list:
        return [
            aiter.get_mla_decode_shape_support(
                heads,
                qlen,
                kv_dtype,
                kv_dtype,
                v_head_dim=KV_LORA_RANK,
                return_lse=lse,
                causal=causal,
            )
            for qlen, lse, causal in shapes
        ]

    def native(heads: int) -> bool:
        return all(s.native and s.supported is True for s in support(heads))

    aligned = AiterMLAHelper.get_actual_mla_num_heads(num_heads)
    broken = any(s.supported is False for s in support(aligned))
    limit = 128 if broken else min(128, 2 * aligned)
    expected = next((h for h in range(aligned, limit + 1, 16) if native(h)), None)
    if expected is None and broken:
        with pytest.raises(ValueError):
            AiterMLAHelper.resolve_padded_mla_num_heads(
                num_heads, shapes, kv_dtype, kv_dtype, KV_LORA_RANK
            )
        return
    heads = AiterMLAHelper.resolve_padded_mla_num_heads(
        num_heads, shapes, kv_dtype, kv_dtype, KV_LORA_RANK
    )
    assert heads == (aligned if expected is None else expected)
