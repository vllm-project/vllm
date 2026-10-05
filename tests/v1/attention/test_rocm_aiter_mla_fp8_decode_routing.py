# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-kernel routing for the ROCm AITER MLA backend.

Gluon exposes a single fp8-KV regime, bh16bn128. It is a bf16-query kernel that
upcasts the cache in registers with a hardcoded scale of 1.0, and it asserts
batch_size == 1, so it cannot serve a decode batch. Every fp8 shape therefore
has to land on the asm kernels, which ship real fp8 variants for gqa=16. These
tests pin that down at the predicates, since the failure it prevents is either a
batch assertion or -- worse -- a silently wrong result.

Gluon is still the route for supported non-DCP decode shapes. DCP multi-token
verification uses segmented MLA instead, so it depends on neither Gluon's
query-head nor its long-context pipeline limits.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm AITER MLA tests", allow_module_level=True)

from vllm.v1.attention.backends.mla import rocm_aiter_mla  # noqa: E402
from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper  # noqa: E402

FP8_DTYPES = ["fp8", "fp8_e4m3", "fp8_e5m2"]
UNQUANTIZED_DTYPES = ["auto", "bfloat16"]


@pytest.fixture
def gluon_available(monkeypatch):
    """Pin the arch gate and the mode knob so only the dtype rules are in play.

    Gluon ships a gfx950 build only, and VLLM_ROCM_AITER_MLA_ASM_PADDING can
    force the asm path on any arch. Without pinning both, the expectations that
    bf16 *keeps* Gluon would pass on a gfx950 host and fail on gfx942 for
    reasons that have nothing to do with what these tests cover.
    """
    monkeypatch.setattr(rocm_aiter_mla, "_gluon_mla_decode_supported", lambda: True)
    monkeypatch.setattr(rocm_aiter_mla, "_aiter_mla_small_head_mode", lambda: "auto")


@pytest.mark.parametrize("kv_cache_dtype", FP8_DTYPES)
@pytest.mark.parametrize("num_heads", [1, 2, 4, 5, 6, 8, 12, 16, 32, 128])
@pytest.mark.parametrize("max_qo_len", [1, 2, 4, 5, 8, 15])
def test_non_dcp_fp8_never_routes_to_gluon(kv_cache_dtype, num_heads, max_qo_len):
    """No non-DCP fp8 shape reaches either Gluon entry point.

    The head count is deliberately swept across divisors of 16 as well as
    non-divisors: the divisor case (e.g. 8 heads at TP8) is the one that stays
    on Gluon for bf16 and so is the one an fp8 guard has to override.
    """
    assert not AiterMLAHelper.use_gluon_decode(num_heads, max_qo_len, kv_cache_dtype)
    assert not AiterMLAHelper.use_gluon_verify(num_heads, max_qo_len, kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", FP8_DTYPES)
@pytest.mark.parametrize("num_heads", [1, 2, 4, 8, 12])
@pytest.mark.parametrize("mode", ["auto", "gluon", "asm"])
def test_fp8_never_routes_to_gluon_under_any_mode(
    monkeypatch, kv_cache_dtype, num_heads, mode
):
    """VLLM_ROCM_AITER_MLA_ASM_PADDING cannot force a non-DCP fp8 cache onto Gluon.

    The dtype guard deliberately precedes the mode knob: honouring an explicit
    "gluon" request under fp8 would hand Gluon the batch it asserts against, so
    it is overridden rather than obeyed. Pinned here because the override is a
    correctness decision, not a preference.
    """
    monkeypatch.setattr(rocm_aiter_mla, "_gluon_mla_decode_supported", lambda: True)
    monkeypatch.setattr(rocm_aiter_mla, "_aiter_mla_small_head_mode", lambda: mode)
    assert not AiterMLAHelper.use_gluon_decode(num_heads, 1, kv_cache_dtype)
    assert not AiterMLAHelper.use_gluon_verify(num_heads, 8, kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", FP8_DTYPES + UNQUANTIZED_DTYPES)
@pytest.mark.parametrize("num_heads", [16, 17, 32, 64, 128])
@pytest.mark.parametrize("max_qo_len", [1, 4, 8])
def test_large_head_counts_never_use_gluon(kv_cache_dtype, num_heads, max_qo_len):
    """>= 16 heads has always been asm-only; the dtype guard must not change it."""
    assert not AiterMLAHelper.use_gluon_decode(num_heads, max_qo_len, kv_cache_dtype)
    assert not AiterMLAHelper.use_gluon_verify(num_heads, max_qo_len, kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", FP8_DTYPES + UNQUANTIZED_DTYPES)
@pytest.mark.parametrize("num_heads", [8, 16, 32, 96])
@pytest.mark.parametrize("max_qo_len", [2, 3, 8])
def test_dcp_multitoken_verify_never_uses_gluon(
    gluon_available, kv_cache_dtype, num_heads, max_qo_len
):
    assert not AiterMLAHelper.use_gluon_verify(
        num_heads, max_qo_len, kv_cache_dtype, dcp_world_size=8
    )


def test_segmented_dcp_verify_does_not_depend_on_gluon(monkeypatch):
    monkeypatch.setattr(rocm_aiter_mla, "_gluon_mla_decode_supported", lambda: False)
    monkeypatch.setattr(rocm_aiter_mla, "_segmented_mla_decode_supported", lambda: True)

    assert rocm_aiter_mla._segmented_dcp_verify_supported(8, 1)
    # Round-robin interleaving other than 1 is not served by this route.
    assert not rocm_aiter_mla._segmented_dcp_verify_supported(8, 4)


@pytest.mark.parametrize("kv_cache_dtype", UNQUANTIZED_DTYPES)
@pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
def test_unquantized_divisor_heads_keep_gluon_decode(
    gluon_available, kv_cache_dtype, num_heads
):
    """Divisor head counts keep the Gluon single-token decode for bf16.

    Gluon does not scale with KV length the way the asm persistent decode does,
    so this is the faster path where it is usable.
    """
    assert AiterMLAHelper.use_gluon_decode(num_heads, 1, kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", UNQUANTIZED_DTYPES)
@pytest.mark.parametrize("num_heads", [3, 5, 6, 7, 9, 12, 15])
def test_unquantized_non_divisor_heads_use_asm_decode(kv_cache_dtype, num_heads):
    """Non-divisor head counts are padded to 16 and take the asm decode."""
    assert not AiterMLAHelper.use_gluon_decode(num_heads, 1, kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", UNQUANTIZED_DTYPES)
@pytest.mark.parametrize("num_heads", [5, 8, 12])
@pytest.mark.parametrize("max_qo_len", [2, 4, 8, 15])
def test_unquantized_small_head_verify_keeps_gluon(
    gluon_available, kv_cache_dtype, num_heads, max_qo_len
):
    """bf16 has no gqa<16, qseqlen>1 asm kernel, so verify still flattens onto Gluon.

    Unlike the decode predicate, this one does not care whether the head count
    divides 16 -- the flatten reshapes to qseqlen=1 either way.
    """
    assert AiterMLAHelper.use_gluon_verify(num_heads, max_qo_len, kv_cache_dtype)
    # The verify flatten is a separate entry point from the single-token decode.
    assert not AiterMLAHelper.use_gluon_decode(num_heads, max_qo_len, kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", FP8_DTYPES + UNQUANTIZED_DTYPES)
@pytest.mark.parametrize("num_heads", [5, 8, 12, 16, 32])
def test_decode_and_verify_are_disjoint(kv_cache_dtype, num_heads):
    """The two Gluon entry points partition on qlen and never both fire."""
    for max_qo_len in (1, 2, 8):
        assert not (
            AiterMLAHelper.use_gluon_decode(num_heads, max_qo_len, kv_cache_dtype)
            and AiterMLAHelper.use_gluon_verify(num_heads, max_qo_len, kv_cache_dtype)
        )


@pytest.mark.parametrize("kv_cache_dtype", UNQUANTIZED_DTYPES)
def test_oversized_kv_cache_refuses_gluon(gluon_available, kv_cache_dtype):
    """Past 2 GiB per layer Gluon drops its KV bounds mask and reads out of range.

    ``None`` is the profiling case, before the cache is sized: nothing to
    refuse yet, so the usual route stands.
    """
    bound = rocm_aiter_mla._GLUON_MAX_KV_CACHE_BYTES
    for kv_cache_bytes in (None, bound):
        assert AiterMLAHelper.use_gluon_decode(8, 1, kv_cache_dtype, kv_cache_bytes)
        assert AiterMLAHelper.use_gluon_verify(
            8, 8, kv_cache_dtype, kv_cache_bytes=kv_cache_bytes
        )
    assert not AiterMLAHelper.use_gluon_decode(8, 1, kv_cache_dtype, bound + 1)
    assert not AiterMLAHelper.use_gluon_verify(
        8, 8, kv_cache_dtype, kv_cache_bytes=bound + 1
    )


@pytest.mark.parametrize("num_heads", [1, 2, 3, 5, 6, 7, 8, 9, 12, 15])
def test_padded_query_is_contiguous(num_heads):
    """asm_mla.cu:805 requires Q.is_contiguous().

    Padding a non-divisor head count to 16 tiles the query and slices it back
    down, which yields a non-contiguous view whenever more than one tile is
    needed (12 heads -> repeat to 24 -> slice to 16). The asm kernel rejects
    that outright, so the padding has to materialize the result.
    """
    q = torch.randn(4, num_heads, 576)
    padded = AiterMLAHelper.get_mla_padded_q(num_heads, q)

    assert padded.shape[1] == max(16, num_heads)
    assert padded.is_contiguous()


@pytest.mark.parametrize("num_heads", [1, 2, 3, 5, 6, 7, 8, 9, 12, 15, 16, 32])
def test_pad_unpad_round_trip_preserves_head_order(num_heads):
    """Unpadding recovers each original head from the padded output, in order."""
    q = torch.arange(num_heads, dtype=torch.float32).view(1, num_heads, 1)
    padded = AiterMLAHelper.get_mla_padded_q(num_heads, q)
    unpadded = AiterMLAHelper.get_mla_unpadded_o(num_heads, padded)

    assert unpadded.shape == q.shape
    torch.testing.assert_close(unpadded, q)


def test_a_non_causal_block_never_routes_to_gluon(gluon_available):
    """A small-head block that normally uses Gluon must use ASM when non-causal."""
    assert not AiterMLAHelper.use_gluon_verify(12, 8, "auto", causal=False)


@pytest.fixture
def pin_gluon_gate(monkeypatch):
    """Set the arch, the Triton probe and the mode knob, then clear both caches.

    ``_gluon_mla_decode_supported`` caches its answer. The Triton probe is
    cached too, and the arch check imports ``on_gfx950`` on each call, so the
    patch has to land on ``vllm.platforms.rocm`` before the gate runs.
    """
    originals = {
        name: getattr(rocm_aiter_mla, name)
        for name in ("_gluon_mla_decode_supported", "_triton_compiles_aiter_gluon_mla")
    }

    def _clear() -> None:
        for fn in originals.values():
            if hasattr(fn, "cache_clear"):
                fn.cache_clear()

    def apply(*, gfx950: bool, triton_ok: bool, mode: str = "auto"):
        _clear()
        monkeypatch.setattr(
            rocm_aiter_mla, "_triton_compiles_aiter_gluon_mla", lambda: triton_ok
        )
        monkeypatch.setattr(rocm_aiter_mla, "_aiter_mla_small_head_mode", lambda: mode)
        import vllm.platforms.rocm as rocm

        monkeypatch.setattr(rocm, "on_gfx950", lambda: gfx950)
        _clear()

    yield apply
    _clear()


@pytest.mark.parametrize("num_heads", [1, 2, 4, 8, 12])
@pytest.mark.parametrize("mode", ["auto", "gluon"])
@pytest.mark.parametrize("kv_cache_dtype", UNQUANTIZED_DTYPES)
def test_uncompilable_triton_never_selects_gluon(
    pin_gluon_gate, num_heads, mode, kv_cache_dtype
):
    """gfx950 still must not select Gluon when this Triton cannot compile the kernel.

    Divisors of 16 are the counts that take Gluon once the arch gate passes.
    An explicit ``gluon`` mode does not override that: the kernel raises
    ``TypeError`` on ``cga_layout`` before any request is served. 12 heads
    (Kimi-K3 at TP8) are included because multi-token verify still uses Gluon
    for non-divisors when the kernel does compile.
    """
    pin_gluon_gate(gfx950=True, triton_ok=False, mode=mode)

    assert not rocm_aiter_mla._gluon_mla_decode_supported()
    assert not AiterMLAHelper.use_gluon_decode(num_heads, 1, kv_cache_dtype)
    assert not AiterMLAHelper.use_gluon_verify(num_heads, 4, kv_cache_dtype)


@pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
@pytest.mark.parametrize("kv_cache_dtype", UNQUANTIZED_DTYPES)
def test_compilable_triton_keeps_gluon_for_divisor_heads(
    pin_gluon_gate, num_heads, kv_cache_dtype
):
    """With a Triton that accepts ``cga_layout``, gfx950 keeps the Gluon path."""
    pin_gluon_gate(gfx950=True, triton_ok=True, mode="auto")

    assert rocm_aiter_mla._gluon_mla_decode_supported()
    assert AiterMLAHelper.use_gluon_decode(num_heads, 1, kv_cache_dtype)
    assert AiterMLAHelper.use_gluon_verify(num_heads, 4, kv_cache_dtype)
    # Single-token decode and multi-token verify stay mutually exclusive.
    assert not AiterMLAHelper.use_gluon_decode(num_heads, 4, kv_cache_dtype)
    assert not AiterMLAHelper.use_gluon_verify(num_heads, 1, kv_cache_dtype)


@pytest.mark.parametrize("kv_cache_dtype", UNQUANTIZED_DTYPES)
def test_compilable_triton_routes_non_divisor_decode_to_asm(
    pin_gluon_gate, kv_cache_dtype
):
    """12 heads decode through padded ASM; verify still uses Gluon when it compiles.

    ``use_gluon_decode`` refuses a head count that does not divide 16 unless
    the mode knob is ``gluon``. ``use_gluon_verify`` has no such divisor check,
    which is why an uncompilable Triton has to fail the shared gate.
    """
    pin_gluon_gate(gfx950=True, triton_ok=True, mode="auto")

    assert not AiterMLAHelper.use_gluon_decode(12, 1, kv_cache_dtype)
    assert AiterMLAHelper.use_gluon_verify(12, 4, kv_cache_dtype)

    pin_gluon_gate(gfx950=True, triton_ok=True, mode="gluon")
    assert AiterMLAHelper.use_gluon_decode(12, 1, kv_cache_dtype)


@pytest.mark.parametrize("num_heads", [8, 12])
@pytest.mark.parametrize("triton_ok", [True, False])
def test_gluon_stays_off_without_gfx950_or_for_fp8_and_asm_mode(
    pin_gluon_gate, num_heads, triton_ok
):
    """The Triton check does not reopen routes that were already refused."""
    pin_gluon_gate(gfx950=False, triton_ok=triton_ok, mode="auto")
    assert not rocm_aiter_mla._gluon_mla_decode_supported()
    assert not AiterMLAHelper.use_gluon_decode(num_heads, 1, "bfloat16")
    assert not AiterMLAHelper.use_gluon_verify(num_heads, 4, "bfloat16")

    pin_gluon_gate(gfx950=True, triton_ok=triton_ok, mode="asm")
    assert not AiterMLAHelper.use_gluon_decode(num_heads, 1, "bfloat16")
    assert not AiterMLAHelper.use_gluon_verify(num_heads, 4, "bfloat16")

    for kv_cache_dtype in FP8_DTYPES:
        pin_gluon_gate(gfx950=True, triton_ok=triton_ok, mode="gluon")
        assert not AiterMLAHelper.use_gluon_decode(num_heads, 1, kv_cache_dtype)
        assert not AiterMLAHelper.use_gluon_verify(num_heads, 4, kv_cache_dtype)


def test_probe_is_false_when_gluon_language_cannot_be_imported(monkeypatch):
    """A Triton build without the Gluon language module is treated as unsupported."""
    import builtins

    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "triton.experimental.gluon.language" or (
            name == "triton.experimental.gluon" and fromlist and "language" in fromlist
        ):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    rocm_aiter_mla._triton_compiles_aiter_gluon_mla.cache_clear()
    try:
        assert not rocm_aiter_mla._triton_compiles_aiter_gluon_mla()
    finally:
        rocm_aiter_mla._triton_compiles_aiter_gluon_mla.cache_clear()


def test_installed_triton_probe_matches_padded_shared_layout_api():
    """The probe is the ``cga_layout`` field, and a missing field raises TypeError.

    AITER calls ``PaddedSharedLayout(..., cga_layout=...)``. Triton 3.6 names
    that parameter ``block_bases``. Newer Triton accepts the keyword, so the
    TypeError check only runs where the field is absent.
    """
    from triton.experimental.gluon import language as gl

    fields = getattr(gl.PaddedSharedLayout, "__dataclass_fields__", {})
    rocm_aiter_mla._triton_compiles_aiter_gluon_mla.cache_clear()
    try:
        assert rocm_aiter_mla._triton_compiles_aiter_gluon_mla() == (
            "cga_layout" in fields
        )
        if "cga_layout" not in fields:
            with pytest.raises(TypeError, match="cga_layout"):
                gl.PaddedSharedLayout(
                    interval_padding_pairs=[[512, 16]],
                    offset_bases=[[0, 1]],
                    cga_layout=[],
                    shape=[64, 512],
                )
            rocm_aiter_mla._gluon_mla_decode_supported.cache_clear()
            assert not rocm_aiter_mla._gluon_mla_decode_supported()
            assert not AiterMLAHelper.use_gluon_decode(8, 1, "bfloat16")
            assert not AiterMLAHelper.use_gluon_verify(12, 4, "bfloat16")
    finally:
        rocm_aiter_mla._triton_compiles_aiter_gluon_mla.cache_clear()
        rocm_aiter_mla._gluon_mla_decode_supported.cache_clear()
