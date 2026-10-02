# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the MXFP4 MLA Triton store kernel against the PyTorch reference.

The reference (:mod:`vllm.v1.attention.ops.mxfp4_mla`) is itself bit-exact
against AITER's ``per_1x32_f4_quant``, so agreement here chains the kernel to
the ecosystem reference.

Requires a GPU. Run with::

    docker run --rm --entrypoint python3 --device /dev/kfd --device /dev/dri \\
      --group-add video --security-opt seccomp=unconfined \\
      -v $PWD:/work -w /work local/glm53-flash:serving \\
      -m pytest test/test_mxfp4_mla_store.py -q
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.attention.ops import mxfp4_mla as mx


def _on_rocm_gpu() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_rocm() and torch.cuda.is_available()


pytestmark = pytest.mark.skipif(not _on_rocm_gpu(), reason="mxfp4_mla is ROCm-only")

LATENT = 512
GROUP = 32
ROW = 272
PAD_SLOT_ID = -1


def _store():
    from vllm.v1.attention.ops.mxfp4_mla_store import store_mxfp4_mla

    return store_mxfp4_mla


def _latent(rows: int, dim: int = LATENT, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, dim, generator=g, dtype=torch.float32)
    x[:, ::97] *= 8.0
    return x


def _empty_cache(num_slots: int) -> torch.Tensor:
    # 0xCD rather than zeros: a byte the kernel never legitimately writes for
    # the fixtures below, so untouched regions are visibly untouched.
    return torch.full((num_slots, ROW), 0xCD, dtype=torch.uint8, device="cuda")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_reference_bit_exactly(seed: int):
    num = 37  # deliberately not a multiple of any tile size
    x = _latent(num, seed=seed)
    slots = torch.randperm(64)[:num]

    cache = _empty_cache(64)
    _store()(x.to(torch.bfloat16).cuda(), slots.cuda().to(torch.int64), cache)

    # The reference consumes the same bf16 the kernel saw, so any difference is
    # the kernel's, not a dtype artifact.
    want_packed, want_scales = mx.quantize_pack(x.to(torch.bfloat16), GROUP)

    got = cache.cpu()
    for i, slot in enumerate(slots.tolist()):
        assert torch.equal(got[slot, :256], want_packed[i]), f"data row {i}"
        assert torch.equal(got[slot, 256:], want_scales[i]), f"scales row {i}"


def test_negative_slots_are_skipped():
    """PAD_SLOT_ID rows must leave the cache untouched, not scribble at -1."""
    x = _latent(8, seed=3)
    slots = torch.tensor([0, PAD_SLOT_ID, 2, PAD_SLOT_ID, 4, -7, 6, PAD_SLOT_ID])

    cache = _empty_cache(8)
    _store()(x.to(torch.bfloat16).cuda(), slots.cuda().to(torch.int64), cache)

    got = cache.cpu()
    written = {0, 2, 4, 6}
    for slot in range(8):
        if slot in written:
            assert not (got[slot] == 0xCD).all(), f"slot {slot} should be written"
        else:
            assert (got[slot] == 0xCD).all(), f"slot {slot} should be untouched"


def test_out_of_range_slots_are_skipped():
    x = _latent(4, seed=4)
    slots = torch.tensor([0, 99, 1, 4])  # 99 and 4 are past a 4-slot cache
    cache = _empty_cache(4)
    _store()(x.to(torch.bfloat16).cuda(), slots.cuda().to(torch.int64), cache)

    got = cache.cpu()
    assert not (got[0] == 0xCD).all()
    assert not (got[1] == 0xCD).all()
    assert (got[2] == 0xCD).all() and (got[3] == 0xCD).all()


def test_writes_stay_inside_their_row():
    """One token into the middle of a cache must not touch its neighbours."""
    x = _latent(1, seed=5)
    cache = _empty_cache(5)
    _store()(
        x.to(torch.bfloat16).cuda(), torch.tensor([2]).cuda().to(torch.int64), cache
    )

    got = cache.cpu()
    assert not (got[2] == 0xCD).all()
    for other in (0, 1, 3, 4):
        assert (got[other] == 0xCD).all(), f"row {other} was disturbed"


def test_roundtrip_through_the_reference_reader():
    """Store with the kernel, read back with the reference: values must match."""
    num = 24
    x = _latent(num, seed=6)
    slots = torch.arange(num)

    cache = _empty_cache(num)
    _store()(x.to(torch.bfloat16).cuda(), slots.cuda().to(torch.int64), cache)

    got = cache.cpu()
    values = mx.unpack_dequantize(
        got[:, :256], got[:, 256:], GROUP, out_dtype=torch.float32
    )
    want = mx.quantize_dequantize(x.to(torch.bfloat16), GROUP).to(torch.float32)
    assert torch.equal(values, want)


def test_zero_latent_writes_zero_codes_and_unit_scale():
    x = torch.zeros(1, LATENT, dtype=torch.bfloat16)
    cache = _empty_cache(1)
    _store()(x.cuda(), torch.tensor([0]).cuda().to(torch.int64), cache)

    got = cache.cpu()
    assert (got[0, :256] == 0).all()
    assert (got[0, 256:] == mx.E8M0_BIAS).all()


def test_no_element_saturates():
    """RoundUp's guarantee, checked on what the kernel actually wrote.

    If any group's peak exceeded 6.0 after scaling, its top element would land
    on code 7 (magnitude 6.0) for a value that was really larger. Instead every
    group's peak magnitude index must be 5 or 6 (values 3.0 or 4.0) or 7 only
    when the true peak really is at the top of the range.
    """
    num = 64
    x = _latent(num, seed=7)
    cache = _empty_cache(num)
    _store()(
        x.to(torch.bfloat16).cuda(),
        torch.arange(num).cuda().to(torch.int64),
        cache,
    )

    got = cache.cpu()
    scales = mx.decode_e8m0_scales(got[:, 256:])
    grouped = x.to(torch.bfloat16).to(torch.float32).reshape(num, -1, GROUP)
    peak = (grouped / scales.unsqueeze(-1)).abs().amax(dim=-1)
    assert (peak <= mx.E2M1_MAX + 1e-6).all(), peak.max()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_accepts_the_model_dtypes(dtype: torch.dtype):
    x = _latent(8, seed=8).to(dtype)
    cache = _empty_cache(8)
    _store()(x.cuda(), torch.arange(8).cuda().to(torch.int64), cache)

    got = cache.cpu()
    want_packed, want_scales = mx.quantize_pack(x, GROUP)
    assert torch.equal(got[:, :256], want_packed)
    assert torch.equal(got[:, 256:], want_scales)


def test_non_contiguous_latent_is_handled():
    """The latent arrives as a slice of a larger buffer in the real call path."""
    big = _latent(16, dim=LATENT * 2, seed=9)
    x = big[:, :LATENT]  # row stride is 1024, not 512
    assert x.stride(0) == LATENT * 2

    cache = _empty_cache(16)
    _store()(
        x.to(torch.bfloat16).cuda(), torch.arange(16).cuda().to(torch.int64), cache
    )

    got = cache.cpu()
    want_packed, want_scales = mx.quantize_pack(x.to(torch.bfloat16), GROUP)
    assert torch.equal(got[:, :256], want_packed)
    assert torch.equal(got[:, 256:], want_scales)


def test_empty_batch_is_a_noop():
    cache = _empty_cache(4)
    _store()(
        torch.zeros(0, LATENT, dtype=torch.bfloat16).cuda(),
        torch.zeros(0, dtype=torch.int64).cuda(),
        cache,
    )
    assert (cache.cpu() == 0xCD).all()


def test_paged_cache_view_and_scale_view_agree():
    """Write flat, read the scales through the aliasing view the kernel uses."""
    num_blocks, block_size = 3, 8
    num = num_blocks * block_size
    x = _latent(num, seed=10)

    paged = _empty_cache(num).reshape(num_blocks, block_size, ROW)
    _store()(
        x.to(torch.bfloat16).cuda(), torch.arange(num).cuda().to(torch.int64), paged
    )

    view = mx.scale_view(paged, LATENT, GROUP).cpu()
    _, want_scales = mx.quantize_pack(x.to(torch.bfloat16), GROUP)
    assert torch.equal(view.reshape(num, -1), want_scales)


def test_token_offset_does_not_overflow_int32():
    """A single launch may cover more than 2**31 source elements.

    ``tl.program_id`` is int32, so a token-major offset of
    ``token * kv_stride_token`` wraps once the latent holds more than 2**31
    elements -- 4,194,304 tokens at a 512-wide latent. Filling a whole cache in
    one launch crosses that line and faulted the GPU rather than degrading, so
    the token index has to be widened before it is scaled.

    Only the rows straddling the boundary are checked; quantization itself is
    covered elsewhere, and materializing a reference for every row of a 4 GiB
    tensor would dominate the runtime.
    """
    boundary = 2**31 // LATENT  # 4,194,304
    num_tokens = boundary + 64
    free, _ = torch.accelerator.get_memory_info()
    needed = num_tokens * (LATENT * 2 + ROW) + (1 << 30)
    if free < needed:
        pytest.skip(
            f"needs ~{needed / 1024**3:.1f} GiB free, has {free / 1024**3:.1f} GiB"
        )

    # A distinct, exactly-representable value per row, so a wrapped offset
    # reads a different row's data and the mismatch is unambiguous.
    latent = torch.empty(num_tokens, LATENT, dtype=torch.bfloat16, device="cuda")
    marks = torch.arange(num_tokens, device="cuda", dtype=torch.float32)
    latent.copy_(torch.exp2((marks % 8) - 4.0)[:, None].expand(-1, LATENT))

    cache = torch.zeros(num_tokens, ROW, dtype=torch.uint8, device="cuda")
    _store()(latent, torch.arange(num_tokens, device="cuda").to(torch.int64), cache)
    torch.accelerator.synchronize()

    probe = [boundary - 1, boundary, boundary + 1, num_tokens - 1]
    rows = torch.tensor(probe, device="cuda")
    want_packed, want_scales = mx.quantize_pack(latent[rows].cpu(), GROUP)
    got = cache[rows].cpu()
    assert torch.equal(got[:, : LATENT // 2], want_packed), (
        "rows past the 2**31-element boundary hold the wrong data: the token "
        "offset wrapped to a different row"
    )
    assert torch.equal(got[:, LATENT // 2 :], want_scales)


def test_mfma_permutation_is_a_bijection():
    """The reorder must move every data byte exactly once.

    A permutation that dropped or duplicated a byte would still produce a
    plausible-looking cache, and the read side would return wrong values rather
    than fail, so this is checked directly rather than inferred from a
    round-trip.
    """
    dst = mx.mfma_chunk_permutation(LATENT, GROUP)
    assert dst.shape == (LATENT // 2,)
    assert torch.equal(torch.sort(dst).values, torch.arange(LATENT // 2))


def test_mfma_permutation_groups_each_lane_contiguously():
    """Lane j's chunks must land in bytes [64j, 64j+64).

    This is the property the HIP read depends on: 64 contiguous bytes is four
    uint4, and those chunks are the stride-4 set MFMA gives lane j. If it does
    not hold, the wide load cannot feed the fragment and the design collapses to
    the 4-byte loads that measure 0.70x of bf16.
    """
    dst = mx.mfma_chunk_permutation(LATENT, GROUP)
    # Natural byte i belongs to chunk i // 4; it lands at dst[i].
    natural_chunk = torch.arange(LATENT // 2) // 4
    for j in range(4):
        block = (dst >= 64 * j) & (dst < 64 * (j + 1))
        chunks = torch.unique(natural_chunk[block])
        assert chunks.numel() == 16, (j, chunks)
        assert torch.equal(chunks, torch.arange(j, 64, 4)), (j, chunks)


def test_permuted_store_matches_the_reference_permutation():
    """The kernel's permuted write must equal permuting the natural layout."""
    x = _latent(16, seed=21)
    natural = _empty_cache(16)
    permuted = _empty_cache(16)
    slots = torch.arange(16).cuda().to(torch.int64)
    xb = x.to(torch.bfloat16).cuda()

    _store()(xb, slots, natural)
    _store()(xb, slots, permuted, permute_for_mfma=True)

    dst = mx.mfma_chunk_permutation(LATENT, GROUP)
    want = natural.cpu().clone()
    want[:, dst] = natural.cpu()[:, : LATENT // 2]
    assert torch.equal(permuted.cpu()[:, : LATENT // 2], want[:, : LATENT // 2])
    # Scales are untouched by the reorder.
    assert torch.equal(
        permuted.cpu()[:, LATENT // 2 :], natural.cpu()[:, LATENT // 2 :]
    )
