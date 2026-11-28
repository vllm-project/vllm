# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused LUT-B weight quantizer: one single-warp Triton program per 8x64 tile.

Mirrors `quantize_lut_b` in `lut_b_utils`: block exponents from the circular
cut of the block amax phases, an exact weighted k-means codebook by dynamic
programming over the sorted tile, and `NUM_SCALE_ROUNDS` rounds of exponent
search plus refit. Results differ from the PyTorch path only through
floating-point summation order.
"""

import torch

from vllm.triton_utils import tl, triton

_BLOCKS = tl.constexpr(16)  # K32 scale blocks per tile, ordered row * 2 + K64 half
_BLOCK_K = tl.constexpr(32)
_BOOK = tl.constexpr(8)
_TILE = tl.constexpr(512)  # values per tile
_WORDS = tl.constexpr(64)  # 24-bit words of eight indices per tile
_LANES = tl.constexpr(32)  # split points scanned per step
_STRIDE = tl.constexpr(520)  # scratch row: prefix positions 0 .. 512, padded
_E4M3_MAX = tl.constexpr(448.0)
_UE8M0_BIAS = tl.constexpr(127.0)
_FP32_TINY = tl.constexpr(1.1754943508222875e-38)
_PROGRAMS_PER_SM = 32


@triton.jit
def _normalize_centers(centers, preserve_zero):
    centers = tl.minimum(tl.maximum(centers, -_E4M3_MAX), _E4M3_MAX)
    centers = centers.to(tl.float8e4nv).to(tl.float32)
    zero_index = tl.argmin(tl.abs(centers), axis=0)
    is_zero = preserve_zero & (tl.arange(0, _BOOK) == zero_index)
    return tl.sort(tl.where(is_zero, 0.0, centers))


@triton.jit
def _circle_exponents(blocks):
    """Block exponents that put every block amax in `[2**cut, 2**(cut + 1))`."""
    block = tl.arange(0, _BLOCKS)
    amax = tl.max(tl.abs(blocks), axis=1)
    valid = amax > 0
    log_amax = tl.log2(tl.maximum(amax, _FP32_TINY))
    phases = tl.sort(tl.where(valid, log_amax - tl.floor(log_amax), float("inf")))
    count = tl.sum(valid.to(tl.int32), axis=0)
    safe_count = tl.maximum(count, 1)
    following = (block + 1) % safe_count
    is_following = following[:, None] == block[None, :]
    following = tl.sum(tl.where(is_following, phases[None, :], 0.0), axis=1)
    gaps = tl.where(block == count - 1, following + 1.0, following) - phases
    gaps = tl.where(block < count, gaps, -1.0)
    start = (tl.argmax(gaps, axis=0) + 1) % safe_count
    cut = tl.sum(tl.where(block == start, phases, 0.0), axis=0)
    cut = tl.where(count > 0, cut, 0.0)
    exponent = tl.where(valid, tl.floor(log_amax - cut), 0.0)
    return tl.minimum(tl.maximum(exponent, -_UE8M0_BIAS), _UE8M0_BIAS)


@triton.jit
def _block_errors(blocks, exponent, tile_exponent, centers):
    """Squared error of each K32 block in units of the tile scale, and indices.

    A value's index counts the midpoints below it, and its decoded value
    accumulates the gaps between the centers it passes. Both are exact.
    """
    values = blocks * tl.exp2(-exponent)[:, None]
    book = tl.arange(0, _BOOK)
    lower = tl.sum(tl.where(book == 0, centers, 0.0), axis=0)
    decoded = tl.zeros_like(values) + lower
    indices = tl.zeros(values.shape, tl.int32)
    for i in tl.static_range(1, _BOOK):
        upper = tl.sum(tl.where(book == i, centers, 0.0), axis=0)
        above = values > (lower + upper) * 0.5
        indices += above.to(tl.int32)
        decoded += tl.where(above, upper - lower, 0.0)
        lower = upper
    error = (decoded - values) * tl.exp2(exponent - tile_exponent)[:, None]
    return tl.sum(error * error, axis=1), indices


@triton.jit
def _segment_cost(p0_end, p1_end, p2_end, p0_start, p1_start, p2_start):
    mass = p0_end - p0_start
    delta = p1_end - p1_start
    cost = p2_end - p2_start - delta * delta / tl.maximum(mass, _FP32_TINY)
    return tl.where(mass > 0, cost, 0.0)


@triton.jit
def _fit_codebook(blocks, exponent, preserve_zero, scratch, splits):
    """Exact weighted k-means codebook of `blocks * 2**-exponent`.

    `scratch` holds this program's prefix sums and two DP rows, `splits` its
    best split per DP stage and prefix length.
    """
    position = tl.arange(0, _TILE)
    values = tl.reshape(blocks * tl.exp2(-exponent)[:, None], (_TILE,))
    # Sort with each value's weight exponent in the low 8 bits of its key;
    # those bits are zero for values scaled from bf16 or fp16 weights.
    shift = (tl.max(exponent, axis=0) - exponent).to(tl.int32)
    shift = tl.reshape(tl.broadcast_to(shift[:, None], (_BLOCKS, _BLOCK_K)), (_TILE,))
    bits = values.to(tl.int32, bitcast=True)
    keys = tl.sort(((bits ^ ((bits >> 31) & 0x7FFFFFFF)) & ~0xFF) | shift)
    bits = tl.where(keys < 0, keys | 0xFF, keys & ~0xFF)
    values = (bits ^ ((bits >> 31) & 0x7FFFFFFF)).to(tl.float32, bitcast=True)
    weights = tl.exp2(-2.0 * (keys & 0xFF).to(tl.float32))

    total = tl.sum(weights, axis=0)
    mean = tl.sum(weights * values, axis=0) / tl.maximum(total, _FP32_TINY)
    centered = values - mean
    p0 = scratch
    p1 = scratch + _STRIDE
    p2 = scratch + 2 * _STRIDE
    tl.store(p0 + 1 + position, tl.cumsum(weights, 0))
    tl.store(p1 + 1 + position, tl.cumsum(weights * centered, 0))
    tl.store(p2 + 1 + position, tl.cumsum(weights * centered * centered, 0))
    tl.store(p0, 0.0)
    tl.store(p1, 0.0)
    tl.store(p2, 0.0)
    tl.debug_barrier()

    # One segment over each prefix. Stage s keeps its costs in row 3 + s % 2.
    end = 1 + position
    first = _segment_cost(
        tl.load(p0 + end), tl.load(p1 + end), tl.load(p2 + end), 0.0, 0.0, 0.0
    )
    tl.store(scratch + 4 * _STRIDE + end, first)
    tl.store(scratch + 4 * _STRIDE, float("inf"))
    tl.debug_barrier()

    # The best last split is monotone in the prefix length, so prefixes are
    # solved coarse to fine, each between the splits of its two neighbors.
    lanes = tl.arange(0, _LANES)
    for stage in range(2, _BOOK):
        previous = scratch + (3 + (stage - 1) % 2) * _STRIDE
        current = scratch + (3 + stage % 2) * _STRIDE
        stage_splits = splits + stage * _STRIDE
        tl.store(current, float("inf"))
        tl.store(stage_splits, 0)
        tl.debug_barrier()
        for level in range(9):
            half = (_TILE // 2) >> level
            for node in range(half, _TILE, 2 * half):
                low = tl.load(stage_splits + node - half)
                right = node + half
                high = tl.load(stage_splits + tl.minimum(right, _TILE - 1))
                high = tl.where(right < _TILE, high, _TILE - 1)
                high = tl.minimum(high, node - 1)
                q0 = tl.load(p0 + node)
                q1 = tl.load(p1 + node)
                q2 = tl.load(p2 + node)
                best = q0 * 0.0 + float("inf")
                best_split = low
                for start in range(low, high + 1, _LANES):
                    split = start + lanes
                    valid = split <= high
                    cost = _segment_cost(
                        q0,
                        q1,
                        q2,
                        tl.load(p0 + split, mask=valid, other=0.0),
                        tl.load(p1 + split, mask=valid, other=0.0),
                        tl.load(p2 + split, mask=valid, other=0.0),
                    )
                    cost += tl.load(previous + split, mask=valid, other=float("inf"))
                    cost = tl.where(valid, cost, float("inf"))
                    chunk, index = tl.min(cost, axis=0, return_indices=True)
                    best_split = tl.where(chunk < best, start + index, best_split)
                    best = tl.minimum(best, chunk)
                tl.store(current + node, best)
                tl.store(stage_splits + node, best_split)
            tl.debug_barrier()

    # The last segment ends at the full tile.
    previous = scratch + (3 + (_BOOK - 1) % 2) * _STRIDE
    last = _segment_cost(
        tl.load(p0 + _TILE),
        tl.load(p1 + _TILE),
        tl.load(p2 + _TILE),
        tl.load(p0 + position),
        tl.load(p1 + position),
        tl.load(p2 + position),
    ) + tl.load(previous + position)
    start = tl.argmin(last, axis=0)
    end = start * 0 + _TILE
    book = tl.arange(0, _BOOK)
    centers = tl.zeros((_BOOK,), tl.float32)
    for stage in range(_BOOK, 0, -1):
        if stage < _BOOK:
            start = tl.load(splits + stage * _STRIDE + end)
        start = tl.where(stage == 1, 0, start)
        mass = tl.load(p0 + end) - tl.load(p0 + start)
        delta = tl.load(p1 + end) - tl.load(p1 + start)
        center = mean + tl.where(mass > 0, delta / tl.maximum(mass, _FP32_TINY), 0.0)
        centers = tl.where(book == stage - 1, center, centers)
        end = start
    tl.debug_barrier()
    return _normalize_centers(centers, preserve_zero)


@triton.jit
def _lut_b_quantize_kernel(
    weight_ptr,
    packed_ptr,
    codebook_ptr,
    scale_ptr,
    scratch_ptr,
    splits_ptr,
    num_tiles,
    k,
    NUM_SCALE_ROUNDS: tl.constexpr,
):
    program = tl.program_id(0)
    scratch = scratch_ptr + program.to(tl.int64) * 5 * _STRIDE
    splits = splits_ptr + program.to(tl.int64) * _BOOK * _STRIDE
    k_tiles = k // (2 * _BLOCK_K)
    block = tl.arange(0, _BLOCKS)
    for index in range(program, num_tiles, tl.num_programs(0)):
        tile = tl.cast(index, tl.int64)
        row = tile // k_tiles * 8 + block // 2
        half = tile % k_tiles * 2 + block % 2
        offsets = (row * k + half * _BLOCK_K)[:, None] + tl.arange(0, _BLOCK_K)[None, :]
        blocks = tl.load(weight_ptr + offsets).to(tl.float32)
        preserve_zero = tl.sum(tl.sum((blocks == 0).to(tl.int32), axis=1), axis=0) > 0

        exponent = _circle_exponents(blocks)
        tile_exponent = tl.max(exponent, axis=0)
        centers = tl.zeros((_BOOK,), tl.float32)
        error = tl.full((_BLOCKS,), float("inf"), tl.float32)
        # Fit 0 is the initial fit. Each later fit first moves every block
        # exponent within +-2 (then +-1) against the current codebook, and the
        # refit codebook is kept only if the error does not grow.
        for fit in range(NUM_SCALE_ROUNDS + 1):
            if fit > 0:
                base = exponent
                radius = tl.where(fit == 1, 2, 1)
                # Moves -2, -1, +1, +2.
                for step in tl.static_range(4):
                    move = step - 2 + step // 2
                    size = 1 + (step % 3 == 0)
                    trial = base + move
                    trial = tl.minimum(tl.maximum(trial, -_UE8M0_BIAS), _UE8M0_BIAS)
                    trial_error, _ = _block_errors(
                        blocks, trial, tile_exponent, centers
                    )
                    better = (trial_error < error) & (size <= radius)
                    exponent = tl.where(better, trial, exponent)
                    error = tl.where(better, trial_error, error)
            refit = _fit_codebook(blocks, exponent, preserve_zero, scratch, splits)
            refit_error, _ = _block_errors(blocks, exponent, tile_exponent, refit)
            keep = tl.sum(refit_error, axis=0) <= tl.sum(error, axis=0)
            centers = tl.where(keep, refit, centers)
            error = tl.where(keep, refit_error, error)
        _, indices = _block_errors(blocks, exponent, tile_exponent, centers)

        # Eight 3-bit indices per little-endian 24-bit word.
        groups = tl.reshape(indices, (_WORDS, 8))
        words = tl.sum(groups << (3 * tl.arange(0, 8))[None, :], axis=1)
        byte = tl.arange(0, 4)
        packed = (words[:, None] >> (8 * byte)[None, :]) & 0xFF
        packed_offsets = tl.arange(0, _WORDS)[:, None] * 3 + byte[None, :]
        tl.store(
            packed_ptr + tile * (_WORDS * 3) + packed_offsets,
            packed.to(tl.uint8),
            mask=(byte < 3)[None, :],
        )
        tl.store(
            codebook_ptr + tile * _BOOK + tl.arange(0, _BOOK),
            centers.to(tl.float8e4nv),
        )
        tl.store(
            scale_ptr + row * (k // _BLOCK_K) + half,
            (exponent + _UE8M0_BIAS).to(tl.uint8),
        )


def quantize_lut_b_triton(
    weight: torch.Tensor, num_scale_rounds: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a `[E, N, K]` CUDA weight; see `lut_b_utils.quantize_lut_b`."""
    num_experts, n, k = weight.shape
    weight = weight.contiguous()
    tiles = (num_experts, n // 8, k // 64)
    num_tiles = tiles[0] * tiles[1] * tiles[2]
    packed = weight.new_empty(*tiles, 192, dtype=torch.uint8)
    codebooks = weight.new_empty(*tiles, 8, dtype=torch.float8_e4m3fn)
    scales = weight.new_empty(num_experts, n, k // 32, dtype=torch.uint8)
    if num_tiles:
        sms = torch.cuda.get_device_properties(weight.device).multi_processor_count
        programs = min(num_tiles, sms * _PROGRAMS_PER_SM)
        scratch = weight.new_empty(programs, 5, _STRIDE, dtype=torch.float32)
        splits = weight.new_empty(programs, _BOOK, _STRIDE, dtype=torch.int32)
        _lut_b_quantize_kernel[(programs,)](
            weight,
            packed,
            codebooks,
            scales,
            scratch,
            splits,
            num_tiles,
            k,
            NUM_SCALE_ROUNDS=num_scale_rounds,
            num_warps=1,
        )
    return packed, codebooks, scales
