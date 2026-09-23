# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

POISON = np.int32(-999)
DEVICE = current_platform.device_type
FP8_DS_MLA_ROW_BYTES = 656

# Straddles both the 32-lane (NVIDIA warp) and 64-lane (CDNA wavefront)
LANE_BOUNDARY_TOP_K = [1, 31, 32, 33, 63, 64, 65, 128]


def _hisparse_ops_available() -> bool:
    import vllm  # noqa: F401  (registers the stable-ABI ops)

    ops = getattr(torch.ops, "_C_cache_ops", None)
    if ops is None:
        return False
    return all(
        hasattr(ops, name)
        for name in (
            "hisparse_resolve_residency",
            "hisparse_invalidate_written_slots",
            "hisparse_gather_plan",
            "hisparse_gather_compact",
        )
    )


requires_hisparse_ops = pytest.mark.skipif(
    not _hisparse_ops_available(),
    reason="HiSparse ops are not compiled into this build",
)


def _dev(a: np.ndarray, dtype=torch.int32) -> torch.Tensor:
    return torch.as_tensor(a.copy()).to(dtype).to(DEVICE)


def _pinned(t: torch.Tensor) -> torch.Tensor:
    return t.pin_memory()


def _byte_pattern(num_rows: int, row_bytes: int, seed: int) -> np.ndarray:
    """Distinct, non-repeating bytes per row so a misaddressed copy shows up."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(num_rows, row_bytes), dtype=np.uint8)


def _distinct_hot_indices(
    rng: np.random.Generator, num_rows: int, top_k: int, hot_rows: int
) -> np.ndarray:
    """Destination slots that are globally unique, padded with -1.

    The gather kernels parallelise over plan entries with no ordering between
    them, so two entries naming the same destination race. The resolver only
    ever emits distinct slots, so unique destinations keep the comparison
    byte-exact while still covering the -1 skip path.
    """
    total = num_rows * top_k
    num_live = min(hot_rows, total)
    flat = np.full(total, -1, dtype=np.int32)
    flat[:num_live] = rng.permutation(hot_rows)[:num_live]
    rng.shuffle(flat)
    return flat.reshape(num_rows, top_k)


class Resolve:
    """One resolver invocation: no source or resident tables, so every valid
    top-k entry is a cold miss and the expected state is constructible."""

    HOT_BLOCK_SIZE = 8

    def __init__(self, top_k: int, hot_size: int, num_rows: int = 2, seed: int = 0):
        self.top_k, self.hot_size, self.num_rows = top_k, hot_size, num_rows
        self.host_rows = max(4096, top_k * 8)
        blocks_per_row = -(-hot_size // self.HOT_BLOCK_SIZE)
        self.hot_block_table = np.arange(
            num_rows * blocks_per_row, dtype=np.int32
        ).reshape(num_rows, blocks_per_row)
        self.hot_num_blocks = num_rows * blocks_per_row
        rng = np.random.default_rng(seed)
        self.global_indices = np.stack(
            [
                rng.choice(self.host_rows, size=top_k, replace=False).astype(np.int32)
                for _ in range(num_rows)
            ]
        )
        self.state_indices = np.arange(num_rows, dtype=np.int32)
        self.dgi = _dev(np.full((num_rows, hot_size), -1, np.int32))
        self.lru = _dev(
            np.tile(np.arange(hot_size, dtype=np.int16), (num_rows, 1)), torch.int16
        )

    def physical_row(self, row: int, slot: int) -> int:
        block = int(self.hot_block_table[row, slot // self.HOT_BLOCK_SIZE])
        return block * self.HOT_BLOCK_SIZE + slot % self.HOT_BLOCK_SIZE

    def step(self) -> dict[str, np.ndarray]:
        n, k = self.num_rows, self.top_k
        outs = {
            name: _dev(np.full((n, k), POISON, np.int32))
            for name in ("hot", "attn", "miss", "resolved", "swap_host", "swap_dev")
        }
        outs["valid"] = _dev(np.full(n, POISON, np.int32))
        outs["swap_counts"] = _dev(np.full(n, POISON, np.int32))

        torch.ops._C_cache_ops.hisparse_resolve_residency(
            _pinned(torch.zeros(self.host_rows, 16, dtype=torch.uint8)),
            torch.zeros(
                self.hot_num_blocks,
                self.HOT_BLOCK_SIZE,
                16,
                dtype=torch.uint8,
                device=DEVICE,
            ),
            _dev(self.hot_block_table),
            _dev(self.global_indices),
            outs["hot"],
            self.dgi,
            self.lru,
            _dev(self.state_indices),
            self.hot_size,
            outs["miss"],
            None,  # stats
            outs["attn"],
            self.HOT_BLOCK_SIZE,  # attention_block_stride
            None,  # request_ids
            None,  # source_block_table
            0,
            outs["resolved"],
            outs["valid"],
            outs["swap_host"],
            outs["swap_dev"],
            outs["swap_counts"],
            None,  # resident_block_table
            0,
            0,
        )
        torch.accelerator.synchronize()
        return {k_: v.cpu().numpy() for k_, v in outs.items()}


@requires_hisparse_ops
@pytest.mark.parametrize("top_k", LANE_BOUNDARY_TOP_K)
def test_resolve_cold_start_fills_slots_in_lru_order(top_k: int) -> None:
    """A cold resolve must take the oldest slots in order and rotate the LRU.

    With every slot stale, miss ``i`` takes LRU position ``i``, so the new
    order is a rotation by ``top_k``. This is exact *order*, not set
    membership: a compaction done 32-wide on a 64-wide wavefront leaves the
    set identical and only permutes it, which is the bug this file exists for.
    """
    case = Resolve(top_k=top_k, hot_size=2 * top_k, seed=top_k)
    out = case.step()

    want_hot = np.array(
        [[case.physical_row(r, i) for i in range(top_k)] for r in range(case.num_rows)],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(out["hot"], want_hot)
    np.testing.assert_array_equal(out["attn"], want_hot)
    np.testing.assert_array_equal(out["resolved"], case.global_indices)
    np.testing.assert_array_equal(
        out["miss"], np.ones((case.num_rows, top_k), np.int32)
    )
    np.testing.assert_array_equal(out["valid"], np.full(case.num_rows, top_k, np.int32))
    np.testing.assert_array_equal(
        out["swap_counts"], np.full(case.num_rows, top_k, np.int32)
    )
    np.testing.assert_array_equal(out["swap_host"], case.global_indices)
    np.testing.assert_array_equal(out["swap_dev"], want_hot)

    rotated = np.roll(np.arange(2 * top_k, dtype=np.int16), -top_k)
    np.testing.assert_array_equal(
        case.lru.cpu().numpy(), np.tile(rotated, (case.num_rows, 1))
    )


@requires_hisparse_ops
@pytest.mark.parametrize("top_k", LANE_BOUNDARY_TOP_K)
def test_resolve_warm_replay_hits_without_touching_state(top_k: int) -> None:
    """Re-resolving the same top-k must hit everything and leave state fixed.

    Hits move to MRU in scan order; since the cold step already left them
    there, a correct second step is a fixed point. An order bug in either step
    breaks it.
    """
    case = Resolve(top_k=top_k, hot_size=2 * top_k, seed=top_k + 1)
    first = case.step()
    lru_after_cold = case.lru.cpu().numpy().copy()
    dgi_after_cold = case.dgi.cpu().numpy().copy()

    second = case.step()

    np.testing.assert_array_equal(second["hot"], first["hot"])
    np.testing.assert_array_equal(
        second["swap_counts"], np.zeros(case.num_rows, np.int32)
    )
    np.testing.assert_array_equal(
        second["miss"], np.zeros((case.num_rows, top_k), np.int32)
    )
    np.testing.assert_array_equal(case.lru.cpu().numpy(), lru_after_cold)
    np.testing.assert_array_equal(case.dgi.cpu().numpy(), dgi_after_cold)


@requires_hisparse_ops
def test_resolve_padding_rows_publish_minus_one() -> None:
    """A CUDA-graph padding row (state index -1) writes -1 and no state."""
    case = Resolve(top_k=64, hot_size=128, num_rows=4, seed=99)
    case.state_indices = np.array([0, -1, 2, -1], dtype=np.int32)
    lru_before = case.lru.cpu().numpy().copy()

    out = case.step()

    for row in (1, 3):
        assert (out["hot"][row] == -1).all()
        assert (out["attn"][row] == -1).all()
        assert (out["resolved"][row] == -1).all()
        assert out["valid"][row] == 0
        assert out["swap_counts"][row] == 0
        np.testing.assert_array_equal(case.lru.cpu().numpy()[row], lru_before[row])


@requires_hisparse_ops
def test_resolve_out_of_range_lru_slot_degrades_to_remiss() -> None:
    """The corruption tripwire must survive the port: an out-of-range slot
    resolves invalid rather than becoming an unbounded read."""
    case = Resolve(top_k=8, hot_size=8, num_rows=1, seed=2)
    lru = case.lru.cpu().numpy()
    lru[0, 0] = np.int16(case.hot_size + 5)
    case.lru = _dev(lru, torch.int16)

    out = case.step()

    assert (out["hot"][0] == -1).any()
    for col in np.where(out["hot"][0] == -1)[0]:
        assert out["miss"][0, col] == 0, "an invalid entry must not be a miss"


@requires_hisparse_ops
@pytest.mark.parametrize("hot_multiple", [2, 5])
def test_resolve_production_shapes(hot_multiple: int) -> None:
    """GLM 5.3 decode (hot=2*top_k) and MTP-3 (hot=5*top_k) at top_k=2048 --
    the real shapes, and the ones whose shared-memory footprint must stay
    under CDNA's hard 64 KB LDS ceiling."""
    top_k = 2048
    case = Resolve(
        top_k=top_k, hot_size=top_k * hot_multiple, num_rows=1, seed=hot_multiple
    )
    out = case.step()

    want_hot = np.array([[case.physical_row(0, i) for i in range(top_k)]], np.int32)
    np.testing.assert_array_equal(out["hot"], want_hot)
    np.testing.assert_array_equal(out["swap_counts"], np.array([top_k], np.int32))
    rotated = np.roll(np.arange(top_k * hot_multiple, dtype=np.int16), -top_k)
    np.testing.assert_array_equal(case.lru.cpu().numpy(), rotated[None, :])


@requires_hisparse_ops
@pytest.mark.parametrize("num_rows", [5, 17, 64])
def test_resolve_many_rows_keeps_per_row_state_independent(num_rows: int) -> None:
    """Each row owns its own LRU region as the batch grows past one wavefront."""
    top_k = 64
    case = Resolve(top_k=top_k, hot_size=2 * top_k, num_rows=num_rows, seed=num_rows)
    out = case.step()

    want_hot = np.array(
        [[case.physical_row(r, i) for i in range(top_k)] for r in range(num_rows)],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(out["hot"], want_hot)
    rotated = np.roll(np.arange(2 * top_k, dtype=np.int16), -top_k)
    np.testing.assert_array_equal(
        case.lru.cpu().numpy(), np.tile(rotated, (num_rows, 1))
    )


def ref_gather_plan(
    host_cache: np.ndarray,
    hot_cache: np.ndarray,
    global_indices: np.ndarray,
    hot_indices: np.ndarray,
    miss_mask: np.ndarray,
    request_state_indices: np.ndarray,
    attention_indices: np.ndarray,
    hot_block_size: int,
    hot_block_stride: int,
) -> None:
    num_rows, top_k = global_indices.shape
    host_rows, row_bytes = host_cache.shape
    hot_rows = (hot_cache.size // hot_block_stride) * hot_block_size

    for row in range(num_rows):
        is_padding = int(request_state_indices[row]) < 0
        for col in range(top_k):
            dst = int(hot_indices[row, col])
            attention_indices[row, col] = (
                -1
                if is_padding or dst < 0
                else (dst // hot_block_size) * hot_block_size + dst % hot_block_size
            )
            if is_padding or int(miss_mask[row, col]) == 0:
                continue
            g = int(global_indices[row, col])
            if g < 0 or dst < 0 or dst >= hot_rows:
                continue
            off = (dst // hot_block_size) * hot_block_stride + (
                dst % hot_block_size
            ) * row_bytes
            # No source row: zero rather than serve stale bytes.
            hot_cache[off : off + row_bytes] = host_cache[g] if g < host_rows else 0


def ref_gather_compact(
    host_cache: np.ndarray,
    hot_cache: np.ndarray,
    miss_global_indices: np.ndarray,
    miss_hot_indices: np.ndarray,
    miss_counts: np.ndarray,
    hot_block_size: int,
    hot_block_stride: int,
) -> None:
    num_rows, top_k = miss_global_indices.shape
    host_rows, row_bytes = host_cache.shape
    hot_rows = (hot_cache.size // hot_block_stride) * hot_block_size

    for row in range(num_rows):
        for col in range(min(max(int(miss_counts[row]), 0), top_k)):
            g = int(miss_global_indices[row, col])
            dst = int(miss_hot_indices[row, col])
            if g < 0 or dst < 0 or dst >= hot_rows:
                continue
            off = (dst // hot_block_size) * hot_block_stride + (
                dst % hot_block_size
            ) * row_bytes
            hot_cache[off : off + row_bytes] = host_cache[g] if g < host_rows else 0


@requires_hisparse_ops
@pytest.mark.parametrize(
    "top_k,row_bytes",
    [(k, FP8_DS_MLA_ROW_BYTES) for k in LANE_BOUNDARY_TOP_K] + [(64, 16)],
)
def test_gather_plan_matches_reference(top_k: int, row_bytes: int) -> None:
    """Byte-exact gather, including the zero-fill path for absent sources."""
    num_rows, host_rows, hot_block_size = 2, 64, 8
    hot_num_blocks = max(4, -(-num_rows * top_k // hot_block_size))
    hot_rows = hot_block_size * hot_num_blocks
    rng = np.random.default_rng(top_k + row_bytes)

    host = _byte_pattern(host_rows, row_bytes, seed=top_k)
    hot = _byte_pattern(hot_rows, row_bytes, seed=top_k + 1).reshape(-1)
    # Mix of real rows, absent sources (g >= host_rows -> zero fill) and -1.
    global_indices = rng.integers(
        -1, host_rows + 8, size=(num_rows, top_k), dtype=np.int32
    )
    hot_indices = _distinct_hot_indices(rng, num_rows, top_k, hot_rows)
    miss_mask = rng.integers(0, 2, size=(num_rows, top_k), dtype=np.int32)
    state_indices = np.array([0, -1], dtype=np.int32)

    want_hot = hot.copy()
    want_attn = np.full((num_rows, top_k), POISON, dtype=np.int32)
    ref_gather_plan(
        host,
        want_hot,
        global_indices,
        hot_indices,
        miss_mask,
        state_indices,
        want_attn,
        hot_block_size,
        hot_block_size * row_bytes,
    )

    d_hot = torch.from_numpy(
        hot.copy().reshape(hot_num_blocks, hot_block_size, row_bytes)
    ).to(DEVICE)
    d_attn = _dev(np.full((num_rows, top_k), POISON, np.int32))
    torch.ops._C_cache_ops.hisparse_gather_plan(
        _pinned(torch.from_numpy(host.copy())),
        d_hot,
        _dev(global_indices),
        _dev(hot_indices),
        _dev(miss_mask),
        _dev(state_indices),
        d_attn,
        hot_block_size,
    )
    torch.accelerator.synchronize()

    np.testing.assert_array_equal(d_hot.cpu().numpy().reshape(-1), want_hot)
    np.testing.assert_array_equal(d_attn.cpu().numpy(), want_attn)


@requires_hisparse_ops
@pytest.mark.parametrize("top_k", [31, 32, 64, 65])
def test_gather_compact_matches_reference(top_k: int) -> None:
    """Compact gather copies exactly miss_count rows and no more."""
    num_rows, host_rows = 3, 64
    row_bytes, hot_block_size = FP8_DS_MLA_ROW_BYTES, 8
    hot_num_blocks = max(4, -(-num_rows * top_k // hot_block_size))
    hot_rows = hot_block_size * hot_num_blocks
    rng = np.random.default_rng(top_k)

    host = _byte_pattern(host_rows, row_bytes, seed=top_k + 2)
    hot = _byte_pattern(hot_rows, row_bytes, seed=top_k + 3).reshape(-1)
    miss_global = rng.integers(
        -1, host_rows + 4, size=(num_rows, top_k), dtype=np.int32
    )
    miss_hot = _distinct_hot_indices(rng, num_rows, top_k, hot_rows)
    # 0, a mid count, and an over-large count that must clamp to top_k.
    miss_counts = np.array([0, top_k // 2, top_k + 5], dtype=np.int32)

    want_hot = hot.copy()
    ref_gather_compact(
        host,
        want_hot,
        miss_global,
        miss_hot,
        miss_counts,
        hot_block_size,
        hot_block_size * row_bytes,
    )

    d_hot = torch.from_numpy(
        hot.copy().reshape(hot_num_blocks, hot_block_size, row_bytes)
    ).to(DEVICE)
    torch.ops._C_cache_ops.hisparse_gather_compact(
        _pinned(torch.from_numpy(host.copy())),
        d_hot,
        _dev(miss_global),
        _dev(miss_hot),
        _dev(miss_counts),
    )
    torch.accelerator.synchronize()

    np.testing.assert_array_equal(d_hot.cpu().numpy().reshape(-1), want_hot)


@requires_hisparse_ops
@pytest.mark.parametrize("region_stride", [31, 32, 64, 65, 256])
def test_invalidate_written_slots_clears_every_copy(region_stride: int) -> None:
    """Every copy of a rewritten slot must be invalidated, in every row."""
    num_states, num_tokens = 4, 16
    rng = np.random.default_rng(region_stride)

    dgi = rng.integers(-1, 40, size=(num_states, region_stride), dtype=np.int32)
    # Plant duplicates: one slot id appearing several times in a row must be
    # cleared everywhere, not just at its first occurrence.
    dgi[0, : min(5, region_stride)] = 7
    state_indices = np.array([0, 1, -1, 9], dtype=np.int32)
    req_id_per_token = rng.integers(-2, 6, size=num_tokens, dtype=np.int32)
    written_slots = rng.integers(-1, 40, size=num_tokens, dtype=np.int64)
    written_slots[0] = 7

    want = dgi.copy()
    for token_idx in range(num_tokens):
        req_idx = int(req_id_per_token[token_idx])
        if not 0 <= req_idx < state_indices.shape[0]:
            continue
        state_idx = int(state_indices[req_idx])
        slot = int(written_slots[token_idx])
        if not (0 <= state_idx < num_states) or slot < 0:
            continue
        want[state_idx][want[state_idx] == slot] = -1

    d_dgi = _dev(dgi)
    torch.ops._C_cache_ops.hisparse_invalidate_written_slots(
        d_dgi,
        _dev(state_indices),
        _dev(req_id_per_token),
        _dev(written_slots, torch.int64),
    )
    torch.accelerator.synchronize()

    np.testing.assert_array_equal(d_dgi.cpu().numpy(), want)
