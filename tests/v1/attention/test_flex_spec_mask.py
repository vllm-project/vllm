# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlexAttention as the reference verifier of speculative plans.

A batch laid out as a plan (``tree_parents`` set) carries, per token, the rows of
its plan it sees (``tree_ancestors``: its ancestors, or the plan's custom mask)
and its position (``positions``). The Flex builder turns them into a ``mask_mod``;
these tests compare it, and Flex's output, with dense attention over the same
visibility computed from the plan directly. The ``mask_mod`` tests run anywhere;
the forward runs Flex eagerly on CPU and compiled on a GPU.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionMetadata,
    FlexAttentionMetadataBuilder,
    flex_attention_compiled,
    get_kernel_options,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

try:
    from torch.nn.attention.flex_attention import flex_attention
except ImportError:  # pragma: no cover - torch without FlexAttention
    flex_attention = None

HAS_GPU = current_platform.is_cuda_alike()
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
BLOCK_SIZE = 16
NUM_BLOCKS = 48
NUM_HEADS, NUM_KV_HEADS, HEAD_SIZE = 4, 2, 16


def _builder() -> FlexAttentionMetadataBuilder:
    model_config = SimpleNamespace(
        get_num_attention_heads=lambda parallel_config: NUM_HEADS,
        get_num_kv_heads=lambda parallel_config: NUM_KV_HEADS,
        get_head_size=lambda: HEAD_SIZE,
        max_model_len=NUM_BLOCKS * BLOCK_SIZE,
        rswa_window=None,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=None,
        cache_config=SimpleNamespace(num_gpu_blocks=NUM_BLOCKS),
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=256),
        attention_config=SimpleNamespace(
            flex_attn_q_block_size=None,
            flex_attn_block_m=None,
            flex_attn_kv_block_size=None,
            flex_attn_block_n=None,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=None),
    )
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.float32,
    )
    return FlexAttentionMetadataBuilder(spec, ["layer"], vllm_config, DEVICE)


@dataclass
class _Batch:
    """Requests verifying plans (and one prefill chunk), with the visibility the
    plans define computed from the plan directly."""

    context: list[int]  # computed tokens before the query
    query_lens: list[int]
    counts: list[int]  # plan rows: the last ``count`` query tokens
    parents: list[list[int]]
    sees: list[np.ndarray]  # [count, count] bool: row i sees row j
    offsets: list[list[int]]  # position - context of each plan row
    plan_mask: bool  # whether ``tree_ancestors`` carries the plans' masks

    def tokens(self):
        """``(request, slot, position, row)`` for each query token in order."""
        out = []
        for r, (ctx, q, count) in enumerate(
            zip(self.context, self.query_lens, self.counts)
        ):
            first_row = ctx + q - count
            for slot in range(ctx, ctx + q):
                row = slot - first_row
                if count > 1 and row >= 0:
                    out.append((r, slot, first_row + self.offsets[r][row], row))
                else:
                    out.append((r, slot, slot, -1))
        return out

    def visible(self, r, slot, row, kv_slot, window=None, position=None):
        """Whether query ``(r, slot, row)`` sees KV slot ``kv_slot`` of request r."""
        if kv_slot > slot:
            return False
        first_row = self.context[r] + self.query_lens[r] - self.counts[r]
        kv_row = kv_slot - first_row
        if row >= 0 and kv_row >= 0 and not self.sees[r][row, kv_row]:
            return False
        if window is None:
            return True
        kv_pos = kv_slot
        if row >= 0 and kv_row >= 0:
            kv_pos = first_row + self.offsets[r][kv_row]
        return abs(position - kv_pos) < window


def _ancestor_sees(parent: list[int]) -> np.ndarray:
    count = len(parent)
    sees = np.eye(count, dtype=bool)
    for i in range(1, count):
        sees[i] |= sees[parent[i]]
    return sees


def _depths(parent: list[int]) -> list[int]:
    depth = [0] * len(parent)
    for i in range(1, len(parent)):
        depth[i] = depth[parent[i]] + 1
    return depth


def _random_batch(seed: int, custom_mask: bool, explicit_positions: bool) -> _Batch:
    gen = np.random.default_rng(seed)
    num_reqs = int(gen.integers(2, 5))
    context, query_lens, counts, parents, sees, offsets = [], [], [], [], [], []
    for r in range(num_reqs):
        if r == 0:  # a prefill chunk: causal, no plan
            q, count, parent = int(gen.integers(3, 20)), 1, [-1]
        else:
            count = int(gen.integers(2, 10))
            # Some decodes put a token before the plan (its committed bonus).
            q = count + int(gen.integers(0, 2))
            parent = [-1] + [int(gen.integers(0, i)) for i in range(1, count)]
        context.append(int(gen.integers(0, 120)))
        query_lens.append(q)
        counts.append(count)
        parents.append(parent)
        if custom_mask and count > 1:
            # Any lower-triangular visibility; a row always sees itself.
            mask = np.tril(gen.random((count, count)) < 0.5)
            np.fill_diagonal(mask, True)
        else:
            mask = _ancestor_sees(parent)
        sees.append(mask)
        if explicit_positions:
            offsets.append([int(x) for x in gen.integers(0, count, count)])
        else:
            offsets.append(_depths(parent))
    return _Batch(
        context, query_lens, counts, parents, sees, offsets, plan_mask=custom_mask
    )


def _common(batch: _Batch, gen: np.random.Generator) -> CommonAttentionMetadata:
    """Metadata for ``batch`` as the runner builds it after ``PlanLayout.apply``."""
    num_reqs = len(batch.counts)
    seq_lens = [c + q for c, q in zip(batch.context, batch.query_lens)]
    max_blocks = -(-max(seq_lens) // BLOCK_SIZE)
    # Block 0 is the null block (padding), never a request's.
    assert num_reqs * max_blocks < NUM_BLOCKS
    block_table = torch.tensor(
        1
        + gen.permutation(NUM_BLOCKS - 1)[: num_reqs * max_blocks].reshape(
            num_reqs, max_blocks
        ),
        dtype=torch.int32,
    )
    tokens = batch.tokens()
    tree_parents, rows_seen, positions, slots = [], [], [], []
    for r, slot, position, row in tokens:
        if row >= 0:
            parent = batch.parents[r][row]
            tree_parents.append(parent)
            if batch.plan_mask:
                seen = batch.sees[r][row]
            else:
                seen = _ancestor_sees(batch.parents[r])[row]
            rows_seen.append(sum(1 << j for j in np.flatnonzero(seen)))
        else:
            tree_parents.append(-1)
            rows_seen.append(0)
        positions.append(position)
        block = block_table[r, slot // BLOCK_SIZE]
        slots.append(int(block) * BLOCK_SIZE + slot % BLOCK_SIZE)
    qsl = torch.tensor(np.concatenate([[0], np.cumsum(batch.query_lens)]))
    qsl = qsl.to(torch.int32)
    seq = torch.tensor(seq_lens, dtype=torch.int32)
    as_long = lambda xs: torch.tensor(xs, dtype=torch.int64, device=DEVICE)  # noqa: E731
    return CommonAttentionMetadata(
        query_start_loc=qsl.to(DEVICE),
        query_start_loc_cpu=qsl,
        seq_lens=seq.to(DEVICE),
        seq_lens_cpu_upper_bound=seq,
        num_reqs=num_reqs,
        num_actual_tokens=len(tokens),
        max_query_len=max(batch.query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=block_table.to(DEVICE),
        slot_mapping=as_long(slots),
        causal=True,
        positions=as_long(positions),
        tree_parents=torch.tensor(tree_parents, dtype=torch.int32, device=DEVICE),
        tree_ancestors=as_long(rows_seen),
    )


def _build(batch: _Batch, seed: int, window: int | None = None):
    gen = np.random.default_rng(1000 + seed)
    common = _common(batch, gen)
    metadata = _builder().build(common_prefix_len=0, common_attn_metadata=common)
    if window is not None:
        # What FlexAttentionImpl.forward does for a sliding-window layer.
        metadata.sliding_window = window
        metadata.mask_mod = metadata.get_mask_mod()
        metadata.block_mask = (
            metadata._build_block_mask_direct()
            if metadata.direct_build
            else metadata.build_block_mask()
        )
    return common, metadata


def _dense_visibility(batch: _Batch, common, window=None) -> torch.Tensor:
    """[num_tokens, NUM_BLOCKS * BLOCK_SIZE] bool over physical KV slots."""
    block_table = common.block_table_tensor.cpu()
    seen = torch.zeros(
        common.num_actual_tokens, NUM_BLOCKS * BLOCK_SIZE, dtype=torch.bool
    )
    for t, (r, slot, position, row) in enumerate(batch.tokens()):
        for kv in range(slot + 1):
            if batch.visible(r, slot, row, kv, window, position):
                physical = int(block_table[r, kv // BLOCK_SIZE]) * BLOCK_SIZE
                seen[t, physical + kv % BLOCK_SIZE] = True
    return seen


def _mask_mod_grid(metadata: FlexAttentionMetadata) -> torch.Tensor:
    q = torch.arange(metadata.num_actual_tokens, device=DEVICE)[:, None]
    kv = torch.arange(NUM_BLOCKS * BLOCK_SIZE, device=DEVICE)[None, :]
    zero = torch.zeros((), dtype=torch.long, device=DEVICE)
    return metadata.mask_mod(zero, zero, q, kv).cpu()


CASES = [
    pytest.param(False, False, id="tree"),
    pytest.param(True, False, id="mask"),
    pytest.param(False, True, id="tree-positions"),
    pytest.param(True, True, id="mask-positions"),
]


@pytest.mark.parametrize("custom_mask, explicit_positions", CASES)
@pytest.mark.parametrize("seed", range(4))
def test_mask_mod_is_the_plan_visibility(custom_mask, explicit_positions, seed):
    """Plan rows see the committed prefix and the rows the plan gives them;
    tokens outside a plan stay causal."""
    batch = _random_batch(seed, custom_mask, explicit_positions)
    common, metadata = _build(batch, seed)
    assert torch.equal(_mask_mod_grid(metadata), _dense_visibility(batch, common))


@pytest.mark.parametrize("custom_mask", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_swa_by_position(custom_mask, seed):
    """A sliding window measures plan rows by their position (depth or explicit
    offset), not by KV slot: a row at depth 1 in slot 7 of its plan is 1 past
    its root, and the block mask keeps every block such a row needs."""
    batch = _random_batch(seed, custom_mask, explicit_positions=seed % 2 == 1)
    window = int(np.random.default_rng(seed).integers(2, 24))
    common, metadata = _build(batch, seed, window)
    expected = _dense_visibility(batch, common, window)
    assert torch.equal(_mask_mod_grid(metadata), expected)
    # By slot, the deepest-slot rows of a wide plan would lose rows they see.
    by_slot = _dense_visibility(
        _Batch(**{**batch.__dict__, "offsets": [list(range(c)) for c in batch.counts]}),
        common,
        window,
    )
    if any(c > window for c in batch.counts):
        assert not torch.equal(by_slot, expected)


def test_swa_by_position_example():
    """A 4-row star (every row a child of the root) at context 20, window 2: by
    position each leaf (position 21) sees the root (20) and itself; by slot the
    last leaf (slot 23) would see neither the root nor the prefix."""
    batch = _Batch(
        context=[20],
        query_lens=[4],
        counts=[4],
        parents=[[-1, 0, 0, 0]],
        sees=[_ancestor_sees([-1, 0, 0, 0])],
        offsets=[[0, 1, 1, 1]],
        plan_mask=False,
    )
    common, metadata = _build(batch, 0, window=2)
    grid = _mask_mod_grid(metadata)
    physical = int(common.block_table_tensor[0, 1]) * BLOCK_SIZE  # slots 16..31
    leaf = grid[3, physical : physical + 8].tolist()
    # Slots 16..23: prefix 16..19 out of the window (|21 - 19| = 2), the root at
    # slot 20 seen, leaves 1-2 not its ancestors, itself seen.
    assert leaf == [False] * 4 + [True, False, False, True]
    assert torch.equal(grid, _dense_visibility(batch, common, window=2))


@pytest.mark.parametrize("seed", range(3))
def test_a_tree_row_sees_what_its_path_sees_as_a_chain(seed):
    """Tree equivalence: each row of a tree sees the committed prefix and the
    rows of its root-to-row path, the KV a chain of that path would see."""
    batch = _random_batch(seed, custom_mask=False, explicit_positions=False)
    common, metadata = _build(batch, seed)
    grid = _mask_mod_grid(metadata)
    block_table = common.block_table_tensor.cpu()
    for t, (r, slot, _, row) in enumerate(batch.tokens()):
        if row < 0:
            continue
        first_row = slot - row
        path, node = [], row
        while node >= 0:
            path.append(node)
            node = batch.parents[r][node]
        expected = set(range(first_row)) | {first_row + j for j in path}
        seen = {
            kv
            for kv in range(slot + 1)
            if grid[
                t,
                int(block_table[r, kv // BLOCK_SIZE]) * BLOCK_SIZE + kv % BLOCK_SIZE,
            ]
        }
        assert seen == expected, (r, row)


def _dense_attention(q, k_cache, v_cache, visible, scale):
    """Softmax attention of every query over the KV slots it sees."""
    group = NUM_HEADS // NUM_KV_HEADS
    k = k_cache.repeat_interleave(group, dim=1)  # [S, H, D]
    v = v_cache.repeat_interleave(group, dim=1)
    scores = torch.einsum("thd,shd->hts", q, k) * scale
    scores = scores.masked_fill(~visible[None], float("-inf"))
    return torch.einsum("hts,shd->thd", scores.softmax(dim=-1), v)


@pytest.mark.skipif(flex_attention is None, reason="torch has no FlexAttention")
@pytest.mark.parametrize("custom_mask, explicit_positions", CASES)
@pytest.mark.parametrize("window", [None, 5])
def test_flex_verifies_plans_like_dense_attention(
    custom_mask, explicit_positions, window
):
    seed = 7 + int(custom_mask) + 2 * int(explicit_positions)
    batch = _random_batch(seed, custom_mask, explicit_positions)
    common, metadata = _build(batch, seed, window)
    torch.manual_seed(seed)
    num_slots = NUM_BLOCKS * BLOCK_SIZE
    num_tokens = common.num_actual_tokens
    q = torch.randn(num_tokens, NUM_HEADS, HEAD_SIZE, device=DEVICE)
    k_cache = torch.randn(num_slots, NUM_KV_HEADS, HEAD_SIZE, device=DEVICE)
    v_cache = torch.randn(num_slots, NUM_KV_HEADS, HEAD_SIZE, device=DEVICE)
    scale = HEAD_SIZE**-0.5

    # The backend's call: its compiled flex with its kernel options (these
    # keep a short query on the main kernel, whose tiles fit 16-token blocks).
    query = q.permute(1, 0, 2)[None]
    assert metadata.block_mask is not None
    block_m, block_n = metadata.block_mask.BLOCK_SIZE
    run = flex_attention if DEVICE.type == "cpu" else flex_attention_compiled
    out = run(
        query,
        k_cache.permute(1, 0, 2)[None],
        v_cache.permute(1, 0, 2)[None],
        block_mask=metadata.block_mask,
        scale=scale,
        enable_gqa=True,
        kernel_options=get_kernel_options(
            query, block_m, block_n, metadata.direct_build
        ),
    )[0].permute(1, 0, 2)

    visible = _dense_visibility(batch, common, window).to(DEVICE)
    expected = _dense_attention(q, k_cache, v_cache, visible, scale)
    torch.testing.assert_close(out, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "features, kind, reason",
    [
        ({}, "mask", "FlexAttentionMetadataBuilder verifies mask"),
        ({"sliding_window": 128}, "mask", "FlexAttentionMetadataBuilder verifies mask"),
        (
            {"logits_soft_cap": 30.0, "has_sinks": True},
            "chain",
            "FlexAttentionMetadataBuilder verifies trees on full and "
            "sliding-window attention only, not with attention sinks, logits "
            "soft cap",
        ),
        ({"has_alibi": True}, "chain", "ALiBi"),
        ({"chunk_size": 64}, "chain", "chunked local attention"),
        ({"kv_sharing_target": "layers.0"}, "chain", "KV sharing"),
        ({"kv_cache_dtype": "fp8"}, "chain", "a quantized KV cache"),
    ],
)
def test_flex_verifies_masks_on_full_and_sliding_layers_only(features, kind, reason):
    builder = object.__new__(FlexAttentionMetadataBuilder)
    got_kind, why = builder.spec_verify_limit(**features)
    assert got_kind == kind
    assert reason in why
