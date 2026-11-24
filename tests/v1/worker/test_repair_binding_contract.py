# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The binding-contract repair matrix (Layout-Contract closure, task 2).

Every case uses distinguishable sentinel values and checks the BACKING
storage (an exact cell-diff against a pre-write snapshot):

  1. Valid Ascend per-K/V binding with num_kv_heads == block_size:
     accepted; the abort re-zero clears exactly the authorized token
     rows in both K and V.
  2. Contradictory or missing binding geometry: refused before any
     write; both buffers remain intact.
  3. Core logical [B, H, N, C] with equal shapes and different physical
     strides: both layouts clear the same logical rows directly in the
     backing storage.
  4. Core cache with exactly two blocks: the binding identifies the
     leading dimension as blocks.
  5. Partial write followed by an exception: cleanup covers the
     throwing layer and preserves neighboring slots and blocks.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.kv_connector.unit.utils import create_vllm_config
from tests.v1.worker.test_repair_commit_checks import (
    _cpu_meta,
)
from vllm.forward_context import set_forward_context
from vllm.v1.core.sched.output import CachedRequestData, RepairWork, SchedulerOutput
from vllm.v1.worker.gpu.repair_executor import (
    RepairViewBinding,
    _repair_cache_views,
    _repair_geometry_from_binding,
    _rezero_repair_span,
    core_logical_repair_binding,
    execute_repair_only,
    per_kv_repair_binding,
)

BS = 128
N_HEADS = 4
HEAD_DIM = 8
N_BLOCKS = 6
SPAN = (16, 80)  # the authorized window: token offsets 16..79 of block 2


def _sentinel(shape: torch.Size, base: float) -> torch.Tensor:
    """Every cell distinct and nonzero, so an exact diff identifies the
    written set without ambiguity."""
    return base + torch.arange(shape.numel(), dtype=torch.float32).reshape(shape)


def _span_work(layer_end: int = 2) -> tuple[dict, dict, torch.Tensor, object]:
    """A repair work over the PARTIAL span (16, 80) of block 2: rows
    0..15 and 80..127 of the same block are un-authorized neighbors."""
    work = RepairWork(
        request_id="req-a",
        repair_span=SPAN,
        layer_end=layer_end,
        restored_extent=512,
        plan_id=1,
        epoch=7,
        expected_write_block_ids=(2,),
    )
    slots = torch.arange(2 * BS + SPAN[0], 2 * BS + SPAN[1], dtype=torch.int64)
    meta = _cpu_meta(block_rows=[torch.tensor([2])], slot_mapping=slots)
    positions = torch.arange(SPAN[0], SPAN[1], dtype=torch.int64)
    return {"req-a": work}, {"req-a": SPAN[1] - SPAN[0]}, positions, meta


def _expected_token_rows_mask(
    like: torch.Tensor, block: int, lo: int, hi: int, token_axis: int
) -> torch.Tensor:
    """The exact cell set the re-zero is authorized to clear."""
    expected = torch.zeros_like(like, dtype=torch.bool)
    if token_axis == 2:
        expected[block, :, lo:hi, :] = True
    else:
        expected[block, lo:hi, :, :] = True
    return expected


class _Kv4DLayer:
    """An Ascend-convention layer: separate K and V views with the
    head-major [n, h, bs, d] binding (token axis 2)."""

    def __init__(self, layer_name: str, fill: float, n_blocks: int = N_BLOCKS):
        k = _sentinel(torch.Size((n_blocks, N_HEADS, BS, HEAD_DIM)), fill)
        v = _sentinel(torch.Size((n_blocks, N_HEADS, BS, HEAD_DIM)), fill + 10_000.0)
        attn = SimpleNamespace(kv_cache=[k, v], layer_name=layer_name)
        self.self_attn = SimpleNamespace(attn=attn)

    def __call__(self, positions, hidden_states, residual):
        return hidden_states + 1.0, residual


class _Kv4DBoomLayer(_Kv4DLayer):
    """Partially writes a few rows of the span IN the head-major
    convention, then raises -- the realistic partial-write shape."""

    def __call__(self, positions, hidden_states, residual):
        if int(positions.min()) < BS:
            cache = self.self_attn.attn.kv_cache[0]
            # Partial writes INSIDE the authorized span (16..31).
            cache[2, :, 16:32, :] = 7.5
            raise RuntimeError("injected partial-write layer failure")
        return hidden_states + 1.0, residual


class _Inner:
    def __init__(self, layers):
        self.layers = layers

    def embed_input_ids(self, input_ids):
        return torch.zeros(input_ids.shape[0], 8)


def _model(layers) -> SimpleNamespace:
    return SimpleNamespace(model=_Inner(layers))


def _sub_output(works, tokens) -> SchedulerOutput:
    return SchedulerOutput(
        repair_works=dict(works),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=dict(tokens),
        total_num_scheduled_tokens=sum(tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _run(model, works, tokens, positions, meta, binding):
    ctx = {f"model.layers.{i}": meta for i in range(len(model.model.layers))}
    with set_forward_context(ctx, create_vllm_config(model="facebook/opt-125m")):
        return execute_repair_only(
            model,
            _sub_output(works, tokens),
            positions.clone(),
            positions,
            list(works),
            view_binding=binding,
        )


class Case1HeadEqualsBlockSizeAccepted:
    def test_accepted_and_slot_exact_in_k_and_v(self):
        k = _sentinel(torch.Size((N_BLOCKS, BS, BS, HEAD_DIM)), 1.0)  # h == bs
        v = _sentinel(torch.Size((N_BLOCKS, BS, BS, HEAD_DIM)), 500_000.0)
        attn = SimpleNamespace(kv_cache=[k, v], layer_name="model.layers.0")
        layer = SimpleNamespace(self_attn=SimpleNamespace(attn=attn))
        binding = per_kv_repair_binding(BS, 2, "LBHNC")
        # ACCEPTED: the binding is authoritative when the shape cannot
        # confirm the axis (num_kv_heads == block_size).
        geo = _repair_geometry_from_binding(binding, _repair_cache_views(layer))
        assert geo.token_axis == 2

        works, tokens, positions, meta = _span_work(layer_end=1)
        k_before, v_before = k.clone(), v.clone()
        _rezero_repair_span([layer], meta, geo)

        expected = _expected_token_rows_mask(
            k, block=2, lo=SPAN[0], hi=SPAN[1], token_axis=2
        )
        assert torch.equal(k == 0, expected)
        assert torch.equal(v == 0, expected)
        # Nothing outside the authorized rows moved in either buffer.
        assert torch.equal(k[~expected], k_before[~expected])
        assert torch.equal(v[~expected], v_before[~expected])


class Case2ContradictoryOrMissingRefusedBeforeWrite:
    def test_missing_binding_refused(self):
        k = _sentinel(torch.Size((N_BLOCKS, N_HEADS, BS, HEAD_DIM)), 1.0)
        v = _sentinel(torch.Size((N_BLOCKS, N_HEADS, BS, HEAD_DIM)), 500_000.0)
        views = [k, v]
        with pytest.raises(ValueError, match="no repair view binding"):
            _repair_geometry_from_binding(None, views)

    def test_contradictory_bindings_refused_buffers_intact(self):
        k = _sentinel(torch.Size((N_BLOCKS, N_HEADS, BS, HEAD_DIM)), 1.0)
        v = _sentinel(torch.Size((N_BLOCKS, N_HEADS, BS, HEAD_DIM)), 500_000.0)
        shared = _sentinel(torch.Size((N_BLOCKS, N_HEADS, BS, HEAD_DIM)), 9.0)
        core_binding = core_logical_repair_binding(BS, "LBHNC")
        per_kv_binding = per_kv_repair_binding(BS, 2, "LBHNC")
        k_before, v_before = k.clone(), v.clone()
        # A core binding on separate K/V views is contradictory...
        with pytest.raises(ValueError, match="contradict the core"):
            _repair_geometry_from_binding(core_binding, [k, v])
        # ...and a per-K/V binding on the shared core view is too.
        with pytest.raises(ValueError, match="contradict the per-K/V"):
            _repair_geometry_from_binding(per_kv_binding, [shared])
        # A layout declaration that contradicts the strides is refused.
        with pytest.raises(ValueError, match="not consistent with the core"):
            _repair_geometry_from_binding(
                core_logical_repair_binding(BS, "LBNHC"), [shared]
            )
        # Both buffers remain intact on every refusal.
        assert torch.equal(k, k_before)
        assert torch.equal(v, v_before)

    def test_executor_refuses_before_any_write(self):
        layers = [_Kv4DLayer(f"model.layers.{i}", fill=100.0 + i) for i in range(4)]
        model = _model(layers)
        works, tokens, positions, meta = _span_work(layer_end=2)
        before = [
            (
                layer.self_attn.attn.kv_cache[0].clone(),
                layer.self_attn.attn.kv_cache[1].clone(),
            )
            for layer in layers
        ]
        # The runner supplies a core binding while the model binds
        # per-K/V views: refused before any layer executes.
        (ra,) = _run(
            model,
            works,
            tokens,
            positions,
            meta,
            core_logical_repair_binding(BS, "LBHNC"),
        )
        assert ra.aborted
        assert ra.executed_layers == 0
        assert not ra.structural_check
        for layer, (kb, vb) in zip(layers, before):
            assert torch.equal(layer.self_attn.attn.kv_cache[0], kb)
            assert torch.equal(layer.self_attn.attn.kv_cache[1], vb)


class Case3CoreEqualShapesDifferentStrides:
    def test_both_layouts_clear_the_same_logical_rows_in_storage(self):
        blocks, heads, chan = 2, BS, 2 * HEAD_DIM  # H == N: equal shapes
        # Token-major (LBNHC) physical storage, bound as the permuted
        # logical [B, H, N, C] view; head-major (LBHNC) physical storage
        # bound as itself (row-major).
        storage_lbnhc = _sentinel(torch.Size((blocks, heads, heads, chan)), 1.0)
        storage_lbhnc = _sentinel(torch.Size((blocks, heads, heads, chan)), 500_000.0)
        view_lbnhc = storage_lbnhc.permute(0, 2, 1, 3)
        view_lbhnc = storage_lbhnc
        assert view_lbnhc.shape == view_lbhnc.shape
        assert view_lbnhc.stride() != view_lbhnc.stride()

        binding_lbnhc = core_logical_repair_binding(BS, "LBNHC")
        binding_lbhnc = core_logical_repair_binding(BS, "LBHNC")
        geo_a = _repair_geometry_from_binding(binding_lbnhc, [view_lbnhc])
        geo_b = _repair_geometry_from_binding(binding_lbhnc, [view_lbhnc])
        assert geo_a == geo_b
        assert geo_a.token_axis == 2

        works, tokens, positions, meta = _span_work(layer_end=1)
        _rezero_repair_span(
            [
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        attn=SimpleNamespace(
                            kv_cache=view_lbnhc, layer_name="model.layers.0"
                        )
                    )
                )
            ],
            meta,
            geo_a,
        )
        _rezero_repair_span(
            [
                SimpleNamespace(
                    self_attn=SimpleNamespace(
                        attn=SimpleNamespace(
                            kv_cache=view_lbhnc, layer_name="model.layers.0"
                        )
                    )
                )
            ],
            meta,
            geo_b,
        )
        # The SAME logical rows cleared in both BACKING storages:
        # the span's tokens of block 1 are zero in both, every other
        # cell keeps its sentinel. storage_lbnhc is token-major (the
        # cleared logical rows live on physical dim 1); storage_lbhnc is
        # head-major (they live on physical dim 2).
        assert torch.equal(
            storage_lbnhc[1, SPAN[0] : SPAN[1]] == 0,
            torch.ones(heads, SPAN[1] - SPAN[0], chan, dtype=torch.bool),
        )
        assert float(storage_lbnhc[1, : SPAN[0]].abs().max()) > 0
        assert float(storage_lbnhc[1, SPAN[1] :].abs().max()) > 0
        assert float(storage_lbnhc[0].abs().max()) > 0
        # storage_lbhnc: head-major -> tokens on physical dim 2
        assert torch.equal(
            storage_lbhnc[1, :, SPAN[0] : SPAN[1]] == 0,
            torch.ones(heads, SPAN[1] - SPAN[0], chan, dtype=torch.bool),
        )
        assert float(storage_lbhnc[1, :, : SPAN[0]].abs().max()) > 0
        assert float(storage_lbhnc[1, :, SPAN[1] :].abs().max()) > 0
        assert float(storage_lbhnc[0].abs().max()) > 0


class Case4TwoBlocksLeadingDimIsBlocks:
    def test_exactly_two_blocks_indexed_as_blocks(self):
        # A core cache with EXACTLY two blocks: the leading dim is
        # blocks by the binding, never a K/V selector.
        cache = _sentinel(torch.Size((2, N_HEADS, BS, HEAD_DIM)), 2.0)
        layer = SimpleNamespace(
            self_attn=SimpleNamespace(
                attn=SimpleNamespace(kv_cache=cache, layer_name="model.layers.0")
            )
        )
        binding = core_logical_repair_binding(BS, "LBHNC")
        geo = _repair_geometry_from_binding(binding, _repair_cache_views(layer))
        assert geo.block_size == BS
        assert binding.blocks_axis == 0

        works, tokens, positions, meta = _span_work(layer_end=1)
        # Slots of block 1 (the second block): bidx = slot // BS = 1.
        slots = torch.arange(BS + SPAN[0], BS + SPAN[1], dtype=torch.int64)
        meta = _cpu_meta(block_rows=[torch.tensor([1])], slot_mapping=slots)
        _rezero_repair_span([layer], meta, geo)
        # Only block 1's span rows were cleared; block 0 is untouched.
        assert torch.equal(
            cache[1, :, SPAN[0] : SPAN[1]] == 0,
            torch.ones(N_HEADS, SPAN[1] - SPAN[0], HEAD_DIM, dtype=torch.bool),
        )
        assert float(cache[1, :, : SPAN[0]].abs().max()) > 0
        assert float(cache[0].abs().max()) > 0
        # A per-K/V binding would read the same leading dim as a K/V
        # selector: declaring it here is refused as contradictory.
        with pytest.raises(ValueError, match="contradict the per-K/V"):
            _repair_geometry_from_binding(
                per_kv_repair_binding(BS, 2, "LBHNC"), _repair_cache_views(layer)
            )

    def test_non_zero_blocks_axis_refused(self):
        cache = _sentinel(torch.Size((2, N_HEADS, BS, HEAD_DIM)), 2.0)
        binding = RepairViewBinding(
            kind="core_logical",
            block_size=BS,
            token_axis=2,
            blocks_axis=1,
            layout_name="LBHNC",
        )
        with pytest.raises(ValueError, match="blocks axis"):
            _repair_geometry_from_binding(binding, [cache])


class Case5PartialWriteExceptionCleanup:
    def test_cleanup_covers_throwing_layer_preserves_neighbors(self):
        layers = [_Kv4DLayer(f"model.layers.{i}", fill=10.0 + i) for i in range(4)]
        layers[1] = _Kv4DBoomLayer("model.layers.1", fill=11.0)
        model = _model(layers)
        works, tokens, positions, meta = _span_work(layer_end=2)
        before = [
            (
                layer.self_attn.attn.kv_cache[0].clone(),
                layer.self_attn.attn.kv_cache[1].clone(),
            )
            for layer in layers
        ]
        binding = per_kv_repair_binding(BS, 2, "LBHNC")

        (ra,) = _run(model, works, tokens, positions, meta, binding)
        assert ra.aborted
        assert ra.executed_layers == 1
        assert not ra.structural_check
        assert "partial-write layer failure" in (ra.error_detail or "")

        # Layer 0 (fully executed) and layer 1 (the RAISING layer,
        # whose partial 7.5 writes fell INSIDE the span) are both
        # cleared exactly on the authorized rows; every other cell --
        # neighboring slots of the same block, neighboring blocks --
        # keeps its snapshot value.
        expected = _expected_token_rows_mask(
            layers[0].self_attn.attn.kv_cache[0], 2, SPAN[0], SPAN[1], 2
        )
        for idx in (0, 1):
            for view, snap in zip(layers[idx].self_attn.attn.kv_cache, before[idx]):
                assert torch.equal(view == 0, expected)
                assert torch.equal(view[~expected], snap[~expected])
        # The layers beyond the work's layer_end never executed.
        for idx in (2, 3):
            for view, snap in zip(layers[idx].self_attn.attn.kv_cache, before[idx]):
                assert torch.equal(view, snap)
