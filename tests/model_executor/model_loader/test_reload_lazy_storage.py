# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Storage allocated on the first forward, not during post-load.

``test_post_load_storage_stability.py`` runs post-load twice and compares
storage. That is blind to anything allocated later: the MoE permute scratch
is created on the FIRST FORWARD, so a post-load-only sweep never touches
it and reports clean.

It is not a hypothetical gap. On stock 0.25.1 with 2xH200, an identity
reload of Qwen3-30B-A3B-W4A8 freed the scratch with the experts object it
belonged to; the next forward reallocated elsewhere while captured graphs
still held the old addresses, and replay died with an illegal memory
access. Re-attaching the pre-reload scratch made the fault disappear with
byte-identical addresses, which is what identified it.

So the lifecycle under test here is the real one:

    build -> post-load -> allocate lazily -> REBUILD -> allocate again

with the rebuild standing in for what a reload does to the owning object.
The property is that the second allocation returns the first one's storage.
"""

import pytest
import torch

from vllm.model_executor.reload_arena import (InitPolicy, ReloadArena,
                                              arena_scope, current_arena,
                                              get_reload_arena)
from vllm.platforms import current_platform

DEVICE = torch.device("cuda" if current_platform.is_cuda_alike() else "cpu")


class _LazyExperts:
    """The CutlassExperts shape: the arena is captured at construction
    because no scope is open at forward time, and the scratch is allocated
    on first use."""

    def __init__(self, arena_backed: bool = True):
        self._arena = current_arena() if arena_backed else None
        self._scratch: dict[str, torch.Tensor] | None = None

    def _alloc(self, slot: str, shape) -> torch.Tensor:
        if self._arena is not None:
            return self._arena.get_or_alloc(f"scratch.{slot}", shape,
                                            torch.int32, DEVICE,
                                            init=InitPolicy.PRESERVE)
        return torch.empty(shape, dtype=torch.int32, device=DEVICE)

    def forward(self) -> dict[str, torch.Tensor]:
        """Lazy allocation, exactly like _get_permute_scratch()."""
        if self._scratch is None:
            self._scratch = {
                name: self._alloc(name, (32, ))
                for name in ("permuted_idx", "inv_permuted_idx",
                             "sort_workspace")
            }
        return self._scratch


def _ptrs(scratch: dict[str, torch.Tensor]) -> dict[str, int]:
    return {k: v.data_ptr() for k, v in scratch.items()}


class TestLazyStorageSurvivesRebuild:

    def test_arena_backed_scratch_survives_the_owning_object_rebuild(self):
        """The property the reproduced W4A8 crash violated."""
        layer = torch.nn.Module()
        arena = get_reload_arena(layer)

        with arena_scope(arena):
            experts = _LazyExperts()
        before = _ptrs(experts.forward())  # first forward: lazy alloc

        with arena_scope(arena):  # what post-load does on reload
            rebuilt = _LazyExperts()
        after = _ptrs(rebuilt.forward())

        assert after == before, (
            "lazily-allocated scratch moved across the owning object's "
            "rebuild; captured graphs hold the previous addresses")

    def test_unmanaged_scratch_moves_and_is_detected(self):
        """Canary: without the arena the storage does move, so a green
        result above is a real property and not an inert assertion."""
        with arena_scope(ReloadArena("unused")):
            experts = _LazyExperts(arena_backed=False)
        before = _ptrs(experts.forward())
        rebuilt = _LazyExperts(arena_backed=False)
        after = _ptrs(rebuilt.forward())
        assert after != before

    def test_old_object_freed_after_rebuild_still_leaves_storage_valid(self):
        """Dropping the pre-reload owner must not free the storage: that is
        precisely what turned the W4A8 case into an illegal access."""
        layer = torch.nn.Module()
        arena = get_reload_arena(layer)

        with arena_scope(arena):
            experts = _LazyExperts()
        before = _ptrs(experts.forward())

        del experts  # reload drops the previous experts object
        with arena_scope(arena):
            rebuilt = _LazyExperts()
        after = _ptrs(rebuilt.forward())

        assert after == before
        # storage is genuinely alive, not merely a recycled address
        for name, tensor in rebuilt.forward().items():
            tensor.fill_(7)
            assert int(tensor[0]) == 7, name

    def test_scratch_is_visible_to_the_commit_gate(self):
        """Lazy allocation happens after capture-time snapshots are taken,
        so the gate must treat a slot that appears later as clean while
        still verifying it on subsequent reloads."""
        layer = torch.nn.Module()
        arena = get_reload_arena(layer)

        snap_before_forward = arena.snapshot()
        with arena_scope(arena):
            experts = _LazyExperts()
        experts.forward()

        # new slots after the snapshot are legitimate, not violations
        assert arena.verify(snap_before_forward) == []

        # but from now on they are covered
        snap = arena.snapshot()
        assert len(snap) == 3
        with arena_scope(arena):
            _LazyExperts().forward()
        assert arena.verify(snap) == []

    def test_shape_change_fails_closed(self):
        """A rebuild asking for a different scratch geometry is not an
        in-place update; refusing beats silently replacing storage."""
        layer = torch.nn.Module()
        arena = get_reload_arena(layer)
        with arena_scope(arena):
            _LazyExperts().forward()
        with pytest.raises(ValueError, match="incompatible spec"):
            arena.get_or_alloc("scratch.permuted_idx", (64, ), torch.int32,
                               DEVICE, init=InitPolicy.PRESERVE)


@pytest.mark.skipif(not current_platform.is_cuda_alike(),
                    reason="needs an accelerator")
def test_real_permute_scratch_reuses_storage_across_rebuild():
    """Drive the production MoEPermuteScratch rather than a stand-in, so
    this tracks the real allocation list instead of a copy of it."""
    from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
        MoEPermuteScratch)

    layer = torch.nn.Module()
    arena = get_reload_arena(layer)

    def build():
        return MoEPermuteScratch(
            max_num_tokens=8, topk=2, num_experts=4, num_local_experts=4,
            device=DEVICE, arena=arena,
        )

    first = build()
    before = {k: v.data_ptr() for k, v in vars(first).items()
              if hasattr(v, "data_ptr")}
    assert before, "no scratch tensors discovered"

    del first
    second = build()  # the reload's rebuild
    after = {k: v.data_ptr() for k, v in vars(second).items()
             if hasattr(v, "data_ptr")}

    moved = sorted(k for k, ptr in before.items() if after.get(k) != ptr)
    assert not moved, f"permute scratch moved across rebuild: {moved}"


@pytest.mark.skipif(not current_platform.is_cuda_alike(),
                    reason="needs an accelerator")
def test_real_permute_scratch_without_arena_moves():
    """Canary for the above: unmanaged, the production class does move."""
    from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
        MoEPermuteScratch)

    def build():
        return MoEPermuteScratch(
            max_num_tokens=8, topk=2, num_experts=4, num_local_experts=4,
            device=DEVICE, arena=None,
        )

    first = build()
    before = {k: v.data_ptr() for k, v in vars(first).items()
              if hasattr(v, "data_ptr")}
    second = build()
    after = {k: v.data_ptr() for k, v in vars(second).items()
             if hasattr(v, "data_ptr")}
    assert any(after.get(k) != ptr for k, ptr in before.items())
