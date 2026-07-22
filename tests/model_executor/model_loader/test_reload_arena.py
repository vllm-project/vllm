# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ReloadArena semantics.

Each test encodes one property a reproduced #48312 failure violated; the
matching case name from the H200 repro suite is noted inline.
"""

import pytest
import torch
from torch import nn

from vllm.model_executor.reload_arena import (
    InitPolicy, ReloadArena, arena_scope, current_arena, get_reload_arena,
    peek_reload_arena, snapshot_model_arenas, verify_model_arenas)


class TestGetOrAlloc:

    def test_same_slot_same_storage_across_rebuilds(self):
        # cat1_cutlass_fp8 / cat1_marlin: rebuilt objects must reacquire
        # the same storage, not allocate fresh
        arena = ReloadArena("layer")
        a = arena.get_or_alloc("ws", (8, ), torch.int64, "cpu")
        b = arena.get_or_alloc("ws", (8, ), torch.int64, "cpu")
        assert a.data_ptr() == b.data_ptr()
        assert a is b

    def test_zero_policy_zeroes_on_reacquire(self):
        # marlin workspace semantics: reuse + zero (the #48438 existing=)
        arena = ReloadArena("layer")
        a = arena.get_or_alloc("ws", (4, ), torch.float32, "cpu",
                               init=InitPolicy.ZERO)
        a.fill_(7.0)
        b = arena.get_or_alloc("ws", (4, ), torch.float32, "cpu",
                               init=InitPolicy.ZERO)
        assert b.data_ptr() == a.data_ptr()
        assert torch.all(b == 0)

    def test_preserve_policy_keeps_contents(self):
        # permute-scratch semantics: contents carried across reload
        # (probe B on H200: pinning old scratch made the fault vanish)
        arena = ReloadArena("layer")
        a = arena.get_or_alloc("scratch", (4, ), torch.int32, "cpu",
                               init=InitPolicy.PRESERVE)
        a.fill_(3)
        b = arena.get_or_alloc("scratch", (4, ), torch.int32, "cpu",
                               init=InitPolicy.PRESERVE)
        assert b.data_ptr() == a.data_ptr()
        assert torch.all(b == 3)

    def test_spec_mismatch_raises_never_reallocates(self):
        arena = ReloadArena("layer")
        arena.get_or_alloc("ws", (8, ), torch.int64, "cpu")
        with pytest.raises(ValueError, match="incompatible spec"):
            arena.get_or_alloc("ws", (16, ), torch.int64, "cpu")
        with pytest.raises(ValueError, match="incompatible spec"):
            arena.get_or_alloc("ws", (8, ), torch.int32, "cpu")


class TestPut:

    def test_first_put_adopts_private_storage(self):
        # cat1_mla: W_UV may be a view of a live parameter; the published
        # tensor must own private storage
        arena = ReloadArena("layer")
        base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        view = base.t()
        stable = arena.put("W_UV", view)
        assert stable.data_ptr() != base.data_ptr()
        assert torch.equal(stable, view)
        assert stable.is_contiguous()

    def test_second_put_copies_into_same_storage(self):
        # cat1_mla + cat2_sinks in one property: address stable, value fresh
        arena = ReloadArena("layer")
        s1 = arena.put("W_UV", torch.ones(3, 4))
        p1 = s1.data_ptr()
        s2 = arena.put("W_UV", torch.full((3, 4), 2.0))
        assert s2.data_ptr() == p1
        assert torch.all(s2 == 2.0)
        assert torch.all(s1 == 2.0)  # same storage

    def test_put_value_refresh_from_updated_source(self):
        # cat2_sinks: recompute-from-source must land in place. The stock
        # bug was `if dtype != fp32: convert` -- a no-op second pass.
        arena = ReloadArena("layer")
        source = torch.arange(4, dtype=torch.bfloat16)
        runtime = arena.put("sinks_f32", source.to(torch.float32))
        source.copy_(torch.arange(4, dtype=torch.bfloat16) * 3 + 1)
        runtime2 = arena.put("sinks_f32", source.to(torch.float32))
        assert runtime2.data_ptr() == runtime.data_ptr()
        assert torch.equal(runtime, source.to(torch.float32))

    def test_put_shape_mismatch_raises(self):
        arena = ReloadArena("layer")
        arena.put("d", torch.ones(3))
        with pytest.raises(ValueError, match="incompatible spec"):
            arena.put("d", torch.ones(4))


class TestSnapshotVerify:

    def test_clean_roundtrip(self):
        arena = ReloadArena("layer")
        arena.get_or_alloc("ws", (8, ), torch.int64, "cpu")
        arena.put("d", torch.ones(3))
        snap = arena.snapshot()
        arena.get_or_alloc("ws", (8, ), torch.int64, "cpu")
        arena.put("d", torch.zeros(3))
        assert arena.verify(snap) == []

    def test_moved_detected(self):
        arena = ReloadArena("layer")
        arena.get_or_alloc("ws", (8, ), torch.int64, "cpu")
        snap = arena.snapshot()
        arena._slots["ws"] = torch.empty(8, dtype=torch.int64)  # simulate bug
        v = arena.verify(snap)
        assert len(v) == 1 and v[0].kind == "moved"

    def test_gone_detected(self):
        # the tolerated_gone lesson: gone is NEVER clean
        arena = ReloadArena("layer")
        arena.get_or_alloc("scratch", (8, ), torch.int32, "cpu")
        snap = arena.snapshot()
        del arena._slots["scratch"]
        v = arena.verify(snap)
        assert len(v) == 1 and v[0].kind == "gone"

    def test_new_lazy_slot_after_snapshot_is_clean(self):
        arena = ReloadArena("layer")
        snap = arena.snapshot()
        arena.get_or_alloc("late", (4, ), torch.int32, "cpu")
        assert arena.verify(snap) == []


class TestAmbientScope:

    def test_scope_resolution_and_reset(self):
        arena = ReloadArena("layer")
        assert current_arena() is None
        with arena_scope(arena) as a:
            assert a is arena
            assert current_arena() is arena
        assert current_arena() is None

    def test_constructor_captures_arena_for_lazy_alloc(self):
        # the permute-scratch pattern: capture at construction, allocate
        # at first forward (outside any scope)
        class Experts:
            def __init__(self):
                self.arena = current_arena()
                self.scratch = None

            def forward_alloc(self):
                if self.scratch is None:
                    if self.arena is not None:
                        self.scratch = self.arena.get_or_alloc(
                            "scratch", (4, ), torch.int32, "cpu",
                            init=InitPolicy.PRESERVE)
                    else:
                        self.scratch = torch.empty(4, dtype=torch.int32)
                return self.scratch

        arena = ReloadArena("layer")
        with arena_scope(arena):
            e1 = Experts()
        s1 = e1.forward_alloc()  # lazy alloc outside the scope
        with arena_scope(arena):
            e2 = Experts()  # PWAL rebuild
        s2 = e2.forward_alloc()
        assert s2.data_ptr() == s1.data_ptr()


class TestModelHelpers:

    def _model(self):
        class Layer(nn.Module):
            pass

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = Layer()

        return Model()

    def test_arena_not_in_state_dict_or_buffers(self):
        model = self._model()
        arena = get_reload_arena(model.layer)
        arena.get_or_alloc("ws", (4, ), torch.float32, "cpu")
        assert "layer.ws" not in model.state_dict()
        assert list(model.layer.buffers()) == []
        assert list(model.layer.parameters()) == []

    def test_snapshot_verify_across_model(self):
        model = self._model()
        arena = get_reload_arena(model.layer)
        arena.get_or_alloc("ws", (4, ), torch.float32, "cpu")
        snaps = snapshot_model_arenas(model)
        assert verify_model_arenas(model, snaps) == []
        arena._slots["ws"] = torch.empty(4)  # simulate rebind bug
        problems = verify_model_arenas(model, snaps)
        assert len(problems) == 1 and "moved" in problems[0]

    def test_peek_does_not_create(self):
        model = self._model()
        assert peek_reload_arena(model.layer) is None
        get_reload_arena(model.layer)
        assert peek_reload_arena(model.layer) is not None
