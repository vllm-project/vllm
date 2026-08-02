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
    InitPolicy, ReloadArena, _canonical_device, arena_scope, current_arena,
    get_reload_arena, peek_reload_arena, snapshot_model_arenas,
    verify_model_arenas)


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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
    def test_unindexed_accelerator_device_matches_indexed(self):
        # observed live: config-level "cuda" vs slot tensor's "cuda:0"
        # must not be treated as a respecification -- the mismatch aborted
        # the first post-reload forward
        arena = ReloadArena("layer")
        a = arena.get_or_alloc("s", (4, ), torch.int32, "cuda")
        b = arena.get_or_alloc("s", (4, ), torch.int32, "cuda:0")
        c = arena.get_or_alloc("s", (4, ), torch.int32,
                               torch.device("cuda"))
        assert a.data_ptr() == b.data_ptr() == c.data_ptr()


class TestDeviceCanonicalization:
    """Device resolution must not be CUDA-specific: the same unindexed-vs-
    indexed mismatch would otherwise recur on every other accelerator."""

    def test_cpu_is_not_given_an_index(self):
        # CPU tensors report a bare "cpu"; canonicalizing to cpu:0 would
        # manufacture the mismatch this function exists to prevent
        assert _canonical_device("cpu") == torch.device("cpu")
        assert _canonical_device(torch.device("cpu")) == torch.device("cpu")
        arena = ReloadArena("layer")
        a = arena.get_or_alloc("s", (4, ), torch.int32, "cpu")
        b = arena.get_or_alloc("s", (4, ), torch.int32, torch.device("cpu"))
        assert a.data_ptr() == b.data_ptr()

    def test_explicit_index_is_left_alone(self):
        assert _canonical_device("cuda:3") == torch.device("cuda", 3)
        assert _canonical_device("xpu:2") == torch.device("xpu", 2)

    def test_unavailable_backend_left_as_given(self):
        # no guessing for a backend torch cannot resolve; a later spec
        # mismatch is loud and safe, a wrong guess is not
        assert _canonical_device("xpu").type == "xpu"

    def test_resolution_is_generic_over_device_type(self, monkeypatch):
        """Any torch-registered accelerator resolves through its own device
        module -- this is what keeps the fix from being CUDA-only."""

        class FakeModule:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def current_device():
                return 7

        monkeypatch.setattr(torch, "get_device_module",
                            lambda t: FakeModule if t == "xpu"
                            else torch.get_device_module(t))
        assert _canonical_device("xpu") == torch.device("xpu", 7)


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

    def test_flashinfer_sinks_use_attention_layer_arena(self):
        from vllm.v1.attention.backends.flashinfer import FlashInferImpl

        layer = nn.Module()
        first_impl = FlashInferImpl.__new__(FlashInferImpl)
        first_impl._sinks_source = torch.arange(4, dtype=torch.bfloat16)

        with arena_scope(get_reload_arena(layer)):
            first_impl.process_weights_after_loading(torch.bfloat16)
        snap = snapshot_model_arenas(layer)
        first_ptr = first_impl.sinks.data_ptr()

        rebuilt_impl = FlashInferImpl.__new__(FlashInferImpl)
        rebuilt_impl._sinks_source = torch.arange(
            4, dtype=torch.bfloat16) + 10
        with arena_scope(get_reload_arena(layer)):
            rebuilt_impl.process_weights_after_loading(torch.bfloat16)

        assert rebuilt_impl.sinks.data_ptr() == first_ptr
        assert torch.equal(
            rebuilt_impl.sinks, rebuilt_impl._sinks_source.to(torch.float32))
        assert verify_model_arenas(layer, snap) == []

    def test_put_shape_mismatch_raises(self):
        arena = ReloadArena("layer")
        arena.put("d", torch.ones(3))
        with pytest.raises(ValueError, match="incompatible spec"):
            arena.put("d", torch.ones(4))


@pytest.mark.parametrize("quant_mode", ["mxfp4", "fp8"])
def test_mla_quantized_decode_weights_reuse_arena_storage(
    monkeypatch, quant_mode
):
    """Cover AITER-only MLA PWAL branches without requiring ROCm hardware."""
    import vllm.model_executor.layers.attention.mla_attention as mla_mod

    layer = mla_mod.MLAAttention.__new__(mla_mod.MLAAttention)
    nn.Module.__init__(layer)
    layer.kv_lora_rank = 2
    layer.num_heads = 2
    layer.qk_nope_head_dim = 3
    layer.v_head_dim = 4
    layer.kv_b_proj = object()
    layer.quant_config = None
    layer.is_aiter_triton_fp4_bmm_enabled = quant_mode == "mxfp4"
    layer.is_aiter_triton_fp8_bmm_enabled = quant_mode == "fp8"

    source = torch.arange(28, dtype=torch.float32).reshape(14, 2)
    monkeypatch.setattr(
        mla_mod,
        "get_and_maybe_dequant_weights",
        lambda *_args, **_kwargs: source,
    )
    monkeypatch.setattr(mla_mod, "should_load_quant_weights", lambda _method: True)

    def fake_quant(value, **_kwargs):
        return value.contiguous() + 1, torch.full(
            value.shape[:-1] + (1,), 2.0, dtype=torch.float32
        )

    if quant_mode == "mxfp4":
        from vllm.model_executor.layers.quantization.quark import (
            utils as quark_utils,
        )

        monkeypatch.setattr(
            quark_utils, "quark_quantize_weight_to_mxfp4", fake_quant
        )
    else:
        monkeypatch.setattr(mla_mod, "dynamic_per_batched_tensor_quant", fake_quant)
        monkeypatch.setattr(mla_mod, "is_global_first_rank", lambda: False)
        monkeypatch.setattr(
            mla_mod.rocm_aiter_ops,
            "triton_fp8_bmm",
            lambda *_args, **_kwargs: None,
        )

    layer.process_weights_after_loading(torch.float32)
    names = ("W_K", "W_K_scale", "W_V", "W_V_scale")
    before = {name: getattr(layer, name).data_ptr() for name in names}

    source.add_(100)
    layer.process_weights_after_loading(torch.float32)
    after = {name: getattr(layer, name).data_ptr() for name in names}

    assert after == before
    assert torch.all(layer.W_K > 50)


def test_wna8o8_detached_activation_scales_reuse_arena_storage():
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_wNa8o8 as wna8o8,
    )

    class NoopKernel:
        def process_weights_after_loading(self, _layer):
            return None

    cls = wna8o8.CompressedTensorsWNA8O8Int
    scheme = cls.__new__(cls)
    scheme.is_int_quantized = False
    scheme.kernel = NoopKernel()
    layer = nn.Module()

    def load_scales(input_value, output_value):
        layer.input_scale = nn.Parameter(torch.tensor([input_value]))
        layer.output_scale = nn.Parameter(torch.tensor([output_value]))

    load_scales(1.0, 2.0)
    scheme.process_weights_after_loading(layer)
    before = (scheme._input_scale.data_ptr(), scheme._output_scale.data_ptr())

    load_scales(3.0, 4.0)
    scheme.process_weights_after_loading(layer)
    after = (scheme._input_scale.data_ptr(), scheme._output_scale.data_ptr())

    assert after == before
    assert scheme._input_scale.item() == 3.0
    assert scheme._output_scale.item() == 4.0


def test_b12x_experts_rebuild_reuses_arena_object_slot(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import (
        FlashInferB12xExperts,
    )

    arena = ReloadArena("routed_experts")
    build_calls = []

    def make_experts():
        experts = FlashInferB12xExperts.__new__(FlashInferB12xExperts)
        experts._wrapper = None
        monkeypatch.setattr(
            experts,
            "_build_wrapper",
            lambda: build_calls.append(1) or object(),
        )
        return experts

    first = make_experts()
    first._bind_arena_wrapper(arena)
    snap = arena.snapshot()
    rebuilt = make_experts()
    rebuilt._bind_arena_wrapper(arena)

    assert rebuilt._wrapper is first._wrapper
    assert rebuilt._wrapper is arena.objects()["flashinfer_b12x.wrapper"]
    assert len(build_calls) == 1
    assert arena.verify(snap) == []

    arena._object_slots["flashinfer_b12x.wrapper"] = object()
    violations = arena.verify(snap)
    assert len(violations) == 1
    assert violations[0].kind == "moved"

    del arena._object_slots["flashinfer_b12x.wrapper"]
    violations = arena.verify(snap)
    assert len(violations) == 1
    assert violations[0].kind == "gone"


@pytest.mark.parametrize(
    "backend_name", ["FLASHINFER_CUTLASS", "VLLM_CUTLASS"]
)
def test_nvfp4_cutlass_quant_config_scales_reuse_arena_storage(backend_name):
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        NvFp4MoeBackend,
        make_nvfp4_moe_quant_config,
    )

    layer = nn.Module()

    def make_config(weight_global_scale, input_scale):
        return make_nvfp4_moe_quant_config(
            backend=NvFp4MoeBackend[backend_name],
            w13_scale=torch.ones(1),
            w2_scale=torch.ones(1),
            w13_scale_2=torch.tensor([weight_global_scale]),
            w2_scale_2=torch.tensor([weight_global_scale + 1]),
            a13_scale=torch.tensor([input_scale]),
            a2_scale=torch.tensor([input_scale * 2]),
            layer=layer,
        )

    with arena_scope(get_reload_arena(layer)):
        first = make_config(2.0, 0.25)
    before = {
        name: getattr(first, name).data_ptr()
        for name in ("g1_alphas", "g2_alphas", "a1_gscale", "a2_gscale")
    }

    # Model the in-place fusion performed by CutlassExpertsFp4. The next
    # PWAL must refresh the slot from newly derived source values first.
    layer.w13_weight_scale_2.mul_(0.25)
    with arena_scope(get_reload_arena(layer)):
        second = make_config(3.0, 0.5)

    after = {
        name: getattr(second, name).data_ptr()
        for name in ("g1_alphas", "g2_alphas", "a1_gscale", "a2_gscale")
    }
    assert after == before
    assert layer.w13_weight_scale_2 is second.g1_alphas
    assert layer.w2_weight_scale_2 is second.g2_alphas
    assert second.g1_alphas.item() == 3.0
    assert second.a1_gscale.item() == 2.0


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

    def test_snapshot_verify_object_slot_across_model(self):
        model = self._model()
        arena = get_reload_arena(model.layer)
        arena.get_or_create_object("wrapper", object)
        snaps = snapshot_model_arenas(model)
        arena._object_slots["wrapper"] = object()

        problems = verify_model_arenas(model, snaps)
        assert len(problems) == 1
        assert "layer" in problems[0] and "wrapper" in problems[0]
        assert "moved" in problems[0]

    def test_peek_does_not_create(self):
        model = self._model()
        assert peek_reload_arena(model.layer) is None
        get_reload_arena(model.layer)
        assert peek_reload_arena(model.layer) is not None
