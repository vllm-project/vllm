# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Buffered online quantization (`reload/online.py`) on tiny CPU layers.

Invariants: a layer is processed exactly when its last expected shard
arrives, on the same bytes as a fully loaded layer; an update equals a cold
start on the new weights, at the startup addresses and parameter objects;
direct reload processes online layers once per update and never twice.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.weight_transfer.base import ParamMeta
from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
from vllm.model_executor.model_loader.reload import (
    finish_reload,
    lifecycle,
    plan_loads,
    start_reload,
)
from vllm.model_executor.model_loader.reload.online import (
    begin_online_reload,
    finalize_online,
    install_online_buffers,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

BF16, FP8 = torch.bfloat16, torch.float8_e4m3fn
_GUARD = "_already_called_process_weights_after_loading"
_MODEL_CONFIG = SimpleNamespace(
    word_embeddings_untied_by_checkpoint=False, quantization=None, dtype=BF16
)


class _ToyOnline(OnlineLinearBase):
    """Per-tensor FP8 of the whole weight, stored transposed, like
    `Fp8PerTensorOnlineLinearMethod`; counts its runs."""

    def __init__(self, name):  # no kernel selection
        self.requantization_source = None
        self.name = name
        self.runs = 0

    def create_weights(self, layer, shape):
        weight = torch.nn.Parameter(
            torch.empty(shape, dtype=BF16, device="meta"), requires_grad=False
        )
        layer.register_parameter(self.name, weight)
        set_weight_attrs(weight, {"weight_loader": layer.weight_loader})
        install_online_buffers(layer)

    def process_weights_after_loading(self, layer):
        if getattr(layer, _GUARD, False):
            return
        self.runs += 1
        weight, scale = _quantize(getattr(layer, self.name))
        replace_parameter(layer, self.name, weight)
        replace_parameter(layer, f"{self.name}_scale", scale)
        setattr(layer, _GUARD, True)

    def apply(self, layer, x, bias=None):
        raise NotImplementedError


def _quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (weight.float().abs().amax() / 448).clamp_min(1e-12).reshape(1)
    return (weight.float() / scale).to(FP8).transpose(-1, -2), scale


def _reference(full: torch.Tensor, name="weight") -> dict[str, torch.Tensor]:
    """The online PWAL on a fully loaded BF16 weight."""
    weight, scale = _quantize(full)
    return {name: weight, f"{name}_scale": scale}


class _Layer(torch.nn.Module):
    name = "weight"

    def __init__(self, shape=(8, 4)):
        super().__init__()
        self.quant_method = _ToyOnline(self.name)
        self.quant_config = SimpleNamespace(online_quantization_config=None)
        self.quant_method.create_weights(self, shape)

    def weight_loader(self, param, loaded_weight):
        param.data.copy_(loaded_weight)


class _QKV(_Layer):
    shard_ids = ("q", "k", "v")
    _rows = {"q": (0, 4), "k": (4, 6), "v": (6, 8)}

    def _get_shard_size_mapping(self, shard):
        start, stop = self._rows[shard]
        return stop - start

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        shards = self.shard_ids if loaded_shard_id is None else loaded_shard_id
        shards = shards if isinstance(shards, tuple) else (shards,)
        start, stop = self._rows[shards[0]][0], self._rows[shards[-1]][1]
        param.data[start:stop].copy_(loaded_weight)


class _MoE(_Layer):
    """Two local experts out of four (global 1 and 3), `w13` rows gate then up."""

    name = "w13_weight"
    _local = {1: 0, 3: 1}

    def __init__(self, gated=True):
        self.gated = gated
        super().__init__(shape=(2, 4 if gated else 2, 4))
        self.moe_config = SimpleNamespace(is_act_and_mul=gated)
        self.local_num_experts = 2

    def _map_global_expert_id_to_local_expert_id(self, expert_id):
        return self._local.get(expert_id, -1)

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        local = self._map_global_expert_id_to_local_expert_id(expert_id)
        if local < 0:
            return False if return_success else None
        if loaded_weight.ndim == 3:
            rows = slice(0, 2) if shard_id == "w1" else slice(2, 4)
            param.data[:, rows].copy_(loaded_weight)
        else:
            rows = slice(0, 2) if shard_id == "w1" else slice(2, 4)
            param.data[local, rows].copy_(loaded_weight)
        return True if return_success else None


def _rand(*shape, seed=0) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(BF16)


def _assert_equal(layer: torch.nn.Module, expected: dict[str, torch.Tensor]):
    for name, value in expected.items():
        got = getattr(layer, name)
        assert got.dtype == value.dtype and got.shape == value.shape, name
        assert torch.equal(got.view(torch.uint8), value.view(torch.uint8)), name


def _load_qkv(layer: _QKV, full: torch.Tensor) -> None:
    for shard, (start, stop) in layer._rows.items():
        layer.weight.weight_loader(layer.weight, full[start:stop], shard)


def test_processes_when_last_shard_arrives():
    layer, full = _QKV(), _rand(8, 4)
    layer.weight.weight_loader(layer.weight, full[0:4], "q")
    layer.weight.weight_loader(layer.weight, full[4:6], "k")
    assert layer.quant_method.runs == 0 and layer.weight.dtype == BF16
    layer.weight.weight_loader(layer.weight, full[6:8], "v")
    assert layer.quant_method.runs == 1
    _assert_equal(layer, _reference(full))  # same bytes as a full load


@pytest.mark.parametrize("shard", [(), (None,), (("q", "k", "v"),)])
def test_fused_tensor_counts_as_every_shard(shard):
    layer, full = _QKV(), _rand(8, 4)
    layer.weight.weight_loader(layer.weight, full, *shard)
    assert layer.quant_method.runs == 1
    _assert_equal(layer, _reference(full))


@pytest.mark.parametrize("gated", [True, False])
def test_moe_expects_each_local_expert_and_shard(gated):
    layer = _MoE(gated)
    full = _rand(2, 4 if gated else 2, 4)
    loader = layer.w13_weight.weight_loader
    assert loader(layer.w13_weight, full[0, :2], "w", "w1", 0, True) is False
    shards = [("w1", slice(0, 2))] + ([("w3", slice(2, 4))] if gated else [])
    for (shard, rows), (global_id, local) in [
        (s, e) for e in layer._local.items() for s in shards
    ]:
        assert layer.quant_method.runs == 0
        assert loader(layer.w13_weight, full[local, rows], "w", shard, global_id, 1)
    assert layer.quant_method.runs == 1
    _assert_equal(layer, _reference(full, "w13_weight"))


def test_moe_full_expert_tensor_covers_every_expert():
    layer, full = _MoE(), _rand(2, 4, 4)
    param = layer.w13_weight
    param.weight_loader(param, full[:, :2], "w", "w1", 1)
    assert layer.quant_method.runs == 0
    param.weight_loader(param, full[:, 2:], "w", "w3", 1)
    assert layer.quant_method.runs == 1
    _assert_equal(layer, _reference(full, "w13_weight"))


def test_load_without_shard_key_defers_to_finalize():
    class _NoShardArg(_QKV):
        def weight_loader(self, param, loaded_weight):  # no loaded_shard_id
            param.data.copy_(loaded_weight)

    layer, full = _NoShardArg(), _rand(8, 4)
    layer.weight.weight_loader(layer.weight, full)
    assert layer.quant_method.runs == 0
    finalize_online(layer, strict=False)
    assert layer.quant_method.runs == 1
    _assert_equal(layer, _reference(full))


def test_cold_start_without_weights_processes_zeros():
    layer = _QKV()
    finalize_online(layer, strict=False)
    assert layer.quant_method.runs == 1
    assert not layer.weight.float().any()
    finalize_online(layer, strict=False)  # processed: not again
    assert layer.quant_method.runs == 1


def test_update_equals_cold_start_at_startup_tensors():
    layer = _QKV()
    _load_qkv(layer, _rand(8, 4, seed=0))
    held = {name: (p, p.data_ptr()) for name, p in layer.named_parameters()}
    new = _rand(8, 4, seed=1)

    begin_online_reload(layer)
    # an override installed for one update (EPLB placement) is undone after it
    layer.weight.weight_loader = lambda *a, wrapped=layer.weight.weight_loader: wrapped(
        *a
    )
    _load_qkv(layer, new)
    assert layer.quant_method.runs == 2
    _assert_equal(layer, _reference(new))
    for name, (param, ptr) in held.items():
        assert getattr(layer, name) is param and param.data_ptr() == ptr, name
    wrapper = layer._online_buffer_state.wrappers["weight"]
    assert layer.weight.weight_loader is wrapper

    begin_online_reload(layer)  # the next update works the same
    _load_qkv(layer, _rand(8, 4, seed=2))
    _assert_equal(layer, _reference(_rand(8, 4, seed=2)))


def test_update_keeps_storage_a_kept_parameter_was_rebound_to():
    """PWAL that rebinds an existing parameter to new storage (MXFP8 writes an
    unswizzled scale, then a swizzled one of another shape) still leaves the
    startup storage holding the result."""

    class _Reswizzle(_ToyOnline):
        def process_weights_after_loading(self, layer):
            if getattr(layer, _GUARD, False):
                return
            had_scale = hasattr(layer, "weight_scale")
            super().process_weights_after_loading(layer)
            if had_scale:  # an update: the kept scale changes shape twice
                scale = layer.weight_scale.data.clone()
                replace_parameter(layer, "weight_scale", scale.repeat(2))
                replace_parameter(layer, "weight_scale", scale)

    layer = _QKV()
    layer.quant_method = _Reswizzle("weight")
    _load_qkv(layer, _rand(8, 4))
    scale = layer.weight_scale
    ptr = scale.data_ptr()
    begin_online_reload(layer)
    _load_qkv(layer, _rand(8, 4, seed=1))
    assert layer.weight_scale is scale and scale.data_ptr() == ptr
    _assert_equal(layer, _reference(_rand(8, 4, seed=1)))


def test_partial_update_raises_when_strict():
    layer, full = _QKV(), _rand(8, 4)
    _load_qkv(layer, full)
    begin_online_reload(layer)
    layer.weight.weight_loader(layer.weight, _rand(4, 4, seed=1), "q")
    with pytest.raises(RuntimeError, match="received only part"):
        finalize_online(layer, strict=True)


def test_layer_left_out_of_an_update_keeps_its_weights():
    layer, full = _QKV(), _rand(8, 4)
    _load_qkv(layer, full)
    begin_online_reload(layer)
    finalize_online(layer, strict=True)
    assert layer.quant_method.runs == 1
    _assert_equal(layer, _reference(full))


def test_deferred_update_processes_only_at_finalize():
    layer = _QKV()
    _load_qkv(layer, _rand(8, 4))
    begin_online_reload(layer, defer=True)
    _load_qkv(layer, _rand(8, 4, seed=1))
    assert layer.quant_method.runs == 1
    finalize_online(layer, strict=False)
    assert layer.quant_method.runs == 2
    _assert_equal(layer, _reference(_rand(8, 4, seed=1)))


class _Model(torch.nn.Module):
    """An online QKV next to a plain layer; loads like a model's
    `stacked_params_mapping`."""

    def __init__(self):
        super().__init__()
        self.qkv = _QKV()
        self.norm = torch.nn.Module()
        self.norm.weight = torch.nn.Parameter(torch.ones(4, dtype=BF16), False)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, weight in weights:
            if name.startswith("qkv."):
                shard = name.split(".")[1]
                self.qkv.weight.weight_loader(params["qkv.weight"], weight, shard)
            else:
                params[name].data.copy_(weight)
        return {name for name, _ in weights}


def _checkpoint(seed):
    full = _rand(8, 4, seed=seed)
    return [
        ("qkv.q", full[0:4]),
        ("qkv.k", full[4:6]),
        ("qkv.v", full[6:8]),
        ("norm.weight", _rand(4, seed=seed + 100)),
    ], full


@pytest.fixture
def direct(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    monkeypatch.setattr(lifecycle, "_weight_update_active", False)


def _cold_start(seed) -> _Model:
    model = _Model()
    weights, _ = _checkpoint(seed)
    model.load_weights(weights)
    finalize_online(model, strict=False)
    process_weights_after_loading(model, _MODEL_CONFIG, torch.device("cpu"))
    return model


def test_direct_reload_processes_online_layers_once(direct):
    model = _cold_start(0)
    assert model.qkv.quant_method.runs == 1
    assert not hasattr(model.qkv.weight, "_ckpt")  # no checkpoint layout record
    assert hasattr(model.norm.weight, "_ckpt")
    held = {n: (p, p.data_ptr()) for n, p in model.named_parameters()}

    weights, full = _checkpoint(1)
    start_reload(model)
    model.load_weights(weights)
    assert model.qkv.quant_method.runs == 2  # on its last shard
    finish_reload(model, _MODEL_CONFIG, torch.device("cpu"))
    assert model.qkv.quant_method.runs == 2  # finish leaves it alone

    fresh = _cold_start(1)
    _assert_equal(model.qkv, _reference(full))
    for name, (param, ptr) in held.items():
        assert torch.equal(param, dict(fresh.named_parameters())[name].data)
        assert dict(model.named_parameters())[name] is param
        assert param.data_ptr() == ptr


def test_direct_reload_of_part_of_a_layer_raises(direct):
    model = _cold_start(0)
    weights, _ = _checkpoint(1)
    start_reload(model)
    model.load_weights(weights[:2])
    with pytest.raises(RuntimeError, match="received only part"):
        finish_reload(model, _MODEL_CONFIG, torch.device("cpu"))


def test_load_plan_sends_online_names_through_load_weights(direct):
    model = _cold_start(0)
    weights, full = _checkpoint(1)
    start_reload(model)
    plan = plan_loads(
        model, [ParamMeta(n, w.dtype, tuple(w.shape)) for n, w in weights]
    )
    for name in ("qkv.q", "qkv.k", "qkv.v"):
        assert plan.get_dest(name) is None, name  # fallback
    assert plan.get_dest("norm.weight") is not None
    assert model.qkv.quant_method.runs == 1  # tracing processed nothing
    model.load_weights(weights)
    finish_reload(model, _MODEL_CONFIG, torch.device("cpu"))
    _assert_equal(model.qkv, _reference(full))
