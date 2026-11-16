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
    fill_online_layer,
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


def test_moe_other_rank_expert_after_completion_keeps_processed_layer():
    # Checkpoints list experts lexicographically (..., 63, 7, 8, 9), so another
    # rank's experts can arrive after this rank's last one (expert parallel).
    layer, full = _MoE(), _rand(2, 4, 4)
    loader = layer.w13_weight.weight_loader
    for global_id, local in layer._local.items():
        loader(layer.w13_weight, full[local, :2], "w", "w1", global_id)
        loader(layer.w13_weight, full[local, 2:], "w", "w3", global_id)
    assert layer.quant_method.runs == 1
    param = layer.w13_weight
    assert loader(param, full[0, :2], "w", "w1", 2, True) is False
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


def test_fill_online_layer_processes_dummy_weights():
    # `load_format=dummy`: no loader runs; the layer is filled and processed.
    layer, full = _QKV(), _rand(8, 4)
    fill_online_layer(layer, lambda: layer.weight.data.copy_(full))
    assert layer.quant_method.runs == 1
    _assert_equal(layer, _reference(full))


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
    assert layer.weight.is_meta and layer.weight.dtype == BF16  # checkpoint format
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
    weight = layer.weight
    begin_online_reload(layer)
    finalize_online(layer, strict=True)
    assert layer.quant_method.runs == 1 and layer.weight is weight
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


class _Biased(_QKV):
    """A QKV whose linear adds a real bias after `create_weights`, loaded in
    place, as `QKVParallelLinear` does; or registers `bias=None`."""

    def __init__(self, bias=True):
        super().__init__()
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(8, dtype=BF16), False)
            set_weight_attrs(self.bias, {"weight_loader": self._bias_loader})
        else:
            self.register_parameter("bias", None)

    def _bias_loader(self, param, loaded_weight, loaded_shard_id=None):
        param.data.copy_(loaded_weight)


def test_update_keeps_a_real_bias_loadable():
    layer = _Biased()
    _load_qkv(layer, _rand(8, 4))
    layer.bias.weight_loader(layer.bias, _rand(8, seed=10))
    bias, ptr = layer.bias, layer.bias.data_ptr()

    begin_online_reload(layer)
    params = dict(layer.named_parameters())
    assert params["bias"] is bias  # a model's load_weights finds it
    bias.weight_loader(params["bias"], _rand(8, seed=11))
    _load_qkv(layer, _rand(8, 4, seed=1))
    assert layer.bias is bias and bias.data_ptr() == ptr
    assert torch.equal(layer.bias, _rand(8, seed=11))
    _assert_equal(layer, _reference(_rand(8, 4, seed=1)))


def test_update_keeps_bias_none():
    layer = _Biased(bias=False)
    _load_qkv(layer, _rand(8, 4))
    begin_online_reload(layer)
    assert "bias" in layer._parameters and layer.bias is None
    _load_qkv(layer, _rand(8, 4, seed=1))
    assert "bias" in layer._parameters and layer.bias is None  # forward reads it


def test_begin_again_after_a_failed_update_keeps_startup_tensors():
    layer = _QKV()
    _load_qkv(layer, _rand(8, 4))
    held = {name: (p, p.data_ptr()) for name, p in layer.named_parameters()}
    begin_online_reload(layer)
    layer.weight.weight_loader(layer.weight, _rand(4, 4, seed=1), "q")  # fails here
    begin_online_reload(layer)  # the next update
    _load_qkv(layer, _rand(8, 4, seed=2))
    _assert_equal(layer, _reference(_rand(8, 4, seed=2)))
    for name, (param, ptr) in held.items():
        assert getattr(layer, name) is param and param.data_ptr() == ptr, name


def test_update_with_a_load_without_shard_key_raises_when_strict():
    class _NoShardArg(_QKV):
        def weight_loader(self, param, loaded_weight):  # no loaded_shard_id
            param.data[:4].copy_(loaded_weight[:4])  # part of the layer

    layer = _NoShardArg()
    layer.weight.weight_loader(layer.weight, _rand(8, 4))
    finalize_online(layer, strict=False)  # cold start: processed as before
    assert layer.quant_method.runs == 1
    begin_online_reload(layer)
    layer.weight.weight_loader(layer.weight, _rand(8, 4, seed=1))
    with pytest.raises(RuntimeError, match="no shard key"):
        finalize_online(layer, strict=True)


def test_update_recomputes_expected_experts():
    # expert parallel rescaling: fewer local experts than at cold start
    layer, full = _MoE(), _rand(2, 4, 4)
    loader = layer.w13_weight.weight_loader
    for global_id, local in layer._local.items():
        loader(layer.w13_weight, full[local, :2], "w", "w1", global_id)
        loader(layer.w13_weight, full[local, 2:], "w", "w3", global_id)
    assert layer.quant_method.runs == 1
    begin_online_reload(layer)
    layer._local, layer.local_num_experts = {1: 0}, 1
    loader = layer.w13_weight.weight_loader
    loader(layer.w13_weight, full[0, :2], "w", "w1", 1)
    loader(layer.w13_weight, full[0, 2:], "w", "w3", 1)
    assert layer.quant_method.runs == 2


def test_update_refills_tensors_pwal_rebuilds():
    """What PWAL builds besides parameters (a derived tensor on the layer, a
    quant config holding the new scale parameter) ends up on the startup
    storage, which captured CUDA graphs read."""

    class _Derived(_ToyOnline):
        def process_weights_after_loading(self, layer):
            if getattr(layer, _GUARD, False):
                return
            super().process_weights_after_loading(layer)
            layer.derived = layer.weight_scale.data * 2  # new tensor each run
            self.config_scale = layer.weight_scale  # e.g. moe_quant_config

    layer = _QKV()
    layer.quant_method = _Derived("weight")
    _load_qkv(layer, _rand(8, 4))
    derived, scale = layer.derived, layer.weight_scale
    ptrs = derived.data_ptr(), scale.data_ptr()
    begin_online_reload(layer)
    _load_qkv(layer, _rand(8, 4, seed=1))
    expected = _reference(_rand(8, 4, seed=1))["weight_scale"]
    assert layer.derived is derived and derived.data_ptr() == ptrs[0]
    assert torch.equal(derived, expected * 2)
    assert layer.quant_method.config_scale.data_ptr() == ptrs[1]
    assert layer.weight_scale is scale and torch.equal(scale, expected)


def test_direct_reload_allows_quantized_lm_head_copy(monkeypatch):
    from vllm.model_executor.layers.quantization.online import lm_head
    from vllm.model_executor.model_loader.reload import direct

    model = _cold_start(0)
    copy = torch.nn.Module()
    copy.quant_method = _ToyOnline("weight")  # online, never loaded: no buffers
    model.lm_head = copy
    monkeypatch.setitem(lm_head._SOURCES, copy, (None, "fp8"))
    direct._check_supported_modules(model)


def test_direct_reload_refuses_online_mxfp4_on_triton():
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import TRITON_BACKENDS
    from vllm.model_executor.model_loader.reload import direct

    model = _cold_start(0)
    model.qkv.quant_method.mxfp4_backend = next(iter(TRITON_BACKENDS))
    with pytest.raises(RuntimeError, match="online MXFP4"):
        direct._check_supported_modules(model)


def test_dummy_load_keeps_non_persistent_buffers():
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

    layer = _QKV()
    layer.register_buffer("table", torch.arange(4.0), persistent=False)
    DummyModelLoader(LoadConfig()).load_weights(layer, _MODEL_CONFIG)
    assert layer.quant_method.runs == 1
    assert torch.equal(layer.table, torch.arange(4.0))


def test_auto_weights_loader_warns_on_narrowing_into_a_buffer(monkeypatch):
    from vllm.model_executor.models import utils

    warned = []
    monkeypatch.setattr(utils.logger, "warning", lambda *a: warned.append(a))
    model = torch.nn.Module()
    model.lin = _Layer()
    loader = utils.AutoWeightsLoader(model)
    list(loader._load_module("", model, [("lin.weight", _rand(8, 4).float())]))
    assert warned and warned[0][2] == torch.float32 and warned[0][3] == BF16
    warned.clear()
    # a BF16 load into the processed (FP8) layer is not a narrowing
    model.lin.weight.weight_loader(model.lin.weight, _rand(8, 4))
    list(loader._load_module("", model, [("lin.weight", _rand(8, 4))]))
    assert not warned
