# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct weight reload (``VLLM_RELOAD_MODE=direct``) on a toy model whose
post-load processing has the hazards found on real models:

(a) a parent derives a separate tensor from a child's weight (MLA W_UK/W_UV
    from a quantized kv_b_proj);
(b) post-load processing changes a weight's layout;
(c) the model's loader writes a weight without its ``weight_loader`` (MiMo);
(d) post-load processing consumes a checkpoint tensor, which stays
    registered as a retired parameter (NVFP4 ``weight_scale_2``);
(e) post-load processing that must never run on converted bytes (packing);
(f) post-load processing grows a weight (padding): a reload writes it into
    the existing storage to keep its address.

Invariant: a reload equals a cold start on the new weights, at the startup
addresses.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerTensorOnlineLinearMethod,
)
from vllm.model_executor.model_loader.reload import finish_reload, start_reload
from vllm.model_executor.model_loader.reload.direct import (
    begin_direct_reload,
    finish_direct_reload,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import replace_parameter, retire_parameter

_MODEL_CONFIG = SimpleNamespace(
    word_embeddings_untied_by_checkpoint=False, quantization=None, dtype=torch.bfloat16
)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _Method(QuantizeMethodBase):
    """Post-load processing given as a function of the layer; counts runs."""

    def __init__(self, fn):
        self.fn = fn
        self.runs = 0

    def create_weights(self, layer, *weight_args, **extra_weight_attrs):
        pass

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        self.runs += 1
        self.fn(layer)


def _derive_w_uk(layer):  # (a): a separate tensor derived from the child
    w_uk = (layer.kv_b.weight.float() * layer.k_scale).t().contiguous()
    if hasattr(layer, "W_UK"):  # refilled in place: graphs keep its address
        layer.W_UK.copy_(w_uk)
    else:
        layer.W_UK = w_uk


def _transpose(layer):  # (b): kernel layout differs from the checkpoint's
    replace_parameter(layer, "weight", layer.weight.t().contiguous())


def _requantize(layer):  # (c): fused-FP8-like rescale
    scale = layer.weight.float().abs().amax()
    replace_parameter(layer, "weight", (layer.weight / scale).to(layer.weight.dtype))
    replace_parameter(layer, "weight_scale", scale.reshape(1))


def _consume_scale_2(layer):  # (d): NVFP4 weight_scale_2 folded in, retired
    ws2 = layer.weight_scale_2
    replace_parameter(layer, "global_scale", (1.0 / ws2).reshape(1))
    replace_parameter(layer, "weight", (layer.weight * ws2).to(layer.weight.dtype))
    retire_parameter(layer, "weight_scale_2")


def _pack(layer):  # (e): not idempotent, must never run on converted bytes
    replace_parameter(layer, "weight", (layer.weight * 2).t().contiguous())


def _pad(layer):  # (f): kernel layout larger than the checkpoint's
    replace_parameter(layer, "weight", torch.nn.functional.pad(layer.weight, (0, 4)))


_SHAPES = {
    "attn.kv_b.weight": ((8, 4), torch.bfloat16),
    "proj.weight": ((4, 6), torch.bfloat16),
    "mimo.weight": ((4, 4), torch.bfloat16),
    "nvfp4.weight": ((4, 4), torch.bfloat16),
    "nvfp4.weight_scale_2": ((), torch.float32),
    "engram.weight": ((3, 5), torch.bfloat16),
    "pad.weight": ((4, 4), torch.bfloat16),
}
_PWAL = {
    "attn.kv_b": None,
    "proj": _transpose,
    "mimo": _requantize,
    "nvfp4": _consume_scale_2,
    "engram": _pack,
    "pad": _pad,
}


class _Layer(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.quant_method = _Method(fn) if fn is not None else None


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Layer(_derive_w_uk)
        self.attn.register_buffer("k_scale", torch.tensor(0.5))  # not in checkpoint
        for name, (shape, dtype) in _SHAPES.items():
            *path, local = name.split(".")
            parent = self
            for i, part in enumerate(path):
                if not hasattr(parent, part):
                    parent.add_module(part, _Layer(_PWAL[".".join(path[: i + 1])]))
                parent = getattr(parent, part)
            param = torch.nn.Parameter(torch.empty(shape, dtype=dtype), False)
            parent.register_parameter(local, param)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            param = params[name]
            if name.startswith("mimo."):
                param.data.copy_(weight)  # (c): bypasses param.weight_loader
            else:
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)
        return loaded


def _checkpoint(scale: float = 1.0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    return {
        name: ((torch.rand(shape, generator=gen) + 0.5) * scale).to(dtype)
        for name, (shape, dtype) in _SHAPES.items()
    }


def _cold_start(weights: dict[str, torch.Tensor]) -> _ToyModel:
    """As ``BaseModelLoader.load_model`` does."""
    with _DEVICE:
        model = _ToyModel()
    model.load_weights(weights.items())
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    return model


def _reload(model: _ToyModel, weights: dict[str, torch.Tensor]) -> None:
    begin_direct_reload(model)
    model.load_weights(weights.items())
    finish_direct_reload(model, _MODEL_CONFIG, _DEVICE)


def _state(model: torch.nn.Module) -> dict[str, tuple]:
    """Every parameter, buffer and tensor attribute: layout, bytes, address."""
    tensors: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for attr, value in vars(module).items():
            if isinstance(value, torch.Tensor):
                tensors[prefix + attr] = value
    tensors.update(model.named_parameters())
    tensors.update(model.named_buffers())
    return {
        name: (
            (tuple(t.shape), t.stride(), t.dtype),
            t.detach().contiguous().reshape(-1).view(torch.uint8).cpu().clone(),
            t.data_ptr(),
        )
        for name, t in tensors.items()
    }


def _assert_same(actual: dict, expected: dict, names=None, addresses=False) -> None:
    """Layout and bytes (and addresses) equal, for ``names`` or every tensor."""
    if names is None:
        assert sorted(actual) == sorted(expected)
        names = sorted(expected)
    for name in names:
        (layout, data, ptr), (exp_layout, exp_data, exp_ptr) = (
            actual[name],
            expected[name],
        )
        assert layout == exp_layout, name
        assert torch.equal(data, exp_data), f"{name} differs"
        assert not addresses or ptr == exp_ptr, f"{name} moved"


def _ptrs(state: dict, names=None) -> dict[str, int]:
    return {n: state[n][2] for n in (state if names is None else names)}


def _runs(model: _ToyModel) -> dict[str, int]:
    return {
        name: m.quant_method.runs
        for name, m in model.named_modules()
        if getattr(m, "quant_method", None) is not None
    }


@pytest.fixture(autouse=True)
def _direct_mode(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")


class TestReloadEqualsColdStart:
    def test_same_checkpoint_is_bit_identical(self):
        model = _cold_start(_checkpoint())
        before = _state(model)
        _reload(model, _checkpoint())
        _assert_same(_state(model), before, addresses=True)
        assert set(_runs(model).values()) == {2}

    def test_new_checkpoint_matches_fresh_cold_start(self):
        model = _cold_start(_checkpoint())
        before = _state(model)
        _reload(model, _checkpoint(1.01))
        after = _state(model)
        _assert_same(after, _state(_cold_start(_checkpoint(1.01))))
        assert _ptrs(after) == _ptrs(before)
        assert model.nvfp4.weight_scale_2.retired  # as at cold start

    @pytest.mark.parametrize(
        "names",
        [
            ["attn.W_UK"],
            ["proj.weight"],
            ["mimo.weight", "mimo.weight_scale"],
            ["nvfp4.weight", "nvfp4.global_scale"],
            ["engram.weight"],
            ["pad.weight"],
        ],
        ids=[
            "a-parent-derived",
            "b-layout-change",
            "c-loader-bypass",
            "d-consumed-scale",
            "e-not-idempotent",
            "f-grows",
        ],
    )
    def test_new_checkpoint_hazard(self, names):
        model = _cold_start(_checkpoint())
        before = _state(model)
        _reload(model, _checkpoint(1.01))
        after = _state(model)
        assert any(not torch.equal(after[n][1], before[n][1]) for n in names)
        _assert_same(after, _state(_cold_start(_checkpoint(1.01))), names=names)
        assert _ptrs(after, names) == _ptrs(before, names)


class TestUnsent:
    def test_tensor_without_post_processing_keeps_its_bytes(self):
        # a frozen table the trainer does not send (Engram, an n-gram
        # embedding): its module has no post-load processing, so a reload
        # keeps its cold-start bytes; derived tensors read them as cold start
        model = _cold_start(_checkpoint())
        old = _checkpoint()
        new = _checkpoint(1.01)
        sent = {n: t for n, t in new.items() if n != "attn.kv_b.weight"}
        before = _state(model)
        _reload(model, sent)
        after = _state(model)
        _assert_same(after, before, names=["attn.kv_b.weight"], addresses=True)
        expected = dict(new, **{"attn.kv_b.weight": old["attn.kv_b.weight"]})
        _assert_same(after, _state(_cold_start(expected)))


class TestRefusals:
    def test_refused_begin_leaves_the_model_untouched(self):
        # proj is viewed back first, in module order; mimo's storage is short
        model = _cold_start(_checkpoint())
        owner, (_, _, dtype, offset), attrs = model.mimo.weight._ckpt
        model.mimo.weight._ckpt = (owner, ((64, 64), (64, 1), dtype, offset), attrs)
        before = _state(model)
        with pytest.raises(RuntimeError, match="checkpoint layout needs"):
            begin_direct_reload(model)
        _assert_same(_state(model), before, addresses=True)

    def test_online_quantization_refused(self, monkeypatch):
        # finish clears the guard that keeps online quantization from
        # quantizing its converted bytes again
        online = object.__new__(Fp8PerTensorOnlineLinearMethod)
        init = _ToyModel.__init__

        def with_online_layer(self):
            init(self)
            self.proj.quant_method = online

        monkeypatch.setattr(_ToyModel, "__init__", with_online_layer)
        with pytest.raises(RuntimeError, match="online quantization"):
            _cold_start(_checkpoint())

    def test_online_shared_experts_refused(self, monkeypatch):
        # post-load processing deletes the staging parameters a reload loads
        init = _ToyModel.__init__

        def with_online_shared_experts(self):
            init(self)
            self.proj.shared_expert_online_loader = object()

        monkeypatch.setattr(_ToyModel, "__init__", with_online_shared_experts)
        with pytest.raises(RuntimeError, match="online quantization"):
            _cold_start(_checkpoint())

    @pytest.mark.parametrize("method_cls", ["HummingLinearMethod", "HummingMoEMethod"])
    def test_humming_refused(self, monkeypatch, method_cls):
        # its post-load processing runs only once per quant method
        from vllm.model_executor.layers.quantization import humming as humming_mod

        humming = object.__new__(getattr(humming_mod, method_cls))
        init = _ToyModel.__init__

        def with_humming_layer(self):
            init(self)
            self.proj.quant_method = humming

        monkeypatch.setattr(_ToyModel, "__init__", with_humming_layer)
        with pytest.raises(RuntimeError, match="humming"):
            _cold_start(_checkpoint())

    def test_replaced_param_refused(self):
        # a new Parameter under a checkpoint name has no checkpoint layout:
        # the loader would write checkpoint bytes into its runtime layout
        model = _cold_start(_checkpoint())
        model.proj.weight = torch.nn.Parameter(model.proj.weight.data, False)
        before = _state(model)
        with pytest.raises(RuntimeError, match=r"replaced .*proj\.weight"):
            begin_direct_reload(model)
        _assert_same(_state(model), before, addresses=True)
        assert "_direct_reload_begun" not in model.__dict__


def test_retry_after_aborted_update_matches_cold_start():
    # Direct reload has no rollback: a failed update leaves the model partly
    # updated and pending (nothing refuses to run it), and the next full
    # update repairs it.
    model = _cold_start(_checkpoint())
    assert "_direct_reload_begun" not in model.__dict__
    begin_direct_reload(model)
    model.load_weights(list(_checkpoint(1.01).items())[:3])  # update aborts
    assert "_direct_reload_begun" in model.__dict__
    _reload(model, _checkpoint(1.01))
    assert "_direct_reload_begun" not in model.__dict__
    _assert_same(_state(model), _state(_cold_start(_checkpoint(1.01))))


def test_start_and_finish_reload_run_direct_reload():
    model = _cold_start(_checkpoint())
    before = _state(model)
    start_reload(model)
    model.load_weights(_checkpoint().items())
    finish_reload(model, _MODEL_CONFIG, _DEVICE)
    _assert_same(_state(model), before, addresses=True)


def test_finish_keeps_guards_of_modules_without_quant_method():
    # e.g. MoERunnerInterface sets the guard in __init__ for other reasons
    model = _cold_start(_checkpoint())
    model.attn.kv_b._already_called_process_weights_after_loading = True
    model.proj._already_called_process_weights_after_loading = True
    _reload(model, _checkpoint())
    assert model.attn.kv_b._already_called_process_weights_after_loading
    assert "_already_called_process_weights_after_loading" not in vars(model.proj)


def test_finish_retry_after_failed_direct_finish_raises(monkeypatch):
    # some modules may already be processed: only a new full update repairs it
    model = _cold_start(_checkpoint())
    start_reload(model)
    model.load_weights(_checkpoint(1.01).items())

    def fail(layer):
        raise ValueError("boom")

    monkeypatch.setattr(model.nvfp4.quant_method, "fn", fail)
    with pytest.raises(ValueError, match="boom"):
        finish_reload(model, _MODEL_CONFIG, _DEVICE)
    with pytest.raises(RuntimeError, match="finish without a begin"):
        finish_reload(model, _MODEL_CONFIG, _DEVICE)


def test_begin_restores_loader_metadata(monkeypatch):
    # AWQ retags the kept parameter's packed/output dims; loaders shard with them
    init = _ToyModel.__init__

    def retag(layer):
        new = layer.weight.t().contiguous()
        new._output_dim, new._packed_dim = 1, 1
        replace_parameter(layer, "weight", new)

    def with_loader_metadata(self):
        init(self)
        self.proj.weight._output_dim = 0
        self.proj.quant_method = _Method(retag)

    monkeypatch.setattr(_ToyModel, "__init__", with_loader_metadata)
    model = _cold_start(_checkpoint())
    weight = model.proj.weight
    assert (weight._output_dim, weight._packed_dim) == (1, 1)
    begin_direct_reload(model)
    assert weight._output_dim == 0 and not hasattr(weight, "_packed_dim")
    model.load_weights(_checkpoint(1.01).items())
    finish_direct_reload(model, _MODEL_CONFIG, _DEVICE)
    assert (weight._output_dim, weight._packed_dim) == (1, 1)
    _assert_same(_state(model), _state(_cold_start(_checkpoint(1.01))))


def test_shared_tensor_with_changed_layout_refused():
    # a draft sharing the target's lm_head after its cold start: the draft's
    # finish would reprocess it although a draft update does not resend it
    target = _cold_start(_checkpoint())
    draft = _cold_start(_checkpoint())
    draft.proj = target.proj  # its post-load processing transposes it
    before = _state(target)
    with pytest.raises(RuntimeError, match="shared from another model"):
        begin_direct_reload(draft)
    assert "_direct_reload_begun" not in draft.__dict__
    _assert_same(_state(target), before, addresses=True)


@pytest.mark.parametrize(
    "fn",
    [
        # ROCm FP8 padding: a view with gaps, padded rows apart
        lambda w: torch.nn.functional.pad(w, (0, 4))[..., :-4],
        # a released weight: its checkpoint bytes stay for the next reload
        lambda w: w.new_empty(0),
    ],
    ids=["padded-view", "emptied"],
)
def test_reload_keeps_address_of_padded_or_emptied_weight(monkeypatch, fn):
    init = _ToyModel.__init__

    def with_fn(self):
        init(self)
        self.proj.quant_method = _Method(
            lambda layer: replace_parameter(layer, "weight", fn(layer.weight))
        )

    monkeypatch.setattr(_ToyModel, "__init__", with_fn)
    model = _cold_start(_checkpoint())
    ptrs = _ptrs(_state(model))
    _reload(model, _checkpoint(1.01))
    _assert_same(_state(model), _state(_cold_start(_checkpoint(1.01))))
    assert _ptrs(_state(model)) == ptrs


def test_replace_parameter_keeps_the_target_checkpoint_layout():
    # post-load processing may pass another checkpoint parameter as new_tensor
    model = _cold_start(_checkpoint())
    ckpt = model.proj.weight._ckpt
    replace_parameter(model.proj, "weight", model.mimo.weight)
    assert model.proj.weight._ckpt == ckpt


def _config(**overrides) -> SimpleNamespace:
    offload = SimpleNamespace(
        offload_backend="auto",
        prefetch=SimpleNamespace(offload_group_size=0),
        uva=SimpleNamespace(cpu_offload_gb=0),
    )
    fields = dict(
        offload_config=offload,
        model_config=SimpleNamespace(quantization=None),
        quant_config=None,
        lora_config=None,
        speculative_config=None,
        weight_transfer_config=None,
    )
    return SimpleNamespace(**(fields | overrides))


_ONLINE = object.__new__(OnlineQuantizationConfig)


@pytest.mark.parametrize(
    "mode,overrides,match",
    [
        ("direct", {}, None),
        ("layerwise", {"lora_config": object()}, None),
        (
            "direct",
            {
                "offload_config": SimpleNamespace(
                    offload_backend="uva",
                    prefetch=SimpleNamespace(offload_group_size=0),
                    uva=SimpleNamespace(cpu_offload_gb=0),
                )
            },
            "weight offloading",
        ),
        (
            "direct",
            {
                "offload_config": SimpleNamespace(
                    offload_backend="auto",
                    prefetch=SimpleNamespace(offload_group_size=2),
                    uva=SimpleNamespace(cpu_offload_gb=0),
                )
            },
            "weight offloading",
        ),
        (
            "direct",
            {
                "offload_config": SimpleNamespace(
                    offload_backend="auto",
                    prefetch=SimpleNamespace(offload_group_size=0),
                    uva=SimpleNamespace(cpu_offload_gb=1),
                )
            },
            "weight offloading",
        ),
        (
            "direct",
            {"model_config": SimpleNamespace(quantization="torchao")},
            "torchao",
        ),
        ("direct", {"quant_config": _ONLINE}, "online quantization"),
        (
            "direct",
            {"quant_config": SimpleNamespace(online_quantization_config=_ONLINE)},
            "online quantization",
        ),
        ("direct", {"lora_config": object()}, "LoRA"),
        (
            "direct",
            {"speculative_config": SimpleNamespace(draft_lm_head_quantization="fp8")},
            "draft_lm_head_quantization",
        ),
        (
            "direct",
            {"weight_transfer_config": SimpleNamespace(backend="sharded_rdt")},
            "sharded_rdt",
        ),
    ],
    ids=[
        "defaults",
        "layerwise-lora",
        "offload-backend",
        "offload-prefetch",
        "offload-uva",
        "torchao",
        "online-config",
        "online-composed",
        "lora",
        "draft-lm-head-quant",
        "sharded-rdt",
    ],
)
def test_verify_direct_reload_compat(monkeypatch, mode, overrides, match):
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    config = _config(**overrides)
    if match is None:
        VllmConfig._verify_direct_reload_compat(config)
    else:
        with pytest.raises(ValueError, match=match):
            VllmConfig._verify_direct_reload_compat(config)
