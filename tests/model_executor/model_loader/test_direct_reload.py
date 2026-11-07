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
    the existing storage to keep its address;
(g) online quantization: the kernel tensor cannot hold the BF16 checkpoint
    tensor, which is staged and quantized as at cold start; its parent derives
    a tensor from it.

Invariant: a reload equals a cold start on the new weights, at the startup
addresses.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import store_params_in_triton
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerTensorOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_processing,
    finalize_layerwise_reload,
    finish_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
    start_reload,
)
from vllm.model_executor.model_loader.reload.direct import (
    _check_supported_modules,
    abort_direct_reload,
    begin_direct_reload,
    finish_direct_reload,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    get_layerwise_info,
    initialize_online_processing,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import (
    refill_derived,
    replace_parameter,
    retire_parameter,
)

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


def _quantize_int8(layer):  # (g): BF16 checkpoint tensor, int8 kernel tensor
    scale = layer.weight.float().abs().amax() / 127
    replace_parameter(layer, "weight", (layer.weight / scale).round().to(torch.int8))
    replace_parameter(layer, "weight_scale", scale)


def _dequantize_child(layer):  # (g): parent reads the quantized child
    refill_derived(layer, "W", layer.online.weight.float() * layer.online.weight_scale)


_SHAPES = {
    "attn.kv_b.weight": ((8, 4), torch.bfloat16),
    "proj.weight": ((4, 6), torch.bfloat16),
    "mimo.weight": ((4, 4), torch.bfloat16),
    "nvfp4.weight": ((4, 4), torch.bfloat16),
    "nvfp4.weight_scale_2": ((), torch.float32),
    "engram.weight": ((3, 5), torch.bfloat16),
    "pad.weight": ((4, 4), torch.bfloat16),
    "deq.online.weight": ((4, 4), torch.bfloat16),
    "deq.online.bias": ((4,), torch.bfloat16),
}
_PWAL = {
    "attn.kv_b": None,
    "proj": _transpose,
    "mimo": _requantize,
    "nvfp4": _consume_scale_2,
    "engram": _pack,
    "pad": _pad,
    "deq": _dequantize_child,
    "deq.online": _quantize_int8,
}


class _OnlineMethod(_Method):
    """As the online quantization methods: the checkpoint tensor is created on
    meta, the layerwise loaders stage it and quantize once the layer is
    complete, and a guard keeps post-load processing from quantizing again."""

    uses_meta_device = True

    def process_weights_after_loading(self, layer):
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        super().process_weights_after_loading(layer)
        layer._already_called_process_weights_after_loading = True


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
            # as the online methods' create_weights: the weight on meta; the
            # bias is registered later by the layer, on the device
            online = path[-1] == "online" and local == "weight"
            param = torch.nn.Parameter(
                torch.empty(shape, dtype=dtype, device="meta" if online else None),
                False,
            )
            parent.register_parameter(local, param)
            if online:
                parent.quant_method = _OnlineMethod(_PWAL["deq.online"])
                initialize_online_processing(parent)

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
        record_metadata_for_reloading(model)
    model.load_weights(weights.items())
    finalize_layerwise_processing(model, _MODEL_CONFIG)
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
            ["deq.online.weight", "deq.online.weight_scale", "deq.W"],
        ],
        ids=[
            "a-parent-derived",
            "b-layout-change",
            "c-loader-bypass",
            "d-consumed-scale",
            "e-not-idempotent",
            "f-grows",
            "g-online-quant",
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
    @pytest.mark.parametrize(
        "unsent,kept",
        [
            (["attn.kv_b.weight"], ["attn.kv_b.weight"]),
            (
                ["deq.online.weight", "deq.online.bias"],
                ["deq.online.weight", "deq.online.weight_scale", "deq.online.bias"],
            ),
        ],
        ids=["no-post-processing", "online-quant"],
    )
    def test_unsent_tensor_keeps_its_bytes(self, unsent, kept):
        # a frozen table the trainer does not send (Engram, an n-gram
        # embedding): its module has no post-load processing, so a reload
        # keeps its cold-start bytes; an unsent online-quantized module gets
        # its kernel tensors back before the post-load rerun, so parents
        # derive from them as at cold start
        model = _cold_start(_checkpoint())
        old = _checkpoint()
        new = _checkpoint(1.01)
        sent = {n: t for n, t in new.items() if n not in unsent}
        before = _state(model)
        _reload(model, sent)
        after = _state(model)
        _assert_same(after, before, names=kept, addresses=True)
        expected = dict(new, **{n: old[n] for n in unsent})
        _assert_same(after, _state(_cold_start(expected)))

    def test_unsent_after_aborted_processing_is_not_requantized(self):
        # an update that failed inside the staged quantization cleared the
        # one-shot guard; the next update leaves the module out: its kernel
        # tensors go back and must not be quantized again
        model = _cold_start(_checkpoint())
        before = _state(model)
        begin_direct_reload(model)
        del model.deq.online._already_called_process_weights_after_loading
        new = _checkpoint(1.01)
        unsent = ["deq.online.weight", "deq.online.bias"]
        _reload(model, {n: t for n, t in new.items() if n not in unsent})
        _assert_same(_state(model), before, names=unsent, addresses=True)
        assert model.deq.online.quant_method.runs == 1


class TestOnlineStaging:
    """Staged quantization runs post-load processing on temporaries."""

    def test_processing_objects_read_the_kernel_tensors(self, monkeypatch):
        # what post-load processing builds around its tensors (a quant
        # config's scale, a triton_kernels tensor on the weight's storage)
        # must read the kernel tensors, as captured CUDA graphs and later
        # kernel-format updates do, not keep the temporaries alive
        def quantize(layer):
            _quantize_int8(layer)
            layer.quant_method.config_scale = layer.weight_scale
            storage = SimpleNamespace(data=layer.weight.data.clone())
            store_params_in_triton(layer, weight=SimpleNamespace(storage=storage))
            layer.weight_triton = storage

        monkeypatch.setitem(_PWAL, "deq.online", quantize)
        model = _cold_start(_checkpoint())
        _reload(model, _checkpoint(1.01))
        layer = model.deq.online
        scale = layer.quant_method.config_scale
        assert scale.data_ptr() == layer.weight_scale.data_ptr()
        assert layer.weight_triton.data.data_ptr() == layer.weight.data_ptr()

    def test_relaid_out_bias_reloads_from_checkpoint_bytes(self, monkeypatch):
        # Marlin pads and permutes an online Linear's bias: it is staged with
        # the weight, so the loader writes the checkpoint layout, not the
        # kernel's, and post-load processing re-lays it out
        def quantize(layer):
            _quantize_int8(layer)
            bias = torch.nn.functional.pad(layer.bias, (0, 4)).flip(0)
            replace_parameter(layer, "bias", bias)

        monkeypatch.setitem(_PWAL, "deq.online", quantize)
        model = _cold_start(_checkpoint())
        ptrs = _ptrs(_state(model))
        _reload(model, _checkpoint(1.01))
        _assert_same(_state(model), _state(_cold_start(_checkpoint(1.01))))
        assert _ptrs(_state(model)) == ptrs

    @pytest.mark.parametrize("has_bias", [False, True])
    def test_padded_moe_is_processed_as_its_experts_arrive(self, has_bias):
        # round-up padding is never loaded, and another rank's experts load
        # nothing: the layer completes on its last local expert, not before
        # (dropping it) and not at finish (buffering every expert)
        class _MoE(OnlineMoEMethodBase):
            def get_fused_moe_quant_config(self, layer):
                return None

            def process_weights_after_loading(self, layer):
                self.processed = True

        method = object.__new__(_MoE)
        method.requantization_source = None
        method.processed = False
        method.moe = SimpleNamespace(
            w13_num_shards=2,
            hidden_dim_unpadded=4,
            intermediate_size_per_partition_unpadded=2,
            has_bias=has_bias,
        )

        def load_expert(param, loaded_weight, expert_id):
            if expert_id < 2:  # local experts 0 and 1
                region = tuple(slice(0, n) for n in loaded_weight.shape)
                param.data[expert_id][region].copy_(loaded_weight)

        layer = torch.nn.Module()
        layer.quant_method = method
        layer.quant_config = SimpleNamespace(online_quantization_config=None)
        with _DEVICE:
            method.create_weights(
                layer,
                num_experts=2,
                hidden_size=8,
                intermediate_size_per_partition=4,
                params_dtype=torch.bfloat16,
                weight_loader=load_expert,
            )
        shapes = {"w13_weight": (4, 4), "w2_weight": (4, 2)}
        if has_bias:
            shapes |= {"w13_bias": (4,), "w2_bias": (4,)}
        loads = [(e, n) for e in (2, 0, 3, 1) for n in shapes]
        info = get_layerwise_info(layer)
        for i, (expert_id, name) in enumerate(loads):
            param = getattr(layer, name)
            param.weight_loader(param, torch.ones(shapes[name]), expert_id=expert_id)
            if expert_id == 2:
                assert info.loaded_weights == []
            assert method.processed == (i == len(loads) - 1)


class TestRefusals:
    def test_refused_begin_leaves_the_model_untouched(self):
        # proj is viewed back first, in module order; mimo's storage is short
        model = _cold_start(_checkpoint())
        owner, (_, _, dtype, offset), attrs, storage = model.mimo.weight._ckpt
        layout = ((64, 64), (64, 1), dtype, offset)
        model.mimo.weight._ckpt = (owner, layout, attrs, storage)
        before = _state(model)
        with pytest.raises(RuntimeError, match="checkpoint layout needs"):
            begin_direct_reload(model)
        _assert_same(_state(model), before, addresses=True)

    @pytest.mark.parametrize("staged", [False, True])
    def test_online_requantization_refused(self, staged):
        # requantization (not staged, as set_requantization_source) releases
        # the checkpoint parameters it dequantizes; online quantization is
        # staged and accepted
        online = object.__new__(Fp8PerTensorOnlineLinearMethod)
        online.uses_meta_device = staged
        with _DEVICE:
            model = _ToyModel()
        model.proj.quant_method = online
        refused = pytest.raises(RuntimeError, match="online requantization")
        with contextlib.nullcontext() if staged else refused:
            _check_supported_modules(model)

    def test_online_shared_experts_refused(self, monkeypatch):
        # post-load processing deletes the staging parameters a reload loads
        init = _ToyModel.__init__

        def with_online_shared_experts(self):
            init(self)
            self.proj.shared_expert_online_loader = object()

        monkeypatch.setattr(_ToyModel, "__init__", with_online_shared_experts)
        with pytest.raises(RuntimeError, match="fused shared experts"):
            _cold_start(_checkpoint())

    def test_load_plan_refuses_online_quantization(self):
        # `begin` stages online modules: their loaders would buffer the traces
        from vllm.model_executor.model_loader.reload import check_plannable

        with pytest.raises(RuntimeError, match="online quantization"):
            check_plannable(_cold_start(_checkpoint()))

    def test_online_quantization_with_eplb_refused(self, monkeypatch):
        # not validated with staging (elastic EP replaces the quant method)
        init = _ToyModel.__init__

        def with_eplb(self):
            init(self)
            parallel = SimpleNamespace(enable_eplb=True)
            self.deq.online.moe_config = SimpleNamespace(moe_parallel_config=parallel)

        monkeypatch.setattr(_ToyModel, "__init__", with_eplb)
        with pytest.raises(RuntimeError, match="EPLB"):
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
    ptrs = _ptrs(_state(model))
    begin_direct_reload(model)
    model.load_weights(list(_checkpoint(1.01).items())[:3])  # update aborts
    assert "_direct_reload_begun" in model.__dict__
    _reload(model, _checkpoint(1.01))
    assert "_direct_reload_begun" not in model.__dict__
    _assert_same(_state(model), _state(_cold_start(_checkpoint(1.01))))
    assert _ptrs(_state(model)) == ptrs


def test_abort_unstages_online_modules():
    # the worker recommits the lent KV cache after an abort: the staged
    # checkpoint tensors must not outlive the update
    model = _cold_start(_checkpoint())
    layer = model.deq.online
    before = _state(model)
    begin_direct_reload(model)
    model.load_weights([("deq.online.weight", _checkpoint(1.01)["deq.online.weight"])])
    assert get_layerwise_info(layer).loaded_weights
    abort_direct_reload(model)
    assert get_layerwise_info(layer).loaded_weights == []
    names = ["deq.online.weight", "deq.online.weight_scale", "deq.online.bias"]
    _assert_same(_state(model), before, names=names, addresses=True)
    _reload(model, _checkpoint(1.01))  # the next full update repairs the model
    _assert_same(_state(model), _state(_cold_start(_checkpoint(1.01))))


def test_layerwise_reload_keeps_a_bias_registered_after_recording():
    # a bias the model's load_weights registers (Transformers-backend
    # lm_head) has no meta copy: it stays live, the layer waits for it, and
    # its loader is unwrapped once the layer is processed
    with _DEVICE:
        model = torch.nn.Module()
        model.add_module("head", torch.nn.Module())
        head = model.head
        head.weight = torch.nn.Parameter(torch.zeros(4, 4), False)
        record_metadata_for_reloading(model)
        head.bias = torch.nn.Parameter(torch.zeros(4), False)
    weight, bias = torch.rand(4, 4), torch.rand(4)
    initialize_layerwise_reload(model)
    for name, value in (("weight", weight), ("bias", bias)):
        param = getattr(head, name)
        param.weight_loader(param, value.to(_DEVICE))
    finalize_layerwise_reload(model, _MODEL_CONFIG)
    assert torch.equal(head.weight.cpu(), weight)
    assert torch.equal(head.bias.cpu(), bias)
    assert getattr(head.bias, "weight_loader", default_weight_loader) is (
        default_weight_loader
    )


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


def test_reload_into_weight_moved_to_an_offset(monkeypatch):
    def move(layer):  # DeepSeek-V4 ROCm GEMM fusion: `set_` into a shared buffer
        w = layer.weight
        if w.storage_offset() == 0:  # once; reloads refill it there
            w.data = torch.cat((torch.zeros_like(w), w))[len(w) :]

    init = _ToyModel.__init__

    def with_move(self):
        init(self)
        self.proj.quant_method = _Method(move)

    monkeypatch.setattr(_ToyModel, "__init__", with_move)
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
        lora_config=None,
        speculative_config=None,
        weight_transfer_config=None,
    )
    return SimpleNamespace(**(fields | overrides))


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
