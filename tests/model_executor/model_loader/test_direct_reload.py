# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct weight reload (`VLLM_RELOAD_MODE=direct`) on a toy model whose
post-load processing has the hazards found on real models:

(a) a parent derives a separate tensor from a child's weight (MLA W_UK/W_UV
    from a quantized kv_b_proj);
(b) post-load processing changes a weight's layout;
(c) the model's loader writes a weight without its `weight_loader` (MiMo);
(d) post-load processing consumes a checkpoint tensor, which stays
    registered as a retired parameter (NVFP4 `weight_scale_2`);
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
import weakref
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import store_params_in_triton
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.online.arrival import OnlineArrivalMixin
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


class _RowwiseOnlineMethod(OnlineArrivalMixin, _OnlineMethod):
    """An online scheme with one int8 scale per `group` rows: a direct
    reload quantizes each arriving shard at once."""

    def __init__(self, group):
        super().__init__(self._quantize)
        self.group = group

    def arrival_row_group(self, layer, name):
        return self.group if name == "weight" else None

    def quantize_param(self, layer, name, weight):
        rows, cols = weight.shape
        blocks = -(-rows // self.group)
        padded = torch.nn.functional.pad(
            weight.float(), (0, 0, 0, blocks * self.group - rows)
        )
        scale = padded.view(blocks, -1).abs().amax(-1, keepdim=True) / 127
        row_scale = scale.repeat_interleave(self.group, 0)[:rows]
        return (weight.float() / row_scale).round().to(torch.int8), scale

    def _quantize(self, layer):
        qweight, scale = self.quantized_param(layer, "weight", layer.weight)
        replace_parameter(layer, "weight", qweight)
        replace_parameter(layer, "weight_scale", scale)


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
    """As `BaseModelLoader.load_model` does."""
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
    """Layout and bytes (and addresses) equal, for `names` or every tensor."""
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

    @staticmethod
    def _online_experts(cls, has_bias):
        """An online MoE layer: 2 local experts of 4, round-up padded, TP rank
        0 of 2; its process_weights_after_loading only records the call."""

        class _MoE(OnlineMoEMethodBase):
            def get_fused_moe_quant_config(self, layer):
                return None

            def process_weights_after_loading(self, layer):
                self.processed = True

        config = SimpleNamespace(
            w13_num_shards=2,
            is_act_and_mul=True,
            hidden_dim_unpadded=4,
            intermediate_size_per_partition_unpadded=3,
            has_bias=has_bias,
            tp_rank=0,
            tp_shard_with_padding=False,
            num_experts=4,
            moe_parallel_config=SimpleNamespace(tp_size=2),
        )
        method = object.__new__(_MoE)
        method.moe, method.requantization_source, method.processed = config, None, False
        layer = object.__new__(cls)
        torch.nn.Module.__init__(layer)
        layer.moe_config, layer.quant_method = config, method
        layer.quant_config = SimpleNamespace(
            get_name=lambda: "online", online_quantization_config=None
        )
        layer.expert_map_manager = SimpleNamespace(
            map_global_to_local=lambda i: i if i < 2 else -1,
            num_fused_shared_experts=0,
        )
        layer._loaded_expert_biases = set()
        layer.shared_expert_online_loader = None
        with _DEVICE:
            method.create_weights(
                layer, 2, 8, 8, torch.bfloat16, weight_loader=layer.weight_loader
            )
        return layer

    @pytest.mark.parametrize("has_bias", [False, True])
    def test_padded_moe_is_processed_as_its_experts_arrive(self, has_bias):
        # round-up padding is never loaded, and another rank's experts load
        # nothing: the layer completes on its last local expert, not before
        # (dropping it) and not at finish (buffering every expert)
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        layer = self._online_experts(RoutedExperts, has_bias)
        shapes = {"w1": (6, 4), "w3": (6, 4), "w2": (4, 6)}  # both TP ranks
        kinds = ["weight", "bias"] if has_bias else ["weight"]
        loads = [(e, s, k) for e in (2, 0, 3, 1) for k in kinds for s in shapes]
        info = get_layerwise_info(layer)
        for i, (expert, shard, kind) in enumerate(loads):
            name = ("w2" if shard == "w2" else "w13") + f"_{kind}"
            shape = shapes[shard] if kind == "weight" else shapes[shard][:1]
            param = getattr(layer, name)
            buffered = len(info.loaded_weights)
            param.weight_loader(param, torch.ones(shape), name, shard, expert)
            if expert >= 2:  # another rank's
                assert len(info.loaded_weights) == buffered
            assert layer.quant_method.processed == (i == len(loads) - 1)

    def test_own_expert_loader_counts_padding(self):
        # gpt-oss's loader copies the padded width: counting only checkpoint
        # elements would process the layer before its last experts
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        class _OwnLoader(RoutedExperts):
            def weight_loader(self, param, loaded_weight, *args, expert_id):  # type: ignore[override]
                if expert_id < 2:  # the whole padded expert, as gpt-oss
                    param.data[expert_id].copy_(loaded_weight)

        layer = self._online_experts(_OwnLoader, has_bias=False)
        loads = [(e, n) for e in (0, 1) for n in ("w13_weight", "w2_weight")]
        for i, (expert, name) in enumerate(loads):
            param = getattr(layer, name)
            param.weight_loader(param, torch.ones(param.shape[1:]), expert_id=expert)
            assert layer.quant_method.processed == (i == len(loads) - 1)

    @pytest.mark.parametrize("group,on_arrival", [(1, True), (2, True), (4, False)])
    def test_row_local_scheme_quantizes_shards_on_arrival(
        self, monkeypatch, group, on_arrival
    ):
        # a scheme whose scales cover only their own rows quantizes each
        # arriving shard (here the two halves of a merged weight) at once,
        # holding no high-precision copy; a shard that splits a group of rows
        # is buffered instead; either way the result equals a cold start
        init = _ToyModel.__init__

        def merged(param, loaded_weight, rows=None):
            param.data[slice(None) if rows is None else rows].copy_(loaded_weight)

        def with_rowwise_scheme(self):
            init(self)
            online = self.deq.online
            online.quant_method = _RowwiseOnlineMethod(group)
            self.deq.quant_method = _Method(lambda layer: None)
            online.weight.weight_loader = merged
            initialize_online_processing(online)

        monkeypatch.setattr(_ToyModel, "__init__", with_rowwise_scheme)
        model = _cold_start(_checkpoint())
        new = _checkpoint(1.01)
        online = model.deq.online
        info = get_layerwise_info(online)
        begin_direct_reload(model)
        received = new["deq.online.weight"][:2].to(_DEVICE, copy=True)
        alive = weakref.ref(received)
        online.weight.weight_loader(online.weight, received, slice(0, 2))
        del received
        assert alive() is None or not on_arrival and _DEVICE.type == "cpu"
        assert bool(info.loaded_weights) is not on_arrival
        assert ("weight" in info.arrival) is on_arrival
        online.weight.weight_loader(
            online.weight, new["deq.online.weight"][2:], slice(2, 4)
        )
        model.load_weights((n, t) for n, t in new.items() if n != "deq.online.weight")
        finish_direct_reload(model, _MODEL_CONFIG, _DEVICE)
        _assert_same(_state(model), _state(_cold_start(new)))

    def test_staged_load_buffers_only_this_ranks_shard(self, monkeypatch):
        # a TP loader copies one shard of the received tensor: only the shard
        # is buffered, so the received tensor (with packed NCCL, a view of
        # the whole receive buffer) is not kept until the module completes
        init = _ToyModel.__init__

        def with_shard_loader(self):
            init(self)
            online = self.deq.online
            online.weight.weight_loader = lambda param, w: default_weight_loader(
                param, w[: len(param)]
            )
            initialize_online_processing(online)

        monkeypatch.setattr(_ToyModel, "__init__", with_shard_loader)
        model = _cold_start(_checkpoint())
        new = _checkpoint(1.01)
        shard = new["deq.online.weight"]
        received = torch.cat([shard, torch.zeros_like(shard)]).to(_DEVICE)
        alive = weakref.ref(received)
        begin_direct_reload(model)
        model.load_weights([("deq.online.weight", received)])
        del received
        if _DEVICE.type != "cpu":  # host sources may stay (file-backed) views
            assert alive() is None
        ((name, copies),) = get_layerwise_info(model.deq.online).loaded_weights
        assert isinstance(copies, list)
        assert sum(source.numel() for *_, source in copies) == shard.numel()
        model.load_weights((n, t) for n, t in new.items() if n != "deq.online.weight")
        finish_direct_reload(model, _MODEL_CONFIG, _DEVICE)
        _assert_same(_state(model), _state(_cold_start(new)))


def _scheme(cls, **attrs):
    method = object.__new__(cls)
    for name, value in attrs.items():
        setattr(method, name, value)
    return method


def _schemes(tp_size=1):
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.model_executor.layers.quantization.online.fp8 import (
        Fp8PerBlockOnlineLinearMethod,
        Fp8PerBlockOnlineMoEMethod,
        Fp8PtpcOnlineLinearMethod,
        Fp8PtpcOnlineMoEMethod,
    )
    from vllm.model_executor.layers.quantization.online.int8 import (
        Int8OnlineMoEMethod,
    )
    from vllm.model_executor.layers.quantization.online.mxfp4 import (
        Mxfp4OnlineLinearMethod,
        Mxfp4OnlineMoEMethod,
    )
    from vllm.model_executor.layers.quantization.online.mxfp8 import (
        Mxfp8OnlineLinearMethod,
        Mxfp8OnlineMoEMethod,
    )

    moe = SimpleNamespace(
        tp_size=tp_size,
        intermediate_size_per_partition_unpadded=256,
        hidden_dim_unpadded=256,
    )
    block = {"weight_block_size": [128, 128]}
    return {
        "fp8-per-channel-linear": _scheme(Fp8PtpcOnlineLinearMethod),
        "fp8-per-block-linear": _scheme(Fp8PerBlockOnlineLinearMethod, **block),
        "mxfp8-linear": _scheme(Mxfp8OnlineLinearMethod),
        "mxfp4-linear": _scheme(Mxfp4OnlineLinearMethod),
        "fp8-per-channel-moe": _scheme(Fp8PtpcOnlineMoEMethod, moe=moe),
        "fp8-per-block-moe": _scheme(Fp8PerBlockOnlineMoEMethod, moe=moe, **block),
        "mxfp8-moe": _scheme(Mxfp8OnlineMoEMethod, moe=moe),
        "mxfp4-moe": _scheme(
            Mxfp4OnlineMoEMethod, moe=moe, mxfp4_backend=Mxfp4MoeBackend.TRITON
        ),
        "int8-moe": _scheme(Int8OnlineMoEMethod, moe=moe),
    }


_SCHEMES = (
    "fp8-per-channel-linear",
    "fp8-per-block-linear",
    "mxfp8-linear",
    "mxfp4-linear",
    "fp8-per-channel-moe",
    "fp8-per-block-moe",
    "mxfp8-moe",
    "mxfp4-moe",
    "int8-moe",
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="quantization kernels")
@pytest.mark.parametrize("scheme", _SCHEMES)
def test_row_local_scheme_quantizes_on_arrival_as_whole(scheme):
    # each real scheme with row-local scales: quantizing the arriving shards
    # into the outputs (no high-precision buffer) gives the bytes quantizing
    # the whole weight gives; rows that never arrive (padding) included
    method = _schemes()[scheme]
    moe = scheme.endswith("moe")
    for name in ("w13_weight", "w2_weight") if moe else ("weight",):
        _check_arrival_equals_whole(method, moe, name)


def _arrival_layer(method, moe, tp_size=1):
    """A layer of `method` with 256 x 256 weights, row parallel at `tp_size`."""
    shape = (2, 256, 256) if moe else (256, 256)
    layer = torch.nn.Module()
    layer.quant_method = method
    layer.tp_size, layer.logical_widths = tp_size, [128, 128]
    layer.input_size, layer.input_size_per_partition = 256 * tp_size, 256
    layer.output_size = layer.output_size_per_partition = 256
    for name in ("w13_weight", "w2_weight") if moe else ("weight",):
        meta = torch.empty(shape, device="meta", dtype=torch.bfloat16)
        param = torch.nn.Parameter(meta, False)
        param.weight_loader_numel = param.numel()
        layer.register_parameter(name, param)
    return layer, shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="quantization kernels")
@pytest.mark.parametrize("scheme", list(_SCHEMES))
def test_scale_reduced_across_ranks_is_buffered(monkeypatch, scheme):
    # at TP 2 (row parallel, expert TP) each real scheme's `quantize_param`
    # reduces a scale across ranks exactly when its `reduces_across_ranks`
    # says so, and such a parameter is not quantized on arrival
    from vllm.model_executor.layers.quantization.utils import quant_utils
    from vllm.model_executor.model_loader.reload.layerwise import (
        _quantize_on_arrival,
    )
    from vllm.model_executor.model_loader.reload.types import (
        LayerReloadingInfo,
        RecordedCopies,
    )

    reduced: list[torch.Tensor] = []
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda t, **_: reduced.append(t)
    )
    group = SimpleNamespace(device_group=None)
    monkeypatch.setattr(quant_utils, "get_tp_group", lambda: group)
    monkeypatch.setattr(quant_utils, "get_ep_group", lambda: group)
    method = _schemes(tp_size=2)[scheme]
    moe = scheme.endswith("moe")
    layer, shape = _arrival_layer(method, moe, tp_size=2)
    for name in ("w13_weight", "w2_weight") if moe else ("weight",):
        reduced.clear()
        weight = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        method.quantize_param(layer, name, weight)
        reduces = method.reduces_across_ranks(layer, name)
        assert bool(reduced) is reduces, name
        info = LayerReloadingInfo(({}, {}), torch.device("cuda"))
        if reduces:
            assert not _quantize_on_arrival(layer, info, name, RecordedCopies())


def _check_arrival_equals_whole(method, moe, name):
    from vllm.model_executor.model_loader.reload.layerwise import (
        _quantize_on_arrival,
    )
    from vllm.model_executor.model_loader.reload.types import (
        LayerReloadingInfo,
        RecordedCopies,
    )

    layer, shape = _arrival_layer(method, moe)
    param = getattr(layer, name)
    group = method.arrival_row_group(layer, name)
    assert group is not None

    weight = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    weight[..., 128:, :] = 0  # never arrives
    cols = slice(0, 192) if moe else slice(None)  # MoE: padded columns
    if moe:
        weight[..., 192:] = 0
    pieces = [(0, 64), (64, 128)] if group == 1 else [(0, 128)]
    copies = RecordedCopies()
    for expert in range(shape[0]) if moe else [None]:
        for first, end in pieces:
            index = (expert, slice(first, end), cols) if moe else slice(first, end)
            view = param.data[index]
            copies.append(
                (
                    view.size(),
                    view.stride(),
                    view.storage_offset(),
                    weight[index].clone(),
                )
            )
    info = LayerReloadingInfo(({}, {}), torch.device("cuda"))
    assert _quantize_on_arrival(layer, info, name, copies)
    for arrived, whole in zip(
        info.arrival[name].outputs, method.quantize_param(layer, name, weight)
    ):
        assert arrived.shape == whole.shape
        assert torch.equal(
            arrived.view(torch.uint8), whole.contiguous().view(torch.uint8)
        )


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
