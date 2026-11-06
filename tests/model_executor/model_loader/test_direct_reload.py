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

from vllm.config import LoadConfig, VllmConfig, WeightTransferConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.input_hadamard import (
    InputHadamardLinearMethod,
)
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerTensorOnlineLinearMethod,
)
from vllm.model_executor.model_loader.reload import finish_reload, start_reload
from vllm.model_executor.model_loader.reload.direct import (
    begin_direct_reload,
    check_direct_reload_support,
    finish_direct_reload,
)
from vllm.model_executor.model_loader.reload.inplace import reload_weights
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsDirectReload
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


class _ToyModel(torch.nn.Module, SupportsDirectReload):
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
        owner, (_, _, dtype, offset), attrs, storage = model.mimo.weight._ckpt
        layout = ((64, 64), (64, 1), dtype, offset)
        model.mimo.weight._ckpt = (owner, layout, attrs, storage)
        before = _state(model)
        with pytest.raises(RuntimeError, match="checkpoint layout needs"):
            begin_direct_reload(model)
        _assert_same(_state(model), before, addresses=True)

    def _assert_begin_refused(self, model, match) -> None:
        before = _state(model)
        with pytest.raises(RuntimeError, match=match):
            begin_direct_reload(model)
        _assert_same(_state(model), before, addresses=True)
        assert "_direct_reload_begun" not in model.__dict__

    @pytest.mark.parametrize(
        "kind,match",
        [
            # finish clears the guard that keeps online quantization from
            # quantizing its converted bytes again
            ("online", "online quantization"),
            # elastic EP wraps the method after load
            ("online-elastic-ep", "online quantization"),
            # post-load processing deletes the staging parameters a reload loads
            ("online-shared-experts", "online quantization"),
            # its post-load processing runs only once per quant method
            ("HummingLinearMethod", "humming"),
            ("HummingMoEMethod", "humming"),
            # Quark online rotation deletes input_rotation after loading
            ("InputHadamardLinearMethod", "input_rotation"),
        ],
    )
    def test_quant_method_refused(self, kind, match):
        from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
            FusedMoEModularMethod,
        )
        from vllm.model_executor.layers.quantization import humming as humming_mod

        model = _cold_start(_checkpoint())
        online = object.__new__(Fp8PerTensorOnlineLinearMethod)
        if kind == "online":
            method = online
        elif kind == "online-elastic-ep":
            method = object.__new__(FusedMoEModularMethod)
            method.__dict__["old_quant_method"] = online
        elif kind == "online-shared-experts":
            model.proj.shared_expert_online_loader = object()
            method = model.proj.quant_method
        elif kind == "InputHadamardLinearMethod":
            method = object.__new__(InputHadamardLinearMethod)
        else:
            method = object.__new__(getattr(humming_mod, kind))
        model.proj.__dict__["quant_method"] = method
        self._assert_begin_refused(model, match)

    @pytest.mark.parametrize(
        "module_path,cls_name",
        [
            ("vllm.models.muse_spark.nvidia.mega_moe", "MuseSparkMegaMoEExperts"),
            ("vllm.models.muse_spark.nvidia.hybrid_moe", "MuseSparkHybridMoEExperts"),
            ("vllm.models.deepseek_v4.amd.mega_moe", "DeepseekV4AiterMegaMoEExperts"),
            ("vllm.models.step5.nvidia.mega_moe", "Step5MegaMoEExperts"),
            # its rows are views of the cold-start checkpoint files
            (
                "vllm.models.qwen4_exp.nvidia.ngram_embedding",
                "Qwen4ExpPLEDiskOffloadEmbedding",
            ),
        ],
    )
    def test_unsupported_component_refused(self, module_path, cls_name):
        # post-load processing with a one-shot guard: a rerun is a no-op
        import importlib

        cls = getattr(importlib.import_module(module_path), cls_name)
        model = _cold_start(_checkpoint())
        experts = object.__new__(cls)
        torch.nn.Module.__init__(experts)
        model.proj.add_module("experts", experts)
        self._assert_begin_refused(
            model, rf"{cls_name} is not supported \(proj\.experts\)"
        )

    @pytest.mark.parametrize(
        "reason,match",
        [
            ("weight copies are built once", "weight copies are built once"),
            (None, "no reason given"),
        ],
    )
    def test_instance_declared_unsupported_refused(self, reason, match):
        # set on a model instance at runtime (MiniMax-M3 AMD mono decode);
        # False refuses even a validated class
        model = _cold_start(_checkpoint())
        model.supports_direct_reload = False
        if reason is not None:
            model.direct_reload_unsupported_reason = reason
        self._assert_begin_refused(
            model, rf"_ToyModel is not supported \(model\): {match}"
        )

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


def _worker(model, draft=None, **config):
    from vllm.v1.worker.gpu_worker import Worker

    worker = object.__new__(Worker)
    worker.vllm_config = _config(parallel_config=None, **config)
    worker.device = _DEVICE
    worker.model_runner = SimpleNamespace(
        get_model=lambda: model, get_draft_model=lambda: draft
    )
    return worker


def _reload_config(**model_config) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(**vars(_MODEL_CONFIG), **model_config),
        load_config=LoadConfig(),
        device_config=SimpleNamespace(device=_DEVICE),
        lora_config=None,
    )


def _refused(model):
    model.supports_direct_reload = False
    model.direct_reload_unsupported_reason = "weight copies are built once"
    return model


class TestCallSites:
    """The check runs when the weight-transfer engine is created and in
    `begin`, which caches only a pass."""

    @pytest.mark.parametrize("refused", ["target", "draft", "lora"])
    def test_refused_model_fails_engine_creation(self, refused):
        # at startup, before a trainer connects and waits in a collective
        target, draft = _cold_start(_checkpoint()), _cold_start(_checkpoint())
        if refused != "lora":
            _refused(target if refused == "target" else draft)
        lora = {"lora_config": object()} if refused == "lora" else {}
        worker = _worker(target, draft, **lora)
        with pytest.raises(RuntimeError, match="built once|LoRA"):
            worker._create_weight_transfer_engine(WeightTransferConfig(backend="ipc"))
        assert not hasattr(worker, "weight_transfer_engine")

    def test_layerwise_engine_creation_not_checked(self, monkeypatch):
        monkeypatch.setenv("VLLM_RELOAD_MODE", "layerwise")
        worker = _worker(_refused(_cold_start(_checkpoint())))
        worker._create_weight_transfer_engine(WeightTransferConfig(backend="ipc"))
        assert worker.weight_transfer_engine is not None

    def test_sparse_nccl_warns_and_is_not_checked(self, caplog_vllm):
        # its patches skip begin/finish: nothing to check, a warning instead
        worker = _worker(_refused(_cold_start(_checkpoint())))
        worker._create_weight_transfer_engine(
            WeightTransferConfig(backend="sparse_nccl")
        )
        assert worker.weight_transfer_engine is not None
        assert any(
            "skip the direct-reload lifecycle" in r.getMessage()
            for r in caplog_vllm.records
        )

    def test_begin_caches_only_a_pass(self, monkeypatch):
        from vllm.model_executor.model_loader.reload import direct

        calls = []
        check = direct.check_direct_reload_support

        def counted(model):
            calls.append(model)
            check(model)

        monkeypatch.setattr(direct, "check_direct_reload_support", counted)
        model = _refused(_cold_start(_checkpoint()))
        for _ in range(2):  # a refusal is not cached
            with pytest.raises(RuntimeError, match="built once"):
                begin_direct_reload(model)
        del model.supports_direct_reload  # e.g. mono decode turned off
        _reload(model, _checkpoint(1.01))
        _reload(model, _checkpoint(1.02))  # the pass is cached: no walk
        assert len(calls) == 3
        _assert_same(_state(model), _state(_cold_start(_checkpoint(1.02))))

    def test_reload_weights_refused_before_config_changes(self):
        model = _refused(_cold_start(_checkpoint()))
        config = _reload_config(model="old", model_weights="w", revision="r")
        before = _state(model)
        with pytest.raises(RuntimeError, match="weight copies are built once"):
            reload_weights(config, model, None, "new", True)
        cfg = config.model_config
        assert (cfg.model, cfg.model_weights, cfg.revision) == ("old", "w", "r")
        _assert_same(_state(model), before, addresses=True)

    @pytest.mark.parametrize("marked", [True, False])
    def test_unmarked_model_warns_once(self, caplog_vllm, marked):
        class _UnmarkedOnce(_ToyModel):  # the mark is not inherited
            pass

        cls = _ToyModel if marked else _UnmarkedOnce
        with _DEVICE:
            model = cls()
        model.load_weights(_checkpoint().items())
        process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
        _worker(model)._create_weight_transfer_engine(
            WeightTransferConfig(backend="ipc")
        )
        for scale in (1.01, 1.02):
            reload_weights(
                _reload_config(), model, iter(_checkpoint(scale).items()), None, True
            )
        warnings = [r for r in caplog_vllm.records if "not validated" in r.getMessage()]
        if marked:
            assert warnings == []
        else:
            assert len(warnings) == 1 and "_UnmarkedOnce" in warnings[0].getMessage()
        _assert_same(_state(model), _state(_cold_start(_checkpoint(1.02))))


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


def _offload(backend="auto", group_size=0, cpu_offload_gb=0) -> SimpleNamespace:
    return SimpleNamespace(
        offload_backend=backend,
        prefetch=SimpleNamespace(offload_group_size=group_size),
        uva=SimpleNamespace(cpu_offload_gb=cpu_offload_gb),
    )


def _config(**overrides) -> SimpleNamespace:
    fields = dict(
        offload_config=_offload(),
        model_config=SimpleNamespace(quantization=None),
        quant_config=None,
        lora_config=None,
        speculative_config=None,
        weight_transfer_config=None,
    )
    return SimpleNamespace(**(fields | overrides))


_ONLINE = object.__new__(OnlineQuantizationConfig)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({}, None),
        ({"offload_config": _offload(backend="uva")}, "weight offloading"),
        ({"offload_config": _offload(group_size=2)}, "weight offloading"),
        ({"offload_config": _offload(cpu_offload_gb=1)}, "weight offloading"),
        ({"model_config": SimpleNamespace(quantization="torchao")}, "torchao"),
        ({"quant_config": _ONLINE}, "online quantization"),
        (
            {"quant_config": SimpleNamespace(online_quantization_config=_ONLINE)},
            "online quantization",
        ),
        ({"lora_config": object()}, "LoRA"),
        (
            {"speculative_config": SimpleNamespace(draft_lm_head_quantization="fp8")},
            "draft_lm_head_quantization",
        ),
    ],
    ids=[
        "defaults",
        "offload-backend",
        "offload-prefetch",
        "offload-uva",
        "torchao",
        "online-config",
        "online-composed",
        "lora",
        "draft-lm-head-quant",
    ],
)
def test_config_refused_by_the_check_not_at_startup(overrides, match):
    # refused where the check runs (weight transfer, begin); a server that
    # never updates starts in direct mode
    config = _config(**overrides)
    VllmConfig._verify_direct_reload_compat(config)
    model = _cold_start(_checkpoint())
    if match is None:
        check_direct_reload_support(model, config)
    else:
        with pytest.raises(RuntimeError, match=match):
            check_direct_reload_support(model, config)


@pytest.mark.parametrize(
    "mode,backend,match",
    [
        ("direct", "sharded_rdt", "sharded_rdt"),
        ("layerwise", "sharded_rdt", None),
        ("direct", "sparse_nccl", None),  # its engine warns instead
    ],
)
def test_verify_direct_reload_compat(monkeypatch, mode, backend, match):
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    config = _config(weight_transfer_config=SimpleNamespace(backend=backend))
    if match is None:
        VllmConfig._verify_direct_reload_compat(config)
    else:
        with pytest.raises(ValueError, match=match):
            VllmConfig._verify_direct_reload_compat(config)
