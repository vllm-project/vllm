# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-FP8 checkpoint weights of layers that run in BF16 (`Fp8DequantConfig`).

The weight and its scale load independently (any order, any call, no pairing in
the model's loader); BF16 rows load as is; post-load processing dequantizes in
the weight's own storage, so cold start, direct reload and layerwise reload all
end with the same bytes at the startup address."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.fp8_dequant import (
    Fp8DequantConfig,
    Fp8DequantLinearMethod,
    load_unneeded_scale,
)
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.utils import get_live_param, weights_already_processed
from vllm.utils.math_utils import cdiv

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_CONFIG = SimpleNamespace(
    word_embeddings_untied_by_checkpoint=False, quantization=None, dtype=torch.bfloat16
)
# q_a / kv_a-like shards: the first spans several dequantization chunks, the
# last ends in a partial block
_OUT = [1280, 192]


@pytest.fixture(autouse=True)
def _tp(monkeypatch, request):
    rank, size = getattr(request, "param", (0, 1))
    for module in (
        "vllm.distributed",  # imported at call time by validate_fp8_block_shape
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.parameter",
    ):
        monkeypatch.setattr(f"{module}.get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(
            f"{module}.get_tensor_model_parallel_world_size", lambda: size
        )


def _block_fp8(rows: int, cols: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=gen).to(torch.float8_e4m3fn)
    scale = torch.rand(-(-rows // 128), -(-cols // 128), generator=gen) + 0.5
    return weight.to(_DEVICE), scale.to(_DEVICE)


def _dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, cols = weight.shape
    s = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
    return (weight.float() * s).to(torch.bfloat16)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = Fp8DequantConfig([128, 128])
        with _DEVICE:
            self.qkv_a = MergedColumnParallelLinear(
                256, _OUT, bias=False, params_dtype=torch.bfloat16, quant_config=config
            )
            self.o = RowParallelLinear(
                256, 128, bias=False, params_dtype=torch.bfloat16, quant_config=config
            )

    def load_weights(self, weights):
        loaded: set[str] = set()
        for name, tensor in weights:
            layer, shard, param_name = name.split(".")
            param = getattr(getattr(self, layer), param_name)
            shard_id = None if shard == "-" else int(shard)
            param.weight_loader(
                param, tensor, *([] if shard_id is None else [shard_id])
            )
            load_unneeded_scale(
                f"{layer}.{param_name}", param, tensor, loaded, shard_id
            )


def _checkpoint(seed: int, bf16_shard: int | None = None) -> dict[str, torch.Tensor]:
    """``layer.shard.param`` -> tensor; shard ``bf16_shard`` of qkv_a is BF16."""
    ckpt = {}
    for shard, rows in enumerate(_OUT):
        weight, scale = _block_fp8(rows, 256, seed + shard)
        if shard == bf16_shard:
            ckpt[f"qkv_a.{shard}.weight"] = _dequant(weight, scale)
        else:
            ckpt[f"qkv_a.{shard}.weight"] = weight
            ckpt[f"qkv_a.{shard}.weight_scale_inv"] = scale
    weight, scale = _block_fp8(128, 256, seed + 9)
    ckpt["o.-.weight"], ckpt["o.-.weight_scale_inv"] = weight, scale
    return ckpt


def _expected(ckpt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    def runtime(prefix):
        weight = ckpt[f"{prefix}.weight"]
        if weight.dtype == torch.bfloat16:
            return weight
        return _dequant(weight, ckpt[f"{prefix}.weight_scale_inv"])

    qkv_a = torch.cat([runtime(f"qkv_a.{shard}") for shard in range(len(_OUT))])
    return {"qkv_a": qkv_a, "o": runtime("o.-")}


def _cold_start(ckpt: dict[str, torch.Tensor]) -> _Model:
    model = _Model()
    record_metadata_for_reloading(model)
    model.load_weights(reversed(list(ckpt.items())))  # scales before weights
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    return model


def _assert_runtime(model: _Model, expected: dict[str, torch.Tensor]) -> None:
    for name, weight in expected.items():
        layer = getattr(model, name)
        assert layer.weight.dtype == torch.bfloat16
        assert layer.weight.is_contiguous()
        torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)
        assert get_live_param(layer, "weight_scale_inv") is None  # retired


def test_layers_use_the_dequant_method():
    model = _Model()
    assert isinstance(model.qkv_a.quant_method, Fp8DequantLinearMethod)
    # the weight is the BF16 weight; its FP8 view spans the same bytes, one
    # row per BF16 row
    weight = model.qkv_a.weight
    assert weight.dtype == torch.bfloat16 and weight.is_contiguous()
    fp8 = weight.fp8_view()
    assert fp8.dtype == torch.float8_e4m3fn
    assert fp8.stride() == (2 * 256, 1)
    assert fp8.data_ptr() == weight.data_ptr()
    assert weight.untyped_storage().nbytes() == sum(_OUT) * 256 * 2


def test_dummy_weights_are_runtime_values():
    """--load-format dummy fills the BF16 weight: no row holds FP8 bytes."""
    from vllm.model_executor.model_loader.weight_utils import (
        initialize_single_dummy_weight,
    )

    model = _Model()
    model.o.weight.view(torch.int16).fill_(0x7FC0)  # BF16 NaN
    for module in model.modules():
        for name, param in module._parameters.items():
            if param is not None:
                initialize_single_dummy_weight(param, name=name)
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    for layer in (model.qkv_a, model.o):
        assert layer.weight.dtype == torch.bfloat16
        assert layer.weight.isfinite().all()
        assert layer.weight.abs().max() <= 1e-3


def test_weight_moved_before_processing():
    """CPU offloading moves the loaded weight (``.to(device)``) before and
    after post-load processing; the copy keeps the FP8 rows."""
    ckpt = _checkpoint(16)
    model = _Model()
    model.load_weights(ckpt.items())
    for layer in (model.qkv_a, model.o):
        layer.weight.data = layer.weight.data.to("cpu").clone().to(_DEVICE)
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    _assert_runtime(model, _expected(ckpt))


def test_raw_load_weights_outside_a_reload():
    """A caller that loads BF16 weights with model.load_weights alone (no
    reload lifecycle, no post-load processing) gets a runnable layer."""
    model = _cold_start(_checkpoint(12))
    new = _expected(_checkpoint(13))
    model.load_weights([("o.-.weight", new["o"])])
    assert model.o.weight.dtype == torch.bfloat16
    torch.testing.assert_close(model.o.weight, new["o"], rtol=0, atol=0)
    x = torch.randn(2, 256, dtype=torch.bfloat16, device=_DEVICE)
    torch.testing.assert_close(model.o(x)[0], x @ new["o"].t())


def test_cold_start_matches_dequantized_checkpoint():
    ckpt = _checkpoint(0)
    model = _Model()
    ptr = model.qkv_a.weight.data_ptr()
    model.load_weights(reversed(list(ckpt.items())))
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    _assert_runtime(model, _expected(ckpt))
    assert model.qkv_a.weight.data_ptr() == ptr  # dequantized in its own storage


def test_weight_and_scale_in_separate_calls():
    ckpt = _checkpoint(1)
    model = _Model()
    for item in ckpt.items():  # one tensor per load_weights call
        model.load_weights([item])
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    _assert_runtime(model, _expected(ckpt))


def test_bf16_shard_next_to_fp8_shard():
    ckpt = _checkpoint(2, bf16_shard=0)
    model = _cold_start(ckpt)
    _assert_runtime(model, _expected(ckpt))


def test_bf16_checkpoint_needs_no_scale():
    ckpt = {k: v for k, v in _checkpoint(3).items() if "scale" not in k}
    for name, weight in list(ckpt.items()):
        prefix = name.removesuffix(".weight")
        ckpt[name] = _dequant(weight, _checkpoint(3)[f"{prefix}.weight_scale_inv"])
    model = _cold_start(ckpt)
    torch.testing.assert_close(model.o.weight, ckpt["o.-.weight"], rtol=0, atol=0)


@pytest.mark.parametrize("_tp", [(1, 2)], indirect=True)
def test_tensor_parallel_slices():
    weight, scale = _block_fp8(128, 256, 4)  # rank 1 of 2 holds columns 128:256
    model = _Model()
    model.load_weights({"o.-.weight": weight, "o.-.weight_scale_inv": scale}.items())
    model.o.quant_method.process_weights_after_loading(model.o)
    expected = _dequant(weight, scale)[:, 128:]
    torch.testing.assert_close(model.o.weight, expected, rtol=0, atol=0)


def test_pre_processed_weights_are_kept():
    # the weight cache hands over plain parameters holding runtime bytes
    runtime = _expected(_checkpoint(11))
    model = _Model()
    for name, weight in runtime.items():
        layer = getattr(model, name)
        layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    with weights_already_processed():
        process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    _assert_runtime(model, runtime)


@pytest.mark.parametrize("bf16_shard", [None, 1])
def test_direct_reload_equals_cold_start(monkeypatch, bf16_shard):
    from vllm.model_executor.model_loader.reload.direct import (
        begin_direct_reload,
        finish_direct_reload,
    )

    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    model = _cold_start(_checkpoint(5))
    ptrs = {n: p.data_ptr() for n, p in model.named_parameters()}

    new = _checkpoint(6, bf16_shard)
    begin_direct_reload(model)
    model.load_weights(new.items())
    finish_direct_reload(model, _MODEL_CONFIG, _DEVICE)

    _assert_runtime(model, _expected(new))
    assert {n: p.data_ptr() for n, p in model.named_parameters()} == ptrs


def test_direct_reload_keeps_weights_left_out(monkeypatch):
    from vllm.model_executor.model_loader.reload.direct import (
        begin_direct_reload,
        finish_direct_reload,
    )

    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    old = _checkpoint(7)
    model = _cold_start(old)
    new = {k: v for k, v in _checkpoint(8).items() if k.startswith("qkv_a.")}
    begin_direct_reload(model)
    model.load_weights(new.items())
    finish_direct_reload(model, _MODEL_CONFIG, _DEVICE)

    # dequantized rows are runtime bytes: a reload that skips them keeps them
    expected = _expected({**old, **new})
    _assert_runtime(model, expected)


@pytest.mark.parametrize("bf16_shard", [None, 0, 1])
def test_layerwise_reload_equals_cold_start(bf16_shard):
    model = _cold_start(_checkpoint(9))
    ptrs = {n: p.data_ptr() for n, p in model.named_parameters()}

    new = _checkpoint(10, bf16_shard)
    initialize_layerwise_reload(model)
    model.load_weights(new.items())
    finalize_layerwise_reload(model, _MODEL_CONFIG)

    _assert_runtime(model, _expected(new))
    assert {n: p.data_ptr() for n, p in model.named_parameters()} == ptrs


@pytest.mark.parametrize("bf16_shard", [None, 0])
@pytest.mark.parametrize("bf16", [False, True])
def test_layerwise_reload_completes_while_loading(bf16_shard, bf16):
    """Every element is loaded, BF16 rows' scales included, so a layerwise
    reload processes each layer as its last tensor arrives instead of holding
    it for finalize."""
    model = _cold_start(_checkpoint(14))
    new = _checkpoint(15, bf16_shard)
    if bf16:  # a BF16 trainer: weights only
        runtime = _expected(new)
        new = {
            "qkv_a.0.weight": runtime["qkv_a"][: _OUT[0]],
            "qkv_a.1.weight": runtime["qkv_a"][_OUT[0] :],
            "o.-.weight": runtime["o"],
        }
    initialize_layerwise_reload(model)
    for item in new.items():
        model.load_weights([item])
    for layer in (model.qkv_a, model.o):
        assert layer.weight.device.type != "meta"
        assert get_layerwise_info(layer).load_numel_total is None  # processed
    finalize_layerwise_reload(model, _MODEL_CONFIG)
    _assert_runtime(model, runtime if bf16 else _expected(new))


# Sparse-attention indexer: `wk` (128 rows) and `weights_proj` (64 rows) fused
# into one replicated `wk_weights_proj`, whose FP8 checkpoint scale layout
# varies between models (`Fp8DequantConfig.for_checkpoint(any_scale_layout)`).
_WK, _WP, _HIDDEN = 128, 64, 512


class _IndexerModel(torch.nn.Module):
    """The indexer part of the DeepSeek-V3.2-style loaders."""

    def __init__(self):
        super().__init__()
        config = Fp8DequantConfig.for_checkpoint(object(), any_scale_layout=True)
        with _DEVICE:
            self.wk_weights_proj = MergedColumnParallelLinear(
                _HIDDEN,
                [_WK, _WP],
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=config,
                disable_tp=True,
            )

    def load_weights(self, weights) -> set[str]:
        from vllm.model_executor.models.deepseek_v2 import _indexer_checkpoint_name

        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            name = _indexer_checkpoint_name(f"layers.0.indexer.{name}")
            proj, _, suffix = name.removeprefix("layers.0.indexer.").partition(".")
            name = f"wk_weights_proj.{suffix}"
            shard = {"wk": 0, "weights_proj": 1}[proj]
            param = params[name]
            param.weight_loader(param, tensor, shard)
            load_unneeded_scale(name, param, tensor, loaded, shard)
            loaded.add(name)
        return loaded


def _scale_layout(layout: str, rows: int, gen: torch.Generator) -> torch.Tensor:
    """A checkpoint scale of a ``[rows, _HIDDEN]`` FP8 weight."""
    shapes = {
        "block": (cdiv(rows, 128), _HIDDEN // 128),
        "channel": (rows,),
        "channel_2d": (rows, 1),
        "tensor": (),
        "mx": (rows, _HIDDEN // 32),
    }
    scale = torch.rand(shapes[layout], generator=gen) + 0.5
    if layout == "mx":  # E8M0 exponents of 2^-2 .. 2^1
        scale = torch.randint(125, 129, shapes[layout], generator=gen)
        return scale.to(torch.uint8).to(_DEVICE)
    return scale.to(_DEVICE)


def _indexer_checkpoint(seed: int, layout: str = "block", fp8_wp: bool = False):
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        scaled_dequantize,
    )

    gen = torch.Generator().manual_seed(seed)
    ckpt, expected = {}, []
    for name, rows, fp8 in (("wk", _WK, True), ("weights_proj", _WP, fp8_wp)):
        weight = torch.randn(rows, _HIDDEN, generator=gen)
        if not fp8:
            ckpt[f"{name}.weight"] = weight.to(torch.bfloat16).to(_DEVICE)
            expected.append(ckpt[f"{name}.weight"])
            continue
        weight = weight.to(torch.float8_e4m3fn).to(_DEVICE)
        scale = _scale_layout(layout, rows, gen)
        # Quark / compressed-tensors call it `weight_scale`
        scale_name = (
            "weight_scale" if layout.startswith("channel") else "weight_scale_inv"
        )
        ckpt[f"{name}.weight"], ckpt[f"{name}.{scale_name}"] = weight, scale
        # what the loaders computed before: dequantize the pair on arrival
        ref = scale.view(torch.float8_e8m0fnu).float() if layout == "mx" else scale
        ref = ref.reshape(ref.shape[0] if ref.ndim else 1, -1)
        group = GroupShape(rows // ref.shape[0], _HIDDEN // ref.shape[1])
        expected.append(scaled_dequantize(weight, ref, group, torch.bfloat16))
    return ckpt, torch.cat(expected)


def _indexer_cold_start(ckpt) -> _IndexerModel:
    model = _IndexerModel()
    record_metadata_for_reloading(model)
    for item in reversed(list(ckpt.items())):  # scales first, one per call
        model.load_weights([item])
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    return model


@pytest.mark.parametrize("layout", ["block", "channel", "channel_2d", "tensor", "mx"])
@pytest.mark.parametrize("scales_first", [True, False])
def test_indexer_wk_any_scale_layout(layout, scales_first):
    ckpt, expected = _indexer_checkpoint(20, layout)
    model = _IndexerModel()
    items = list(ckpt.items())
    for item in reversed(items) if scales_first else items:
        model.load_weights([item])  # weight and scale in separate calls
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    layer = model.wk_weights_proj
    torch.testing.assert_close(layer.weight, expected, rtol=0, atol=0)
    assert get_live_param(layer, "weight_scale_inv") is None


def test_indexer_fp8_weights_proj():
    # dots3 stores weights_proj as FP8 too
    ckpt, expected = _indexer_checkpoint(21, fp8_wp=True)
    model = _indexer_cold_start(ckpt)
    torch.testing.assert_close(model.wk_weights_proj.weight, expected, rtol=0, atol=0)


def test_indexer_bf16_wk():
    ckpt, _ = _indexer_checkpoint(22)
    bf16 = {"wk.weight": torch.randn(_WK, _HIDDEN).to(torch.bfloat16).to(_DEVICE)}
    bf16["weights_proj.weight"] = ckpt["weights_proj.weight"]
    model = _IndexerModel()
    loaded = model.load_weights(bf16.items())
    process_weights_after_loading(model, _MODEL_CONFIG, _DEVICE)
    expected = torch.cat([bf16["wk.weight"], bf16["weights_proj.weight"]])
    torch.testing.assert_close(model.wk_weights_proj.weight, expected, rtol=0, atol=0)
    # nothing left unloaded for the load check
    assert loaded == {n for n, _ in model.named_parameters()}


@pytest.mark.parametrize("_tp", [(1, 2)], indirect=True)
def test_indexer_replicated_under_tp():
    ckpt, expected = _indexer_checkpoint(23)
    model = _indexer_cold_start(ckpt)
    torch.testing.assert_close(model.wk_weights_proj.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("update", ["fp8", "bf16", "left_out"])
def test_indexer_direct_reload_equals_cold_start(monkeypatch, update):
    from vllm.model_executor.model_loader.reload.direct import (
        begin_direct_reload,
        finish_direct_reload,
    )

    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    old, old_expected = _indexer_checkpoint(24, "mx")
    model = _indexer_cold_start(old)
    ptrs = {n: p.data_ptr() for n, p in model.named_parameters()}

    new, expected = _indexer_checkpoint(25, "mx")
    if update == "bf16":
        new = {k: v for k, v in new.items() if "scale" not in k}
        new["wk.weight"] = expected[:_WK].clone()
    elif update == "left_out":
        new, expected = {}, old_expected
    begin_direct_reload(model)
    for item in new.items():
        model.load_weights([item])
    finish_direct_reload(model, _MODEL_CONFIG, _DEVICE)

    torch.testing.assert_close(model.wk_weights_proj.weight, expected, rtol=0, atol=0)
    assert {n: p.data_ptr() for n, p in model.named_parameters()} == ptrs


@pytest.mark.parametrize("layout", ["block", "mx"])
def test_indexer_layerwise_reload_completes_while_loading(layout):
    """The BF16 weights_proj rows load ones into their scale rows, so an FP8
    update completes the layer while loading."""
    model = _indexer_cold_start(_indexer_checkpoint(26, layout)[0])
    new, expected = _indexer_checkpoint(27, layout)
    initialize_layerwise_reload(model)
    for item in new.items():
        model.load_weights([item])
    assert get_layerwise_info(model.wk_weights_proj).load_numel_total is None
    finalize_layerwise_reload(model, _MODEL_CONFIG)
    torch.testing.assert_close(model.wk_weights_proj.weight, expected, rtol=0, atol=0)


def _sharded_rdt_sync(model, ckpt: dict[str, torch.Tensor]) -> None:
    """Bake a sharded RDT plan for ``ckpt``'s names, then replay it as the
    engine does: materialize each planned layer, apply its fills, scatter the
    recorded slices (no loader runs), post-process; finalize at the end."""
    from vllm.distributed.weight_transfer.sharded_rdt_engine import (
        ShardedRDTWeightTransferEngine,
        ShardedRDTWeightTransferInitInfo,
        _apply_fills,
    )
    from vllm.model_executor.model_loader.reload.layerwise import (
        LAYERWISE_INFO,
        _copy_and_restore_kernel_tensors,
    )
    from vllm.model_executor.model_loader.reload.meta import materialize_layer

    engine = object.__new__(ShardedRDTWeightTransferEngine)
    engine.model, engine.device = model, _DEVICE
    engine._name_to_plan, engine._name_meta, engine._live_names = {}, {}, set()
    names = list(ckpt)
    engine._bake(
        ShardedRDTWeightTransferInitInfo(
            names=names,
            dtype_names=[str(ckpt[n].dtype).removeprefix("torch.") for n in names],
            shapes=[list(ckpt[n].shape) for n in names],
        )
    )
    assert engine._live_names == set(names)  # every name baked: no plan failure
    assert engine._live_names <= set(engine._name_to_plan)

    initialize_layerwise_reload(model)
    with torch.no_grad():
        for plan in {id(p): p for p in engine._name_to_plan.values()}.values():
            layer = plan[0].layer
            info = LAYERWISE_INFO[layer]
            materialize_layer(layer, info)
            _apply_fills(layer, engine._layer_fills.get(layer, ()))
            for sc in plan:
                src_name, chain = sc.src
                tensor = ckpt[src_name]
                for op, args, kwargs in chain:
                    tensor = getattr(tensor, op)(*args, **dict(kwargs))
                param = getattr(layer, sc.param_name)
                param.data.as_strided(sc.shape, sc.stride, sc.offset).copy_(tensor)
            layer.quant_method.process_weights_after_loading(layer)
            _copy_and_restore_kernel_tensors(layer, info)
            info.reset()
    finalize_layerwise_reload(model, _MODEL_CONFIG)


def test_sharded_rdt_bf16_update():
    """A BF16 trainer syncing into block-FP8-config layers bakes under sharded
    RDT: the ones the BF16 rows load into their scale rows are recorded as
    fills, so each layer is fully covered without fetching them."""
    model = _cold_start(_checkpoint(30))
    runtime = _expected(_checkpoint(31))
    update = {
        "qkv_a.0.weight": runtime["qkv_a"][: _OUT[0]].clone(),
        "qkv_a.1.weight": runtime["qkv_a"][_OUT[0] :].clone(),
        "o.-.weight": runtime["o"],
    }
    ptrs = {n: p.data_ptr() for n, p in model.named_parameters()}
    _sharded_rdt_sync(model, update)
    _assert_runtime(model, runtime)
    assert {n: p.data_ptr() for n, p in model.named_parameters()} == ptrs
    for layer in (model.qkv_a, model.o):  # the scale rows got the fill
        assert torch.all(layer.weight_scale_inv == 1)


def test_sharded_rdt_bf16_indexer_update():
    model = _indexer_cold_start(_indexer_checkpoint(32)[0])
    _, expected = _indexer_checkpoint(33)
    update = {
        "wk.weight": expected[:_WK].clone(),
        "weights_proj.weight": expected[_WK:].clone(),
    }
    _sharded_rdt_sync(model, update)
    torch.testing.assert_close(model.wk_weights_proj.weight, expected, rtol=0, atol=0)


def test_sharded_rdt_refuses_fp8():
    """FP8 rows would need the loader at replay: refused at bake."""
    model = _cold_start(_checkpoint(34))
    with pytest.raises(NotImplementedError, match="sharded RDT"):
        _sharded_rdt_sync(model, {"o.-.weight": _checkpoint(35)["o.-.weight"]})
