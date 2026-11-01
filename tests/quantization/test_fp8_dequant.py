# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-FP8 checkpoint weights of layers that run in BF16 (`Fp8DequantConfig`).

The weight and its scale load independently (any order, any call, no pairing in
the model's loader); BF16 rows load as is; post-load processing dequantizes in
the weight's own storage, so cold start, direct reload and layerwise reload all
end with the same bytes at the startup address."""

import pytest
import torch

from vllm.config import ModelConfig
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.fp8_dequant import (
    Fp8DequantConfig,
    load_unneeded_scale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.reload.direct import (
    begin_direct_reload,
    finish_direct_reload,
)
from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import (
    initialize_single_dummy_weight,
)
from vllm.model_executor.utils import get_live_param, weights_already_processed
from vllm.utils.math_utils import cdiv

pytestmark = pytest.mark.usefixtures("dist_init")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# q_a / kv_a-like shards: the first spans several dequantization chunks, the
# last ends in a partial block
_OUT = [1280, 192]


@pytest.fixture(scope="module")
def model_config() -> ModelConfig:
    return ModelConfig("facebook/opt-125m", dtype="bfloat16", skip_tokenizer_init=True)


def _block_fp8(rows: int, cols: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=gen).to(torch.float8_e4m3fn)
    scale = torch.rand(cdiv(rows, 128), cdiv(cols, 128), generator=gen) + 0.5
    return weight.to(_DEVICE), scale.to(_DEVICE)


def _dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, cols = weight.shape
    s = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
    return (weight.float() * s).to(torch.bfloat16)


class _Model(torch.nn.Module):
    """A fused (q_a / kv_a-like) and a row-parallel block-FP8 projection, loaded
    the way the model loaders do: one tensor at a time, no pairing."""

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

    def load_weights(self, weights) -> set[str]:
        loaded: set[str] = set()
        for name, tensor in weights:
            layer, shard, param_name = name.split(".")
            param = getattr(getattr(self, layer), param_name)
            shard_id = None if shard == "-" else int(shard)
            param.weight_loader(
                param, tensor, *(() if shard_id is None else (shard_id,))
            )
            loaded.add(f"{layer}.{param_name}")
            load_unneeded_scale(
                f"{layer}.{param_name}", param, tensor, loaded, shard_id
            )
        return loaded


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


def _bf16(ckpt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The same weights as a BF16 trainer sends them: no scales."""
    return {
        name: _dequant(w, ckpt[name + "_scale_inv"]) if w.dtype != torch.bfloat16 else w
        for name, w in ckpt.items()
        if name.endswith(".weight")
    }


def _expected(ckpt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    runtime = _bf16(ckpt)
    qkv_a = torch.cat([runtime[f"qkv_a.{shard}.weight"] for shard in range(len(_OUT))])
    return {"qkv_a": qkv_a, "o": runtime["o.-.weight"]}


def _cold_start(ckpt: dict[str, torch.Tensor], model_config: ModelConfig) -> _Model:
    model = _Model()
    record_metadata_for_reloading(model)
    model.load_weights(reversed(list(ckpt.items())))  # scales before weights
    process_weights_after_loading(model, model_config, _DEVICE)
    return model


def _assert_runtime(model: _Model, expected: dict[str, torch.Tensor]) -> None:
    for name, weight in expected.items():
        layer = getattr(model, name)
        assert layer.weight.dtype == torch.bfloat16
        assert layer.weight.is_contiguous()
        torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)
        assert get_live_param(layer, "weight_scale_inv") is None  # retired


def _ptrs(model: torch.nn.Module) -> dict[str, int]:
    return {n: p.data_ptr() for n, p in model.named_parameters()}


@pytest.mark.parametrize("bf16_shard", [None, 0])
def test_cold_start_dequantizes_in_place(model_config, bf16_shard):
    """FP8 rows load into the first half of their BF16 rows' bytes (no extra
    memory) and are dequantized at the same address; BF16 rows load as is."""
    ckpt = _checkpoint(0, bf16_shard)
    model = _Model()
    storage = model.qkv_a.weight.untyped_storage()
    assert storage.nbytes() == sum(_OUT) * 256 * 2  # the BF16 weight's bytes
    ptrs = _ptrs(model)
    model.load_weights(reversed(list(ckpt.items())))
    process_weights_after_loading(model, model_config, _DEVICE)
    _assert_runtime(model, _expected(ckpt))
    assert _ptrs(model) == ptrs


@pytest.mark.parametrize("scales_first", [True, False])
def test_weight_and_scale_in_separate_calls(model_config, scales_first):
    ckpt = _checkpoint(1)
    model = _Model()
    items = list(ckpt.items())
    for item in reversed(items) if scales_first else items:
        model.load_weights([item])
    process_weights_after_loading(model, model_config, _DEVICE)
    _assert_runtime(model, _expected(ckpt))


def test_missing_scale_is_rejected(model_config):
    """FP8 rows whose scale never arrived are not dequantized with whatever
    the scale holds."""
    ckpt = _checkpoint(17)
    del ckpt["qkv_a.1.weight_scale_inv"]
    model = _Model()
    model.load_weights(ckpt.items())
    with pytest.raises(ValueError, match="without their weight_scale_inv"):
        process_weights_after_loading(model, model_config, _DEVICE)


def test_bf16_checkpoint(model_config):
    """A BF16 checkpoint needs no scales: every parameter counts as loaded."""
    bf16 = _bf16(_checkpoint(3))
    model = _Model()
    loaded = model.load_weights(bf16.items())
    assert loaded == {n for n, _ in model.named_parameters()}
    process_weights_after_loading(model, model_config, _DEVICE)
    _assert_runtime(model, _expected(bf16))


def test_dummy_weights_are_runtime_values(model_config):
    """--load-format dummy fills the BF16 weight; no row holds FP8 bytes."""
    model = _Model()
    model.o.weight.view(torch.int16).fill_(0x7FC0)  # BF16 NaN
    for module in model.modules():
        for name, param in module._parameters.items():
            if param is not None:
                initialize_single_dummy_weight(param, name=name)
    process_weights_after_loading(model, model_config, _DEVICE)
    for layer in (model.qkv_a, model.o):
        assert layer.weight.isfinite().all()
        assert layer.weight.abs().max() <= 1e-3


def test_weight_copied_before_processing(model_config):
    """CPU offloading moves the loaded weight (``.to(device)``) before
    post-load processing; the copy keeps its FP8 rows."""
    ckpt = _checkpoint(16)
    model = _Model()
    model.load_weights(ckpt.items())
    for layer in (model.qkv_a, model.o):
        layer.weight.data = layer.weight.data.to("cpu").clone().to(_DEVICE)
    process_weights_after_loading(model, model_config, _DEVICE)
    _assert_runtime(model, _expected(ckpt))


def test_bf16_load_weights_outside_a_reload(model_config):
    """A bare ``model.load_weights`` of BF16 tensors leaves a runnable layer."""
    model = _cold_start(_checkpoint(12), model_config)
    new = _expected(_checkpoint(13))
    model.load_weights([("o.-.weight", new["o"])])
    x = torch.randn(2, 256, dtype=torch.bfloat16, device=_DEVICE)
    torch.testing.assert_close(model.o(x)[0], x @ new["o"].t())


def test_pre_processed_weights_are_kept(model_config):
    """The weight cache hands over plain parameters holding runtime bytes."""
    runtime = _expected(_checkpoint(11))
    model = _Model()
    for name, weight in runtime.items():
        layer = getattr(model, name)
        layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    with weights_already_processed():
        process_weights_after_loading(model, model_config, _DEVICE)
    _assert_runtime(model, runtime)


@pytest.mark.parametrize("update", ["fp8", "mixed", "bf16", "left_out"])
def test_direct_reload_equals_cold_start(monkeypatch, model_config, update):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    old = _checkpoint(5)
    model = _cold_start(old, model_config)
    ptrs = _ptrs(model)
    new = {
        "fp8": _checkpoint(6),
        "mixed": _checkpoint(6, bf16_shard=1),
        "bf16": _bf16(_checkpoint(6)),
        # dequantized rows are runtime bytes: a reload that skips o keeps it
        "left_out": {k: v for k, v in _checkpoint(6).items() if k.startswith("qkv_a")},
    }[update]
    begin_direct_reload(model)
    for item in new.items():
        model.load_weights([item])
    finish_direct_reload(model, model_config, _DEVICE)
    _assert_runtime(model, _expected({**old, **new}))
    assert _ptrs(model) == ptrs


def test_direct_reload_rejects_fp8_rows_without_scale(monkeypatch, model_config):
    """FP8 rows a reload sends without their scale are not dequantized with
    the previous load's scale."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    model = _cold_start(_checkpoint(7), model_config)
    new = _checkpoint(8)
    del new["qkv_a.1.weight_scale_inv"]
    begin_direct_reload(model)
    model.load_weights(new.items())
    with pytest.raises(ValueError, match="without their weight_scale_inv"):
        finish_direct_reload(model, model_config, _DEVICE)


@pytest.mark.parametrize("update", ["fp8", "mixed", "bf16"])
def test_layerwise_reload_equals_cold_start(model_config, update):
    """Every element is loaded, BF16 rows' scales included, so each layer is
    processed as its last tensor arrives rather than held for finalize."""
    model = _cold_start(_checkpoint(9), model_config)
    ptrs = _ptrs(model)
    new = {
        "fp8": _checkpoint(10),
        "mixed": _checkpoint(10, bf16_shard=0),
        "bf16": _bf16(_checkpoint(10)),
    }[update]
    initialize_layerwise_reload(model)
    for item in new.items():
        model.load_weights([item])
    for layer in (model.qkv_a, model.o):
        assert get_layerwise_info(layer).load_numel_total is None  # processed
    finalize_layerwise_reload(model, model_config)
    _assert_runtime(model, _expected(new))
    assert _ptrs(model) == ptrs


@pytest.mark.parametrize("bf16", [False, True])
def test_replicated_layer(model_config, bf16):
    """A ReplicatedLinear (GLM5-next kv_a_proj_with_mqa without q_lora_rank)
    loads FP8 through the FP8 view too."""
    with _DEVICE:
        layer = ReplicatedLinear(
            256,
            192,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=Fp8DequantConfig([128, 128]),
        )
    weight, scale = _block_fp8(192, 256, 40)
    expected = _dequant(weight, scale)
    if bf16:
        layer.weight.weight_loader(layer.weight, expected)
        load_unneeded_scale("weight", layer.weight, expected, set())
    else:
        layer.weight_scale_inv.weight_loader(layer.weight_scale_inv, scale)
        layer.weight.weight_loader(layer.weight, weight)
    layer.quant_method.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_column_parallel_tp_slice(model_config, tp_rank, bf16):
    """Under TP 2, a rank loads its rows of a column-parallel weight (q_b_proj)
    from the full FP8 or BF16 checkpoint tensor, with the matching scale rows."""
    with _DEVICE:
        layer = ColumnParallelLinear(
            256,
            512,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=Fp8DequantConfig([128, 128]),
            tp_rank=tp_rank,
            tp_size=2,
        )
    weight, scale = _block_fp8(512, 256, 3)
    full = _dequant(weight, scale)
    ckpt = {"weight": full} if bf16 else {"weight_scale_inv": scale, "weight": weight}
    loaded: set[str] = set()
    for name, tensor in ckpt.items():
        param = getattr(layer, name)
        param.weight_loader(param, tensor)
        loaded.add(name)
        load_unneeded_scale(name, param, tensor, loaded)
    assert loaded == {"weight", "weight_scale_inv"}
    process_weights_after_loading(layer, model_config, _DEVICE)
    rows = slice(256 * tp_rank, 256 * (tp_rank + 1))
    torch.testing.assert_close(layer.weight, full[rows], rtol=0, atol=0)


# Sparse-attention indexer: `wk` (128 rows) and `weights_proj` (64 rows) fused
# into one replicated `wk_weights_proj`, whose FP8 checkpoint scale layout
# varies between models (`Fp8DequantConfig(None)`).
_WK, _WP, _HIDDEN = 128, 64, 512


class _IndexerModel(torch.nn.Module):
    """The indexer part of the DeepSeek-V3.2-style loaders."""

    def __init__(self, hidden: int = _HIDDEN):
        super().__init__()
        with _DEVICE:
            self.wk_weights_proj = MergedColumnParallelLinear(
                hidden,
                [_WK, _WP],
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=Fp8DequantConfig(None),
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
            loaded.add(name)
            load_unneeded_scale(name, param, tensor, loaded, shard)
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
    if layout == "mx":  # E8M0 exponents of 2^-2 .. 2^1
        scale = torch.randint(125, 129, shapes[layout], generator=gen)
        return scale.to(torch.uint8).to(_DEVICE)
    return (torch.rand(shapes[layout], generator=gen) + 0.5).to(_DEVICE)


def _indexer_checkpoint(seed: int, layout: str = "block", fp8_wp: bool = False):
    """A checkpoint and the BF16 weight the paired loaders made of it."""
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
        ref = scale.view(torch.float8_e8m0fnu).float() if layout == "mx" else scale
        ref = ref.reshape(ref.shape[0] if ref.ndim else 1, -1)
        group = GroupShape(rows // ref.shape[0], _HIDDEN // ref.shape[1])
        expected.append(scaled_dequantize(weight, ref, group, torch.bfloat16))
    return ckpt, torch.cat(expected)


def _indexer_cold_start(ckpt, model_config: ModelConfig) -> _IndexerModel:
    model = _IndexerModel()
    record_metadata_for_reloading(model)
    for item in reversed(list(ckpt.items())):  # scales first, one per call
        model.load_weights([item])
    process_weights_after_loading(model, model_config, _DEVICE)
    return model


@pytest.mark.parametrize("layout", ["block", "channel", "channel_2d", "tensor", "mx"])
@pytest.mark.parametrize("scales_first", [True, False])
def test_indexer_wk_any_scale_layout(model_config, layout, scales_first):
    """Bit-equal to main's paired `scaled_dequantize`, for every scale layout
    and with weight and scale in separate calls."""
    ckpt, expected = _indexer_checkpoint(20, layout)
    model = _IndexerModel()
    items = list(ckpt.items())
    for item in reversed(items) if scales_first else items:
        model.load_weights([item])
    process_weights_after_loading(model, model_config, _DEVICE)
    torch.testing.assert_close(model.wk_weights_proj.weight, expected, rtol=0, atol=0)


def test_indexer_fp8_weights_proj(model_config):
    # dots3 stores weights_proj as FP8 too
    ckpt, expected = _indexer_checkpoint(21, fp8_wp=True)
    model = _indexer_cold_start(ckpt, model_config)
    torch.testing.assert_close(model.wk_weights_proj.weight, expected, rtol=0, atol=0)


def test_indexer_bf16_rows_of_odd_width(model_config):
    """BF16 rows load ones into their scale rows for any width."""
    model = _IndexerModel(hidden=200)
    rows = torch.randn(_WK + _WP, 200).to(torch.bfloat16).to(_DEVICE)
    loaded = model.load_weights(
        [("wk.weight", rows[:_WK]), ("weights_proj.weight", rows[_WK:])]
    )
    assert loaded == {n for n, _ in model.named_parameters()}
    assert torch.all(model.wk_weights_proj.weight_scale_inv == 1)
    process_weights_after_loading(model, model_config, _DEVICE)
    torch.testing.assert_close(model.wk_weights_proj.weight, rows, rtol=0, atol=0)


@pytest.mark.parametrize("update", ["fp8", "bf16", "left_out"])
def test_indexer_direct_reload_equals_cold_start(monkeypatch, model_config, update):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    old, old_expected = _indexer_checkpoint(24, "mx")
    model = _indexer_cold_start(old, model_config)
    ptrs = _ptrs(model)
    new, expected = _indexer_checkpoint(25, "mx")
    if update == "bf16":
        new = {
            "wk.weight": expected[:_WK].clone(),
            "weights_proj.weight": expected[_WK:],
        }
    elif update == "left_out":
        new, expected = {}, old_expected
    begin_direct_reload(model)
    for item in new.items():
        model.load_weights([item])
    finish_direct_reload(model, model_config, _DEVICE)
    torch.testing.assert_close(model.wk_weights_proj.weight, expected, rtol=0, atol=0)
    assert _ptrs(model) == ptrs


@pytest.mark.parametrize("layout", ["block", "mx"])
def test_indexer_layerwise_reload_equals_cold_start(model_config, layout):
    """The BF16 weights_proj rows load ones into their scale rows, so an FP8
    update completes the layer while loading."""
    model = _indexer_cold_start(_indexer_checkpoint(26, layout)[0], model_config)
    new, expected = _indexer_checkpoint(27, layout)
    initialize_layerwise_reload(model)
    for item in new.items():
        model.load_weights([item])
    assert get_layerwise_info(model.wk_weights_proj).load_numel_total is None
    finalize_layerwise_reload(model, model_config)
    torch.testing.assert_close(model.wk_weights_proj.weight, expected, rtol=0, atol=0)
