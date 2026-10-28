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
)
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.utils import get_live_param, weights_already_processed

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
        for name, tensor in weights:
            layer, shard, param_name = name.split(".")
            param = getattr(getattr(self, layer), param_name)
            if shard == "-":
                param.weight_loader(param, tensor)
            else:
                param.weight_loader(param, tensor, int(shard))


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
    # the FP8 view spans the bytes of the BF16 weight, one row per BF16 row
    weight = model.qkv_a.weight
    assert weight.dtype == torch.float8_e4m3fn
    assert weight.stride() == (2 * 256, 1)
    assert weight.untyped_storage().nbytes() == sum(_OUT) * 256 * 2


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


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@requires_gpu
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
    finish_direct_reload(model, _MODEL_CONFIG)

    _assert_runtime(model, _expected(new))
    assert {n: p.data_ptr() for n, p in model.named_parameters()} == ptrs


@requires_gpu
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
    finish_direct_reload(model, _MODEL_CONFIG)

    # dequantized rows are runtime bytes: a reload that skips them keeps them
    expected = _expected({**old, **new})
    _assert_runtime(model, expected)


@requires_gpu
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
