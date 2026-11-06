# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint weights that are stored in FP8 but run in BF16.

- `Fp8DequantLinearMethod` (GLM5-next MLA projections): the checkpoint's own
  block-FP8 ``weight`` and ``weight_scale_inv``, dequantized after loading.
- The sparse-attention indexer: ``wk`` in the checkpoint's layout (block,
  per-channel or MX scales), ``weights_proj`` in BF16, fused into one BF16
  weight after loading.

Each checkpoint tensor loads on its own (any order, one per ``load_weights``
call), and an update may send the weight whole in BF16 instead, without its
scale. Direct and layerwise reloads end with the bytes of a cold start, at the
cold-start addresses."""

import pytest
import torch

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    dequantized_fp8_config,
)
from vllm.model_executor.model_loader.reload.direct import (
    begin_direct_reload,
    finish_direct_reload,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.utils import get_live_param
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FP8 linear kernels need a GPU"
)

_DEVICE = torch.device("cuda")
# q_a / kv_a-like shards: the first spans several dequantization chunks, the
# last ends in a partial block
_OUT = [1280, 192]


@pytest.fixture
def model_config(dist_init):
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config

    config = ModelConfig(
        "facebook/opt-125m", dtype="bfloat16", skip_tokenizer_init=True
    )
    with set_current_vllm_config(VllmConfig(model_config=config)):
        yield config


def _block_fp8(rows: int, cols: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=gen).to(torch.float8_e4m3fn)
    scale = torch.rand(cdiv(rows, 128), cdiv(cols, 128), generator=gen) + 0.5
    return weight.to(_DEVICE), scale.to(_DEVICE)


def _dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, cols = weight.shape
    s = scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
    return (weight.float() * s).to(torch.bfloat16)


def _ptrs(model: torch.nn.Module) -> dict[str, int]:
    return {n: p.data_ptr() for n, p in model.named_parameters()}


def _reload(model, ckpt, mode, model_config, monkeypatch, reverse=False):
    """Send every tensor in its own ``load_weights`` call, as RL updates do
    (``reverse``: weights before their scales)."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    if mode == "direct":
        begin_direct_reload(model)
    else:
        initialize_layerwise_reload(model)
    items = list(ckpt.items())
    for item in reversed(items) if reverse else items:
        model.load_weights([item])
    if mode == "direct":
        finish_direct_reload(model, model_config, _DEVICE)
    else:
        finalize_layerwise_reload(model, model_config)


class _MlaModel(torch.nn.Module):
    """A fused (q_a / kv_a-like) and a row-parallel projection that a
    block-FP8 checkpoint stores in FP8 and the model runs in BF16."""

    def __init__(self):
        super().__init__()
        config = dequantized_fp8_config(Fp8Config(weight_block_size=[128, 128]))
        with _DEVICE:
            self.qkv_a = MergedColumnParallelLinear(
                256, _OUT, bias=False, params_dtype=torch.bfloat16, quant_config=config
            )
            self.o = RowParallelLinear(
                256, 128, bias=False, params_dtype=torch.bfloat16, quant_config=config
            )

    def load_weights(self, weights) -> None:
        for name, tensor in weights:
            layer, shard, param_name = name.split(".")
            param = getattr(getattr(self, layer), param_name)
            shard_ids = () if shard == "-" else (int(shard),)
            param.weight_loader(param, tensor, *shard_ids)


def _mla_checkpoint(seed: int) -> dict[str, torch.Tensor]:
    """``layer.shard.param`` -> tensor, scales first."""
    ckpt = {}
    for shard, rows in enumerate(_OUT):
        weight, ckpt[f"qkv_a.{shard}.weight_scale_inv"] = _block_fp8(
            rows, 256, seed + shard
        )
        ckpt[f"qkv_a.{shard}.weight"] = weight
    ckpt["o.-.weight_scale_inv"], ckpt["o.-.weight"] = reversed(
        _block_fp8(128, 256, seed + 9)
    )
    return ckpt


def _mla_expected(ckpt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    def dq(prefix: str) -> torch.Tensor:
        return _dequant(ckpt[f"{prefix}.weight"], ckpt[f"{prefix}.weight_scale_inv"])

    qkv_a = torch.cat([dq(f"qkv_a.{shard}") for shard in range(len(_OUT))])
    return {"qkv_a": qkv_a, "o": dq("o.-")}


def _as_bf16(ckpt: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """What an RL trainer sends instead: each weight dequantized, no scales."""
    return {
        name: _dequant(tensor, ckpt[f"{name}_scale_inv"])
        for name, tensor in ckpt.items()
        if name.endswith(".weight")
    }


def _mla_cold_start(ckpt, model_config, scales_first=True) -> _MlaModel:
    model = _MlaModel()
    ptrs = _ptrs(model)
    record_metadata_for_reloading(model)
    items = list(ckpt.items())
    for item in items if scales_first else reversed(items):
        model.load_weights([item])
    process_weights_after_loading(model, model_config, _DEVICE)
    # dequantized in the loaded (BF16-sized) storage: no second allocation
    assert _ptrs(model) == ptrs
    return model


def _assert_mla(model: _MlaModel, expected: dict[str, torch.Tensor]) -> None:
    for name, weight in expected.items():
        layer = getattr(model, name)
        assert layer.weight.dtype == torch.bfloat16
        torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)
        assert get_live_param(layer, "weight_scale_inv") is None  # retired
    x = torch.randn(3, 256, dtype=torch.bfloat16, device=_DEVICE)
    torch.testing.assert_close(model.o(x)[0], x @ expected["o"].t())


@pytest.mark.parametrize("scales_first", [True, False])
def test_mla_cold_start_separate_calls(model_config, scales_first):
    ckpt = _mla_checkpoint(0)
    model = _mla_cold_start(ckpt, model_config, scales_first)
    _assert_mla(model, _mla_expected(ckpt))


@pytest.mark.parametrize("mode", ["direct", "layerwise"])
def test_mla_reload_equals_cold_start(monkeypatch, model_config, mode):
    """Direct reload views the BF16 storage back to the FP8 checkpoint layout
    (a dtype and size change); layerwise reload restages the FP8 layer. Both
    dequantize the new tensors at the cold-start addresses. An update may
    send the weights in BF16 instead, without scales."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    model = _mla_cold_start(_mla_checkpoint(1), model_config)
    ptrs = _ptrs(model)
    for seed, bf16 in ((2, False), (3, True), (4, False)):
        new = _mla_checkpoint(seed)
        expected = _mla_expected(new)
        _reload(model, _as_bf16(new) if bf16 else new, mode, model_config, monkeypatch)
        _assert_mla(model, expected)
        assert _ptrs(model) == ptrs


# Sparse-attention indexer: `wk` (128 rows) and `weights_proj` (64 rows).
_WK, _WP, _HIDDEN = 128, 64, 512
# `wk` layouts of published checkpoints
_LAYOUTS = ["bf16", "block", "block_e8m0", "channel", "mxfp8"]


def _quant_config(layout: str):
    if layout.startswith("block"):  # DeepSeek-V3.2, GLM-5.x FP8; e8m0: Hy4 MTP
        config = Fp8Config(weight_block_size=[128, 128])
        config.is_scale_e8m0 = layout == "block_e8m0"  # type: ignore[attr-defined]
        return config
    if layout == "channel":  # amd/GLM-5.2-Quark-MXFP4-AttnFP8
        from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig

        fp8 = {"dtype": "fp8_e4m3", "qscheme": "per_channel"}
        tensors = {"weight": fp8 | {"is_dynamic": False}}
        tensors["input_tensors"] = fp8 | {"is_dynamic": True}
        return QuarkConfig(
            {
                "global_quant_config": tensors,
                "layer_type_quant_config": {},
                "exclude": [],
            }
        )
    if layout == "mxfp8":  # tencent/Hy4-preview-FP8 (ModelOpt MXFP8)
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptMxFp8Config,
        )

        return ModelOptMxFp8Config(True, None, [])
    if layout == "online":  # a BF16 checkpoint quantized on load
        from vllm.config.quantization import QuantizationConfigArgs
        from vllm.model_executor.layers.quantization.online.base import (
            OnlineQuantizationConfig,
        )

        return OnlineQuantizationConfig(QuantizationConfigArgs(linear="fp8_per_tensor"))
    return None


class _IndexerModel(torch.nn.Module):
    def __init__(self, layout: str):
        super().__init__()
        from vllm.model_executor.layers.sparse_attn_indexer import (
            fuse_indexer_wk_weights_proj,
            init_indexer_wk_weights_proj,
        )

        self.indexer = torch.nn.Module()
        self.indexer.head_dim, self.indexer.n_head = _WK, _WP
        self.indexer.process_weights_after_loading = lambda: (
            fuse_indexer_wk_weights_proj(self.indexer)
        )
        # the model's dtype, as model construction sets it
        with set_default_torch_dtype(torch.bfloat16), _DEVICE:
            init_indexer_wk_weights_proj(
                self.indexer, _HIDDEN, _quant_config(layout), "layers.0.indexer"
            )

    def load_weights(self, weights) -> None:
        params = dict(self.named_parameters())
        for name, tensor in weights:
            param = params[f"indexer.{name}"]
            param.weight_loader(param, tensor)


def _wk_checkpoint(layout: str, gen: torch.Generator):
    """``wk`` checkpoint tensors (scale first) and their dequantization."""
    if layout == "block":
        weight, scale = _block_fp8(_WK, _HIDDEN, int(gen.initial_seed()))
        return {"wk.weight_scale_inv": scale, "wk.weight": weight}, _dequant(
            weight, scale
        )
    weight = torch.randn(_WK, _HIDDEN, generator=gen)
    if layout == "bf16":
        weight = weight.to(torch.bfloat16).to(_DEVICE)
        return {"wk.weight": weight}, weight
    weight = weight.to(torch.float8_e4m3fn).to(_DEVICE)
    name = "wk.weight_scale_inv" if layout == "block_e8m0" else "wk.weight_scale"
    if layout == "channel":
        scale = torch.rand(_WK, generator=gen) + 0.5
        full = scale[:, None]
    else:  # E8M0 exponents 2^-2 .. 2^1, per 128x128 block or per 1x32 group
        rows, cols = (
            (1, _HIDDEN // 128) if layout == "block_e8m0" else (_WK, _HIDDEN // 32)
        )
        scale = torch.randint(125, 129, (rows, cols), generator=gen).to(torch.uint8)
        full = scale.view(torch.float8_e8m0fnu).float()
        full = full.repeat_interleave(_WK // rows, 0)
        full = full.repeat_interleave(_HIDDEN // cols, 1)
        if layout == "block_e8m0":  # the scale parameter's dtype
            scale = scale.view(torch.float8_e8m0fnu)
    expected = (weight.float() * full.to(_DEVICE)).to(torch.bfloat16)
    return {name: scale.to(_DEVICE), "wk.weight": weight}, expected


def _indexer_checkpoint(seed: int, layout: str):
    """Checkpoint tensors (scales first) and the fused BF16 weight."""
    gen = torch.Generator().manual_seed(seed)
    ckpt, wk = _wk_checkpoint(layout, gen)
    wp = torch.randn(_WP, _HIDDEN, generator=gen).to(torch.bfloat16).to(_DEVICE)
    ckpt["weights_proj.weight"] = wp
    return ckpt, torch.cat([wk, wp])


def _assert_parts_in_fused(indexer: torch.nn.Module) -> None:
    """Wk and weights_proj are views of the fused weight's rows, so filling
    it copies nothing."""
    from vllm.model_executor.layers.sparse_attn_indexer import (
        fuse_indexer_wk_weights_proj,
    )
    from vllm.model_executor.model_loader.reload.meta import CopyCounter

    fused = indexer.wk_weights_proj.weight
    assert indexer.wk.weight.data_ptr() == fused.data_ptr()
    assert indexer.weights_proj.weight.data_ptr() == fused[_WK:].data_ptr()
    with CopyCounter() as counter:
        fuse_indexer_wk_weights_proj(indexer)
    assert counter.copied_numel == 0


def _indexer_cold_start(ckpt, layout, model_config, scales_first=True):
    model = _IndexerModel(layout)
    record_metadata_for_reloading(model)
    items = list(ckpt.items())
    for item in items if scales_first else reversed(items):
        model.load_weights([item])
    process_weights_after_loading(model, model_config, _DEVICE)
    return model


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("scales_first", [True, False])
def test_indexer_fused_weight(model_config, layout, scales_first):
    """Wk loads in each checkpoint layout with its scale in a separate call,
    and the fused weight equals the exact dequantization of the checkpoint."""
    ckpt, expected = _indexer_checkpoint(10, layout)
    model = _indexer_cold_start(ckpt, layout, model_config, scales_first)
    fused = model.indexer.wk_weights_proj
    torch.testing.assert_close(fused.weight, expected, rtol=0, atol=0)
    assert "weight" not in fused._parameters  # derived, not loaded
    _assert_parts_in_fused(model.indexer)
    x = torch.randn(3, _HIDDEN, dtype=torch.bfloat16, device=_DEVICE)
    torch.testing.assert_close(fused(x)[0], x @ expected.t())


@pytest.mark.parametrize("layout", _LAYOUTS)
@pytest.mark.parametrize("cold", ["fp8", "bf16"])
@pytest.mark.parametrize("mode", ["direct", "layerwise"])
def test_indexer_reload_fp8_or_bf16(monkeypatch, model_config, layout, cold, mode):
    """Each load sends ``wk`` whole, either in the checkpoint's FP8 layout or
    in BF16 without a scale (as RL trainers do). Every reload ends with the
    dequantized weight in the fused rows, at the cold-start addresses."""
    monkeypatch.setenv("VLLM_RELOAD_MODE", mode)
    sent = [layout, "bf16", layout] if cold == "fp8" else ["bf16", layout, "bf16"]
    old, _ = _indexer_checkpoint(11, sent[0])
    model = _indexer_cold_start(old, layout, model_config)
    fused_ptr = model.indexer.wk_weights_proj.weight.data_ptr()
    ptrs = _ptrs(model)
    for seed, fmt in enumerate(sent[1:] * 2, start=12):
        new, expected = _indexer_checkpoint(seed, fmt)
        _reload(model, new, mode, model_config, monkeypatch, reverse=seed % 2 == 1)
        fused = model.indexer.wk_weights_proj.weight
        torch.testing.assert_close(fused, expected, rtol=0, atol=0)
        assert fused.data_ptr() == fused_ptr
        assert _ptrs(model) == ptrs
        _assert_parts_in_fused(model.indexer)


def _glm5_mla_params() -> dict:
    """Params of a GLM5-next MLA layer whose projections load a block-FP8
    checkpoint (see `dequantized_fp8_config`): only the names matter here."""
    prefix = "layers.3.self_attn"
    return {
        f"{prefix}.{target}.weight_scale_inv": None
        for target in ("fused_qkv_a_proj", "q_b_proj", "o_proj")
    }


def test_glm5next_attn_loader_accepts_quark_weight_scale():
    # Quark names the block scale ``weight_scale``; the projection's scale
    # parameter is ``weight_scale_inv``.
    from transformers import Glm5NextTextConfig

    from vllm.models.glm5next.common.model import _mla_proj_checkpoint_tensor

    config = Glm5NextTextConfig(kv_lora_rank=512)
    scale = torch.rand(4, 2)
    name, tensor = _mla_proj_checkpoint_tensor(
        "layers.3.self_attn.kv_a_proj_with_mqa.weight_scale",
        scale,
        _glm5_mla_params(),
        config,
        0,
    )
    assert name == "layers.3.self_attn.kv_a_proj_with_mqa.weight_scale_inv"
    assert tensor is scale
    # a projection the model does not load from block-FP8 keeps its name
    name, _ = _mla_proj_checkpoint_tensor(
        "layers.0.self_attn.o_proj.weight_scale", scale, {}, config, 0
    )
    assert name == "layers.0.self_attn.o_proj.weight_scale"


def test_glm5next_nope_kv_a_padding_block_scale():
    # NoPE checkpoints omit kv_a's RoPE rows: the weight gets zero rows and the
    # scale its extra blocks, each on its own (no weight/scale pairing).
    from transformers import Glm5NextTextConfig

    from vllm.models.glm5next.common.model import _mla_proj_checkpoint_tensor

    config = Glm5NextTextConfig(kv_lora_rank=512)
    prefix = "layers.3.self_attn.kv_a_proj_with_mqa"
    weight = torch.randn(512, 256).to(torch.float8_e4m3fn)
    _, padded = _mla_proj_checkpoint_tensor(
        f"{prefix}.weight", weight, _glm5_mla_params(), config, 64
    )
    assert padded.shape == (576, 256) and padded.dtype == torch.float8_e4m3fn
    assert torch.equal(padded[:512].view(torch.uint8), weight.view(torch.uint8))
    assert not padded[512:].float().any()
    scale = torch.rand(4, 2)
    _, padded = _mla_proj_checkpoint_tensor(
        f"{prefix}.weight_scale_inv", scale, _glm5_mla_params(), config, 64
    )
    assert padded.shape == (5, 2)  # ceil(576 / 128) blocks
    assert torch.equal(padded[:4], scale)


def test_indexer_wk_stays_bf16_under_online_quantization(model_config):
    """The fused GEMM runs in BF16, so online quantization must not round wk."""
    ckpt, expected = _indexer_checkpoint(10, "bf16")
    model = _indexer_cold_start(ckpt, "online", model_config)
    fused = model.indexer.wk_weights_proj.weight
    torch.testing.assert_close(fused, expected, rtol=0, atol=0)


def test_indexer_in_place_reload_refreshes_fused_weight(model_config):
    """An in-place reload of the runtime weights (no post-load processing)
    refills the derived fused weight through `refresh_derived_weights`."""
    from vllm.model_executor.model_loader.weight_utils import refresh_derived_weights

    model = _indexer_cold_start(
        _indexer_checkpoint(10, "block")[0], "block", model_config
    )
    _, expected = _indexer_checkpoint(11, "bf16")
    model.indexer.wk.weight.data.copy_(expected[:_WK])
    model.indexer.weights_proj.weight.data.copy_(expected[_WK:])
    refresh_derived_weights(model)
    torch.testing.assert_close(
        model.indexer.wk_weights_proj.weight, expected, rtol=0, atol=0
    )


def test_layerwise_metadata_does_not_keep_layers_alive(model_config):
    """`wk`'s weight loader must not pin the layer through the layerwise
    reload metadata (a WeakKeyDictionary keyed by the layer)."""
    import gc
    import weakref

    model = _IndexerModel("block")
    record_metadata_for_reloading(model)
    wk = weakref.ref(model.indexer.wk)
    del model
    gc.collect()
    assert wk() is None
