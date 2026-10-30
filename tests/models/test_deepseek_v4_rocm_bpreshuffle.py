# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4's B-preshuffled FP8 linears on ROCm survive a weight reload.

The attention's fused_wqa_wkv / wo_b and the standalone shared experts'
gate_up_proj run aiter's B-preshuffle GEMM. Their linear kernel shuffles the
weight while processing it after loading, so a reload (an RL weight sync) with
new weights shuffles those again, in the storage the CUDA graphs captured.
"""

import importlib.util

import pytest
import torch
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterFp8BlockScaledMMKernel,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.utils import weights_already_processed
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype


def _on_gfx942_or_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942, on_gfx950

    return on_gfx942() or on_gfx950()


pytestmark = pytest.mark.skipif(
    not (_on_gfx942_or_gfx950() and importlib.util.find_spec("aiter") is not None),
    reason="aiter's B-preshuffle GEMM runs on gfx942 / gfx950",
)

N, K, BLOCK = 512, 1024, 128
# DeepSeek-V4-Flash stores E8M0 linear scales, -Flash-Base float32 ones.
CHECKPOINTS = {
    "e8m0-scales": "deepseek-ai/DeepSeek-V4-Flash",
    "fp32-scales": "deepseek-ai/DeepSeek-V4-Flash-Base",
}


@pytest.fixture(params=list(CHECKPOINTS.values()), ids=list(CHECKPOINTS))
def vllm_config(request, monkeypatch, dist_init):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR", "1")
    rocm_aiter_ops.refresh_env_variables()
    config = EngineArgs(
        model=request.param, load_format="dummy", max_model_len=4096
    ).create_engine_config()
    with set_current_vllm_config(config):
        yield config
    rocm_aiter_ops.refresh_env_variables()


def _linear(vllm_config: VllmConfig, bpreshuffle: bool) -> ReplicatedLinear:
    with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        layer = ReplicatedLinear(
            K, N, bias=False, quant_config=vllm_config.quant_config, return_bias=False
        )
    assert isinstance(layer.quant_method.fp8_linear, AiterFp8BlockScaledMMKernel)
    layer.use_aiter_bpreshuffle = bpreshuffle
    return layer


def _checkpoint(vllm_config: VllmConfig, seed: int):
    """An e4m3 weight and its 128 x 128 block scales in the checkpoint's scale
    dtype, all differing by seed."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(N, K, generator=gen, device="cuda").clamp(-6, 6)
    exponent = torch.randint(
        117, 122, (N // BLOCK, K // BLOCK), generator=gen, device="cuda"
    )
    if vllm_config.quant_config.is_scale_e8m0:
        scale = exponent.to(torch.uint8).view(torch.float8_e8m0fnu)
    else:
        scale = torch.exp2(exponent.float() - 127)
    return weight.to(torch.float8_e4m3fn), scale


def _load(layer: nn.Module, checkpoint: tuple[torch.Tensor, torch.Tensor]) -> None:
    for name, tensor in zip(("weight", "weight_scale_inv"), checkpoint):
        param = getattr(layer, name)
        param.weight_loader(param, tensor)


def _assert_matches(
    out: torch.Tensor,
    x: torch.Tensor,
    checkpoint,
    quantize=rocm_aiter_ops.group_fp8_quant,
) -> None:
    """``out`` is the GEMM of the checkpoint's weights and of ``x`` as
    ``quantize`` quantizes it, up to the bf16 output. The B-preshuffle GEMM
    takes aiter's group quant."""
    weight, scale = checkpoint
    scale = scale.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    x_q, x_scale = quantize(x)
    x_deq = x_q.float() * x_scale.repeat_interleave(BLOCK, 1)
    ref = x_deq @ (weight.float() * scale).T
    assert ((out.float() - ref).norm() / ref.norm()).item() < 1e-2


def _inputs() -> list[torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(100)
    return [
        torch.randn(m, K, generator=gen, device="cuda", dtype=torch.bfloat16)
        for m in (1, 33, 256)
    ]


def test_reload_reshuffles_new_weights_in_place(vllm_config):
    layer = _linear(vllm_config, bpreshuffle=True)
    record_metadata_for_reloading(layer)
    first = _checkpoint(vllm_config, seed=0)
    _load(layer, first)
    layer.quant_method.process_weights_after_loading(layer)

    assert layer.aiter_bpreshuffled and layer.weight.is_shuffled
    assert layer.weight_scale_inv.dtype == torch.float32
    for x in _inputs():
        _assert_matches(layer(x), x, first)

    weight, scale = layer.weight, layer.weight_scale_inv
    addresses = weight.data_ptr(), scale.data_ptr()
    second = _checkpoint(vllm_config, seed=1)
    initialize_layerwise_reload(layer)
    _load(layer, second)
    finalize_layerwise_reload(layer, vllm_config.model_config)

    assert layer.weight is weight and layer.weight_scale_inv is scale
    assert (weight.data_ptr(), scale.data_ptr()) == addresses
    for x in _inputs():
        _assert_matches(layer(x), x, second)


def test_other_layers_keep_the_blockscale_gemm(vllm_config):
    checkpoint = _checkpoint(vllm_config, seed=0)
    plain = _linear(vllm_config, bpreshuffle=False)
    shuffled = _linear(vllm_config, bpreshuffle=True)
    for layer in (plain, shuffled):
        _load(layer, checkpoint)
        layer.quant_method.process_weights_after_loading(layer)

    assert not getattr(plain, "aiter_bpreshuffled", False)
    assert not torch.equal(
        plain.weight.view(torch.uint8), shuffled.weight.view(torch.uint8)
    )
    for x in _inputs():
        _assert_matches(
            plain(x), x, checkpoint, plain.quant_method.fp8_linear.quant_fp8
        )


def test_weights_loaded_already_processed(vllm_config):
    """Weights that arrive processed (the weight cache IPC loader) are already
    shuffled, and the GEMM still reads them as such."""
    checkpoint = _checkpoint(vllm_config, seed=0)
    layer = _linear(vllm_config, bpreshuffle=True)
    _load(layer, checkpoint)
    layer.quant_method.process_weights_after_loading(layer)
    copy = _linear(vllm_config, bpreshuffle=True)
    for name, param in layer.named_parameters():
        getattr(copy, name).data = param.data.clone()

    with weights_already_processed():
        copy.quant_method.process_weights_after_loading(copy)

    for x in _inputs():
        _assert_matches(copy(x), x, checkpoint)
