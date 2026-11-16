# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import partial
from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils import nvfp4_emulation_utils as nv
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


def _rename(arg: Any, old: str, new: str) -> Any:
    return arg.replace(old, new) if isinstance(arg, str) else arg


def _load_quantized(layer: torch.nn.Module, name: str, *scale_names: str):
    """Load `(values, *scales)` as a checkpoint of that layout is loaded."""
    load = getattr(getattr(layer, name), "weight_loader", default_weight_loader)
    layer.quantizes_on_load = True

    def load_quantized(param, quantized, *args, **kwargs):
        values, *scales = quantized
        for scale_name, scale in zip(scale_names, scales):
            scale_param = getattr(layer, scale_name)
            # the MoE loader picks how to load a scale by its parameter name
            getattr(scale_param, "weight_loader", default_weight_loader)(
                scale_param,
                scale,
                *(_rename(a, name, scale_name) for a in args),
                **{k: _rename(v, name, scale_name) for k, v in kwargs.items()},
            )
        return load(param, values, *args, **kwargs)

    return load_quantized


def quantize_on_load(
    layer: torch.nn.Module, name: str, quantize: Callable, *scale_names: str
) -> None:
    """Make quantized `layer.<name>` load high-precision tensors via `quantize`."""
    load = _load_quantized(layer, name, *scale_names)

    def loader(param, loaded_weight, *args, **kwargs):
        weight = loaded_weight.to(param.device, layer.params_dtype)
        return load(param, quantize(weight), *args, **kwargs)

    getattr(layer, name).weight_loader = loader


def _quantize_nvfp4(weight: torch.Tensor, amax: torch.Tensor) -> tuple:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    global_scale = nv.FLOAT4_E2M1_MAX * fp8_max / amax.float().clamp_min(1e-8)
    values, scales = ops.scaled_fp4_quant(
        weight.contiguous(), global_scale, is_sf_swizzled_layout=False
    )
    return values, scales, (1.0 / global_scale).reshape(1)


def _nvfp4_on_load(layer: torch.nn.Module) -> None:
    """w1 and w3 of an expert share one global scale: the first waits."""
    pending: dict[int, tuple[str, torch.Tensor]] = {}

    def loader(load, param, loaded_weight, name, shard_id, expert_id, **kw):
        weight = loaded_weight.to(param.device, layer.params_dtype, copy=True)
        shards = [(shard_id, weight)]
        if shard_id in ("w1", "w3") and layer.moe_config.is_act_and_mul:
            if expert_id not in pending:
                pending[expert_id] = shards[0]
                return True
            shards.insert(0, pending.pop(expert_id))
        amax = max(w.abs().amax() for _, w in shards)
        for shard, w in shards:
            ret = load(param, _quantize_nvfp4(w, amax), name, shard, expert_id, **kw)
        return ret

    for p in ("w13_", "w2_"):
        names = (f"{p}weight", f"{p}weight_scale", f"{p}weight_scale_2")
        load = _load_quantized(layer, *names)
        getattr(layer, names[0]).weight_loader = partial(loader, load)


def checkpoint_method(layer: torch.nn.Module, spec: Any, default: Any) -> Any:
    """The pre-quantized checkpoint method that `spec` stands for, if any."""
    from compressed_tensors.quantization import QuantizationArgs

    import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 as ct_fp8  # noqa: E501
    import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_mxfp8 as ct_mxfp8  # noqa: E501
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization import fp8, modelopt, mxfp4
    from vllm.model_executor.layers.quantization.utils import mxfp4_utils, mxfp8_utils
    from vllm.model_executor.layers.quantization.utils import quant_utils as qu
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    is_moe, moe = isinstance(layer, RoutedExperts), getattr(layer, "moe_config", None)
    if spec.activation != default or not (is_moe or isinstance(layer, LinearBase)):
        return None
    weight, ckpt = spec.weight, modelopt.CkptCtx()
    block = weight == qu.kFp8Static128BlockSym
    if weight in (qu.kFp8StaticTensorSym, qu.kFp8Static128BlockSym):
        cfg = fp8.Fp8Config(weight_block_size=[128, 128] if block else None)
        method = fp8.Fp8MoEMethod(cfg, layer) if is_moe else fp8.Fp8LinearMethod(cfg)
        quantize = per_block_cast_to_fp8 if block else ops.scaled_fp8_quant
    elif weight == qu.kFp8StaticChannelSym and is_moe:
        channel = QuantizationArgs(type="float", strategy="channel")
        token = QuantizationArgs(type="float", strategy="token", dynamic=True)
        method = ct_fp8.CompressedTensorsW8A8Fp8MoEMethod(channel, token, moe)
        quantize = partial(ops.scaled_fp8_quant, use_per_token_if_dynamic=True)
    elif weight == qu.kFp8StaticChannelSym:
        token_spec = QuantSpec(weight=qu.kFp8StaticTokenSym, activation=spec.activation)
        method = modelopt.ModelOptLinearMethod(token_spec, ckpt)

        def quantize(w):  # its channel scales are 1-D
            values, scales = ops.scaled_fp8_quant(w, use_per_token_if_dynamic=True)
            return values, scales.flatten()

    elif weight == qu.kMxfp8Static and is_moe:
        method = ct_mxfp8.CompressedTensorsW8A8Mxfp8MoEMethod(moe)
        quantize = mxfp8_utils.mxfp8_e4m3_quantize
    elif weight == qu.kMxfp8Static:
        mx_spec = QuantSpec(weight=qu.kMxfp8Static, activation=qu.kMxfp8Dynamic)
        method = modelopt.ModelOptLinearMethod(mx_spec, ckpt)
        quantize = mxfp8_utils.mxfp8_e4m3_quantize
    elif weight == qu.kMxfp4Static and is_moe:
        method = mxfp4.Mxfp4MoEMethod(moe, spec.activation)
        quantize = mxfp4_utils.mxfp4_quantize
    elif weight == qu.kNvfp4Static and is_moe:
        config = modelopt.ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
        config.moe_activation_override = spec.activation
        method = modelopt.ModelOptNvFp4FusedMoE(config, moe)
    else:
        return None
    scale = "weight_scale_inv" if block else "weight_scale"
    create_weights = method.create_weights

    def create_weights_quantized_on_load(layer, *args, **kwargs):
        create_weights(layer, *args, **kwargs)
        if weight == qu.kNvfp4Static:
            return _nvfp4_on_load(layer)
        for p in ("w13_", "w2_") if is_moe else ("",):
            quantize_on_load(layer, f"{p}weight", quantize, f"{p}{scale}")

    method.create_weights = create_weights_quantized_on_load  # type: ignore[method-assign]
    return method
