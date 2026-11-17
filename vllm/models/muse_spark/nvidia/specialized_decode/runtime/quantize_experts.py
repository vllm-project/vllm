# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 quantization at load of the routed experts an NVFP4 checkpoint keeps in
BF16 (Muse Spark: layers 0 and 61), enabled by
`additional_config={"quantize_bf16_experts": "nvfp4"}` with the specialized decode
(`MuseOptions`).

The weights are rounded to nearest as the checkpoint's own NVFP4 experts are
stored (per-16 E4M3 block scales over a per-tensor FP32 global scale of
amax / (6 * 448)), so the layers then run the NVFP4 experts path of the other
layers. NVFP4 activations need static input scales, which a BF16 layer has not
been calibrated for: each such layer takes those of the nearest layer whose
experts the checkpoint ships in NVFP4.
"""

import copy
from collections.abc import Iterator

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    FLOAT4_E2M1_MAX,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import weight_amax
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

OPTION = "quantize_bf16_experts"
# The largest e2m1 value times the largest e4m3 block scale.
NVFP4_RANGE = FLOAT4_E2M1_MAX * torch.finfo(torch.float8_e4m3fn).max
# The smallest amax a global scale is taken of: an all-zero tensor gets a
# finite scale.
MIN_AMAX = 1e-12
INPUT_SCALES = ("w13_input_scale", "w2_input_scale")


def check_gpu_capability(capability: DeviceCapability | None) -> None:
    """Raise unless `capability` is in the range of ops.scaled_fp4_quant's
    compiled kernels (SM100-SM12x)."""
    if capability is None or not 100 <= capability.to_int() < 130:
        raise ValueError(f"{OPTION} needs an SM100-SM12x GPU")


def check_quantize_bf16_experts(vllm_config) -> None:
    """Raise unless the deployment can quantize its BF16 experts at load."""
    check_gpu_capability(current_platform.get_device_capability())
    if not isinstance(vllm_config.quant_config, ModelOptNvFp4Config):
        raise ValueError(f"{OPTION} needs a ModelOpt NVFP4 checkpoint")
    if vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe":
        raise NotImplementedError(f"{OPTION} does not support MegaMoE")


def without_exclusion(
    quant_config: ModelOptNvFp4Config, prefix: str
) -> QuantizationConfig:
    """A copy of `quant_config` that quantizes the module at `prefix`: its
    exclude patterns minus those that match `prefix`."""

    def excludes(pattern: str) -> bool:
        single = copy.copy(quant_config)
        single.exclude_modules = [pattern]
        return single.is_layer_excluded(prefix)

    lifted = copy.copy(quant_config)
    lifted.exclude_modules = [
        p for p in quant_config.exclude_modules if not excludes(p)
    ]
    return lifted


def quantize_nvfp4_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RTN NVFP4 of a BF16 `[rows, cols]` weight in the ModelOpt layout.

    Returns:
        The packed e2m1 codes (uint8 `[rows, cols // 2]`, element `2i` in the
        low nibble), the E4M3 block scales (`[rows, cols // 16]`, row-major)
        and the FP32 global scale `weight_scale_2`.

    """
    scale_2 = weight_amax(weight).float().clamp(min=MIN_AMAX) / NVFP4_RANGE
    packed, scale = ops.scaled_fp4_quant(
        weight, 1.0 / scale_2, is_sf_swizzled_layout=False
    )
    return packed, scale.view(torch.float8_e4m3fn).view(weight.shape[0], -1), scale_2


def load_quantized(
    experts: nn.Module, name: str, tensor: torch.Tensor
) -> Iterator[str]:
    """Loads a BF16 consolidated expert tensor (`w13_weight` `[E, 2I, H]` or
    `w2_weight` `[E, H, I]`) into NVFP4 parameters, quantizing this rank's
    experts only, one at a time on the parameters' device."""
    base = name.removesuffix("_weight")
    suffixes = ("weight", "weight_scale", "weight_scale_2")
    params = [getattr(experts, f"{base}_{s}") for s in suffixes]
    shard_ids = ("w1", "w3") if base == "w13" else ("w2",)
    device = params[0].device
    for expert_id in range(tensor.shape[0]):
        if experts._map_global_expert_id_to_local_expert_id(expert_id) < 0:
            continue
        weight = tensor[expert_id].to(device)
        for shard_id, shard in zip(shard_ids, weight.chunk(len(shard_ids))):
            for suffix, param, loaded in zip(
                suffixes, params, quantize_nvfp4_weight(shard)
            ):
                param.weight_loader(
                    param=param,
                    loaded_weight=loaded,
                    weight_name=f"{base}_{suffix}",
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
    # The input scales are set once every layer has loaded
    # (`adopt_input_scales`); the loader requires every parameter reported.
    yield from (f"{base}_{s}" for s in (*suffixes, "input_scale"))


def adopt_input_scales(moes: dict[int, nn.Module]) -> None:
    """Sets the input scales of the experts quantized at load to those of the
    nearest layer whose NVFP4 experts came from the checkpoint (the largest
    over its experts, as its own kernels take them). `moes`: the MuseSparkMoE
    of each MoE layer, by layer index."""
    quantized = [i for i, moe in moes.items() if moe.quantizes_experts_at_load]
    if not quantized:
        return
    shipped = [
        i
        for i, moe in moes.items()
        if not moe.quantizes_experts_at_load
        and hasattr(moe.experts.routed_experts, INPUT_SCALES[0])
    ]
    for index in quantized:
        if not shipped:
            raise RuntimeError(
                f"Layer {index}: no layer with checkpoint NVFP4 experts to take "
                "the input scales of"
            )
        source = min(shipped, key=lambda i: (abs(i - index), i))
        target = moes[index].experts.routed_experts
        source_experts = moes[source].experts.routed_experts
        for name in INPUT_SCALES:
            scale = getattr(source_experts, name).data.max()
            getattr(target, name).data.fill_(scale)
