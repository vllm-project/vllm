# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load ModelOpt MIXED_PRECISION GLM-5.3 checkpoints into BF16-built modules.

GLM-5.3-Flash builds its attention (KDA, MLA, the MTP layer's MLA and
indexer) and its vision tower in BF16. ModelOpt mixed-precision checkpoints
such as `nvidia/GLM-5.3-Flash-FP4-Local` quantize those modules (MXFP8,
FP8_PB_WO, W4A16_NVFP4), so their `weight` comes with `weight_scale`
companions that have no parameter to load into. This module dequantizes those
weights to BF16 while they stream in and hands the result to the model's normal
loading path, so fusion, sharding and padding are unchanged.

Dequantizing each checkpoint module before fusion lets a fused layer mix
formats: KDA's `in_proj_qkvbfg_a` fuses MXFP8 `q/k/v_proj` with BF16
`b/f_a/g_a_proj`, and the indexer's `weights_proj` (MXFP8 in the MTP layer) is
a BF16 view of the fused `wk_weights_proj` weight. Layers that load their
checkpoint scales (`bf16_linear`, `DequantLinearMethod`) are left to them.
"""

from collections.abc import Callable, Iterable, Iterator

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    dequant_mxfp8_to_bf16,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)

logger = init_logger(__name__)

# Longest suffix first, so ".weight_scale_2" is not read as ".weight_scale".
_PARTS = (".weight_scale_2", ".weight_scale", ".input_scale", ".weight")

_NVFP4_BLOCK = 16
_FP8_BLOCK = 128


def is_modelopt_mixed(quant_config: QuantizationConfig | None) -> bool:
    return quant_config is not None and quant_config.get_name() == "modelopt_mixed"


def _split(name: str) -> tuple[str, str] | None:
    for suffix in _PARTS:
        if name.endswith(suffix):
            return name[: -len(suffix)], suffix[1:]
    return None


def _is_quantized_weight(t: torch.Tensor) -> bool:
    return t.dtype in (torch.float8_e4m3fn, torch.uint8)


def _block_scale(scale: torch.Tensor, rows: int, cols: int, block: int) -> torch.Tensor:
    """Expand a [ceil(rows/block), ceil(cols/block)] scale to [rows, cols]."""
    full = scale.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    return full[:rows, :cols]


def dequantize_to_bf16(parts: dict[str, torch.Tensor]) -> torch.Tensor:
    """Dequantize one ModelOpt linear weight to BF16.

    The format follows from the tensors themselves:
      * FP8 weight + uint8 E8M0 scale per 32 columns: MXFP8.
      * FP8 weight + float32 2D scale: FP8_PB_WO (128x128 blocks).
      * FP8 weight + scalar scale: per-tensor FP8.
      * uint8 packed E2M1 weight + FP8 scale per 16 columns + `weight_scale_2`:
        NVFP4 / W4A16_NVFP4.
    """
    weight, scale = parts["weight"], parts["weight_scale"]
    device = weight.device
    accelerator = torch.accelerator.current_accelerator(check_available=True)
    if device.type == "cpu" and accelerator is not None and accelerator.type == "cuda":
        # Weights stream in on the CPU; dequantize them on the current GPU.
        device = torch.device("cuda", torch.accelerator.current_device_index())
    weight, scale = weight.to(device), scale.to(device)

    if weight.dtype == torch.float8_e4m3fn:
        rows, cols = weight.shape
        if scale.dtype == torch.uint8:
            if scale.shape != (rows, cols // MXFP8_BLOCK_SIZE):
                raise ValueError(f"MXFP8 scale {tuple(scale.shape)} for {(rows, cols)}")
            if bool((scale == 255).any()):
                raise ValueError("MXFP8 scale holds the E8M0 NaN code")
            return dequant_mxfp8_to_bf16(weight, scale)
        w = weight.to(torch.float32)
        if scale.numel() == 1:
            w = w * scale.to(torch.float32).reshape(())
        else:
            # FP8_PB_WO: one scale per 128x128 block, stored with any shape
            # holding ceil(rows/128) * ceil(cols/128) values.
            blocks = (-(-rows // _FP8_BLOCK), -(-cols // _FP8_BLOCK))
            if scale.numel() != blocks[0] * blocks[1]:
                raise ValueError(
                    f"FP8 block scale {tuple(scale.shape)} for {(rows, cols)}"
                )
            w = w * _block_scale(
                scale.to(torch.float32).reshape(blocks), rows, cols, _FP8_BLOCK
            )
        return w.to(torch.bfloat16)

    if weight.dtype == torch.uint8:
        if "weight_scale_2" not in parts:
            raise ValueError("NVFP4 weight without weight_scale_2")
        rows, packed = weight.shape
        cols = packed * 2
        if scale.shape != (rows, cols // _NVFP4_BLOCK):
            raise ValueError(f"NVFP4 scale {tuple(scale.shape)} for {(rows, cols)}")
        # weight_scale_2 multiplies the per-block scales (a Triton kernel on GPU)
        global_scale = parts["weight_scale_2"].to(device, torch.float32).reshape(1)
        return dequantize_to_dtype(
            weight.contiguous(),
            scale.contiguous(),
            global_scale,
            torch.bfloat16,
            block_size=_NVFP4_BLOCK,
            swizzle=False,
        )

    raise ValueError(f"unsupported quantized weight dtype {weight.dtype}")


def dequantize_bf16_targets(
    weights: Iterable[tuple[str, torch.Tensor]],
    is_bf16_target: Callable[[str], bool],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield `weights` with quantized BF16-target modules dequantized.

    `is_bf16_target(module)` says whether the checkpoint module (name without
    the `.weight` suffix) loads into a parameter the model holds in BF16.
    For those modules the quantized `weight` and its scales are buffered until
    complete, then yielded as one BF16 `<module>.weight`; their
    `input_scale` is dropped. Every other tensor passes through unchanged.
    """
    pending: dict[str, dict[str, torch.Tensor]] = {}
    done: set[str] = set()
    num_dequantized = 0
    for item in weights:
        name, tensor = item[0], item[1]
        split = _split(name)
        if split is None:
            yield item
            continue
        module, part = split
        if module in done:
            if part == "weight":
                raise ValueError(f"{name} seen twice")
            continue  # a late input_scale or extra scale of a finished module
        entry = pending.get(module)
        if entry is None:
            if part == "weight" and not _is_quantized_weight(tensor):
                yield item  # already BF16
                continue
            if not is_bf16_target(module):
                yield item
                continue
            entry = pending[module] = {}
        entry[part] = tensor
        if "weight" not in entry or "weight_scale" not in entry:
            continue
        if entry["weight"].dtype == torch.uint8 and "weight_scale_2" not in entry:
            continue
        del pending[module]
        done.add(module)
        num_dequantized += 1
        yield (f"{module}.weight", dequantize_to_bf16(entry))
    if pending:
        missing = {m: sorted(p) for m, p in list(pending.items())[:5]}
        raise ValueError(
            f"{len(pending)} quantized modules are incomplete in the checkpoint "
            f"(module: parts present), e.g. {missing}"
        )
    if num_dequantized:
        logger.info_once(
            "Dequantized %d ModelOpt-quantized modules that this model holds in "
            "BF16 (attention / indexer / vision).",
            num_dequantized,
        )


def stacked_bf16_target_fn(
    params_dict: dict[str, torch.nn.Parameter],
    stacked_params_mapping: Iterable[tuple[str, str, int]],
) -> Callable[[str], bool]:
    """`is_bf16_target` for a loader with a stacked-params mapping.

    The module is mapped to the parameter it loads into (fused targets via the
    mapping); it is a BF16 target when that parameter exists and has no
    `weight_scale` / `weight_scale_inv` companion. Modules with no
    parameter of their own (e.g. routed experts) are not.
    """
    mapping = [(p.lstrip("."), w.lstrip(".")) for p, w, _ in stacked_params_mapping]

    def is_bf16_target(module: str) -> bool:
        target = module
        base, _, leaf = module.rpartition(".")
        for param_name, weight_name in mapping:
            if leaf == weight_name:
                candidate = f"{base}.{param_name}" if base else param_name
                if f"{candidate}.weight" in params_dict:
                    target = candidate
                    break
        if f"{target}.weight" not in params_dict:
            return False
        return (
            f"{target}.weight_scale" not in params_dict
            and f"{target}.weight_scale_inv" not in params_dict
        )

    return is_bf16_target
