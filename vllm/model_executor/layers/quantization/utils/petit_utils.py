# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

try:
    from petit_kernel import mul_nvfp4_a16, process_nvfp4_scales, repack_nvfp4
    _PETIT_AVAILABLE = True
except ImportError:
    _PETIT_AVAILABLE = False

_PETIT_INSTALL_MSG = ("Petit is not installed. Please install it with "
                      "`pip install petit-kernel`.")


def _require_petit() -> None:
    if not _PETIT_AVAILABLE:
        raise ImportError(_PETIT_INSTALL_MSG)


def _check_petit_nvfp4_supported(
        quant_method: str,
        group_size: Optional[int]) -> tuple[bool, Optional[str]]:
    if quant_method != "NVFP4":
        return (
            False,
            ("Petit currently only supports: NVFP4 quantizations in sglang. "
             "Please check the `hf_quant_config.json` file for your model's "
             "quant configuration."),
        )
    if group_size is not None and group_size != 16:
        return (
            False,
            "Petit currently only supports: group_size=16 quantizations.",
        )
    return (True, None)


def verify_petit_nvfp4_supported(quant_method: str,
                                 group_size: Optional[int]) -> None:
    supported, error_msg = _check_petit_nvfp4_supported(
        quant_method, group_size)
    if not supported:
        assert error_msg is not None
        raise ValueError(error_msg)


def prepare_nvfp4_layer_for_petit(layer: torch.nn.Module) -> None:
    _require_petit()

    # Repack weights to petit format
    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    qweight = layer.weight.view(torch.int32).contiguous()
    petit_qweight = repack_nvfp4(qweight,
                                 size_n=part_size_n,
                                 size_k=part_size_k)
    layer.weight = torch.nn.Parameter(petit_qweight, requires_grad=False)

    # Permute scales
    weight_scale = process_nvfp4_scales(scales=layer.weight_scale,
                                        size_k=part_size_k,
                                        size_n=part_size_n)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)


def apply_petit_nvfp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _require_petit()

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n, )

    # TODO: Use auto-tuning to find the performant solution_id
    output = mul_nvfp4_a16(
        a=reshaped_x,
        b=weight,
        s=weight_scale,
        global_scale=weight_scale_2,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        solution_id=-1,
    )
    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)
