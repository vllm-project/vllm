# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

# TYPE_CHECKING is used for static type analysis to prevent circular imports.
if TYPE_CHECKING:
    from types import ModuleType

# 1. Create a global variable as a placeholder for the module
_petit_kernel: "ModuleType | None" = None

_PETIT_INSTALL_MSG = (
    "Petit is not installed. Please install it with `pip install petit-kernel`."
)


def _import_petit_kernel() -> "ModuleType":
    """
    A helper function to handle the lazy import.
    The first time this function is called, it will import the petit_kernel
    library and store it in the global _petit_kernel variable.
    Subsequent calls will return the already-loaded module directly.
    """
    global _petit_kernel
    if _petit_kernel is not None:
        return _petit_kernel

    try:
        import petit_kernel

        _petit_kernel = petit_kernel
        return _petit_kernel
    except ImportError:
        # The 'from None' syntax prevents chaining the original ImportError,
        # making the traceback cleaner.
        raise ImportError(_PETIT_INSTALL_MSG) from None


def _check_petit_nvfp4_supported(
    quant_method: str, group_size: int | None
) -> tuple[bool, str | None]:
    if quant_method != "NVFP4":
        return (
            False,
            (
                "Petit currently only supports: NVFP4 quantizations in vLLM. "
                "Please check the `hf_quant_config.json` file for your model's "
                "quant configuration."
            ),
        )
    if group_size is not None and group_size != 16:
        return (
            False,
            "Petit currently only supports: group_size=16 quantizations.",
        )
    return (True, None)


def verify_petit_nvfp4_supported(quant_method: str, group_size: int | None) -> None:
    supported, error_msg = _check_petit_nvfp4_supported(quant_method, group_size)
    if not supported:
        assert error_msg is not None
        raise ValueError(error_msg)


def _check_petit_mxfp4_supported(
    quant_method: str, group_size: int | None
) -> tuple[bool, str | None]:
    if quant_method != "MXFP4":
        return (
            False,
            (
                "Petit currently only supports: MXFP4 quantizations for this "
                "backend. Please check the `hf_quant_config.json` file for "
                "your model's quant configuration."
            ),
        )
    if group_size is not None and group_size != 32:
        return (
            False,
            "Petit currently only supports: group_size=32 quantizations for MXFP4.",
        )
    return (True, None)


def verify_petit_mxfp4_supported(quant_method: str, group_size: int | None) -> None:
    supported, error_msg = _check_petit_mxfp4_supported(quant_method, group_size)
    if not supported:
        assert error_msg is not None
        raise ValueError(error_msg)


def _prepare_fp4_layer_for_petit(
    layer: torch.nn.Module,
    repack_fn_name: str,
    process_scales_fn_name: str,
) -> None:
    petit_kernel = _import_petit_kernel()
    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    qweight = layer.weight.view(torch.int32).contiguous()

    repack_fn = getattr(petit_kernel, repack_fn_name)
    process_scales_fn = getattr(petit_kernel, process_scales_fn_name)

    petit_qweight = repack_fn(qweight, size_n=part_size_n, size_k=part_size_k)
    layer.weight = torch.nn.Parameter(petit_qweight, requires_grad=False)

    weight_scale = process_scales_fn(
        scales=layer.weight_scale, size_k=part_size_k, size_n=part_size_n
    )
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)


def _apply_petit_fp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    size_n: int,
    size_k: int,
    gemm_fn_name: str,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    petit_kernel = _import_petit_kernel()
    gemm_fn = getattr(petit_kernel, gemm_fn_name)

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    # TODO: Use auto-tuning to find the performant solution_id
    output = gemm_fn(
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


def prepare_nvfp4_layer_for_petit(layer: torch.nn.Module) -> None:
    _prepare_fp4_layer_for_petit(
        layer,
        repack_fn_name="repack_nvfp4",
        process_scales_fn_name="process_nvfp4_scales",
    )


def apply_petit_nvfp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return _apply_petit_fp4_linear(
        input=input,
        weight=weight,
        weight_scale=weight_scale,
        weight_scale_2=weight_scale_2,
        size_n=size_n,
        size_k=size_k,
        gemm_fn_name="mul_nvfp4_a16",
        bias=bias,
    )


def prepare_mxfp4_layer_for_petit(layer: torch.nn.Module) -> None:
    _prepare_fp4_layer_for_petit(
        layer,
        repack_fn_name="repack_mxfp4",
        process_scales_fn_name="process_mxfp4_scales",
    )


def apply_petit_mxfp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return _apply_petit_fp4_linear(
        input=input,
        weight=weight,
        weight_scale=weight_scale,
        weight_scale_2=weight_scale_2,
        size_n=size_n,
        size_k=size_k,
        gemm_fn_name="mul_mxfp4_a16",
        bias=bias,
    )
