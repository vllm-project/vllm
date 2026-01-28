# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Optional

import torch

# TYPE_CHECKING is used for static type analysis to prevent circular imports.
if TYPE_CHECKING:
    from types import ModuleType

# 1. Create a global variable as a placeholder for the module
_petit_kernel: Optional["ModuleType"] = None

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
                "Petit currently only supports: NVFP4 quantizations in sglang. "
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


def prepare_nvfp4_layer_for_petit(layer: torch.nn.Module) -> None:
    # 2. Call _import_petit_kernel() to trigger (or get) the import.
    petit_kernel = _import_petit_kernel()

    # Repack weights to petit format
    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    qweight = layer.weight.view(torch.int32).contiguous()

    # 3. Call functions through the imported module variable.
    petit_qweight = petit_kernel.repack_nvfp4(
        qweight, size_n=part_size_n, size_k=part_size_k
    )
    layer.weight = torch.nn.Parameter(petit_qweight, requires_grad=False)

    # Permute scales
    weight_scale = petit_kernel.process_nvfp4_scales(
        scales=layer.weight_scale, size_k=part_size_k, size_n=part_size_n
    )
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)


def apply_petit_nvfp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # Trigger (or get) the import here as well.
    petit_kernel = _import_petit_kernel()

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    # TODO: Use auto-tuning to find the performant solution_id
    # Call the function via the module variable.
    output = petit_kernel.mul_nvfp4_a16(
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
