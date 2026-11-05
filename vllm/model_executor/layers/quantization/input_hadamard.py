# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block Hadamard rotation of linear-layer inputs, independent of the weight
format.

Rotating each contiguous block of the input dimension by an orthonormal
Hadamard matrix ``H`` on both operands leaves the product unchanged,
``(x H)(w H)^T = x w^T``, while spreading outliers across each block, which
shrinks the error of block-scaled quantization (MXFP4, MXFP8, ...).
"""

import functools

import torch

from vllm.model_executor.layers.linear import (
    WEIGHT_LOADER_V2_SUPPORTED,
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.utils import set_weight_attrs


@functools.cache
def hadamard_matrix(size: int, device: torch.device) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard matrix, ``scipy.linalg.hadamard(size)``
    scaled by ``1 / sqrt(size)``, in fp32. ``size`` must be a power of two."""
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {size}")
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    while h.shape[0] < size:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / size**0.5


def block_hadamard_rotate(x: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    """Rotate each contiguous ``hadamard.shape[0]``-wide block of the last
    dimension of ``x`` by ``hadamard``, computed in fp32 and returned in
    ``x.dtype``."""
    size = hadamard.shape[0]
    rotated = x.float().unflatten(-1, (-1, size)) @ hadamard
    return rotated.flatten(-2).to(x.dtype)


@register_weight_loader_v2_supported_method
class InputHadamardLinearMethod(LinearMethodBase):
    """Wraps a linear method and rotates the layer input by an orthonormal
    block Hadamard matrix before the wrapped `apply`, in the manner of
    `CompressedTensorsLinearTransformMethod`.

    With ``rotate_weight``, the weight (``[out, in]``) is rotated the same way
    right before the wrapped method processes it, for example before online
    quantization. Otherwise the checkpoint stores the weight already rotated,
    together with an int8 ``input_rotation`` tensor holding the unnormalized
    (+-1) Sylvester matrix, which is checked and then dropped.
    """

    # TODO: the rotation runs in PyTorch before the wrapped method; let kernels
    # that can rotate inside their activation quantization (AITER's MXFP4 HIP
    # quant, CUDA hadacore) take it over.

    def __init__(
        self, quant_method: LinearMethodBase, size: int, rotate_weight: bool
    ) -> None:
        self.quant_method = quant_method
        self.size = size
        self.rotate_weight = rotate_weight

    @property
    def uses_meta_device(self) -> bool:  # type: ignore[override]
        return self.quant_method.uses_meta_device

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if input_size_per_partition % self.size != 0:
            raise ValueError(
                f"Input size per partition {input_size_per_partition} is not "
                f"divisible by the Hadamard size {self.size}."
            )
        # This class is registered for weight_loader_v2; hand v1 to wrapped
        # methods that are not, as CompressedTensorsLinearTransformMethod does.
        if self.quant_method.__class__.__name__ not in WEIGHT_LOADER_V2_SUPPORTED:
            extra_weight_attrs["weight_loader"] = layer.weight_loader
        self.quant_method.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        if not self.rotate_weight:
            input_rotation = torch.nn.Parameter(
                torch.zeros(self.size, self.size, dtype=torch.int8),
                requires_grad=False,
            )
            set_weight_attrs(
                input_rotation, {"weight_loader": self._load_input_rotation}
            )
            layer.register_parameter("input_rotation", input_rotation)

    def _load_input_rotation(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, *args, **kwargs
    ) -> None:
        """Every shard of a fused layer carries the same rotation, and it must
        be the Sylvester matrix `apply` rotates by."""
        hadamard = hadamard_matrix(self.size, loaded_weight.device)
        if not torch.equal(loaded_weight.float(), (hadamard * self.size**0.5).round()):
            raise NotImplementedError(
                "Only the Sylvester Hadamard input rotation is supported; random "
                "or trained rotations are not."
            )
        param.data.copy_(loaded_weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Online methods may be called again after quantizing; rotate once.
        weight: torch.Tensor = layer.weight
        if not getattr(layer, "_already_called_process_weights_after_loading", False):
            if self.rotate_weight:
                hadamard = hadamard_matrix(self.size, weight.device)
                weight.data.copy_(block_hadamard_rotate(weight.data, hadamard))
            else:
                input_rotation: torch.Tensor = layer.input_rotation
                if not input_rotation.any():
                    raise ValueError(
                        "The checkpoint did not provide the input_rotation of an "
                        "online-rotated layer."
                    )
                del layer.input_rotation
        self.quant_method.process_weights_after_loading(layer)
        layer.input_hadamard = hadamard_matrix(self.size, layer.weight.device)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hadamard: torch.Tensor = layer.input_hadamard
        x = block_hadamard_rotate(x, hadamard)
        return self.quant_method.apply(layer, x, bias)
