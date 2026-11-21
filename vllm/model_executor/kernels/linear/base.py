# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Self, TypeVar

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.model_executor.utils import refill_derived


def store_partial_sum_prescaled(
    layer: torch.nn.Module, name: str, tensor: torch.Tensor
) -> None:
    """With FP16 partial sums, store `tensor` times the 2^-k prescale as
    `layer.<name>`. A weight reload refills it in place (captured CUDA
    graphs keep its address)."""
    if getattr(layer, "partial_sum_scale", 1.0) != 1.0:
        refill_derived(layer, name, tensor * layer.partial_sum_scale)


def partial_sum_prescaled(
    layer: torch.nn.Module, name: str, tensor: torch.Tensor
) -> torch.Tensor:
    """`tensor` times the 2^-k prescale of FP16 partial sums, from the copy
    stored after loading if there is one."""
    stored = getattr(layer, name, None)
    return tensor * layer.partial_sum_scale if stored is None else stored


# TODO: extend similarly to oracle/base.py on MOE side, to remove code duplication in
# the different linear oracles (#65978)
class LinearKernelOracle:
    """Shared behaviour of linear kernel selection (dispatch).

    Linear kernels have no oracle abstraction yet; this only hosts the logic
    shared by the `init_*_linear_kernel` selectors, to be extended as they are
    migrated.
    """

    @staticmethod
    def maybe_skip_for_requantization_source() -> bool:
        """Whether to return a placeholder instead of selecting a compute kernel.

        A checkpoint method created as an online requantization source only
        dequantizes its weights, so it needs no linear kernel.
        """
        from vllm.model_executor.layers.quantization.online.requantization import (
            is_requantization_source,
        )

        return is_requantization_source()


@dataclass(slots=True)
class MMLinearLayerConfig: ...


@dataclass(slots=True)
class Params:
    """Base class for quantized layer parameters.

    This class provides a typed interface for accessing quantized weights and scales
    from layer modules. It serves as a parameter container that can be extracted from
    layers and passed to kernel implementations.

    Attributes:
        weight: The quantized weight tensor
        weight_scale: weight scaling factors
        input_scale: Optional input scaling factors

    Class Variables:
        WEIGHT: Attribute name for weight tensor on the layer module
        WEIGHT_SCALE: Attribute name for weight scale tensor on the layer module
        INPUT_SCALE: Attribute name for input scale tensor on the layer module

    Important:
        The string values of WEIGHT, WEIGHT_SCALE, and INPUT_SCALE class variables
        MUST match the attribute names used in the corresponding quantization method's
        create_weights() implementation.
        For example, if FP8LinearMethod.create_weights()
        sets layer.weight and layer.weight_scale,
        then WEIGHT="weight" and
        WEIGHT_SCALE="weight_scale" must be used here.

    Usage:
        ```python
        # Extract parameters from a quantized layer
        params = Params.from_layer(layer)

        # Access typed parameters
        output = func(input, params.weight, params.weight_scale)
        ```

    """

    weight: torch.Tensor
    weight_scale: torch.Tensor
    input_scale: torch.Tensor | None

    # Attribute names on the layer
    WEIGHT: ClassVar[str] = "weight"
    WEIGHT_SCALE: ClassVar[str] = "weight_scale"
    INPUT_SCALE: ClassVar[str] = "input_scale"

    @classmethod
    def from_layer(cls, layer: torch.nn.Module) -> Self:
        return cls(
            weight=getattr(layer, cls.WEIGHT),
            weight_scale=getattr(layer, cls.WEIGHT_SCALE),
            input_scale=getattr(layer, cls.INPUT_SCALE, None),
        )


@dataclass(slots=True)
class FP8Params(Params):
    """FP8 layer parameters with typed fields."""

    input_scale_ub: torch.Tensor | None

    INPUT_SCALE_UB: ClassVar[str] = "input_scale_ub"

    @classmethod
    def from_layer(cls, layer: torch.nn.Module) -> "FP8Params":
        """Extract parameters from layer."""
        return cls(
            weight=getattr(layer, cls.WEIGHT),
            weight_scale=getattr(layer, cls.WEIGHT_SCALE),
            input_scale=getattr(layer, cls.INPUT_SCALE, None),
            input_scale_ub=getattr(layer, cls.INPUT_SCALE_UB, None),
        )


_ParamsT = TypeVar("_ParamsT", bound=Params)
_ConfigT = TypeVar("_ConfigT", bound=MMLinearLayerConfig)
_LinearConfigT = TypeVar("_LinearConfigT")


class LinearKernel(ABC, Generic[_LinearConfigT]):
    """Base class of every quantized linear kernel."""

    # True if apply_weights honors layer.partial_sum_dtype/partial_sum_scale
    # (FP16 partial sums of RowParallelLinear, VLLM_TP_FP16_PARTIAL_SUMS).
    supports_partial_sum_dtype: ClassVar[bool] = False

    def __init__(self, config: _LinearConfigT) -> None:
        assert self.can_implement(config)[0]
        assert self.is_supported()[0]
        self.config = config

    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        """Whether the kernel runs on this device (the current one if
        `compute_capability` is None), and if not, why."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, config: _LinearConfigT) -> tuple[bool, str | None]:
        """Whether the kernel handles `config`, and if not, why."""
        raise NotImplementedError

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        """Whether a row's output is independent of the other rows in the batch."""
        return False

    @classmethod
    def executed_activation_key(
        cls, activation_quant_key: QuantKey | None
    ) -> QuantKey | None:
        """The activation quantization the kernel runs for a layer requesting
        `activation_quant_key`. Weight-only kernels return None; kernel
        selection then warns about, or under VLLM_STRICT_QUANT_SCHEME forbids,
        the fallback."""
        return activation_quant_key

    def input_quant_key(self) -> QuantKey | None:
        """The activation quant key the kernel accepts pre-quantized (via
        `as_quantized_activation`), or None if it quantizes its own input."""
        return None

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert the loaded weights on `layer` in place to the kernel's format."""
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the quantized GEMM."""
        raise NotImplementedError


class MMLinearKernel(LinearKernel[_ConfigT], Generic[_ConfigT, _ParamsT]):
    """Abstract base class for quantized matrix multiplication kernels.

    This class provides the interface for implementing custom quantized linear layer
    kernels in vLLM. Subclasses should implement specific quantization strategies
    (e.g., FP8, INT8) and their corresponding compute kernels.

    Generic Type Parameters:
        _ConfigT: Configuration type for the kernel (subclass of MMLinearLayerConfig).
                  Contains kernel-specific settings like quantization keys, dtypes, etc.
        _ParamsT: Parameter type for the kernel (subclass of Params).
                  Defines the quantized weights and scales needed by the kernel.

    Typical Usage:
        1. Define a config dataclass inheriting from MMLinearLayerConfig
        2. Define a params dataclass inheriting from Params
        3. Subclass MMLinearKernel with your config and params types
        4. Implement all abstract methods
        5. Register the kernel with the quantization method

    Example:
        ```python
        @dataclass
        class MyKernelConfig(MMLinearLayerConfig):
            static: bool
            output_dtype: torch.dtype


        @dataclass
        class MyKernelParams(FP8Params):
            custom_scale: torch.Tensor
            CUSTOM_SCALE: ClassVar[str] = "custom_scale"


        class MyKernel(MMLinearKernel[MyKernelConfig, MyKernelParams]):
            @classmethod
            def is_supported(cls, compute_capability=None):
                if compute_capability and compute_capability < 90:
                    return False, "Requires compute capability >= 9.0"
                return True, None

            @classmethod
            def can_implement(cls, config):
                if not config.static:
                    return False, "Only static quantization supported"
                return True, None

            def process_weights_after_loading(self, layer):
                # Preprocess weights for the kernel
                params = self._get_layer_params(layer)
                processed = preprocess_weights(params.weight)
                replace_parameter(layer, params.WEIGHT, processed)

            def _get_layer_params(self, layer, **kwargs):
                return MyKernelParams.from_layer(layer)

            def apply_weights(self, layer, x, bias=None, **kwargs):
                params = self._get_layer_params(layer)
                # Call your custom kernel
                output = my_custom_kernel(x, params.weight, params.weight_scale)
                if bias is not None:
                    output += bias
                return output
        ```

    """

    def __init__(self, config: _ConfigT) -> None:
        """Initialize the kernel with the given configuration.

        Args:
            config: Kernel-specific configuration containing settings like
                   quantization keys, output dtypes, etc.

        """
        self.config = config

    # return a covariant type in the subclass
    @abstractmethod
    def _get_layer_params(self, layer: torch.nn.Module, **kwargs: Any) -> _ParamsT:
        """Extract typed parameters from the layer module.

        This internal method retrieves the quantized weights and scales from
        the layer as a typed parameter object. Subclasses should typically
        delegate to ParamsClass.from_layer().

        Args:
            layer: The layer module containing the parameters
            **kwargs: Additional arguments

        Returns:
            A typed parameter object containing weights, scales, and other
            quantization parameters

        Example:
            ```python
            def _get_layer_params(self, layer, **kwargs):
                return MyKernelParams.from_layer(layer)
            ```

        """
        raise NotImplementedError

    def get_output_padding(self) -> int | None:
        """Get the number of output tokens to pad for this kernel.

        Some kernels require input padding for optimal performance.
        Override this method to specify padding requirements.

        Returns:
            Number of tokens to pad, or None for no padding (default)

        """
        return None

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError
