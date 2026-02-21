# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

import torch
from typing_extensions import Self


@dataclass
class MMLinearLayerConfig: ...


@dataclass
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


@dataclass
class FP8Params(Params):
    """FP8 layer parameters with typed fields"""

    input_scale_ub: torch.Tensor | None

    INPUT_SCALE_UB: ClassVar[str] = "input_scale_ub"

    @classmethod
    def from_layer(cls, layer: torch.nn.Module) -> "FP8Params":
        """Extract parameters from layer"""
        return cls(
            weight=getattr(layer, cls.WEIGHT),
            weight_scale=getattr(layer, cls.WEIGHT_SCALE),
            input_scale=getattr(layer, cls.INPUT_SCALE, None),
            input_scale_ub=getattr(layer, cls.INPUT_SCALE_UB, None),
        )


@dataclass
class Int8Params(Params):
    """Int8 layer parameters with typed fields"""

    input_zero_point: torch.Tensor | None
    azp_adj: torch.Tensor | None

    INPUT_ZERO_POINT: ClassVar[str] = "input_zero_point"
    AZP_ADJ: ClassVar[str] = "azp_adj"

    @classmethod
    def from_layer(cls, layer: torch.nn.Module) -> "Int8Params":
        """Extract parameters from layer"""
        return cls(
            weight=getattr(layer, cls.WEIGHT),
            weight_scale=getattr(layer, cls.WEIGHT_SCALE),
            input_scale=getattr(layer, cls.INPUT_SCALE, None),
            input_zero_point=getattr(layer, cls.INPUT_ZERO_POINT, None),
            azp_adj=getattr(layer, cls.AZP_ADJ, None),
        )


_ParamsT = TypeVar("_ParamsT", bound=Params)
_ConfigT = TypeVar("_ConfigT", bound=MMLinearLayerConfig)


class MMLinearKernel(ABC, Generic[_ConfigT, _ParamsT]):
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
        2. Define a params dataclass inheriting from Params (or FP8Params/Int8Params)
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

    Lifecycle:
        1. Kernel selection: is_supported() and can_implement() check compatibility
        2. Initialization: __init__() creates kernel instance with config
        3. Weight loading: process_weights_after_loading() preprocesses weights
        4. Inference: apply_weights() executes the quantized matmul
    """

    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        """Check if this kernel is supported on the current hardware.

        This method checks hardware-level compatibility (e.g., GPU architecture,
        compute capability, available instructions). It's called during kernel
        selection to filter out kernels that cannot run on the current device.

        Args:
            compute_capability: GPU compute capability (e.g., 80 for A100, 90 for H100).
                               If None, should check the current device.

        Returns:
            A tuple of (is_supported, reason):
                - is_supported: True if the kernel can run on this hardware
                - reason: If not supported, a string explaining why; otherwise None
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, config: _ConfigT) -> tuple[bool, str | None]:
        """Check if this kernel can implement the given configuration.

        This method checks configuration-level compatibility (e.g., quantization
        scheme, group sizes, static vs dynamic quantization). It's called after
        is_supported() to determine if this kernel can handle the specific
        quantization configuration.

        Args:
            config: The kernel configuration to check

        Returns:
            A tuple of (can_implement, reason):
                - can_implement: True if this kernel supports the config
                - reason: If not supported, a string explaining why; otherwise None
            ```
        """
        raise NotImplementedError

    def __init__(self, config: _ConfigT) -> None:
        """Initialize the kernel with the given configuration.

        Args:
            config: Kernel-specific configuration containing settings like
                   quantization keys, output dtypes, etc.
        """
        self.config = config

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process and transform weights after loading from checkpoint.

        This method is called once after weights are loaded but before inference.
        Use it to preprocess weights into the format required by your kernel
        (e.g., reordering, padding, format conversion).

        Modifications should be done in-place using replace_parameter() to ensure
        the layer's parameters are properly updated.

        Args:
            layer: The layer module containing the weights to process

        Example:
            ```python
            def process_weights_after_loading(self, layer):
                params = self._get_layer_params(layer)
                # Reorder weights for better memory access
                weight_reordered = reorder_weights(params.weight)
                replace_parameter(layer, params.WEIGHT, weight_reordered)
            ```
        """
        raise NotImplementedError

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
        """Apply the quantized weights to the input tensor.

        This is the main inference method that performs the quantized matrix
        multiplication. It should handle input quantization (if needed), call
        the underlying kernel, and apply bias.

        Args:
            layer: The layer module containing the quantized weights
            x: Input tensor of shape [..., in_features]
            bias: Optional bias tensor of shape [out_features]
            **kwargs: Additional kernel-specific arguments

        Returns:
            Output tensor of shape [..., out_features]
        """
        raise NotImplementedError


_BaseKernelT = TypeVar("_BaseKernelT", bound=MMLinearKernel)
_FallbackKernelT = TypeVar("_FallbackKernelT", bound=MMLinearKernel)


class DynamicMMLinearKernel(
    MMLinearKernel, Generic[_ConfigT, _ParamsT, _BaseKernelT, _FallbackKernelT]
):
    """Dynamic kernel that switches between base and fallback
    implementations at runtime.

    This class provides a wrapper around two kernel implementations,
    allowing dynamic dispatch based on
    runtime conditions (e.g., input shape, batch size).
    The predicate() method determines which kernel
    to use for each forward pass.

    This is useful for:
    - Switching between optimized kernels based on input characteristics
    - Providing fallback paths for edge cases
    - Supporting both static and dynamic quantization in one kernel

    Generic Type Parameters:
        _ConfigT: Configuration type for the kernel
        _ParamsT: Parameter type for the kernel
        _BaseKernelT: Type of the primary/optimized kernel
        _FallbackKernelT: Type of the fallback/compatibility kernel

    Class Attributes:
        base_type: The kernel class to use for optimized execution
        fallback_type: The kernel class to use when base kernel is not suitable

    Usage:
        Subclasses must define `base_type` and `fallback_type` class attributes,
        and implement `predicate()` and `process_weights_after_loading()`.

    Example:
        ```python
        class MyDynamicKernel(
            DynamicMMLinearKernel[MyConfig, MyParams, OptimizedKernel, FallbackKernel]
        ):
            base_type = OptimizedKernel
            fallback_type = FallbackKernel

            def predicate(self, layer, x, bias=None, **kwargs):
                # Use optimized kernel for large batches
                return x.shape[0] * x.shape[1] > 1024

            def process_weights_after_loading(self, layer):
                # Process weights for both kernels
                self.base.process_weights_after_loading(layer)
                self.fallback.process_weights_after_loading(layer)
        ```

    Note:
        The apply_weights() method uses torch.cond() to enable torch.compile support.
        Both code paths are registered in the computation graph, allowing the compiled
        graph to handle dynamic dispatch efficiently.
    """

    base_type: type[_BaseKernelT]
    fallback_type: type[_FallbackKernelT]

    def __init__(self, config: _ConfigT):
        self.base = self.base_type(config)
        self.fallback = self.fallback_type(config)

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        is_base_supported, reason_1 = cls.base_type.is_supported(compute_capability)
        is_fallback_supported, reason_2 = cls.fallback_type.is_supported(
            compute_capability
        )
        if is_base_supported and is_fallback_supported:
            return True, None

        # Both unsupported: include both reasons
        if not is_base_supported and not is_fallback_supported:
            return (
                False,
                f"base is not supported due to {reason_1}; "
                f"fallback is not supported due to {reason_2}",
            )

        # Exactly one unsupported: report that one
        if not is_base_supported:
            return False, f"base is not supported due to {reason_1}"

        # Here: base is supported but fallback is not
        return False, f"fallback is not supported due to {reason_2}"

    def _get_layer_params(self, layer: torch.nn.Module, **kwargs) -> _ParamsT:
        get_fallback_params = kwargs.get("get_fallback_params", False)
        if get_fallback_params:
            return self.fallback._get_layer_params(layer)
        return self.base._get_layer_params(layer)

    @classmethod
    def can_implement(cls, config: _ConfigT) -> tuple[bool, str | None]:
        can_implement_base, reason_1 = cls.base_type.can_implement(config)
        can_implement_fallback, reason_2 = cls.fallback_type.can_implement(config)
        if can_implement_base and can_implement_fallback:
            return True, None

        if not can_implement_base and not can_implement_fallback:
            return (
                False,
                f"base is not supported due to {reason_1}; "
                f"fallback is not supported due to {reason_2}",
            )

        if not can_implement_base:
            return False, f"base is not supported due to {reason_1}"

        return False, f"fallback is not supported due to {reason_2}"

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def predicate(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> bool:
        """Determine which kernel to use at runtime.

        This method is called during forward pass to decide whether to use
        the base (optimized) kernel or the fallback kernel based on runtime
        conditions such as input shape, batch size, or other characteristics.

        Args:
            layer: The layer module containing the quantized weights
            x: Input tensor
            bias: Optional bias tensor
            **kwargs: Additional kernel-specific arguments

        Returns:
            True to use the base kernel, False to use the fallback kernel
        """
        raise NotImplementedError

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # PyTorch's torch.compile cannot handle input-dependent control flow in standard
        # Python conditionals. torch.cond() explicitly registers both code paths in the
        # computation graph,
        # allowing torch.compile to capture both branches.
        # without torch.cond, the predicate condition
        # won't be able to be captured by torch
        # compile

        return torch.cond(
            self.predicate(layer, x, bias),
            self.base.apply_weights,
            self.fallback.apply_weights,
            (layer, x, bias),
        )
