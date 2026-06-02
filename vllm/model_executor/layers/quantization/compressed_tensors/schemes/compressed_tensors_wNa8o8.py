# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight N-bit INT scheme with static INT8 input/output activation quant.

Handles pack-quantized INT weight checkpoints that carry static per-tensor INT8
``input_activations`` and/or ``output_activations``. An output-activation scale is
the distinguishing signal: it means the activation quantization is reproduced as a
float fake-quant on the layer input and output, around a weight-only matmul, rather
than a fused int8 GEMM.

The scheme owns the shared weight parameters and the activation fake-quant; the
weight matmul is delegated to a backend, chosen in this order: humming (CUDA, any
bit width), the MP linear kernels (marlin etc.; 4/8-bit only), then a torch dequant
reference.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.nn.functional as F
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa: E501
    WNA16_SUPPORTED_TYPES_MAP,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_repeat_scales_on_all_ranks,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import _has_module

logger = init_logger(__name__)

__all__ = ["CompressedTensorsWNA8O8Int", "fake_quant_static_int8"]


def fake_quant_static_int8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Static per-tensor symmetric INT8 quantize-dequantize, in x's dtype."""
    scale = scale.to(x.dtype)
    q = torch.clamp(torch.round(x / scale), -128.0, 127.0)
    return q * scale


class _WeightBackend(ABC):
    """Owns the weight transform and matmul for a pack-quantized INT weight.

    The shared ``weight_packed`` / ``weight_scale`` / ``weight_shape`` parameters
    are registered by the scheme before ``maybe_create`` is called.
    """

    @classmethod
    @abstractmethod
    def maybe_create(cls, layer: torch.nn.Module, cfg: "_WeightConfig"):
        """Return a backend instance if it can serve ``cfg``, else ``None``."""
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...

    @abstractmethod
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor: ...


class _WeightConfig:
    """Layer dimensions and quant params needed to build a weight backend."""

    def __init__(self, scheme: "CompressedTensorsWNA8O8Int", **dims):
        self.num_bits = scheme.num_bits
        self.strategy = scheme.strategy
        self.group_size = scheme.group_size  # -1 for channel
        self.quant_format = scheme.quant_format
        self.is_int_quantized = scheme.is_int_quantized
        self.input_size = dims["input_size"]
        self.output_size = dims["output_size"]
        self.output_partition_sizes = dims["output_partition_sizes"]
        self.input_size_per_partition = dims["input_size_per_partition"]
        self.output_size_per_partition = sum(self.output_partition_sizes)
        self.params_dtype = dims["params_dtype"]


class _HummingBackend(_WeightBackend):
    """humming GEMM; reads the compressed-tensors pack-quantized format natively."""

    def __init__(self, method):
        self._method = method

    @classmethod
    def maybe_create(cls, layer, cfg):
        if not (current_platform.is_cuda() and _has_module("humming")):
            return None

        from humming.schema.compressed_tensors import CompressedTensorsWeightSchema

        from vllm.model_executor.layers.quantization.humming import (
            HummingLayerQuantizationConfig,
            HummingLinearMethod,
        )

        schema = CompressedTensorsWeightSchema(
            format=cfg.quant_format,
            type="int",
            num_bits=cfg.num_bits,
            strategy=cfg.strategy,
            symmetric=True,
            group_size=None if cfg.group_size == -1 else cfg.group_size,
        )
        # TODO(mgoin): wire an int8 input_schema here (gated on input-activation
        # presence) to run the GEMM with int8 activations, once its performance is
        # validated. Today the input quant stays a fake-quant and weight-only GEMM.
        method = HummingLinearMethod(
            HummingLayerQuantizationConfig(weight_schema=schema)
        )

        # State HummingLinearMethod.process_weights_after_loading reads.
        layer.register_buffer("locks", torch.zeros(1024, dtype=torch.int32))
        layer.is_fallback = False
        layer.param_dtype = cfg.params_dtype
        layer.output_partition_sizes = cfg.output_partition_sizes
        layer.output_partition_sizes_sum = cfg.output_size_per_partition
        layer.has_bias = False
        return cls(method)

    def process_weights_after_loading(self, layer):
        self._method.process_weights_after_loading(layer)

    def apply(self, layer, x, bias):
        return self._method.apply(layer, x, bias)


class _MarlinBackend(_WeightBackend):
    """MP linear kernels (marlin, machete, ...). Supports 4/8-bit only."""

    def __init__(self, kernel):
        self._kernel = kernel

    @classmethod
    def maybe_create(cls, layer, cfg):
        # MP kernels consume the int32-packed layout only.
        if cfg.is_int_quantized:
            return None
        quant_type = WNA16_SUPPORTED_TYPES_MAP.get(cfg.num_bits)
        if quant_type is None:
            return None

        mp_config = MPLinearLayerConfig(
            full_weight_shape=(cfg.input_size, cfg.output_size),
            partition_weight_shape=(
                cfg.input_size_per_partition,
                cfg.output_size_per_partition,
            ),
            weight_type=quant_type,
            act_type=cfg.params_dtype,  # activation quant applied externally
            group_size=cfg.group_size,
            zero_points=False,
            has_g_idx=False,
        )
        try:
            kernel_cls = choose_mp_linear_kernel(mp_config)
        except ValueError:
            return None
        kernel = kernel_cls(
            mp_config,
            w_q_param_name="weight_packed",
            w_s_param_name="weight_scale",
            w_zp_param_name="weight_zero_point",
            w_gidx_param_name="weight_g_idx",
        )
        return cls(kernel)

    def process_weights_after_loading(self, layer):
        self._kernel.process_weights_after_loading(layer)

    def apply(self, layer, x, bias):
        return self._kernel.apply_weights(layer, x, bias)


class _TorchBackend(_WeightBackend):
    """Reference path: get int8 weights and dequantize in float."""

    def __init__(self, num_bits: int, is_int_quantized: bool):
        self.num_bits = num_bits
        self.is_int_quantized = is_int_quantized

    @classmethod
    def maybe_create(cls, layer, cfg):
        return cls(cfg.num_bits, cfg.is_int_quantized)

    def process_weights_after_loading(self, layer):
        if self.is_int_quantized:
            layer.weight_unpacked = layer.weight  # already int8
            return
        shape = torch.Size(
            [layer.output_size_per_partition, layer.input_size_per_partition]
        )
        int8 = unpack_from_int32(layer.weight_packed.data, self.num_bits, shape)
        layer.weight_unpacked = torch.nn.Parameter(
            int8.contiguous(), requires_grad=False
        )
        del layer.weight_packed

    def apply(self, layer, x, bias):
        w = layer.weight_unpacked.to(x.dtype)
        scale = layer.weight_scale.to(x.dtype)
        if scale.shape[1] != 1:  # group: broadcast each group over the input dim
            out_f, in_f = w.shape
            ng = scale.shape[1]
            w = (w.view(out_f, ng, in_f // ng) * scale.unsqueeze(-1)).view(out_f, in_f)
        else:
            w = w * scale
        return F.linear(x, w, bias)


# Tried in priority order; the first that can serve the layer wins.
_BACKENDS = (_HummingBackend, _MarlinBackend, _TorchBackend)


class CompressedTensorsWNA8O8Int(CompressedTensorsScheme):
    def __init__(
        self,
        num_bits: int,
        strategy: str,
        group_size: int | None = None,
        has_input_act: bool = False,
        has_output_act: bool = False,
        layer_name: str | None = None,
        quant_format: str = "pack-quantized",
    ):
        self.num_bits = num_bits
        self.pack_factor = 32 // num_bits
        self.strategy = strategy
        self.group_size = -1 if group_size is None else group_size
        self.has_input_act = has_input_act
        self.has_output_act = has_output_act
        self.layer_name = layer_name
        # "pack-quantized" (sub-byte, int32-packed) or "int-quantized" (8-bit int8).
        self.quant_format = quant_format
        self.is_int_quantized = quant_format == "int-quantized"
        self._input_scale: torch.Tensor | None = None
        self._output_scale: torch.Tensor | None = None

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = sum(output_partition_sizes)
        self._register_weight(
            layer, input_size, input_size_per_partition, params_dtype, weight_loader
        )

        cfg = _WeightConfig(
            self,
            input_size=input_size,
            output_size=output_size,
            output_partition_sizes=output_partition_sizes,
            input_size_per_partition=input_size_per_partition,
            params_dtype=params_dtype,
        )
        for backend_cls in _BACKENDS:
            backend = backend_cls.maybe_create(layer, cfg)
            if backend is not None:
                break
        assert backend is not None, f"No backend found for {layer=} {cfg=}"
        self.backend: _WeightBackend = backend

    def _register_weight(
        self, layer, input_size, input_size_per_partition, params_dtype, weight_loader
    ):
        out = layer.output_size_per_partition
        if self.is_int_quantized:
            # 8-bit int weight stored directly as int8 (no packing, no shape).
            layer.register_parameter(
                "weight",
                ModelWeightParameter(
                    data=torch.empty(out, input_size_per_partition, dtype=torch.int8),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                ),
            )
        else:
            layer.register_parameter(
                "weight_packed",
                PackedvLLMParameter(
                    input_dim=1,
                    output_dim=0,
                    packed_dim=1,
                    packed_factor=self.pack_factor,
                    weight_loader=weight_loader,
                    data=torch.empty(
                        out,
                        input_size_per_partition // self.pack_factor,
                        dtype=torch.int32,
                    ),
                ),
            )

        # Scale: per-output-channel, or per group along the input dim under TP.
        group_size = self.group_size if self.group_size != -1 else input_size
        partitioned = not marlin_repeat_scales_on_all_ranks(
            False, self.group_size, input_size != input_size_per_partition
        )
        scales = (input_size_per_partition if partitioned else input_size) // group_size
        scale_data = torch.empty(out, scales, dtype=params_dtype)
        if partitioned:
            assert input_size_per_partition % group_size == 0
            weight_scale = GroupQuantScaleParameter(
                data=scale_data, output_dim=0, input_dim=1, weight_loader=weight_loader
            )
        else:
            weight_scale = ChannelQuantScaleParameter(
                data=scale_data, output_dim=0, weight_loader=weight_loader
            )
        layer.register_parameter("weight_scale", weight_scale)
        if not self.is_int_quantized:
            layer.register_parameter(
                "weight_shape",
                BasevLLMParameter(
                    data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader
                ),
            )

        for name, present in (
            ("input_scale", self.has_input_act),
            ("output_scale", self.has_output_act),
        ):
            if present:
                layer.register_parameter(
                    name,
                    BasevLLMParameter(
                        data=torch.empty(1, dtype=torch.float32),
                        weight_loader=weight_loader,
                    ),
                )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Lift the static activation scales off the layer (applied externally) so
        # the backend only sees weight tensors. Drop uncalibrated (zero) scales.
        self._input_scale = self._take_act_scale(layer, "input_scale")
        self._output_scale = self._take_act_scale(layer, "output_scale")
        self.has_input_act = self._input_scale is not None
        self.has_output_act = self._output_scale is not None
        self.backend.process_weights_after_loading(layer)

    @staticmethod
    def _take_act_scale(layer, name: str) -> torch.Tensor | None:
        param = getattr(layer, name, None)
        if param is None:
            return None
        scale = param.data.clone()
        delattr(layer, name)
        return None if float(scale.reshape(-1)[0]) == 0.0 else scale

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        if self.has_input_act:
            x = fake_quant_static_int8(x, self._input_scale)
        out = self.backend.apply(layer, x, bias)
        if self.has_output_act:
            out = fake_quant_static_int8(out, self._output_scale)
        return out
