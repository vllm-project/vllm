# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_marlin_supported,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from vllm.scalar_type import scalar_types

from .inc_scheme import INCLinearScheme

logger = init_logger(__name__)

if TYPE_CHECKING:
    from ..config_parser import INCLayerConfig


@lru_cache(maxsize=1)
def get_ark_state() -> tuple[bool, str | None, Any | None, Any | None]:
    """Return ARK availability, error details, cached module, and QuantLinear."""
    try:
        import auto_round_kernel as ark
        from auto_round_kernel.qlinear import QuantLinear

        logger.info("Successfully imported auto_round_kernel.")
    except ImportError as error:
        return False, str(error), None, None

    if getattr(ark, "cpu_lib", None) is None and getattr(ark, "xpu_lib", None) is None:
        return (
            False,
            "No ARK backend library is available.",
            None,
            None,
        )
    logger.info("Successfully loaded auto_round_kernel backend library.")

    return True, None, ark, QuantLinear


class INCWNA16LinearScheme(INCLinearScheme):
    def __init__(self, layer_config: "INCLayerConfig") -> None:
        self.layer_config = layer_config
        self.inner_method = self._build_inner_method()

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    def _build_inner_method(self):
        if self.layer_config.is_gptq:
            return self._build_gptq_method()
        if self.layer_config.is_awq:
            return self._build_awq_method()
        raise NotImplementedError(
            f"WNA16 linear scheme does not support {self.layer_config}"
        )

    def _build_gptq_method(self):
        group_size = self.layer_config.group_size
        if not isinstance(group_size, int):
            raise ValueError(
                "INC WNA16 linear requires scalar group_size, "
                f"but found {group_size!r}."
            )
        gptq_type_map = {
            (4, True): scalar_types.uint4b8,
            (8, True): scalar_types.uint8b128,
        }
        use_marlin = (
            self.layer_config.backend == "auto" or "marlin" in self.layer_config.backend
        ) and (self.layer_config.bits, self.layer_config.sym) in gptq_type_map
        if use_marlin:
            use_marlin = check_marlin_supported(
                gptq_type_map[(self.layer_config.bits, self.layer_config.sym)],
                group_size,
                has_zp=not self.layer_config.sym,
            )

        if use_marlin:
            from vllm.model_executor.layers.quantization.auto_gptq import (
                AutoGPTQLinearMethod,
            )

            return AutoGPTQLinearMethod(
                AutoGPTQConfig(
                    weight_bits=self.layer_config.bits,
                    group_size=group_size,
                    desc_act=False,
                    is_sym=self.layer_config.sym,
                    lm_head_quantized=False,
                    dynamic={},
                    full_config={},
                )
            )

        raise NotImplementedError(
            f"INC quantization with bits={self.layer_config.bits}, "
            f"sym={self.layer_config.sym} is not supported. "
            "Only 4-bit and 8-bit symmetric quantization is supported "
            "with Marlin kernels."
        )

    def _build_awq_method(self):
        group_size = self.layer_config.group_size
        if not isinstance(group_size, int):
            raise ValueError(
                "INC WNA16 linear requires scalar group_size, "
                f"but found {group_size!r}."
            )

        awq_type_map = {
            4: scalar_types.uint4,
            8: scalar_types.uint8,
        }
        use_marlin = (
            self.layer_config.backend == "auto" or "marlin" in self.layer_config.backend
        ) and self.layer_config.bits in awq_type_map
        if use_marlin:
            use_marlin = check_marlin_supported(
                awq_type_map[self.layer_config.bits],
                group_size,
                not self.layer_config.sym,
            )

        if use_marlin:
            from vllm.model_executor.layers.quantization.auto_awq import (
                AutoAWQMarlinLinearMethod,
            )

            return AutoAWQMarlinLinearMethod(
                AutoAWQConfig(
                    weight_bits=self.layer_config.bits,
                    group_size=group_size,
                    zero_point=not self.layer_config.sym,
                    lm_head_quantized=False,
                    modules_to_not_convert=[],
                    full_config={},
                )
            )

        from vllm.model_executor.layers.quantization.auto_awq import (
            AutoAWQLinearMethod,
        )

        return AutoAWQLinearMethod(
            AutoAWQConfig(
                weight_bits=self.layer_config.bits,
                group_size=group_size,
                zero_point=not self.layer_config.sym,
                lm_head_quantized=False,
            )
        )

    def create_weights(
        self,
        layer: "torch.nn.Module",
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: "torch.dtype",
        **extra_weight_attrs,
    ) -> None:
        return self.inner_method.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: "torch.nn.Module") -> None:
        return self.inner_method.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: "torch.nn.Module",
        x: "torch.Tensor",
        bias: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        return self.inner_method.apply(layer, x, bias)


class INCXPULinearBase(INCLinearScheme):
    # AWQ packs nibbles within each int32 in the order [0, 2, 4, 6, 1, 3, 5, 7];
    # this permutation undoes that ordering so values can be repacked in
    # standard sequential (GPTQ) order.
    _REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]

    def __init__(self, layer_config: "INCLayerConfig") -> None:
        self.weight_bits = layer_config.bits
        group_size = layer_config.group_size
        if not isinstance(group_size, int):
            raise ValueError(
                f"INC XPU WNA16 requires scalar group_size, but found {group_size!r}."
            )
        self.group_size = group_size

        self.sym = layer_config.sym
        self.pack_factor = 32 // self.weight_bits
        self.is_awq_packed = layer_config.is_awq

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    def _create_inc_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        params_dtype: torch.dtype,
        weight_loader: Any,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        scales_and_zp_size = input_size_per_partition // self.group_size

        if self.is_awq_packed:
            # AWQ: qweight [in, out // pack_factor] packed along output dim
            qweight = PackedvLLMParameter(
                data=torch.empty(
                    input_size_per_partition,
                    output_size_per_partition // self.pack_factor,
                    dtype=torch.int32,
                ),
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.pack_factor,
                weight_loader=weight_loader,
            )
        else:
            # GPTQ: qweight [in // pack_factor, out] packed along input dim
            qweight = PackedvLLMParameter(
                data=torch.empty(
                    input_size_per_partition // self.pack_factor,
                    output_size_per_partition,
                    dtype=torch.int32,
                ),
                input_dim=0,
                output_dim=1,
                packed_dim=0,
                packed_factor=self.pack_factor,
                weight_loader=weight_loader,
            )
        scales = GroupQuantScaleParameter(
            data=torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )
        # Both AWQ and GPTQ checkpoints store qzeros with this shape; for
        # symmetric quantization the values are ignored downstream.
        qzeros = PackedvLLMParameter(
            data=torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)

        g_idx = RowvLLMParameter(
            data=torch.tensor(
                [i // self.group_size for i in range(input_size_per_partition)],
                dtype=torch.int32,
            ),
            input_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("g_idx", g_idx)

    def _convert_awq_qweight_to_gptq(self, qw: torch.Tensor) -> torch.Tensor:
        """Convert AWQ qweight [K, N // pf] to GPTQ qweight [K // pf, N].

        AWQ packs along the output dim with a non-standard nibble order; GPTQ
        packs along the input dim with sequential nibble order. The conversion
        is lossless — it only reshuffles bits.
        """
        size_bits = self.weight_bits
        pack_factor = self.pack_factor
        mask = (1 << size_bits) - 1
        device = qw.device
        reverse_order = torch.tensor(
            self._REVERSE_AWQ_PACK_ORDER, dtype=torch.long, device=device
        )
        shifts = torch.arange(0, 32, size_bits, dtype=torch.int32, device=device)

        K, N_packed = qw.shape
        N = N_packed * pack_factor

        # Unpack int32 → individual values, fix AWQ nibble ordering
        unpacked = (qw.unsqueeze(-1) >> shifts) & mask  # (K, N_packed, pf)
        unpacked = unpacked[:, :, reverse_order]
        unpacked = unpacked.reshape(K, N)  # (K, N)

        # Repack along input dim (dim 0) in sequential nibble order
        unpacked = unpacked.reshape(K // pack_factor, pack_factor, N)
        new_qw = (unpacked.to(torch.int32) << shifts[None, :, None]).sum(
            dim=1, dtype=torch.int32
        )
        return new_qw.contiguous()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        self._create_inc_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            params_dtype=params_dtype,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )


class INCXPULinearMethod(INCXPULinearBase):
    """XPU linear method for INC w4a16 quantization (symmetric only).

    Supports both GPTQ-packed (``auto_round:auto_gptq``) and AWQ-packed
    (``auto_round:auto_awq``) AutoRound checkpoints. AWQ-packed qweights are
    losslessly repacked into the GPTQ-style nibble layout during
    ``process_weights_after_loading``, before the final oneDNN "NT" transpose
    that ``torch.ops._xpu_C.int4_gemm_w4a16`` expects.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.qweight.data.device

        qweight_data = layer.qweight.data
        if self.is_awq_packed:
            # Lossless repack: AWQ [K, N // pf] → GPTQ [K // pf, N]
            qweight_data = self._convert_awq_qweight_to_gptq(qweight_data)

        qweight_ct = qweight_data.t().contiguous()
        layer.qweight = Parameter(qweight_ct.t(), requires_grad=False)
        layer.scales = Parameter(layer.scales.data, requires_grad=False)
        layer.qzeros = Parameter(
            torch.tensor([8], dtype=torch.int8, device=device),
            requires_grad=False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (layer.qweight.shape[1],)
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = torch.ops._xpu_C.int4_gemm_w4a16(
            reshaped_x,
            layer.qweight,
            bias,
            layer.scales,
            layer.qzeros,
            self.group_size,
            None,
        )
        return out.reshape(out_shape)


class INCARKLinearMethod(INCXPULinearBase):
    def __init__(self, layer_config: "INCLayerConfig") -> None:
        super().__init__(layer_config)

        is_available, error_str, _, quant_linear_cls = get_ark_state()
        if not is_available or quant_linear_cls is None:
            reason = error_str or "unknown error"
            raise ImportError(f"Failed to import auto_round_kernel. {reason}")

        self.quant_linear_cls = quant_linear_cls

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        super().create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )
        layer.in_features = input_size_per_partition
        layer.out_features = sum(output_partition_sizes)
        layer.params_dtype = params_dtype

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(layer, "input_size_per_partition"):
            in_features = layer.input_size_per_partition
        elif hasattr(layer, "input_size"):
            in_features = layer.input_size
        else:
            raise AttributeError("Cannot determine in_features for layer.")

        if hasattr(layer, "output_partition_sizes"):
            out_features = sum(layer.output_partition_sizes)
        elif hasattr(layer, "output_size_per_partition"):
            out_features = layer.output_size_per_partition
        elif hasattr(layer, "output_size"):
            out_features = layer.output_size
        else:
            out_features = layer.scales.shape[-1]

        ark_linear = self.quant_linear_cls(
            bits=self.weight_bits,
            group_size=self.group_size,
            sym=self.sym,
            in_features=in_features,
            out_features=out_features,
            bias=layer.bias is not None,
            weight_dtype=layer.params_dtype,
        )
        ark_linear.to(layer.qweight.device)

        with torch.no_grad():
            qweight_src = layer.qweight.detach()
            if self.is_awq_packed:
                # ARK consumes GPTQ-style packed nibbles; convert AWQ losslessly.
                qweight_src = self._convert_awq_qweight_to_gptq(qweight_src)
            ark_linear.qweight.copy_(qweight_src)
            if hasattr(layer, "qzeros") and layer.qzeros is not None:
                ark_linear.qzeros.copy_(layer.qzeros.detach())
            else:
                ark_linear.qzeros = None
            ark_linear.scales.copy_(layer.scales.detach())
            if hasattr(layer, "bias") and layer.bias is not None:
                ark_linear.bias.copy_(layer.bias.detach())

        ark_linear.post_init()
        layer.ark_linear = ark_linear

        del layer.qweight
        if hasattr(layer, "qzeros"):
            del layer.qzeros
        del layer.scales

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del bias
        return layer.ark_linear.forward(x)


class INCXPUW4A16LinearScheme(INCXPULinearMethod):
    pass
