# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fractions import Fraction
from typing import TYPE_CHECKING, Any

import regex as re
import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig,
    QuantizationMethods,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


class INCConfig(QuantizationConfig):
    """Config class for Intel Neural Compressor (INC).
    Repo: https://github.com/intel/neural-compressor
    """

    SUPPORTED_BITS = {2, 3, 4, 8}
    SUPPORTED_DTYPES = {"int"}
    SUPPORTED_FORMATS = {"auto_round:auto_gptq", "auto_round:auto_awq"}
    SUPPORTED_BACKENDS = {
        "auto",
        "gptq",
        "gptq:marlin",
        "awq",
        "awq:marlin",
        "marlin",
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        sym: bool = True,
        packing_format: str = "auto_round:auto_gptq",
        block_name_to_quantize: str | list[str] | None = None,
        extra_config: dict[str, Any] | None = None,
        data_type: str = "int",
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if weight_bits not in self.SUPPORTED_BITS:
            raise ValueError(
                f"Unsupported weight_bits: {weight_bits}, "
                f"currently only support {self.SUPPORTED_BITS}."
            )
        if data_type not in self.SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported data_type: {data_type},"
                f" currently only support  {self.SUPPORTED_DTYPES}."
            )
        if packing_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported packing_format: {packing_format}, "
                f"currently only support {self.SUPPORTED_FORMATS}."
            )
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend: {backend},  "
                f"currently only support {self.SUPPORTED_BACKENDS}."
            )

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.sym = sym
        self.packing_format = packing_format
        self.block_name_to_quantize = (
            block_name_to_quantize.split(",")
            if isinstance(block_name_to_quantize, str)
            else block_name_to_quantize
        )
        self.extra_config = extra_config
        self.data_type = data_type
        self.backend = backend
        self.pack_factor = Fraction(32, weight_bits)

    def __repr__(self) -> str:
        return (
            f"INCConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, sym={self.sym})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "inc"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "INCConfig":
        return cls(
            weight_bits=cls.get_from_keys(config, ["bits"]),
            group_size=cls.get_from_keys(config, ["group_size"]),
            sym=cls.get_from_keys(config, ["sym"]),
            packing_format=cls.get_from_keys_or(
                config, ["packing_format"], "auto_round:auto_gptq"
            ),
            block_name_to_quantize=cls.get_from_keys_or(
                config, ["block_name_to_quantize", "to_quant_block_names"], None
            ),
            extra_config=cls.get_from_keys_or(config, ["extra_config"], None),
            data_type=cls.get_from_keys_or(config, ["data_type"], "int"),
            backend=cls.get_from_keys_or(config, ["backend", "vllm_backend"], "auto"),
        )

    def get_layer_config(self, layer, layer_name: str):
        def get_config(name: str, quantized: bool = True):
            if not self.extra_config:
                return (
                    self.weight_bits if quantized else 16,
                    self.group_size if quantized else -1,
                    self.sym if quantized else True,
                )

            # exact match first
            if name in self.extra_config:
                cfg = self.extra_config[name]
                return (
                    cfg.get("bits", self.weight_bits if quantized else 16),
                    cfg.get("group_size", self.group_size if quantized else -1),
                    cfg.get("sym", self.sym if quantized else True),
                )

            REGEX_SPECIAL_CHARS = set(r"*+?^$()[]{}|\\")
            for pattern, cfg in self.extra_config.items():
                if not isinstance(pattern, str) or not any(
                    c in REGEX_SPECIAL_CHARS for c in pattern
                ):
                    continue

                try:
                    if re.search(re.compile(pattern), name) is not None:
                        return (
                            cfg.get("bits", self.weight_bits if quantized else 16),
                            cfg.get("group_size", self.group_size if quantized else -1),
                            cfg.get("sym", self.sym if quantized else True),
                        )
                except re.error:
                    # Invalid regex, ignore.
                    continue

            return (
                self.weight_bits if quantized else 16,
                self.group_size if quantized else -1,
                self.sym if quantized else True,
            )

        # 1. Exact match from config
        if self.extra_config and layer_name in self.extra_config:
            return get_config(layer_name)

        # 2. Determine whether layer should be quantized
        quantized = not isinstance(layer, ParallelLMHead)
        if self.block_name_to_quantize:
            quantized = any(
                layer_name.startswith(name) for name in self.block_name_to_quantize
            )

        # 3. Handle fused MoE
        if self.extra_config and "fusedmoe" in layer.__class__.__name__.lower():
            moe_configs = [
                get_config(name, quantized)
                for name in self.extra_config
                if name.startswith(layer_name)
            ]
            if moe_configs:
                if len(set(moe_configs)) == 1:
                    return moe_configs[0]
                raise ValueError(
                    f"Fused MoE layer '{layer_name}' requires "
                    f"consistent quant config for all sub-layers"
                )

        # 4. Handle fused QKV or other patterns
        if self.extra_config:
            for fusion_key, sub_keys in self.packed_modules_mapping.items():
                if fusion_key in layer_name and layer_name.count(fusion_key) == 1:
                    sub_names = [
                        layer_name.replace(fusion_key, sub_key) for sub_key in sub_keys
                    ]
                    sub_configs = [get_config(name, quantized) for name in sub_names]
                    if len(set(sub_configs)) == 1:
                        return sub_configs[0]
                    raise ValueError(
                        f"Fused module '{layer_name}' requires "
                        f"consistent quant config for {sub_names}"
                    )

        # 5. Fallback or try a regular expression match
        return get_config(layer_name, quantized)

    def check_quantized(self, weight_bits: int) -> bool:
        return weight_bits < 16

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.block_name_to_quantize is not None:
            self.block_name_to_quantize = hf_to_vllm_mapper.apply_list(
                self.block_name_to_quantize
            )
        if self.extra_config is not None:
            self.extra_config = hf_to_vllm_mapper.apply_dict(self.extra_config)

    def apply_awq_quant_layer(self, layer, prefix: str, backend: str = "auto"):
        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            check_marlin_supported,
            check_moe_marlin_supports_layer,
        )

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None

        logger.debug(
            "[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",
            prefix,
            layer.__class__.__name__,
            weight_bits,
            group_size,
            sym,
        )
        if backend == "auto" or "marlin" in backend:
            AWQ_TYPE_MAP = {
                4: scalar_types.uint4,
                8: scalar_types.uint8,
            }
            use_marlin = (weight_bits in AWQ_TYPE_MAP) and check_marlin_supported(
                AWQ_TYPE_MAP[weight_bits], group_size, not sym
            )

            if isinstance(layer, FusedMoE):
                use_marlin = use_marlin and check_moe_marlin_supports_layer(
                    layer, group_size
                )

        else:
            use_marlin = False
        if use_marlin:
            from vllm.model_executor.layers.quantization.awq_marlin import (
                AWQMarlinConfig,
                AWQMarlinLinearMethod,
                AWQMarlinMoEMethod,
            )

            quant_args_marlin = AWQMarlinConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                zero_point=not sym,
                lm_head_quantized=False,
                full_config={},
                modules_to_not_convert=[],
            )
        else:
            from vllm.model_executor.layers.quantization.awq import (
                AWQConfig,
                AWQLinearMethod,
            )

            quant_args = AWQConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                zero_point=not sym,
            )

        if isinstance(layer, FusedMoE):
            if use_marlin:
                return AWQMarlinMoEMethod(quant_args_marlin, layer.moe_config)
            from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config

            config = {
                "quant_method": "awq",
                "bits": weight_bits,
                "group_size": group_size,
                "zero_point": not sym,
                "lm_head": False,
            }
            return MoeWNA16Config.from_config(config).get_quant_method(layer, prefix)

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if use_marlin:
                return AWQMarlinLinearMethod(quant_args_marlin)
            else:
                return AWQLinearMethod(quant_args)
        return None

    def apply_gptq_quant_layer(self, layer, prefix: str, backend: str = "auto"):
        from vllm.model_executor.layers.fused_moe import FusedMoE
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            check_marlin_supported,
            check_moe_marlin_supports_layer,
        )

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None

        logger.debug(
            "[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",
            prefix,
            layer.__class__.__name__,
            weight_bits,
            group_size,
            sym,
        )
        if backend == "auto" or "marlin" in backend:
            GPTQ_TYPE_MAP = {
                (4, True): scalar_types.uint4b8,
                (8, True): scalar_types.uint8b128,
            }
            use_marlin = (weight_bits, sym) in GPTQ_TYPE_MAP and check_marlin_supported(
                GPTQ_TYPE_MAP[(weight_bits, sym)], group_size, has_zp=not sym
            )
            if isinstance(layer, FusedMoE):
                use_marlin = use_marlin and check_moe_marlin_supports_layer(
                    layer, group_size
                )
        else:
            use_marlin = False
        if use_marlin:
            from vllm.model_executor.layers.quantization.gptq_marlin import (
                GPTQMarlinConfig,
                GPTQMarlinLinearMethod,
                GPTQMarlinMoEMethod,
            )

            quant_args_marlin = GPTQMarlinConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                is_sym=sym,
                lm_head_quantized=False,
                desc_act=False,
                dynamic={},
                full_config={},
            )
        else:
            from vllm.model_executor.layers.quantization.gptq import (
                GPTQConfig,
                GPTQLinearMethod,
            )

            quant_args = GPTQConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                lm_head_quantized=False,
                desc_act=False,
                dynamic={},
            )

        if isinstance(layer, FusedMoE):
            if use_marlin:
                return GPTQMarlinMoEMethod(quant_args_marlin, layer.moe_config)
            else:
                from vllm.model_executor.layers.quantization.moe_wna16 import (
                    MoeWNA16Config,
                )

                config = {
                    "quant_method": "gptq",
                    "bits": weight_bits,
                    "group_size": group_size,
                    "sym": sym,
                    "lm_head": False,
                }
                return MoeWNA16Config.from_config(config).get_quant_method(
                    layer, prefix
                )

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if use_marlin:
                return GPTQMarlinLinearMethod(quant_args_marlin)
            else:
                return GPTQLinearMethod(quant_args)

        return None

    def apply_xpu_w4a16_quant_layer(self, layer, prefix: str):
        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None

        if weight_bits != 4:
            raise NotImplementedError(
                f"INC on XPU only supports 4-bit quantization, "
                f"got weight_bits={weight_bits}."
            )
        if not sym:
            raise NotImplementedError(
                "INC W4A16 on XPU only supports symmetric quantization for now."
            )

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            try:
                return INCXPULinearARKMethod(
                    weight_bits=weight_bits,
                    group_size=group_size,
                    sym=sym,
                )
            except ImportError as error:
                logger.warning(
                    "Failed to initialize ARK backend for layer %s; "
                    "falling back to the default XPU INC path. Error: %s",
                    prefix,
                    error,
                )

            return INCXPULinearMethod(
                weight_bits=weight_bits,
                group_size=group_size,
                sym=sym,
            )
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if prefix and self.extra_config:
            for layer_name in self.extra_config:
                if (
                    layer_name == prefix or layer_name == f"model.{prefix}"
                ) and self.extra_config[layer_name].get("bits", 16) >= 16:
                    return UnquantizedLinearMethod()
        if current_platform.is_xpu():
            return self.apply_xpu_w4a16_quant_layer(layer, prefix)
        if "gptq" in self.packing_format or "gptq" in self.backend:
            return self.apply_gptq_quant_layer(layer, prefix)
        if "awq" in self.packing_format or "awq" in self.backend:
            return self.apply_awq_quant_layer(layer, prefix)

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> "QuantizationMethods | None":
        """Override the `auto-round` method to `inc`."""
        is_auto_round_format = hf_quant_cfg.get("quant_method", None) == "auto-round"
        if is_auto_round_format:
            return cls.get_name()
        return None


class _INCXPULinearBase(LinearMethodBase):
    _set_layer_attrs_on_create = False

    def __init__(self, weight_bits: int, group_size: int, sym: bool):
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.sym = sym
        self.pack_factor = 32 // weight_bits

    @classmethod
    def _create_inc_weights(
        cls,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        params_dtype: torch.dtype,
        weight_loader: Any,
        group_size: int,
        pack_factor: int,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        scales_and_zp_size = input_size_per_partition // group_size

        if cls._set_layer_attrs_on_create:
            layer.in_features = input_size_per_partition
            layer.out_features = output_size_per_partition
            layer.params_dtype = params_dtype

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=pack_factor,
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

        qzeros = PackedvLLMParameter(
            data=torch.empty(
                scales_and_zp_size,
                output_size_per_partition // pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=pack_factor,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)

        g_idx = RowvLLMParameter(
            data=torch.tensor(
                [i // group_size for i in range(input_size_per_partition)],
                dtype=torch.int32,
            ),
            input_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("g_idx", g_idx)

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
        del input_size, output_size
        type(self)._create_inc_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            params_dtype=params_dtype,
            weight_loader=extra_weight_attrs.get("weight_loader"),
            group_size=self.group_size,
            pack_factor=self.pack_factor,
        )


def _get_ark_type_str(dtype: torch.dtype) -> str:
    """Helper: Convert PyTorch's dtype to a string format recognized by ARK"""
    if dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif dtype == torch.float32:
        return "fp32"
    else:
        raise ValueError(f"Unsupported dtype for ARK: {dtype}")


class INCXPULinearARKMethod(_INCXPULinearBase):
    """XPU linear method for INC quantization utilizing the ARK backend.

    Repacks GPTQ/INC weights into ARK's layout.
    """

    _set_layer_attrs_on_create = True
    _ark_instance: Any | None = None

    def __init__(self, weight_bits: int, group_size: int, sym: bool):
        super().__init__(weight_bits=weight_bits, group_size=group_size, sym=sym)

        self.weight_type = f"int{weight_bits}"
        self.asym = not sym

        self.ark = self._get_ark_instance()

    @classmethod
    def _get_ark_instance(cls):
        if cls._ark_instance is None:
            try:
                import auto_round_kernel

                ark_inst = auto_round_kernel._ark_instance()
                cls._ark_instance = ark_inst
            except ImportError as e:
                raise ImportError("Failed to import auto_round_kernel.") from e

        assert cls._ark_instance is not None
        return cls._ark_instance

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = layer.qweight.device

        compute_type = _get_ark_type_str(layer.params_dtype)
        scale_type = _get_ark_type_str(layer.scales.dtype)
        bits = self.weight_bits

        # ==========================================
        # Core logic: unpack vLLM int32-packed weights into the full int8
        # tensor layout expected by ARK.
        # ==========================================
        qw = layer.qweight.data  # Shape: [K // pack_factor, N]
        K_packed, N = qw.shape

        # Precompute bit shifts, e.g. int4 -> [0, 4, 8, 12, 16, 20, 24, 28].
        # int8 -> [0, 8, 16, 24]
        shifts = torch.arange(0, 32, bits, device=device)

        # 1. Broadcast shifts and apply bit masking to extract values in
        # the range [0, (1 << bits) - 1].
        # Shape change: [K_packed, N, 1] -> [K_packed, N, pack_factor]
        unpacked_w = (qw.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)

        # 2. Convert into an int8 container.
        unpacked_w = unpacked_w.to(torch.int8)

        # 3. Restore the sign bits.
        # For symmetric quantization, the real int4 range is [-8, 7].
        if self.sym:
            # INC/GPTQ symmetric quantization stores values with a default
            # offset of 2**(bits - 1). For INT4, stored_value = real_value + 8,
            # so decoding must subtract 8.
            offset = 1 << (bits - 1)  # For bits=4, offset = 8.
            unpacked_w = unpacked_w - offset

        # 4. Reorder dimensions to restore the real [K, N] layout.
        # transpose(1, 2) keeps values from the same packed group contiguous
        # along the K dimension.
        unpacked_w = unpacked_w.transpose(1, 2).reshape(-1, N).contiguous()

        scale = layer.scales.data.contiguous()

        if self.asym:
            qz = layer.qzeros.data  # [groups, N // pack_factor]
            groups, N_packed = qz.shape
            unpacked_z = (qz.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
            zp = unpacked_z.view(groups, -1).to(torch.int8).contiguous()
        else:
            # Per the ARK C++ implementation, symmetric quantization passes
            # an empty tensor here.
            zp = torch.empty(0, dtype=torch.int8, device=device)

        assert self.ark is not None
        packw = self.ark.repack_quantized_weight(
            unpacked_w,
            scale,
            zp,
            self.group_size,
            compute_type,
            self.weight_type,
            scale_type,
            self.asym,
        )

        # Wrap as a Parameter and disable gradients.
        layer.packed_weight = Parameter(packw, requires_grad=False)

        layer.compute_type = compute_type
        layer.scale_type = scale_type

        # Release the original temporary tensors loaded by vLLM.
        del layer.qweight
        del layer.scales
        del layer.qzeros
        if hasattr(layer, "g_idx"):
            del layer.g_idx

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Preserve the original shape and flatten the input to a 2D matrix
        # of shape [batch * seq_len, K].
        out_shape = x.shape[:-1] + (layer.out_features,)
        reshaped_x = x.reshape(-1, x.shape[-1]).contiguous()

        # Ensure bias has no grad requirement and is contiguous to avoid C++
        # pointer handling issues.
        if bias is not None:
            safe_bias = bias.detach().contiguous()
        else:
            safe_bias = torch.empty(0, dtype=x.dtype, device=x.device)

        assert self.ark is not None
        out = self.ark.woqgemm(
            reshaped_x,  # Input activations [M, K]
            layer.packed_weight,  # Repacked low-level weight buffer
            safe_bias,  # Bias [1, N] or empty
            layer.out_features,  # N
            layer.in_features,  # K
            self.group_size,  # Block size
            layer.compute_type,  # fp16 / bf16
            self.weight_type,  # int4
            layer.scale_type,  # fp16 / bf16
            self.asym,  # False
        )

        return out.reshape(out_shape)


class INCXPULinearMethod(_INCXPULinearBase):
    """XPU linear method for INC w4a16 GPTQ quantization (symmetric only).

    Repacks GPTQ weights from [in_packed, out] to oneDNN [out, in_packed]
    layout and calls torch.ops._xpu_C.int4_gemm_w4a16.

    GPTQ format: qweight [in_packed, out] with sequential nibble order.

    Note: Asymmetric quantization (sym=false) is not for now.

    FIXME(yiliu30): Refine the implementation to reuse XPUwNa16LinearKernel.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Repack GPTQ weights into kernel-ready NT layout."""
        device = layer.qweight.data.device

        # oneDNN int4 kernel requires strides[0]==1 ("NT format"), but GPTQ
        # checkpoint is [K_packed, N] contiguous with strides (N, 1).
        # Two transposes are needed — neither alone can achieve this:
        #   1. .t().contiguous() → [N, K_packed] contiguous in memory
        #   2. .t()              → [K_packed, N] view with strides (1, K_packed)
        # The result has the same logical shape but strides[0]==1 as required.
        qweight_ct = layer.qweight.data.t().contiguous()
        layer.qweight = Parameter(qweight_ct.t(), requires_grad=False)

        # Scales: [num_groups, out] — no change needed
        layer.scales = Parameter(layer.scales.data, requires_grad=False)

        # Symmetric: GPTQ v1 stores qzeros=7, effective zp = 7+1 = 8
        # Kernel expects int8 scalar = 8
        layer.qzeros = Parameter(
            torch.tensor([8], dtype=torch.int8, device=device),
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # qweight is already in NT layout [K_packed, N] (strides (1, K_packed))
        # from process_weights_after_loading — pass directly to kernel.
        out_shape = x.shape[:-1] + (layer.qweight.shape[1],)
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = torch.ops._xpu_C.int4_gemm_w4a16(
            reshaped_x,
            layer.qweight,
            bias,
            layer.scales,
            layer.qzeros,
            self.group_size,
            None,  # g_idx not needed: desc_act is always False for INC models
        )
        return out.reshape(out_shape)
