# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import torch
from packaging import version
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.kernels.mixed_precision import (
    BitBLASLinearKernel, MPLinearLayerConfig)
from vllm.model_executor.layers.quantization.utils.bitblas_utils import (
    BITBLAS_SUPPORTED_NUM_BITS as GPTQ_BITBLAS_SUPPORTED_NUM_BITS)
from vllm.model_executor.layers.quantization.utils.bitblas_utils import (
    BITBLAS_SUPPORTED_SYM as GPTQ_BITBLAS_SUPPORTED_SYM)
from vllm.model_executor.layers.quantization.utils.bitblas_utils import (
    MINIMUM_BITBLAS_VERSION, bitblas_repeat_scales_on_all_ranks,
    check_bitblas_supported, verify_bitblas_supported)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           RowvLLMParameter)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)


class GPTQBitBLASConfig(QuantizationConfig):
    """Config class for GPTQ BitBLAS"""

    # (num_bits, is_sym) -> quant_type
    TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
    }

    TORCH_DTYPE = torch.float16
    GPTQ_CKPT_STORAGE_DTYPE = (
        "int32"  # GPTQ Default Checkpoints use int32 as storage dtype
    )
    GPTQ_BITBLAS_STORAGE_DTYPE = "int8"  # BitBLAS uses int8 as storage dtype
    TORCH_BITBLAS_STORAGE_DTYPE = getattr(torch, GPTQ_BITBLAS_STORAGE_DTYPE)
    # "original" or "rescale" or "quantized",
    # the gptq_bitblas prefer "quantized"
    ZEROS_MODE = "quantized"

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        is_sym: bool,
        quant_method: Optional[str],
        lm_head_quantized: bool,
    ) -> None:

        try:
            import bitblas
            if version.parse(bitblas.__version__) < version.parse(
                    MINIMUM_BITBLAS_VERSION):
                raise ImportError(
                    "bitblas version is wrong. Please "
                    f"install bitblas>={MINIMUM_BITBLAS_VERSION}")
        except ImportError as e:
            bitblas_import_exception = e
            raise ValueError(
                "Trying to use the bitblas backend, but could not import"
                f"with the following error: {bitblas_import_exception}. "
                "Please install bitblas through the following command: "
                f"`pip install bitblas>={MINIMUM_BITBLAS_VERSION}`"
            ) from bitblas_import_exception

        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        super().__init__()
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.is_sym = is_sym
        self.quant_method = quant_method
        self.lm_head_quantized = lm_head_quantized

        # Verify
        if self.weight_bits not in GPTQ_BITBLAS_SUPPORTED_NUM_BITS:
            raise ValueError(
                f"BitBLAS does not support weight_bits = {self.weight_bits}. "
                f"Only weight_bits = {GPTQ_BITBLAS_SUPPORTED_NUM_BITS} "
                "are supported.")

        if self.is_sym not in GPTQ_BITBLAS_SUPPORTED_SYM:
            raise ValueError(
                f"BitBLAS does not support is_sym = {self.is_sym}. "
                f"Only sym = {GPTQ_BITBLAS_SUPPORTED_SYM} are supported.")

        self.storage_dtype = self.GPTQ_BITBLAS_STORAGE_DTYPE

        storage_nbit = int("".join(c for c in self.GPTQ_CKPT_STORAGE_DTYPE
                                   if c.isdigit()))

        # 4 Bits packed into 32 bit datatype.
        self.pack_factor = storage_nbit // weight_bits
        self.nbits = weight_bits

        # Zeros type for the quantized weights.
        self.zeros_mode = self.ZEROS_MODE

        if (weight_bits, is_sym) not in self.TYPE_MAP:
            raise ValueError("Unsupported quantization config: "
                             f"bits={weight_bits}, sym={is_sym}")

        self.quant_type = self.TYPE_MAP[(weight_bits, is_sym)]

    def __repr__(self) -> str:
        return (f"GPTQBitBLASConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act})"
                f"is_sym={self.is_sym}, "
                f"quant_method={self.quant_method})")

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "gptq_bitblas"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GPTQBitBLASConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        is_sym = cls.get_from_keys(config, ["sym"])
        quant_method = cls.get_from_keys(config, ["quant_method"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                 default=False)
        return cls(weight_bits, group_size, desc_act, is_sym, quant_method,
                   lm_head_quantized)

    @classmethod
    def override_quantization_method(
            cls, hf_quant_cfg, user_quant) -> Optional[QuantizationMethods]:
        can_convert = cls.is_gptq_bitblas_compatible(hf_quant_cfg)

        is_valid_user_quant = (user_quant is None or user_quant == "bitblas"
                               or user_quant == "gptq_bitblas")

        if can_convert and is_valid_user_quant:
            msg = ("The model is convertible to {} during runtime."
                   " Using {} kernel.".format(cls.get_name(), cls.get_name()))
            logger.info(msg)
            return cls.get_name()

        if can_convert and user_quant == "gptq":
            logger.info("Detected that the model can run with gptq_bitblas"
                        ", however you specified quantization=gptq explicitly,"
                        " so forcing gptq. Use quantization=gptq_bitblas for"
                        " faster inference")
        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["GPTQBitBLASLinearMethod"]:
        if isinstance(layer, LinearBase) or (isinstance(layer, ParallelLMHead)
                                             and self.lm_head_quantized):
            return GPTQBitBLASLinearMethod(self)
        return None

    @property
    def torch_storage_dtype(self) -> torch.dtype:
        return self.TORCH_BITBLAS_STORAGE_DTYPE

    @classmethod
    def is_gptq_bitblas_compatible(cls, quant_config: dict[str, Any]):
        # Extract data from quant config.
        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        sym = quant_config.get("sym")
        desc_act = quant_config.get("desc_act")

        # temporarily disable on ROCm platform
        if not current_platform.is_cuda():
            return False

        # If we cannot find the info needed in the config, cannot convert.
        if (num_bits is None or group_size is None or sym is None
                or desc_act is None):
            return False

        if (num_bits, sym) not in cls.TYPE_MAP:
            return False

        # If the capability of the device is too low, cannot convert.
        major, minor = torch.cuda.get_device_capability()
        device_capability = major * 10 + minor
        if device_capability < cls.get_min_capability():
            return False

        # Otherwise, can convert if model satisfies bitblas constraints.
        return check_bitblas_supported(quant_type=cls.TYPE_MAP[(num_bits,
                                                                sym)],
                                       group_size=group_size)


class GPTQBitBLASLinearMethod(LinearMethodBase):
    """Linear method for GPTQ BitBLAS.

    Args:
        quant_config: The GPTQ BitBLAS quantization config.
    """

    kernel_type = BitBLASLinearKernel
    _kernel_backends_being_used: set[str] = set()

    def __init__(self, quant_config: GPTQBitBLASConfig) -> None:
        self.quant_config = quant_config
        # Verify supported on platform.
        verify_bitblas_supported(quant_type=self.quant_config.quant_type,
                                 group_size=self.quant_config.group_size)

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
        """Creates quantized weights for use in linear operations.

        The function initializes and returns a dictionary containing 
        quantized weights, scales, and zeros
        for performing quantized matrix multiplication operations.

        Args:
            input_size_per_partition: The size of the input partition.
            output_partition_sizes: The size of the output partition.
            input_size: The total size of the input (unused).
            output_size: The total size of the output (unused).
            params_dtype: 
                The data type of the parameters (expected to be torch.float16).

        Returns:
            A dictionary containing the quantized weights ('qweight'), 
            scales ('scales'), and zeros ('zeros').

        Raises:
            ValueError: If `params_dtype` is not `torch.float16` or 
            if the input size per partition is not divisible by the 
            group size in `quant_config`.
        """
        if params_dtype != torch.float16:
            raise ValueError("Parameter data type must be torch.float16, "
                             f"but got {params_dtype}")

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        if input_size_per_partition % group_size != 0:
            raise ValueError(
                f"Input size per partition ({input_size_per_partition}) must "
                f"be divisible by group size ({self.quant_config.group_size})."
            )

        kernel_type = self.kernel_type
        # Validate output_size_per_partition
        output_size_per_partition = sum(output_partition_sizes)

        is_row_parallel = input_size != input_size_per_partition
        weight_loader = extra_weight_attrs.get("weight_loader")

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=\
                (input_size_per_partition, output_size_per_partition),
            weight_type=self.quant_config.quant_type,
            act_type=params_dtype,
            group_size=self.quant_config.group_size,
            zero_points=False,
            has_g_idx=self.quant_config.desc_act
        )

        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for GPTQBitBLASLinearMethod",
                        kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        # Determine sharding
        if bitblas_repeat_scales_on_all_ranks(self.quant_config.desc_act,
                                              self.quant_config.group_size,
                                              is_row_parallel):
            # By setting scale_dim == None, weight_loader will
            # repeat the scales on each GPU in TP>1 case.
            scales_and_zp_input_dim = None
            scales_and_zp_size = input_size // group_size
        else:
            # By setting scale_dim == 0, weight_loader will
            # shard the scales in TP>1 case.
            scales_and_zp_input_dim = 0
            scales_and_zp_size = input_size_per_partition // group_size

        # Init buffers
        # Quantized weights
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        # Activation order
        # Ignore warning from fused linear layers such as QKVParallelLinear.
        g_idx = RowvLLMParameter(data=torch.empty(
            input_size_per_partition,
            dtype=torch.int32,
        ),
                                 input_dim=0,
                                 weight_loader=weight_loader)

        # Scales
        scales = Parameter(
            torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                **extra_weight_attrs,
                "input_dim": scales_and_zp_input_dim,
                "output_dim": 1,
            },
        )

        # Quantized zero-points
        qzeros_args = {
            "data":
            torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader":
            weight_loader
        }
        weight_scale_args = {
            "data":
            torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader":
            weight_loader
        }

        if scales_and_zp_input_dim is None:
            scales = ChannelQuantScaleParameter(output_dim=1,
                                                **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args)

        else:
            scales = GroupQuantScaleParameter(output_dim=1,
                                              input_dim=0,
                                              **weight_scale_args)
            qzeros = PackedvLLMParameter(
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)

        self.kernel = kernel_type(
            mp_linear_kernel_config,
            w_q_param_name="qweight",
            w_s_param_name="scales",
            w_zp_param_name="qzeros",
            w_gidx_param_name="g_idx",
            bitblas_quant_config=self.quant_config,
        )

        # Initialize or retrieve the BitBLAS matrix multiplication operator.
        self.kernel.configure_bitblas_matmul(
            input_size_per_partition,
            output_size_per_partition,
            params_dtype=params_dtype,
            bias=False,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.kernel.apply_gptq_bitblas_linear(layer, x)
        if bias is not None:
            out.add_(bias)
        return out
