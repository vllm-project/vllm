# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import torch
from packaging import version

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization import (
    QuantizationConfig,
    QuantizationMethods,
)
from vllm.model_executor.layers.quantization.utils.bitblas_utils import (
    BITBLAS_OPTIMIZE_FEATURES,
    BITBLAS_SUPPORTED_NUM_BITS,
    BITBLAS_SUPPORTED_SYM,
    MINIMUM_BITBLAS_VERSION,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from vllm.model_executor.utils import set_weight_attrs

logger = init_logger(__name__)


class BitBLASConfig(QuantizationConfig):
    """Config class for BitBLAS.

    Reference: https://github.com/Microsoft/BitBLAS
    """

    TORCH_DTYPE = torch.float16
    STORAGE_DTYPE = "int8"  # assume int8 storage
    TORCH_STORAGE_DTYPE = getattr(torch, STORAGE_DTYPE)
    # "original" or "rescale" or "quantized",
    # gptq_with_bitblas prefer "quantized implementation"
    ZEROS_MODE = "quantized"

    def __init__(
        self,
        weight_bits: int,
        group_size: int | None,
        desc_act: bool | None,
        is_sym: bool | None,
        quant_method: str | None,
        lm_head_quantized: bool,
    ) -> None:
        try:
            import bitblas

            if version.parse(bitblas.__version__) < version.parse(
                MINIMUM_BITBLAS_VERSION
            ):
                raise ImportError(
                    "bitblas version is wrong. Please "
                    f"install bitblas>={MINIMUM_BITBLAS_VERSION}"
                )
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
        if self.weight_bits not in BITBLAS_SUPPORTED_NUM_BITS:
            raise ValueError(
                f"BitBLAS does not support weight_bits = {self.weight_bits}. "
                f"Only weight_bits = {BITBLAS_SUPPORTED_NUM_BITS} "
                "are supported."
            )

        if self.is_sym not in BITBLAS_SUPPORTED_SYM:
            raise ValueError(
                f"BitBLAS does not support is_sym = {self.is_sym}. "
                f"Only sym = {BITBLAS_SUPPORTED_SYM} are supported."
            )

        storage_dtype = self.STORAGE_DTYPE
        storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))

        self.storage_dtype = storage_dtype
        self.storage_torch_dtype = self.TORCH_STORAGE_DTYPE
        # 4 Bits packed into 32 bit datatype.
        self.pack_factor = storage_nbit // weight_bits
        self.nbits = weight_bits

        # Zeros type for the quantized weights.
        self.zeros_mode = self.ZEROS_MODE

    def __repr__(self) -> str:
        return (
            f"BitBLASConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}, "
            f"is_sym={self.is_sym}, "
            f"quant_method={self.quant_method})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "bitblas"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @staticmethod
    def get_from_keys(
        config: dict[str, Any], keys: list[str], default: Any = None
    ) -> Any:
        """Get a value from the model's quantization config."""
        for key in keys:
            if key in config:
                return config[key]
        return default

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BitBLASConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"], -1)
        desc_act = cls.get_from_keys(config, ["desc_act"], False)
        is_sym = cls.get_from_keys(config, ["sym"], False)
        quant_method = cls.get_from_keys(config, ["quant_method"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        return cls(
            weight_bits, group_size, desc_act, is_sym, quant_method, lm_head_quantized
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None:
        # compat: autogptq >=0.8.0 use checkpoint_format: str
        # compat: autogptq <=0.7.1 is_bitblas_format: bool
        is_bitblas_format = hf_quant_cfg.get(
            "checkpoint_format"
        ) == "bitblas" or hf_quant_cfg.get("is_bitblas_format", False)

        is_valid_user_quant = (
            user_quant is None or user_quant == "gptq" or user_quant == "bitblas"
        )

        if is_bitblas_format and is_valid_user_quant:
            msg = "The model is serialized in {} format. Using {} kernel.".format(
                cls.get_name(), cls.get_name()
            )
            logger.info(msg)
            return cls.get_name()

        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["BitBLASLinearMethod"]:
        if isinstance(layer, LinearBase) or (
            isinstance(layer, ParallelLMHead) and self.lm_head_quantized
        ):
            return BitBLASLinearMethod(self)
        return None


class BitBLASLinearMethod(LinearMethodBase):
    """Linear method for BitBLAS.

    Args:
        quant_config: The BitBLAS quantization config.
    """

    # USE BITBLAS_OPTIMIZE_FEATURES_CONTIGUOUS
    # Instead of BITBLAS_OPTIMIZE_FEATURES
    # If you want to high contiguous batching
    # performance
    OPT_FEATURES = BITBLAS_OPTIMIZE_FEATURES
    ENABLE_TUNING = True
    BITBLAS_DTYPES = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.half: "float16",
        torch.int8: "int8",
    }

    def __init__(self, quant_config: BitBLASConfig):
        self.quant_config = quant_config

    def create_weights_gptq(
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

        The function initializes and returns a dictionary containing quantized
        weights, scales, and zeros
        for performing quantized matrix multiplication operations.

        Args:
            input_size_per_partition: The size of the input partition.
            output_partition_sizes: List of output partition sizes.
            input_size: The total size of the input (unused).
            output_size: The total size of the output (unused).
            params_dtype:
                The data type of the parameters (expected to be torch.float16).

        Returns:
            A dictionary containing the quantized weights ('qweight'),
            scales ('scales'), and zeros ('zeros').

        Raises:
            ValueError: If `params_dtype` is not `torch.float16` or if the input
                size per partition is not divisible by the group size
                in `quant_config`.
        """
        del input_size, output_size  # Unused arguments.
        weight_loader = extra_weight_attrs["weight_loader"]

        if params_dtype not in self.quant_config.get_supported_act_dtypes():
            raise ValueError(
                f"Parameter data type must be torch.float16, but got {params_dtype}"
            )
        group_size = self.quant_config.group_size
        if group_size is None:
            group_size = -1
        # Validate output_size_per_partition
        output_size_per_partition = sum(output_partition_sizes)
        if group_size != -1 and input_size_per_partition % group_size != 0:
            raise ValueError(
                f"Input size per partition ({input_size_per_partition}) must "
                f"be divisible by group size ({group_size})."
            )

        # Initialize or retrieve the BitBLAS matrix multiplication operator.
        self._configure_bitblas_matmul(
            input_size_per_partition,
            output_size_per_partition,
            params_dtype=params_dtype,
            enable_tuning=self.ENABLE_TUNING,
            bias=False,
            layout="nt",
            bits=self.quant_config.weight_bits,
        )

        # Initialize quantized weights with dimensions
        # Quantized 4Bit weights packed.
        qweight = PackedvLLMParameter(
            data=torch.empty(
                self.bitblas_matmul.retrieve_weight_shape(),
                device="cuda",
                dtype=self.quant_config.storage_torch_dtype,
                requires_grad=False,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=self.quant_config.pack_factor,
            bitblas_tile_size=(
                self.bitblas_matmul.retrieve_weight_shape()[-2]
                if self.bitblas_matmul.propagate_b
                else None
            ),
            weight_loader=weight_loader,
        )

        # Compute the number of input groups for channel-wise quantization.
        input_groups = 1 if group_size == -1 else input_size_per_partition // group_size

        # Initialize scales and zeros for the quantized weights.
        weight_scale_args = {
            "data": torch.empty(
                output_size_per_partition,
                input_groups,
                device="cuda",
                dtype=params_dtype,
            ),
            "weight_loader": weight_loader,
        }
        if input_groups == 1:
            scales = ChannelQuantScaleParameter(output_dim=0, **weight_scale_args)
        else:
            scales = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **weight_scale_args
            )

        if self.quant_config.zeros_mode == "quantized":
            zeros = PackedvLLMParameter(
                data=torch.empty(
                    input_groups,
                    output_size_per_partition // self.quant_config.pack_factor,
                    device="cuda",
                    dtype=self.quant_config.storage_torch_dtype,
                    requires_grad=False,
                ),
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                weight_loader=weight_loader,
            )

        else:
            zeros = BasevLLMParameter(
                torch.empty(
                    output_size_per_partition,
                    input_groups,
                    device="cuda",
                    dtype=params_dtype,
                ),
                weight_loader=weight_loader,
            )
            # Set attributes to indicate how scales and zeros are applied.
            set_weight_attrs(
                zeros,
                {
                    "input_dim": None if input_groups == 1 else 1,
                    "output_dim": 0,
                },
            )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("zeros", zeros)

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
        if self.quant_config.quant_method == "gptq":
            return self.create_weights_gptq(
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                **extra_weight_attrs,
            )
        else:
            raise ValueError(
                f"Unsupported quant_method {self.quant_config.quant_method}"
            )

    def _configure_bitblas_matmul(
        self,
        infeatures,
        outfeatures,
        params_dtype,
        enable_tuning,
        bias,
        layout,
        bits,
        out_dtype="float16",
    ):
        from bitblas import MatmulConfig

        bitblas_dtype = self.BITBLAS_DTYPES[params_dtype]

        with_scaling = False
        with_zeros = False
        group_size = self.quant_config.group_size
        zeros_mode = self.quant_config.zeros_mode
        if self.quant_config.quant_method == "gptq":
            with_scaling = True
            with_zeros = True
            W_dtype = f"uint{bits}"
            if self.quant_config.is_sym:
                with_zeros = False
                W_dtype = f"int{bits}"
        else:
            raise ValueError(
                f"Unsupported quant_method {self.quant_config.quant_method}"
            )

        matmul_config = MatmulConfig(
            N=outfeatures,
            K=infeatures,
            A_dtype=bitblas_dtype,
            W_dtype=W_dtype,
            out_dtype=out_dtype,
            accum_dtype="int32" if bitblas_dtype == "int8" else bitblas_dtype,
            storage_dtype=self.quant_config.STORAGE_DTYPE,
            with_scaling=with_scaling,
            with_zeros=with_zeros,
            group_size=group_size,
            with_bias=bias,
            layout=layout,
            zeros_mode=zeros_mode,
        )
        self.bitblas_matmul = self._get_or_create_bitblas_operator(
            matmul_config, enable_tuning
        )

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        from bitblas import Matmul, auto_detect_nvidia_target
        from bitblas.cache import get_database_path, global_operator_cache

        BITBLAS_DATABASE_PATH = get_database_path()
        BITBLAS_TARGET = auto_detect_nvidia_target()
        if global_operator_cache.size() == 0:
            global_operator_cache.load_from_database(
                BITBLAS_DATABASE_PATH, BITBLAS_TARGET
            )

        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            bitblas_matmul = Matmul(config, target=BITBLAS_TARGET, enable_tuning=False)
            if enable_tuning:
                TUNING_MESSAGE = f"BitBLAS Operator {config} is tuning ..."
                logger.info(TUNING_MESSAGE)
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(
                    BITBLAS_DATABASE_PATH, BITBLAS_TARGET
                )
                TUNED_MESSAGE = (
                    f"BitBLAS Operator {config} tuned and saved to database."
                )
                logger.info(TUNED_MESSAGE)
            else:
                _message = f"BitBLAS Operator {config} created."
                logger.info(_message)
        else:
            _message = f"BitBLAS Operator {config} found in global_operator_cache."
            logger.info(_message)
        return bitblas_matmul

    def apply_gptq(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        scales = layer.scales
        qzeros = layer.zeros

        x_2d = x.view(-1, x.shape[-1])

        if self.quant_config.is_sym:
            output_2d = self.bitblas_matmul(x_2d, qweight, scales)
        else:
            output_2d = self.bitblas_matmul(x_2d, qweight, scales, qzeros)

        output = output_2d.view(x.shape[:-1] + (output_2d.shape[1],))

        if bias is not None:
            output.add_(bias)  # In-place add

        return output

    def apply(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.quant_config.quant_method == "gptq":
            return self.apply_gptq(*args, **kwargs)
        else:
            raise ValueError(
                f"Unsupported quant_method {self.quant_config.quant_method}"
            )
