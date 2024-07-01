import enum
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

import bitblas.cache
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    set_weight_attrs,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

logger = init_logger(__name__)


try:
    import bitblas
except ImportError as e:
    bitblas_import_exception = e
    raise ValueError(
        f"Trying to use the bitblas backend, but could not import dependencies with the following error: {bitblas_import_exception}"
    )

import bitblas
from bitblas.utils import auto_detect_nvidia_target

bitblas.set_log_level("Debug")
BITBLAS_TARGET = auto_detect_nvidia_target()
BITBLAS_DATABASE_PATH = bitblas.cache.get_database_path()

GPTQ_BITBLAS_SUPPORTED_NUM_BITS = [1, 2, 4, 8]
GPTQ_BITBLAS_SUPPORTED_SYM = [False, True]


def unpack_qzeros(qzeros, bits):
    qzeros = qzeros.view(torch.int32)
    elems_per_int32 = 32 // bits
    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * elems_per_int32),
        dtype=torch.int8,
        device=qzeros.device,
        requires_grad=False,
    )

    for col in range(unpacked_zeros.shape[1]):
        i = col % elems_per_int32
        unpacked_zeros[:, col] = (qzeros[:, col // elems_per_int32] >> (bits * i)) & 0xF

    return unpacked_zeros + 1



class GPTQBitBLASConfig(QuantizationConfig):
    """Config class for GPTQ BitBLAS"""

    TORCH_DTYPE = torch.float16
    GPTQ_CKPT_STORAGE_DTYPE = "int32"  # GPTQ Default Checkpoints use int32 as storage dtype
    GPTQ_BITBLAS_STORAGE_DTYPE = "int8"  # BitBLAS uses int8 as storage dtype    
    TORCH_BITBLAS_STORAGE_DTYPE = getattr(torch, GPTQ_BITBLAS_STORAGE_DTYPE)
    ZEROS_MODE = "quantized"  # "original" or "rescale" or "quantized", the gptq_bitblas prefer "quantized"

    def __init__(
        self, weight_bits: int, group_size: int, desc_act: bool, is_sym: bool
    ) -> None:
        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.is_sym = is_sym

        # Verify
        if self.weight_bits not in GPTQ_BITBLAS_SUPPORTED_NUM_BITS:
            raise ValueError(
                f"BitBLAS does not support weight_bits = {self.weight_bits}. "
                f"Only weight_bits = {GPTQ_BITBLAS_SUPPORTED_NUM_BITS} "
                "are supported."
            )

        if self.is_sym not in GPTQ_BITBLAS_SUPPORTED_SYM:
            raise ValueError(
                f"BitBLAS does not support is_sym = {self.is_sym}. "
                f"Only sym = {GPTQ_BITBLAS_SUPPORTED_SYM} are supported."
            )

        
        self.storage_dtype = self.GPTQ_BITBLAS_STORAGE_DTYPE

        storage_nbit = int("".join(c for c in self.GPTQ_CKPT_STORAGE_DTYPE if c.isdigit()))

        # 4 Bits packed into 32 bit datatype.
        self.pack_factor = storage_nbit // weight_bits
        self.nbits = weight_bits

        # Zeros type for the quantized weights.
        self.zeros_mode = self.ZEROS_MODE

    def __repr__(self) -> str:
        return (
            f"GPTQBitBLASConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "gptq_bitblas"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQBitBLASConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        is_sym = cls.get_from_keys(config, ["sym"])
        return cls(weight_bits, group_size, desc_act, is_sym)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        can_convert = cls.is_bitblas_compatible(hf_quant_cfg)

        is_valid_user_quant = user_quant is None or user_quant == "bitblas"

        if can_convert and is_valid_user_quant:
            msg = (
                "The model is convertible to {} during runtime."
                " Using {} kernel.".format(cls.get_name(), cls.get_name())
            )
            logger.info(msg)
            return cls.get_name()

        if can_convert and user_quant == "gptq":
            logger.info(
                "Detected that the model can run with gptq_bitblas"
                ", however you specified quantization=gptq explicitly,"
                " so forcing gptq. Use quantization=gptq_bitblas for"
                " faster inference"
            )
        return None

    def get_quant_method(
        self, layer: torch.nn.Module
    ) -> Optional["GPTQBitBLASLinearMethod"]:
        if isinstance(layer, LinearBase):
            return GPTQBitBLASLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def is_bitblas_compatible(cls, quant_config: Dict[str, Any]):
        # Extract data from quant config.
        num_bits = quant_config.get("bits", None)
        group_size = quant_config.get("group_size", None)
        sym = quant_config.get("sym", None)
        desc_act = quant_config.get("desc_act", None)

        # If we cannot find the info needed in the config, cannot convert.
        if num_bits is None or group_size is None or sym is None or desc_act is None:
            return False

        # If the capability of the device is too low, cannot convert.
        major, minor = torch.cuda.get_device_capability()
        device_capability = major * 10 + minor
        if device_capability < cls.get_min_capability():
            return False

        # Otherwise, can convert if model satisfies bitblas constraints.
        return (
            num_bits in GPTQ_BITBLAS_SUPPORTED_NUM_BITS
            and sym in GPTQ_BITBLAS_SUPPORTED_SYM
        )


class GPTQBitBLASState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


class GPTQBitBLASLinearMethod(LinearMethodBase):
    """Linear method for GPTQ BitBLAS.

    Args:
        quant_config: The GPTQ BitBLAS quantization config.
    """

    OPT_FEATURES = [1, 16, 32, 64, 128, 256, 512]
    ENABLE_TUNING = True
    BITBLAS_DTYPES = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.half: "float16",
        torch.int8: "int8",
    }

    def __init__(self, quant_config: GPTQBitBLASConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Creates quantized weights for use in linear operations.

        The function initializes and returns a dictionary containing quantized weights, scales, and zeros
        for performing quantized matrix multiplication operations.

        Args:
            input_size_per_partition: The size of the input partition.
            output_partition_sizes: The size of the output partition.
            input_size: The total size of the input (unused).
            output_size: The total size of the output (unused).
            params_dtype: The data type of the parameters (expected to be torch.float16).

        Returns:
            A dictionary containing the quantized weights ('qweight'), scales ('scales'), and zeros ('zeros').

        Raises:
            ValueError: If `params_dtype` is not `torch.float16` or if the input size per partition
                        is not divisible by the group size in `quant_config`.
        """
        del output_size  # Unused arguments.
        if params_dtype != torch.float16:
            raise ValueError(
                f"Parameter data type must be torch.float16, but got {params_dtype}"
            )

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        if input_size_per_partition % group_size != 0:
            raise ValueError(
                f"Input size per partition ({input_size_per_partition}) must be divisible by "
                f"group size ({self.quant_config.group_size})."
            )

        # Validate output_size_per_partition
        output_size_per_partition = sum(output_partition_sizes)
        # By default, no sharding over "input dim"
        scales_and_zp_size = input_size // group_size
        scales_and_zp_input_dim = None
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

        # Init buffers
        # Quantized weights
        qweight = Parameter(
            torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                **extra_weight_attrs,
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        # Activation order
        g_idx = Parameter(
            torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        # Ignore warning from fused linear layers such as QKVParallelLinear.
        set_weight_attrs(
            g_idx,
            {**extra_weight_attrs, "input_dim": 0, "ignore_warning": True},
        )

        g_idx_sort_indices = torch.empty(
            g_idx.shape,
            dtype=torch.int32,
        )

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
        qzeros = Parameter(
            torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros,
            {
                **extra_weight_attrs,
                "input_dim": scales_and_zp_input_dim,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            },
        )
        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)
        layer.g_idx_sort_indices = g_idx_sort_indices
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.input_size = input_size
        layer.bitblas_state = GPTQBitBLASState.REPACK

    def _configure_bitblas_matmul(
        self,
        infeatures,
        outfeatures,
        params_dtype,
        enable_tuning,
        bias,
        layout,
        bits,
    ):
        from bitblas import MatmulConfig

        bitblas_dtype = self.BITBLAS_DTYPES[params_dtype]

        W_dtype = f"uint{bits}"

        matmul_config = MatmulConfig(
            M=self.OPT_FEATURES,
            N=outfeatures,
            K=infeatures,
            A_dtype=bitblas_dtype,
            W_dtype=W_dtype,
            out_dtype=bitblas_dtype,
            accum_dtype="int32" if bitblas_dtype == "int8" else bitblas_dtype,
            storage_dtype=self.quant_config.GPTQ_BITBLAS_STORAGE_DTYPE,
            with_scaling=True,
            with_zeros=True,
            group_size=self.quant_config.group_size,
            with_bias=bias,
            layout=layout,
            zeros_mode=self.quant_config.zeros_mode,
        )
        self.bitblas_matmul = self._get_or_create_bitblas_operator(
            matmul_config, enable_tuning
        )

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        from bitblas import Matmul
        from bitblas.cache import global_operator_cache

        if global_operator_cache.size() == 0:
            global_operator_cache.load_from_database(
                BITBLAS_DATABASE_PATH, BITBLAS_TARGET
            )

        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            bitblas_matmul = Matmul(config, target=BITBLAS_TARGET)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(
                    BITBLAS_DATABASE_PATH, BITBLAS_TARGET
                )
                print(
                    "BitBLAS Tuning done, appended operator to global_operator_cache."
                )
            else:
                print(f"BitBLAS Operator {config} created.")
        else:
            print(f"BitBLAS Operator {config} found in global_operator_cache.")
        return bitblas_matmul
    
    def repack_bitblas_from_gptq(self, b_q_weight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor):
        from bitblas.quantization.utils import general_compress

        # qweight in gptq old quant linear stored with (outfeatures, infeatures), should be transposed.
        qweight = b_q_weight.T.contiguous().view(self.quant_config.TORCH_BITBLAS_STORAGE_DTYPE)
        if self.bitblas_matmul.weight_transform is not None:
            qweight = self.bitblas_matmul.weight_transform(qweight.cpu()).cuda()
        # scales in gptq old quant linear stored with (infeatures // group_size, outfeatures), should be transposed.
        scales = scales.T.contiguous()
        # qzeros should be de-quantized to int zeros.
        intzeros = unpack_qzeros(qzeros, self.quant_config.weight_bits).T.contiguous()
        zeros = None
        if self.bitblas_matmul.config.zeros_mode == "original":
            zeros = intzeros.to(torch.float16).contiguous()
        elif self.bitblas_matmul.config.zeros_mode == "rescale":
            zeros[:, :] = intzeros.to(torch.float16)[:, :] * scales[:, :]
        elif self.bitblas_matmul.config.zeros_mode == "quantized":
            zeros = (
                torch.Tensor(
                    general_compress(intzeros.T.contiguous().cpu().numpy(), self.quant_config.weight_bits)
                )
                .to(qweight.device)
                .to(self.quant_config.TORCH_BITBLAS_STORAGE_DTYPE)
                .contiguous()
            )
        else:
            raise ValueError(
                f"Unsupported zeros type: {self.bitblas_matmul.config.zeros_mode}"
            )

        return qweight, scales, zeros

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        part_size_n = layer.output_size_per_partition
        out_shape = x.shape[:-1] + (part_size_n,)

        if layer.bitblas_state == GPTQBitBLASState.REPACK:
            layer.bitblas_state = GPTQBitBLASState.READY

            # Newly generated tensors need to replace existing tensors that are
            # already registered as parameters by vLLM (and won't be freed)
            def replace_tensor(name, new_t):
                # It is important to use copy_() here since it ensures
                # the same buffer is reused
                getattr(layer, name).copy_(new_t.view(getattr(layer, name).dtype).view(getattr(layer, name).shape))
                del new_t

            # Repack weights
            bitblas_qweight, bitblas_scales, bitblas_qzeros = self.repack_bitblas_from_gptq(
                layer.qweight,
                layer.scales,
                layer.qzeros,
            )
            replace_tensor("qweight", bitblas_qweight)
            replace_tensor("scales", bitblas_scales)
            replace_tensor("qzeros", bitblas_qzeros)

        output = self.bitblas_matmul(x, layer.qweight, layer.scales, layer.qzeros)

        if bias is not None:
            output.add_(bias)  # In-place add
        return output.reshape(out_shape)
