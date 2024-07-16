import enum
from enum import Enum
from typing import Any, Dict, List, Optional

import bitblas.cache
import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizationConfig)

logger = init_logger(__name__)

try:
    import bitblas
    from bitblas.utils import auto_detect_nvidia_target
except ImportError as e:
    bitblas_import_exception = e
    error_message = (
        "Trying to use the bitblas backend, but could not import dependencies "
        f"with the following error: {bitblas_import_exception}")
    raise ValueError(error_message) from bitblas_import_exception

bitblas.set_log_level("Debug")
BITBLAS_TARGET = auto_detect_nvidia_target()
BITBLAS_DATABASE_PATH = bitblas.cache.get_database_path()

BITNET_BITBLAS_SUPPORTED_NUM_BITS = [1, 2, 4, 8]


class BITNETBitBLASConfig(QuantizationConfig):
    """Config class for BITNET BitBLAS"""

    TORCH_DTYPE = torch.int8
    BITNET_CKPT_STORAGE_DTYPE = (
        "float16"  # BITNET Default Checkpoints use float16 as storage dtype
    )
    BITNET_BITBLAS_STORAGE_DTYPE = "int8"  # BitBLAS uses int8 as storage dtype
    TORCH_BITBLAS_STORAGE_DTYPE = getattr(torch, BITNET_BITBLAS_STORAGE_DTYPE)

    def __init__(self, weight_bits: int, is_sym: bool) -> None:
        self.input_bits = 8
        self.weight_bits = weight_bits
        self.is_sym = is_sym

        # Verify
        if self.weight_bits not in BITNET_BITBLAS_SUPPORTED_NUM_BITS:
            raise ValueError(
                f"BitBLAS does not support weight_bits = {self.weight_bits}. "
                f"Only weight_bits = {BITNET_BITBLAS_SUPPORTED_NUM_BITS} "
                "are supported.")

        self.storage_dtype = self.BITNET_BITBLAS_STORAGE_DTYPE
        self.nbits = weight_bits

    def __repr__(self) -> str:
        return (f"BITNETBitBLASConfig(weight_bits={self.weight_bits}, "
                f"is_sym={self.is_sym})")

    @classmethod
    def get_name(cls) -> str:
        return "bitnet_bitblas"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.int8]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BITNETBitBLASConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        is_sym = cls.get_from_keys(config, ["sym"])
        return cls(weight_bits, is_sym)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        can_convert = cls.is_bitblas_compatible(hf_quant_cfg)

        is_valid_user_quant = user_quant is None or user_quant == "bitblas"

        if can_convert and is_valid_user_quant:
            msg = ("The model is convertible to {} during runtime."
                   " Using {} kernel.".format(cls.get_name(), cls.get_name()))
            logger.info(msg)
            return cls.get_name()

        if can_convert and user_quant == "bitnet":
            logger.info(
                "Detected that the model can run with bitnet_bitblas"
                ", however you specified quantization=bitnet explicitly,"
                " so forcing bitnet. Use quantization=bitnet_bitblas for"
                " faster inference")
        return None

    def get_quant_method(
            self,
            layer: torch.nn.Module) -> Optional["BITNETBitBLASLinearMethod"]:
        if isinstance(layer, LinearBase):
            return BITNETBitBLASLinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def is_bitblas_compatible(cls, quant_config: Dict[str, Any]):
        # Extract data from quant config.
        num_bits = quant_config.get("bits", None)
        sym = quant_config.get("sym", None)

        # If we cannot find the info needed in the config, cannot convert.
        if num_bits is None or sym is None:
            return False

        # If the capability of the device is too low, cannot convert.
        major, minor = torch.cuda.get_device_capability()
        device_capability = major * 10 + minor
        if device_capability < cls.get_min_capability():
            return False

        # Otherwise, can convert if model satisfies bitblas constraints.
        return num_bits in BITNET_BITBLAS_SUPPORTED_NUM_BITS


class BITNETBitBLASState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


class BITNETBitBLASLinearMethod(LinearMethodBase):
    """Linear method for BITNET BitBLAS.

    Args:
        quant_config: The BITNET BitBLAS quantization config.
    """

    ENABLE_TUNING = True
    BITBLAS_DTYPES = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.half: "float16",
        torch.int8: "int8",
    }

    def __init__(self, quant_config: BITNETBitBLASConfig) -> None:
        self.quant_config = quant_config
        self.Qp = 2**(quant_config.input_bits - 1) - 1

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

        The function initializes and returns a dictionary containing 
        quantized weights, scales, and zeros for performing quantized 
        matrix multiplication operations.

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
            ValueError: If `params_dtype` is not `torch.float16` or if the 
            input size per partition is not divisible by the group size 
            in `quant_config`.
        """
        del output_size  # Unused arguments.
        if params_dtype != torch.float16:
            raise ValueError("Parameter data type must be torch.float16, "
                             f"but got {params_dtype}")

        # Validate output_size_per_partition
        output_size_per_partition = sum(output_partition_sizes)
        bitblas_dtype = "int8"
        # Initialize or retrieve the BitBLAS matrix multiplication operator.
        self._configure_bitblas_matmul(
            input_size_per_partition,
            output_size_per_partition,
            bitblas_dtype=bitblas_dtype,
            enable_tuning=self.ENABLE_TUNING,
            bias=False,
            layout="nt",
            bits=self.quant_config.weight_bits,
        )

        # Init buffers
        # Quantized weights
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight,
            {
                **extra_weight_attrs,
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        qweight = Parameter(
            torch.empty(
                *self.bitblas_matmul.retrieve_weight_shape(),
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                **extra_weight_attrs,
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        layer.register_parameter("weight", weight)
        layer.register_parameter("qweight", qweight)
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.input_size = input_size
        layer.bitblas_state = BITNETBitBLASState.REPACK

    def _configure_bitblas_matmul(
        self,
        infeatures,
        outfeatures,
        bitblas_dtype,
        enable_tuning,
        bias,
        layout,
        bits,
    ):
        from bitblas import MatmulConfig

        W_dtype = f"int{bits}"

        matmul_config = MatmulConfig(
            N=outfeatures,
            K=infeatures,
            A_dtype=bitblas_dtype,
            W_dtype=W_dtype,
            out_dtype="float32",
            accum_dtype="int32" if bitblas_dtype == "int8" else bitblas_dtype,
            storage_dtype=self.quant_config.BITNET_BITBLAS_STORAGE_DTYPE,
            with_scaling=False,
            with_zeros=False,
            with_bias=bias,
            layout=layout,
        )
        self.bitblas_matmul = self._get_or_create_bitblas_operator(
            matmul_config, enable_tuning)

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        from bitblas import Matmul
        from bitblas.cache import global_operator_cache

        if global_operator_cache.size() == 0:
            global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH,
                                                     BITBLAS_TARGET)

        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            bitblas_matmul = Matmul(config,
                                    target=BITBLAS_TARGET,
                                    enable_tuning=False)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(
                    BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
                logger.info("BitBLAS Tuning done, appended operator to "
                            "global_operator_cache.")
            else:
                _message = (
                    f"BitBLAS Operator {config} created without tuning. ")
                logger.info(_message)
        else:
            _message = (f"BitBLAS Operator {config} retrieved from cache.")
            logger.info(_message)
        return bitblas_matmul

    def weight_quant(self, weight):
        weight = weight.float()
        s = 1 / weight.abs().mean().clamp(min=1e-5)
        result = (weight * s).round().clamp(-1, 1)
        return result.type(torch.int8)

    def activation_quant(self, x, num_bits=8):
        x = x.float()
        Qn = -(2**(num_bits - 1))
        Qp = 2**(num_bits - 1) - 1
        s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (x * s).round().clamp(Qn, Qp)
        return result.type(torch.int8)

    def repack_bitblas_from_bitnet(self,
                                   b_q_weight: torch.Tensor,
                                   is_qkv_packed: bool = False):
        if is_qkv_packed:
            hidden_size = b_q_weight.size(0)
            sw_q = 1 / b_q_weight[:hidden_size //
                                  3].abs().mean().clamp(min=1e-5)
            sw_k = 1 / b_q_weight[hidden_size // 3:2 * hidden_size //
                                  3].abs().mean().clamp(min=1e-5)
            sw_v = 1 / b_q_weight[2 * hidden_size //
                                  3:].abs().mean().clamp(min=1e-5)
            self.sw_q = sw_q
            self.sw_k = sw_k
            self.sw_v = sw_v
            qweight_q = self.weight_quant(b_q_weight[:hidden_size //
                                                     3]).detach()
            qweight_k = self.weight_quant(
                b_q_weight[hidden_size // 3:2 * hidden_size // 3]).detach()
            qweight_v = self.weight_quant(b_q_weight[2 * hidden_size //
                                                     3:]).detach()
            qweight = torch.cat([qweight_q, qweight_k, qweight_v], dim=0)
        else:
            sw = 1 / b_q_weight.abs().mean().clamp(min=1e-5)
            self.sw = sw
            qweight = self.weight_quant(b_q_weight).detach()
        qweight = self.bitblas_matmul.transform_weight(qweight)
        return qweight

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        part_size_n = layer.output_size_per_partition
        out_shape = x.shape[:-1] + (part_size_n, )
        quant_input = self.activation_quant(
            x, self.quant_config.input_bits).detach()

        if layer.bitblas_state == BITNETBitBLASState.REPACK:
            layer.bitblas_state = BITNETBitBLASState.READY

            # Newly generated tensors need to replace existing tensors that are
            # already registered as parameters by vLLM (and won't be freed)
            def free_tensor(name):
                # free the original weight tensor
                delattr(layer, name)

            def replace_tensor(name, new_t):
                # Cannot use copy_() as gptq because the storage
                # shape and dtype are different
                delattr(layer, name)
                setattr(layer, name, new_t)

            # Repack weights
            bitblas_qweight = self.repack_bitblas_from_bitnet(layer.weights)
            # free the original weight tensor
            free_tensor("weight")
            replace_tensor("qweight", bitblas_qweight)

        fp32_out = self.bitblas_matmul(quant_input, layer.qweight)
        sw = self.sw
        Qp = self.Qp
        si = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        # if / (si * sw) it will inf in some cases
        output = fp32_out / si
        output = output / sw
        output = output.half()
        output = output.type(x.dtype)
        if bias is not None:
            output.add_(bias)  # In-place add

        return output.reshape(out_shape)
