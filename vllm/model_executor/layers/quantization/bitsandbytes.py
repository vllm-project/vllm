# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Union

import torch
from packaging import version

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_router import FusedMoERouter
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
    set_weight_attrs,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig,
    QuantizationMethods,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


class BitsAndBytesConfig(QuantizationConfig):
    """Config class for BitsAndBytes Quantization.

    Reference: https://arxiv.org/abs/2305.14314
    """

    def __init__(
        self,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float32",
        bnb_4bit_quant_storage: str = "uint8",
        bnb_4bit_quant_type: str = "fp4",
        bnb_4bit_use_double_quant: bool = False,
        llm_int8_enable_fp32_cpu_offload: bool = False,
        llm_int8_has_fp16_weight: bool = False,
        llm_int8_skip_modules: list[str] | None = None,
        llm_int8_threshold: float = 6.0,
    ) -> None:
        super().__init__()
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_storage = bnb_4bit_quant_storage
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        self.llm_int8_skip_modules = llm_int8_skip_modules or []
        self.llm_int8_threshold = llm_int8_threshold

        if self.bnb_4bit_quant_storage not in ["uint8"]:
            raise ValueError(
                f"Unsupported bnb_4bit_quant_storage: {self.bnb_4bit_quant_storage}"
            )

    def __repr__(self) -> str:
        return (
            f"BitsAndBytesConfig(load_in_8bit={self.load_in_8bit}, "
            f"load_in_4bit={self.load_in_4bit}, "
            f"bnb_4bit_compute_dtype={self.bnb_4bit_compute_dtype}, "
            f"bnb_4bit_quant_storage={self.bnb_4bit_quant_storage}, "
            f"bnb_4bit_quant_type={self.bnb_4bit_quant_type}, "
            f"llm_int8_skip_modules={self.llm_int8_skip_modules})"
        )

    @classmethod
    def get_name(self) -> QuantizationMethods:
        return "bitsandbytes"

    @classmethod
    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BitsAndBytesConfig":
        def get_safe_value(config, keys, default_value=None):
            try:
                value = cls.get_from_keys(config, keys)
                return value if value is not None else default_value
            except ValueError:
                return default_value

        load_in_8bit = get_safe_value(config, ["load_in_8bit"], default_value=False)
        load_in_4bit = get_safe_value(config, ["load_in_4bit"], default_value=True)
        bnb_4bit_compute_dtype = get_safe_value(
            config, ["bnb_4bit_compute_dtype"], default_value="float32"
        )
        bnb_4bit_quant_storage = get_safe_value(
            config, ["bnb_4bit_quant_storage"], default_value="uint8"
        )
        bnb_4bit_quant_type = get_safe_value(
            config, ["bnb_4bit_quant_type"], default_value="fp4"
        )
        bnb_4bit_use_double_quant = get_safe_value(
            config, ["bnb_4bit_use_double_quant"], default_value=False
        )
        llm_int8_enable_fp32_cpu_offload = get_safe_value(
            config, ["llm_int8_enable_fp32_cpu_offload"], default_value=False
        )
        llm_int8_has_fp16_weight = get_safe_value(
            config, ["llm_int8_has_fp16_weight"], default_value=False
        )
        llm_int8_skip_modules = get_safe_value(
            config, ["llm_int8_skip_modules"], default_value=[]
        )
        llm_int8_threshold = get_safe_value(
            config, ["llm_int8_threshold"], default_value=6.0
        )

        return cls(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_quant_storage=bnb_4bit_quant_storage,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload,
            llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,
            llm_int8_skip_modules=llm_int8_skip_modules,
            llm_int8_threshold=llm_int8_threshold,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Union["LinearMethodBase", "BitsAndBytesMoEMethod"] | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped_bnb(prefix, self.llm_int8_skip_modules):
                return UnquantizedLinearMethod()
            return BitsAndBytesLinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return BitsAndBytesMoEMethod(self, layer.moe_config)
        return None


def is_layer_skipped_bnb(prefix: str, llm_int8_skip_modules: list[str]):
    # Split the prefix into its dot-separated components
    components = prefix.split(".")

    # Check if any of the skip modules exactly matches any component
    substr_check = any(
        module_name in components for module_name in llm_int8_skip_modules
    )

    # Allow certain layers to not be quantized
    set_components = set(".".join(components[: i + 1]) for i in range(len(components)))
    set_llm_int8_skip_modules = set(llm_int8_skip_modules)
    prefix_check = len(set_llm_int8_skip_modules & set_components) != 0

    return substr_check or prefix_check


def calculate_quant_ratio(dtype):
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits // torch.iinfo(torch.uint8).bits
    else:
        return torch.iinfo(dtype).bits // torch.iinfo(torch.uint8).bits


class BitsAndBytesLinearMethod(LinearMethodBase):
    """Linear method for BitsAndBytes.

    Args:
       quant_config: The BitsAndBytes quantization config.
    """

    def __init__(self, quant_config: BitsAndBytesConfig):
        try:
            import bitsandbytes

            if version.parse(bitsandbytes.__version__) < version.parse("0.46.1"):
                raise ImportError(
                    "bitsandbytes version is wrong. Please "
                    "install bitsandbytes>=0.46.1."
                )
        except ImportError as err:
            raise ImportError(
                "Please install bitsandbytes>=0.46.1 via "
                "`pip install bitsandbytes>=0.46.1` to use "
                "bitsandbytes quantizer."
            ) from err

        self.quant_config = quant_config

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
        from bitsandbytes.nn import Int8Params

        def create_qweight_for_8bit():
            qweight = Int8Params(
                data=torch.empty(
                    sum(output_partition_sizes),
                    input_size_per_partition,
                    dtype=torch.int8,
                ),
                has_fp16_weights=self.quant_config.llm_int8_has_fp16_weight,
                requires_grad=False,
            )
            set_weight_attrs(
                qweight,
                {
                    "input_dim": 0,
                    "output_dim": 0,
                    "pack_factor": 1,
                    "use_bitsandbytes_8bit": True,
                    "generation": 0,
                },
            )
            return qweight

        def create_qweight_for_4bit():
            quant_ratio = calculate_quant_ratio(params_dtype)

            total_size = input_size_per_partition * sum(output_partition_sizes)
            if total_size % quant_ratio != 0:
                raise ValueError(
                    "The input size is not aligned with the quantized weight shape."
                )

            qweight = torch.nn.Parameter(
                torch.empty(total_size // quant_ratio, 1, dtype=torch.uint8),
                requires_grad=False,
            )
            set_weight_attrs(
                qweight,
                {
                    "input_dim": 0,
                    "output_dim": 0,
                    "pack_factor": quant_ratio,
                    "use_bitsandbytes_4bit": True,
                },
            )
            return qweight

        if self.quant_config.load_in_8bit:
            qweight = create_qweight_for_8bit()
        else:
            qweight = create_qweight_for_4bit()
        # Enable parameters to have the same name as in the BNB
        # checkpoint format.
        layer.register_parameter("weight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.quant_config.load_in_8bit:
            return self._apply_8bit_weight(layer, x, bias)
        else:
            return self._apply_4bit_weight(layer, x, bias)

    def _apply_8bit_weight(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # only load the bitsandbytes module when needed
        from bitsandbytes import MatmulLtState, matmul

        original_type = x.dtype
        original_shape = x.shape
        reshape_after_matmul = False
        if x.ndim > 2:
            x = x.reshape(-1, x.size(-1))
            reshape_after_matmul = True
        bf_x = x.to(torch.bfloat16)

        qweight = layer.weight
        offsets = qweight.bnb_shard_offsets
        quant_states = qweight.bnb_quant_state
        matmul_states = qweight.matmul_state
        generation = qweight.generation

        out_dim_0 = x.shape[0]
        out_dim_1 = sum(
            [quant_state[1].shape[0] for quant_state in quant_states.items()]
        )
        out = torch.empty(out_dim_0, out_dim_1, dtype=torch.float16, device=x.device)

        current_index = 0
        for i in range(len(quant_states)):
            output_size = quant_states[i].shape[0]

            # in profile_run or the first generation of inference,
            # create new matmul_states
            if generation == 0 or generation == 1:
                matmul_states[i] = MatmulLtState()
                matmul_states[i].CB = qweight[offsets[i] : offsets[i + 1]]
                matmul_states[i].SCB = quant_states[i].to(x.device)
                matmul_states[i].threshold = self.quant_config.llm_int8_threshold
                matmul_states[
                    i
                ].has_fp16_weights = self.quant_config.llm_int8_has_fp16_weight
                matmul_states[i].is_training = False
                if (
                    matmul_states[i].threshold > 0.0
                    and not matmul_states[i].has_fp16_weights
                ):
                    matmul_states[i].use_pool = True

            new_x = bf_x.unsqueeze(0)

            out[:, current_index : current_index + output_size] = matmul(
                new_x, qweight[offsets[i] : offsets[i + 1]], state=matmul_states[i]
            )

            current_index += output_size

            # only update the matmul_states if it is not profile_run
            if (
                generation > 0
                and not self.quant_config.llm_int8_has_fp16_weight
                and matmul_states[i].CB is not None
                and matmul_states[i].CxB is not None
            ):
                del matmul_states[i].CB
                qweight[offsets[i] : offsets[i + 1]] = matmul_states[i].CxB

        out = out.to(original_type)

        if reshape_after_matmul:
            out = out.view(*original_shape[:-1], out.size(-1))

        if bias is not None:
            out += bias

        qweight.generation += 1

        return out

    def _apply_4bit_weight(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_type = x.dtype
        original_shape = x.shape
        reshape_after_matmul = False
        if x.ndim > 2:
            x = x.reshape(-1, x.size(-1))
            reshape_after_matmul = True
        bf_x = x.to(torch.bfloat16)

        qweight = layer.weight
        quant_states = qweight.bnb_quant_state
        offsets = qweight.bnb_shard_offsets

        out_dim_0 = x.shape[0]
        out_dim_1 = sum(
            [quant_state[1].shape[0] for quant_state in quant_states.items()]
        )
        out = torch.empty(out_dim_0, out_dim_1, dtype=torch.bfloat16, device=x.device)
        apply_bnb_4bit(bf_x, qweight, offsets, out)
        out = out.to(original_type)

        if reshape_after_matmul:
            out = out.view(*original_shape[:-1], out.size(-1))

        if bias is not None:
            out += bias

        return out


def _apply_bnb_4bit(
    x: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    out: torch.Tensor,
) -> None:
    # only load the bitsandbytes module when needed
    from bitsandbytes import matmul_4bit

    quant_states = weight.bnb_quant_state
    current_index = 0
    for i in range(len(quant_states)):
        output_size = quant_states[i].shape[0]
        # It is more efficient to use out kwarg like
        # matmul_4bit(..., out = ...).  Infeasible now due to the bug
        # https://github.com/TimDettmers/bitsandbytes/issues/1235.
        # Need to change  after the bug is fixed.
        out[:, current_index : current_index + output_size] = matmul_4bit(
            x, weight[offsets[i] : offsets[i + 1]].t(), quant_states[i]
        )
        current_index += output_size


def _apply_bnb_4bit_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    offsets: torch.Tensor,
    out: torch.Tensor,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="apply_bnb_4bit",
        op_func=_apply_bnb_4bit,
        mutates_args=["out"],
        fake_impl=_apply_bnb_4bit_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    apply_bnb_4bit = torch.ops.vllm.apply_bnb_4bit

except AttributeError as error:
    raise error


class BitsAndBytesMoEMethod(FusedMoEMethodBase):
    """MoE method for BitsAndBytes.

    Args:
       quant_config: The BitsAndBytes quantization config.
    """

    def __init__(
        self,
        quant_config: BitsAndBytesConfig,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        try:
            import bitsandbytes

            if version.parse(bitsandbytes.__version__) < version.parse("0.46.1"):
                raise ImportError(
                    "bitsandbytes version is wrong. Please "
                    "install bitsandbytes>=0.46.1."
                )
        except ImportError as err:
            raise ImportError(
                "Please install bitsandbytes>=0.46.1 via "
                "`pip install bitsandbytes>=0.46.1` to use "
                "bitsandbytes quantizer."
            ) from err
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if self.quant_config.load_in_8bit:
            call_fun = self._create_weights_8bit
        else:
            call_fun = self._create_weights_4bit
        call_fun(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

    def apply(
        self,
        layer: FusedMoE,
        router: FusedMoERouter,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids = router.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )
        # TODO(bnell): Do these need to be called on the hot path?
        if self.quant_config.load_in_8bit:
            w13, w2 = self._apply_8bit_dequant(layer)
        else:
            w13, w2 = self._apply_4bit_dequnt(layer)
        return fused_experts(
            hidden_states=x,
            w1=w13,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )

    def _create_weights_4bit(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        quant_ratio = calculate_quant_ratio(params_dtype)
        # Fused gate_up_proj (column parallel)
        w13_total_size = (
            hidden_size * 2 * intermediate_size_per_partition
        ) // quant_ratio
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_total_size,
                1,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        set_weight_attrs(
            w13_qweight,
            {
                "num_experts": num_experts,
                "input_dim": hidden_size,
                "output_dim": 2 * intermediate_size_per_partition,
                "experts_shape": (
                    num_experts,
                    intermediate_size_per_partition * 2,
                    hidden_size,
                ),
                "pack_factor": quant_ratio,
                "use_bitsandbytes_4bit": True,
            },
        )
        # down_proj (row parallel)
        w2_total_size = (hidden_size * intermediate_size_per_partition) // quant_ratio
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w2_total_size,
                1,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            w2_qweight,
            {
                "num_experts": num_experts,
                "input_dim": intermediate_size_per_partition,
                "output_dim": hidden_size,
                "experts_shape": (
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                ),
                "pack_factor": quant_ratio,
                "use_bitsandbytes_4bit": True,
            },
        )
        layer.register_parameter("w2_weight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

    def _create_weights_8bit(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    def _apply_4bit_dequnt(
        self, layer: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from bitsandbytes.functional import dequantize_4bit

        w13 = dequantize_4bit(
            layer.w13_weight.reshape(-1, 1),
            layer.w13_weight.bnb_quant_state,
        )
        w2 = dequantize_4bit(
            layer.w2_weight.reshape(-1, 1),
            layer.w2_weight.bnb_quant_state,
        )
        w13 = w13.reshape(layer.w13_weight.experts_shape)
        w2 = w2.reshape(layer.w2_weight.experts_shape)
        return w13, w2

    def _apply_8bit_dequant(
        self, layer: torch.nn.Module
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
