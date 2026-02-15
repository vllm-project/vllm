# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from importlib.util import find_spec
from types import ModuleType
from typing import Any

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir.ops
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import MatcherFusedAddRMSNorm, MatcherQuantFP8

FP8_DTYPE = current_platform.fp8_dtype()

logger = init_logger(__name__)

flashinfer_comm: ModuleType | None = None
if find_spec("flashinfer"):
    try:
        import flashinfer.comm as _flashinfer_comm

        if hasattr(_flashinfer_comm, "allreduce_fusion") and hasattr(
            _flashinfer_comm, "create_allreduce_fusion_workspace"
        ):
            flashinfer_comm = _flashinfer_comm
    except ImportError:
        pass

logger = init_logger(__name__)

if hasattr(torch.ops._C, "scaled_fp4_quant"):
    STATIC_FP4_QUANT_OP = torch.ops._C.scaled_fp4_quant.default

# Max size of the input tensor per world size per device capability
# to use flashinfer fused allreduce
FI_ALLREDUCE_FUSION_MAX_SIZE_MB: dict[int, dict[int, float]] = {
    90: {
        2: 64,  # 64MB
        4: 2,  # 2MB
        8: 0.5,  # 0.5MB
    },
    100: {
        2: 64,  # 64MB
        4: 32,  # 32MB
        8: 1,  # 1MB
    },
}

# Max size of the input tensor per world size per device capability
# to use flashinfer one shot fused allreduce
# OneShot max size is at most 64MB / world size (FlashInfer restriction)
_FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB: dict[int, dict[int, float]] = {
    90: {
        2: 32,  # 32MB
        4: 2,  # 2MB
        8: 0.5,  # 0.5MB
    },
    100: {
        2: 32,  # 32MB
        4: 4,  # 4MB
        8: 1,  # 1MB
    },
}


if flashinfer_comm is not None:
    _FI_WORKSPACE = None
    MiB = 1024 * 1024

    def call_trtllm_fused_allreduce_norm(
        allreduce_in: torch.Tensor,
        residual: torch.Tensor,
        rms_gamma: torch.Tensor,
        rms_eps: float,
        world_size: int,
        launch_with_pdl: bool,
        fp32_acc: bool,
        max_token_num: int,
        pattern_code: int,
        norm_out: torch.Tensor | None = None,
        quant_out: torch.Tensor | None = None,
        scale_out: torch.Tensor | None = None,
        scale_factor: torch.Tensor | None = None,
    ) -> None:
        num_tokens, hidden_size = allreduce_in.shape
        element_size = allreduce_in.element_size()
        current_tensor_size = num_tokens * hidden_size * element_size
        max_tensor_size = max_token_num * hidden_size * element_size
        assert current_tensor_size <= max_tensor_size, (
            f"Current tensor size {current_tensor_size} is larger than "
            f"max token num {max_token_num} * hidden size {hidden_size} * "
            f"element size {element_size}"
        )
        curr_device = current_platform.get_device_capability()
        device_capability = curr_device.to_int() if curr_device is not None else None
        # Get one shot input size limit for the current world size
        # for the current device capability
        max_one_shot_size = _FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB.get(
            device_capability,  # type: ignore[arg-type, unused-ignore]
            {},
        ).get(world_size, None)
        # Use one shot if no max size is specified
        use_oneshot = (
            max_one_shot_size is None or current_tensor_size <= max_one_shot_size * MiB
        )

        assert _FI_WORKSPACE is not None, (
            "Flashinfer must be enabled when using flashinfer"
        )
        if norm_out is None:
            norm_out = allreduce_in
            residual_out = residual
        else:
            # return residual_out as allreduce_out with zeroed residual_in
            # as flashinfer does not support rms_norm
            # and allreduce_out together
            residual_out = allreduce_in
        # For the sizes that are smaller than the max size,
        # we only use flashinfer one shot allreduce
        flashinfer_comm.allreduce_fusion(
            input=allreduce_in,
            workspace=_FI_WORKSPACE,
            pattern=pattern_code,
            residual_in=residual,
            residual_out=residual_out,
            norm_out=norm_out,
            rms_gamma=rms_gamma,
            rms_eps=rms_eps,
            launch_with_pdl=launch_with_pdl,
            use_oneshot=use_oneshot,
            fp32_acc=fp32_acc,
            quant_out=quant_out,
            scale_out=scale_out,
            # in vllm we only support swizzled layout
            layout_code=flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4,
            scale_factor=scale_factor,
        )

    def call_trtllm_fused_allreduce_norm_fake(
        allreduce_in: torch.Tensor,
        residual: torch.Tensor,
        rms_gamma: torch.Tensor,
        rms_eps: float,
        world_size: int,
        launch_with_pdl: bool,
        fp32_acc: bool,
        max_token_num: int,
        pattern_code: int,
        norm_out: torch.Tensor | None = None,
        quant_out: torch.Tensor | None = None,
        scale_out: torch.Tensor | None = None,
        scale_factor: torch.Tensor | None = None,
    ) -> None:
        pass

    direct_register_custom_op(
        op_name="flashinfer_trtllm_fused_allreduce_norm",
        op_func=call_trtllm_fused_allreduce_norm,
        mutates_args=[
            "allreduce_in",
            "residual",
            "norm_out",
            "quant_out",
            "scale_out",
        ],
        fake_impl=call_trtllm_fused_allreduce_norm_fake,
    )
    flashinfer_trtllm_fused_allreduce_norm = (
        torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default
    )


class FlashInferFusedAllReduceParams:
    """Parameters for FlashInfer fused allreduce operations."""

    def __init__(
        self,
        world_size: int,
        max_token_num: int = 1024,
    ) -> None:
        self.world_size = world_size
        self.launch_with_pdl = True
        self.fp32_acc = True
        self.max_token_num = max_token_num

    def get_trtllm_fused_allreduce_kwargs(self) -> dict[str, bool | int]:
        return {
            "world_size": self.world_size,
            "launch_with_pdl": self.launch_with_pdl,
            "fp32_acc": self.fp32_acc,
            "max_token_num": self.max_token_num,
        }


# TODO(luka): unify
class BasePattern:
    def __init__(self, dtype: torch.dtype, device: str | None) -> None:
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def empty(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=self.dtype, device=self.device, **kwargs)

    def empty_f32(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=torch.float32, device=self.device, **kwargs)


class AllReduceRMSNormPattern(BasePattern):
    """
    This pattern replaces the allreduce + rms norm (without residual)
    with fused flashinfer implementation.
    Applies to allreduce + rmsnorm before attn in the first Transformer block.
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        allreduce_params: FlashInferFusedAllReduceParams,
    ) -> None:
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.allreduce_params = allreduce_params

    def get_inputs(self) -> list[torch.Tensor]:
        # input, weight
        return [self.empty(5, 16), self.empty(16)]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms = vllm.ir.ops.rms_norm(allreduce_output, weight, self.epsilon)

            return rms, allreduce_output

        def replacement(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual = torch.zeros_like(input)
            rms_result = torch.empty_like(input)
            assert flashinfer_comm is not None, "FlashInfer must be enabled"
            allreduce = auto_functionalized(
                flashinfer_trtllm_fused_allreduce_norm,
                allreduce_in=input,
                residual=residual,
                norm_out=rms_result,
                quant_out=None,
                scale_out=None,
                rms_gamma=weight,
                rms_eps=self.epsilon,
                pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )
            # rms_result, allreduce_in
            return allreduce[3], allreduce[1]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllReduceFusedAddRMSNormPattern(BasePattern):
    """
    This pattern replaces the allreduce + rms norm (with residual)
    with fused flashinfer implementation.
    Applies to o_proj + rmsnorm after attn and mlp + rmsnorm before attn.
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        allreduce_params: FlashInferFusedAllReduceParams,
    ) -> None:
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.allreduce_params = allreduce_params
        self.rmsnorm_matcher = MatcherFusedAddRMSNorm(epsilon)

    def get_inputs(self) -> list[torch.Tensor]:
        input, residual, weight = self.rmsnorm_matcher.inputs()

        # input goes through allreduce first, always 16-bit
        return [residual, input.to(self.dtype), weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms, residual = self.rmsnorm_matcher(allreduce_output, weight, residual)
            return rms, residual

        def replacement(
            residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert flashinfer_comm is not None, "FlashInfer must be enabled"
            allreduce = auto_functionalized(
                flashinfer_trtllm_fused_allreduce_norm,
                allreduce_in=input,
                residual=residual,
                norm_out=None,
                quant_out=None,
                scale_out=None,
                rms_gamma=weight,
                rms_eps=self.epsilon,
                pattern_code=flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )
            # allreduce_in, residual
            return allreduce[1], allreduce[2]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )

        # Same pattern, but only return the output and not residual
        # (helpful for end of graph where residual is not used again)
        first_return_only = lambda fn: lambda a, b, c: fn(a, b, c)[0]

        pm.register_replacement(
            first_return_only(pattern),  # type: ignore[no-untyped-call]
            first_return_only(replacement),  # type: ignore[no-untyped-call]
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
        )


class AllReduceFusedRMSNormStaticQuantFP8Pattern(BasePattern):
    """
    This pattern replaces the allreduce + rms norm (without residual)
    + static fp8 quant with fused flashinfer implementation.
    Applies to allreduce + rmsnorm + quant before attn
    in the first Transformer block.
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        allreduce_params: FlashInferFusedAllReduceParams,
    ) -> None:
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.allreduce_params = allreduce_params
        self.quant_dtype = torch.float8_e4m3fn
        self.quant_matcher = MatcherQuantFP8(kFp8StaticTensorSym)

    def get_inputs(self) -> list[torch.Tensor]:
        _, scale = self.quant_matcher.inputs()

        # input, weight
        return [self.empty(5, 16), self.empty(16), scale]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_reduce = tensor_model_parallel_all_reduce(input)
            rms = vllm.ir.ops.rms_norm(all_reduce, weight, self.epsilon)
            quant, _ = self.quant_matcher(rms, scale)
            return quant, all_reduce

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual = torch.zeros_like(input)
            result_rms = torch.empty_like(input)
            result_quant = torch.empty_like(input, dtype=self.quant_dtype)
            assert flashinfer_comm is not None, "FlashInfer must be enabled"
            allreduce = auto_functionalized(
                flashinfer_trtllm_fused_allreduce_norm,
                allreduce_in=input,
                residual=residual,
                norm_out=result_rms,
                quant_out=result_quant,
                scale_out=None,
                rms_gamma=weight,
                rms_eps=self.epsilon,
                # We don't use norm_out afterwards
                pattern_code=(
                    flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant
                ),
                scale_factor=scale,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )

            # quant_out, allreduce_output
            return allreduce[4], allreduce[1]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllReduceFusedAddRMSNormStaticQuantFP8Pattern(BasePattern):
    """
    This pattern replaces the allreduce + rms norm (with residual)
    + static fp8 quant with fused flashinfer implementation.
    Applies to o_proj + rmsnorm after attn + quant and
    mlp + rmsnorm + quant before attn.
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        allreduce_params: FlashInferFusedAllReduceParams,
    ) -> None:
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.allreduce_params = allreduce_params
        self.quant_dtype = torch.float8_e4m3fn

        self.rmsnorm_matcher = MatcherFusedAddRMSNorm(epsilon)
        self.quant_matcher = MatcherQuantFP8(kFp8StaticTensorSym)

    def get_inputs(self) -> list[torch.Tensor]:
        input, residual, weight = self.rmsnorm_matcher.inputs()
        _, scale = self.quant_matcher.inputs()

        # input goes through allreduce first, always 16-bit
        return [residual, input.to(self.dtype), weight, scale]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms, res = self.rmsnorm_matcher(allreduce_output, weight, residual)
            quant, _ = self.quant_matcher(rms, scale)

            return quant, res

        def replacement(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_quant = torch.empty_like(input, dtype=self.quant_dtype)
            assert flashinfer_comm is not None, "FlashInfer must be enabled"
            allreduce = auto_functionalized(
                flashinfer_trtllm_fused_allreduce_norm,
                allreduce_in=input,
                residual=residual,
                norm_out=None,
                quant_out=result_quant,
                scale_out=None,
                rms_gamma=weight,
                rms_eps=self.epsilon,
                # We don't use norm_out afterwards
                pattern_code=(
                    flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNormFP8Quant
                ),
                scale_factor=scale,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )
            # quant_out, rms_norm_residual
            return allreduce[4], allreduce[2]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllReduceFusedRMSNormStaticQuantNVFP4Pattern(BasePattern):
    """
    This pattern replaces the allreduce + rms norm (without residual)
    + static nvfp4 quant with fused flashinfer implementation.
    Applies to allreduce + rmsnorm + quant before attn
    in the first Transformer block.
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        allreduce_params: FlashInferFusedAllReduceParams,
    ) -> None:
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.allreduce_params = allreduce_params

    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([1, 16, 16], device=self.device, dtype=self.dtype)
        quant_result = torch.empty((16, 8), device=self.device, dtype=torch.uint8)
        input_global_scale = torch.empty(
            [1, 1], device=self.device, dtype=torch.float32
        )
        weight = torch.empty([16], device=self.device, dtype=self.dtype)
        output_scale = torch.empty([128, 4], device=self.device, dtype=torch.int32)

        return [input, quant_result, weight, input_global_scale, output_scale]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            quant_result: torch.Tensor,
            weight: torch.Tensor,
            input_global_scale: torch.Tensor,
            output_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            all_reduce = tensor_model_parallel_all_reduce(input)
            rms = vllm.ir.ops.rms_norm(all_reduce, weight, self.epsilon)
            quant_out_tuple = auto_functionalized(
                STATIC_FP4_QUANT_OP,
                output=quant_result,
                input=rms,
                output_scale=output_scale,
                input_scale=input_global_scale,
                is_sf_swizzled_layout=True,
            )

            # quant_out, allreduce_output, output_scale
            return quant_out_tuple[1], all_reduce, quant_out_tuple[2]

        def replacement(
            input: torch.Tensor,
            quant_result: torch.Tensor,
            weight: torch.Tensor,
            input_global_scale: torch.Tensor,
            output_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            residual = torch.zeros_like(input)
            result_rms = torch.empty_like(input)
            assert flashinfer_comm is not None, "FlashInfer must be enabled"
            allreduce = auto_functionalized(
                flashinfer_trtllm_fused_allreduce_norm,
                allreduce_in=input,
                residual=residual,
                norm_out=result_rms,
                quant_out=quant_result,
                scale_out=output_scale,
                rms_gamma=weight,
                rms_eps=self.epsilon,
                # We don't use norm_out afterwards
                pattern_code=(
                    flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant
                ),
                scale_factor=input_global_scale,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )

            # quant_out, allreduce_output, output_scale
            return allreduce[4], allreduce[1], allreduce[5]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllReduceFusedAddRMSNormStaticQuantNVFP4Pattern(BasePattern):
    """
    This pattern replaces the allreduce + rms norm (with residual)
    + static nvfp4 quant with fused flashinfer implementation.
    Applies to o_proj + rmsnorm after attn + quant and
    mlp + rmsnorm + quant before attn.
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        allreduce_params: FlashInferFusedAllReduceParams,
    ) -> None:
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.allreduce_params = allreduce_params
        self.rmsnorm_matcher = MatcherFusedAddRMSNorm(epsilon)

    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([16, 16], device=self.device, dtype=self.dtype)

        residual = torch.empty([16, 16], device=self.device, dtype=self.dtype)
        weight = torch.empty([16, 16], device=self.device, dtype=self.dtype)
        quant_result = torch.empty((16, 8), device=self.device, dtype=torch.uint8)
        input_global_scale = torch.empty(
            [1, 1], device=self.device, dtype=torch.float32
        )
        output_scale = torch.empty([128, 4], device=self.device, dtype=torch.int32)

        return [
            quant_result,
            residual,
            input,
            output_scale,
            weight,
            input_global_scale,
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            quant_result: torch.Tensor,
            residual: torch.Tensor,
            input: torch.Tensor,
            output_scale: torch.Tensor,
            weight: torch.Tensor,
            input_global_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms, residual = self.rmsnorm_matcher(allreduce_output, weight, residual)
            quant_out_tuple = auto_functionalized(
                STATIC_FP4_QUANT_OP,
                output=quant_result,
                input=rms,
                output_scale=output_scale,
                input_scale=input_global_scale,
                is_sf_swizzled_layout=True,
            )

            # quant_out, allreduce_output, output_scale
            return quant_out_tuple[1], residual, quant_out_tuple[2]

        def replacement(
            quant_result: torch.Tensor,
            residual: torch.Tensor,
            input: torch.Tensor,
            output_scale: torch.Tensor,
            weight: torch.Tensor,
            input_global_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            assert flashinfer_comm is not None, "FlashInfer must be enabled"
            allreduce = auto_functionalized(
                flashinfer_trtllm_fused_allreduce_norm,
                allreduce_in=input,
                residual=residual,
                norm_out=None,
                quant_out=quant_result,
                scale_out=output_scale,
                rms_gamma=weight,
                rms_eps=self.epsilon,
                # We don't use norm_out afterwards
                pattern_code=(
                    flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNormFP4Quant
                ),
                scale_factor=input_global_scale,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )
            # quant_out, rms_norm_residual, output_scale
            return allreduce[4], allreduce[2], allreduce[5]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )


class AllReduceFusionPass(VllmPatternMatcherPass):
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.disabled = True
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size <= 1:
            logger.warning_once("AllReduce fusion pass is disabled for tp_size <= 1.")
            return
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="all_reduce_fusion_pass"
        )
        if config.model_config is None:
            logger.warning_once(
                "AllReduce fusion pass is disabled for missing model_config."
            )
            return
        self.hidden_dim = config.model_config.get_hidden_size()
        self.group = get_tp_group().device_group
        rank = get_tensor_model_parallel_rank()
        if flashinfer_comm is None:
            logger.warning(
                "Flashinfer is not installed or comm module not found, "
                "skipping allreduce fusion pass"
            )
            return
        max_size = config.compilation_config.pass_config.flashinfer_max_size(
            self.tp_size
        )
        if max_size is None:
            # Flashinfer doesn't support current world size
            logger.warning(
                "Flashinfer allreduce fusion is not supported for world size %s"
                " or max size is not provided",
                self.tp_size,
            )
            return
        element_size = torch.tensor([], dtype=self.model_dtype).element_size()
        self.max_token_num = max_size // (self.hidden_dim * element_size)
        # take the min to save workspace size and we'll never use more
        # than max_num_batched_tokens anyways
        self.max_token_num = min(
            self.max_token_num, config.scheduler_config.max_num_batched_tokens
        )
        logger.debug_once(
            f"Flashinfer max size: {max_size // (1024 * 1024)} MB,"
            "Maximal number of tokens used by "
            f"Flashinfer Allreduce Fusion: {self.max_token_num}",
            scope="global",
        )

        self.workspace = flashinfer_comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=self.tp_size,
            rank=rank,
            max_token_num=self.max_token_num,
            hidden_dim=self.hidden_dim,
            dtype=self.model_dtype,
        )

        global _FI_WORKSPACE
        _FI_WORKSPACE = self.workspace
        self.allreduce_params = FlashInferFusedAllReduceParams(
            world_size=self.tp_size,
            max_token_num=self.max_token_num,
        )

        self.register_patterns()
        self.dump_patterns(config, self.patterns)

    @enable_fake_mode
    def register_patterns(self) -> None:
        for epsilon in [1e-5, 1e-6]:
            AllReduceFusedRMSNormStaticQuantFP8Pattern(
                epsilon,
                self.model_dtype,
                self.device,
                self.allreduce_params,
            ).register(self.patterns)
            AllReduceFusedAddRMSNormStaticQuantFP8Pattern(
                epsilon,
                self.model_dtype,
                self.device,
                self.allreduce_params,
            ).register(self.patterns)
            if current_platform.has_device_capability(100):
                AllReduceFusedRMSNormStaticQuantNVFP4Pattern(
                    epsilon,
                    self.model_dtype,
                    self.device,
                    self.allreduce_params,
                ).register(self.patterns)
                AllReduceFusedAddRMSNormStaticQuantNVFP4Pattern(
                    epsilon,
                    self.model_dtype,
                    self.device,
                    self.allreduce_params,
                ).register(self.patterns)
            AllReduceRMSNormPattern(
                epsilon,
                self.model_dtype,
                self.device,
                self.allreduce_params,
            ).register(self.patterns)
            AllReduceFusedAddRMSNormPattern(
                epsilon,
                self.model_dtype,
                self.device,
                self.allreduce_params,
            ).register(self.patterns)

            # WARNING: This is a hack to clear the pattern matcher cache
            # and allow multiple values of epsilon.
            torch._inductor.pattern_matcher._seen_patterns.clear()

        self.disabled = False

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        if self.disabled:
            logger.warning_once("AllReduce fusion pass is disabled.")
            return False
        return bool(compile_range.end <= self.max_token_num)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        if self.disabled:
            logger.debug("AllReduceFusionPass disabled")
            return

        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def __del__(self) -> None:
        if getattr(self, "disabled", True):
            return
        if getattr(self, "workspace", None) is not None:
            with contextlib.suppress(Exception):
                self.workspace.destroy()
