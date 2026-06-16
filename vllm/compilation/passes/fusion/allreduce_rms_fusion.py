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
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.passes.fusion.rms_quant_fusion import (
    _rms_input_weight_dtype_match,
)
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    direct_register_custom_op,
)

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmInductorPass,
    VllmPatternMatcherPass,
    VllmPatternReplacement,
)
from .matcher_utils import MatcherQuantFP8

FP8_DTYPE = current_platform.fp8_dtype()

_IR_RMS_NORM_OP = torch.ops.vllm_ir.rms_norm.default
_IR_FUSED_ADD_RMS_NORM_OP = torch.ops.vllm_ir.fused_add_rms_norm.default


def _norm_input_weight_dtype_match(match: pm.Match) -> bool:
    """Prevent fusion when the norm input and weight dtypes differ (e.g. a Gemma
    fp32 weight.float()+1 gamma), covering rms_norm and fused_add_rms_norm."""
    for node in match.nodes:
        if node.target == _IR_RMS_NORM_OP:
            x, weight = node.args[0], node.args[1]
        elif node.target == _IR_FUSED_ADD_RMS_NORM_OP:
            x, weight = node.args[0], node.args[2]
        else:
            continue
        if isinstance(x, fx.Node) and isinstance(weight, fx.Node):
            return x.meta["val"].dtype == weight.meta["val"].dtype
    return True


# The empirical value for small batch
PDL_ADVANCE_LAUNCH_TOKENS = 16

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

if hasattr(torch.ops._C, "scaled_fp4_quant"):
    STATIC_FP4_QUANT_OP = torch.ops._C.scaled_fp4_quant.out

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
    103: {
        2: 64,  # 64MB
        4: 64,  # 64MB
        8: 2,  # 2MB
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
    103: {
        2: 32,  # 32MB
        4: 4,  # 4MB
        8: 2,  # 2MB
    },
}


if flashinfer_comm is not None:
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        destroy_fi_ar_workspace,
        get_fi_ar_quant_workspace,
        get_fi_ar_workspace,
    )

    ar_fusion_patterns = flashinfer_comm.AllReduceFusionPattern

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
        weight_bias: float = 0.0,
    ) -> None:
        # handle transformers backend passing outer batch dim.
        if allreduce_in.dim() != 2:
            hidden = allreduce_in.shape[-1]
            allreduce_in = allreduce_in.view(-1, hidden)
            residual = residual.view(-1, hidden)
            if norm_out is not None:
                norm_out = norm_out.view(-1, hidden)
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

        # Select workspace based on pattern: quant patterns use the
        # trtllm quant workspace, non-quant patterns use the primary workspace.
        is_quant_pattern = pattern_code in (
            ar_fusion_patterns.kARResidualRMSNormFP8Quant,
            ar_fusion_patterns.kARResidualRMSNormFP4Quant,
        )
        get_workspace_fn = (
            get_fi_ar_quant_workspace if is_quant_pattern else get_fi_ar_workspace
        )
        workspace = get_workspace_fn(
            world_size=world_size,
            rank=get_tensor_model_parallel_rank(),
            max_token_num=max_token_num,
            hidden_dim=hidden_size,
            dtype=allreduce_in.dtype,
            group=get_tp_group().device_group,
        )
        assert workspace is not None, (
            "Flashinfer allreduce workspace must be initialized when using flashinfer"
        )
        assert flashinfer_comm is not None
        if norm_out is None:
            norm_out = allreduce_in
            residual_out = residual
        else:
            # return residual_out as allreduce_out with zeroed residual_in
            # as flashinfer does not support rms_norm
            # and allreduce_out together
            residual_out = allreduce_in

        layout_code = None
        # layout_code only supported by trtllm backend
        if workspace.backend == "trtllm":
            # in vllm we only support swizzled layout
            layout_code = flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4

        flashinfer_comm.allreduce_fusion(
            input=allreduce_in,
            workspace=workspace,
            pattern=pattern_code,
            launch_with_pdl=launch_with_pdl,
            output=None,
            residual_out=residual_out,
            norm_out=norm_out,
            quant_out=quant_out,
            scale_out=scale_out,
            residual_in=residual,
            rms_gamma=rms_gamma,
            rms_eps=rms_eps,
            scale_factor=scale_factor,
            layout_code=layout_code,
            use_oneshot=use_oneshot,
            fp32_acc=fp32_acc,
            weight_bias=weight_bias,
            trigger_completion_at_end=num_tokens > PDL_ADVANCE_LAUNCH_TOKENS,
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
        weight_bias: float = 0.0,
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
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
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

    def get_inputs(self) -> list[torch.Tensor]:
        input = self.empty(5, 16)
        residual = self.empty(5, 16)
        weight = self.empty(16)

        # input goes through allreduce first, always 16-bit
        return [residual, input.to(self.dtype), weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms, residual = vllm.ir.ops.fused_add_rms_norm(
                allreduce_output, residual, weight, self.epsilon
            )
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

        # extra_check routes a Gemma fp32 gamma to AllReduceFusedAddGemmaRMSNormPattern.
        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            extra_check=_norm_input_weight_dtype_match,
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
            extra_check=_norm_input_weight_dtype_match,
        )


class AllReduceGemmaRMSNormPattern(BasePattern):
    """Gemma-style variant of AllReduceRMSNormPattern (no residual)."""

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
        return [self.empty(5, 16), self.empty(16)]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms = vllm.ir.ops.rms_norm(
                allreduce_output, weight.float() + 1.0, self.epsilon
            )
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
                weight_bias=1.0,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )
            return allreduce[3], allreduce[1]

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
        )


class AllReduceFusedAddGemmaRMSNormPattern(BasePattern):
    """Gemma-style variant of AllReduceFusedAddRMSNormPattern (with residual)."""

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
        input = self.empty(5, 16)
        residual = self.empty(5, 16)
        weight = self.empty(16)
        return [residual, input.to(self.dtype), weight]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms, residual = vllm.ir.ops.fused_add_rms_norm(
                allreduce_output, residual, weight.float() + 1.0, self.epsilon
            )
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
                weight_bias=1.0,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )
            return allreduce[1], allreduce[2]

        pm.register_replacement(
            pattern, replacement, self.get_inputs(), pm.fwd_only, pm_pass
        )

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
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
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

        self.quant_matcher = MatcherQuantFP8(kFp8StaticTensorSym)

    def get_inputs(self) -> list[torch.Tensor]:
        input = self.empty(5, 16)
        residual = self.empty(5, 16)
        weight = self.empty(16)
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
            rms, res = vllm.ir.ops.fused_add_rms_norm(
                allreduce_output, residual, weight, self.epsilon
            )
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
                input=rms,
                input_scale=input_global_scale,
                is_sf_swizzled_layout=True,
                output=quant_result,
                output_scale=output_scale,
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
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
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
            rms, residual = vllm.ir.ops.fused_add_rms_norm(
                allreduce_output, residual, weight, self.epsilon
            )
            quant_out_tuple = auto_functionalized(
                STATIC_FP4_QUANT_OP,
                input=rms,
                input_scale=input_global_scale,
                is_sf_swizzled_layout=True,
                output=quant_result,
                output_scale=output_scale,
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

        workspace_kwargs = dict(
            world_size=self.tp_size,
            rank=rank,
            max_token_num=self.max_token_num,
            hidden_dim=self.hidden_dim,
            dtype=self.model_dtype,
            group=self.group,
        )
        if get_fi_ar_workspace(**workspace_kwargs) is None:
            logger.warning_once(
                "Failed to initialize Flashinfer allreduce workspace. "
                "Flashinfer allreduce-norm fusion will be disabled."
            )
            return

        self.supports_quant_fusion = (
            get_fi_ar_quant_workspace(**workspace_kwargs) is not None
        )
        if not self.supports_quant_fusion:
            logger.warning_once(
                "Failed to initialize Flashinfer allreduce workspace. "
                "Flashinfer allreduce-norm-quant fusion will be disabled."
            )

        self.allreduce_params = FlashInferFusedAllReduceParams(
            world_size=self.tp_size,
            max_token_num=self.max_token_num,
        )

        self.register_patterns()
        self.dump_patterns(config, self.patterns)

    @enable_fake_mode
    def register_patterns(self) -> None:
        for epsilon in [1e-5, 1e-6]:
            if self.supports_quant_fusion:
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
            AllReduceGemmaRMSNormPattern(
                epsilon,
                self.model_dtype,
                self.device,
                self.allreduce_params,
            ).register(self.patterns)
            AllReduceFusedAddGemmaRMSNormPattern(
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
        with contextlib.suppress(Exception):
            destroy_fi_ar_workspace()


# TODO: make BasePattern to inherit from VllmPatternReplacement
class AiterAllreduceFusedRMSNormPattern(BasePattern, VllmPatternReplacement):
    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        use_aiter_rmsnorm: bool = True,
    ) -> None:
        super().__init__(dtype, device)
        self.dtype = dtype
        self.epsilon = epsilon
        self.FUSED_AR_RMSNORM_OP = rocm_aiter_ops.get_fused_allreduce_rmsnorm_op()

    def get_inputs(self) -> list[torch.Tensor]:
        return [self.empty(5, 16), self.empty(16)]

    @property
    def pattern(self):
        def _pattern(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms = vllm.ir.ops.rms_norm(allreduce_output, weight, self.epsilon)

            return rms, allreduce_output

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual = torch.zeros_like(input)
            allreduce = self.FUSED_AR_RMSNORM_OP(
                input_=input,
                residual=residual,
                weight=weight.to(input.dtype),
                epsilon=self.epsilon,
            )
            return allreduce[0], allreduce[1]

        return _replacement


class AiterAllreduceFusedAddRMSNormPattern(BasePattern, VllmPatternReplacement):
    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        use_aiter_rmsnorm: bool = True,
    ) -> None:
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.dtype = dtype
        self.FUSED_AR_RMSNORM_OP = rocm_aiter_ops.get_fused_allreduce_rmsnorm_op()

    def get_inputs(self) -> list[torch.Tensor]:
        # input, residual, weight
        return [self.empty(5, 16), self.empty(5, 16), self.empty(16)]

    @property
    def pattern(self):
        def _pattern(
            residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms, residual = vllm.ir.ops.fused_add_rms_norm(
                allreduce_output, residual, weight, self.epsilon
            )
            return rms, residual

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            residual: torch.Tensor, input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            allreduce = self.FUSED_AR_RMSNORM_OP(
                input_=input,
                residual=residual,
                weight=weight.to(input.dtype),
                epsilon=self.epsilon,
            )
            return allreduce[0], allreduce[1]

        return _replacement


class AiterAllreduceFusedRMSNormGroupQuantFP8Pattern(
    BasePattern, VllmPatternReplacement
):
    """Fuse AllReduce + RMSNorm + per-group FP8 quant into a single AITER
    custom op.

    Matches the AR-side analogue of ``AiterRMSFp8GroupQuantPattern`` in
    ``rocm_aiter_fusion.py``: ``all_reduce -> rms_norm -> group_fp8_quant``
    fans out into ``rocm_aiter_fused_allreduce_rmsnorm_quant_per_group``.

    Without this pattern, ``RocmAiterAllReduceFusionPass`` would fuse the
    ``all_reduce + rms_norm`` half (PR #41825 wires that), but the trailing
    ``rocm_aiter_group_fp8_quant`` would still launch as a separate kernel.
    That standalone quant accounts for ~535us / decode step on DSv3.2 MI355X
    TP4 -- this pattern eliminates it by absorbing the quant into the AR
    epilogue. Group size 128 matches the FP8 block-scaled MM kernel used by
    DSv3.2's linear weights.
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        group_size: int = 128,
    ) -> None:
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.dtype = dtype
        self.group_size = group_size
        self.FUSED_AR_RMS_QUANT_OP = (
            rocm_aiter_ops.get_fused_allreduce_rmsnorm_quant_per_group_op()
        )
        self.quant_dtype = current_platform.fp8_dtype()
        self.quant_matcher = MatcherQuantFP8(
            QuantKey(
                dtype=self.quant_dtype,
                scale=ScaleDesc(torch.float32, False, GroupShape(1, group_size)),
                symmetric=True,
            ),
            match_rocm_aiter=True,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        # input, weight; hidden dim must be a group_size multiple so the
        # group quant matcher's example trace is well-defined.
        return [self.empty(5, self.group_size), self.empty(self.group_size)]

    @property
    def pattern(self):
        def _pattern(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms = vllm.ir.ops.rms_norm(allreduce_output, weight, self.epsilon)
            quant, scale = self.quant_matcher(rms)
            return quant, scale

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual = torch.zeros_like(input)
            result = self.FUSED_AR_RMS_QUANT_OP(
                input_=input,
                residual=residual,
                weight=weight.to(input.dtype),
                epsilon=self.epsilon,
                group_size=self.group_size,
            )
            # quant_out, scale_out (residual is unused on the no-add path,
            # mirroring how AiterAllreduceFusedRMSNormPattern drops the
            # residual output)
            return result[0], result[2]

        return _replacement


class AiterAllreduceFusedAddRMSNormGroupQuantFP8Pattern(
    BasePattern, VllmPatternReplacement
):
    """``fused_add`` variant of ``AiterAllreduceFusedRMSNormGroupQuantFP8Pattern``.

    Targets the dominant DSv3.2-style post-attention / post-MLP path:
    ``all_reduce -> fused_add_rms_norm -> group_fp8_quant``. Returns the
    FP8 quant output, the residual carry-over, and the per-group scale.
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        group_size: int = 128,
    ) -> None:
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.dtype = dtype
        self.group_size = group_size
        self.FUSED_AR_RMS_QUANT_OP = (
            rocm_aiter_ops.get_fused_allreduce_rmsnorm_quant_per_group_op()
        )
        self.quant_dtype = current_platform.fp8_dtype()
        self.quant_matcher = MatcherQuantFP8(
            QuantKey(
                dtype=self.quant_dtype,
                scale=ScaleDesc(torch.float32, False, GroupShape(1, group_size)),
                symmetric=True,
            ),
            match_rocm_aiter=True,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        # residual, input, weight
        return [
            self.empty(5, self.group_size),
            self.empty(5, self.group_size),
            self.empty(self.group_size),
        ]

    @property
    def pattern(self):
        def _pattern(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms, residual_out = vllm.ir.ops.fused_add_rms_norm(
                allreduce_output, residual, weight, self.epsilon
            )
            quant, scale = self.quant_matcher(rms)
            return quant, scale, residual_out

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result = self.FUSED_AR_RMS_QUANT_OP(
                input_=input,
                residual=residual,
                weight=weight.to(input.dtype),
                epsilon=self.epsilon,
                group_size=self.group_size,
            )
            # quant_out, scale_out, residual_out
            return result[0], result[2], result[1]

        return _replacement


class AiterAllreduceFusedAddRMSNormGroupQuantWithIndexerPattern(
    BasePattern, VllmPatternReplacement
):
    """Indexer-fan-out variant of ``AiterAllreduceFusedAddRMSNormGroupQuantFP8Pattern``.

    Targets the DSv3.2 post-attention / post-MLP path where the post-AR normed
    activation has two consumers: a per-group FP8 quant for ``fused_qkv_a_proj``
    *and* a bf16 ``rocm_unquantized_gemm`` for the indexer ``wk_weights_proj``.
    The single-consumer pattern above cannot fire when this fan-out is present,
    so without this pattern the standalone FP8 quant kernel survives unfused
    (~535us / decode step on DSv3.2 MI355X TP4).

    Lowers to ``rocm_aiter_fused_allreduce_rmsnorm_quant_per_group_with_bf16_norm``
    (the ``emit_bf16=True`` variant of the AR+RMS+QUANT launcher, which returns
    FP8 quant + scales + bf16 normed activations in one kernel) and rewires the
    indexer GEMM onto the emitted bf16 norm output. The RMS output is also a
    graph output in DSv3.2's residual carry; it is returned as a pattern output
    so the matcher can substitute the bf16 norm in its place.

    The trailing FP8 group-quant is matched via ``MatcherQuantFP8`` (consistent
    with the sibling patterns above), which traces both ``QuantFP8.forward_hip``
    and ``forward_native`` paths and so matches whichever op the call site
    lowers to (``vllm.triton_per_token_group_quant_fp8`` or
    ``vllm.rocm_aiter_group_fp8_quant``).
    """

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        group_size: int = 128,
    ) -> None:
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.dtype = dtype
        self.group_size = group_size
        self.FUSED_AR_RMS_QUANT_BF16_OP = (
            rocm_aiter_ops.get_fused_allreduce_rmsnorm_quant_per_group_with_bf16_norm_op()  # noqa: E501
        )
        self.quant_dtype = current_platform.fp8_dtype()
        self.quant_matcher = MatcherQuantFP8(
            QuantKey(
                dtype=self.quant_dtype,
                scale=ScaleDesc(torch.float32, False, GroupShape(1, group_size)),
                symmetric=True,
            ),
            match_rocm_aiter=True,
        )

    def get_inputs(self) -> list[torch.Tensor]:
        h = self.group_size
        indexer_out = 8
        return [
            self.empty(5, h),
            self.empty(5, h),
            self.empty(h),
            self.empty(indexer_out, h),
        ]

    @property
    def pattern(self):
        eps = self.epsilon

        def _pattern(
            residual: torch.Tensor,
            input_: torch.Tensor,
            norm_weight: torch.Tensor,
            indexer_weight: torch.Tensor,
        ) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            ar_out = tensor_model_parallel_all_reduce(input_)
            rms, res_out = vllm.ir.ops.fused_add_rms_norm(
                ar_out, residual, norm_weight, eps
            )
            q, s = self.quant_matcher(rms)
            idx = torch.ops.vllm.rocm_unquantized_gemm(rms, indexer_weight)
            return q, s, res_out, idx, rms

        return _pattern

    @property
    def replacement(self):
        gs = self.group_size
        eps = self.epsilon

        def _replacement(
            residual: torch.Tensor,
            input_: torch.Tensor,
            norm_weight: torch.Tensor,
            indexer_weight: torch.Tensor,
        ) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            fused = self.FUSED_AR_RMS_QUANT_BF16_OP(
                input_=input_,
                residual=residual,
                weight=norm_weight.to(input_.dtype),
                epsilon=eps,
                group_size=gs,
            )
            quant_out, residual_out, scale_out, bf16_norm = (
                fused[0],
                fused[1],
                fused[2],
                fused[3],
            )
            idx = torch.ops.vllm.rocm_unquantized_gemm(bf16_norm, indexer_weight)
            return quant_out, scale_out, residual_out, idx, bf16_norm

        return _replacement


class RocmAiterAllReduceFusionPass(VllmFusionPatternMatcherPass):
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "rocm_aiter_allreduce_fusion_pass")
        self.disabled = True
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size <= 1:
            logger.warning_once("AllReduce fusion pass is disabled for tp_size <= 1.")
            return

        if config.model_config is None:
            logger.warning_once(
                "AllReduce fusion pass is disabled for missing model_config."
            )
            return

        device_comm = get_tp_group().device_communicator
        if device_comm is None:
            logger.warning_once("Device communicator is required.")
            return

        ca_comm = getattr(device_comm, "ca_comm", None)
        if ca_comm is None:
            logger.warning_once("Custom Allreduce is required.")
            return
        self.ca_comm = ca_comm

        assert isinstance(ca_comm, CustomAllreduce)

        group = get_tp_group().cpu_group
        rocm_aiter_ops.initialize_aiter_allreduce(group, self.device)
        hidden_dim = config.model_config.get_hidden_size()
        element_size = torch.tensor([], dtype=self.model_dtype).element_size()
        max_size = rocm_aiter_ops.get_aiter_allreduce_max_size()
        if max_size is None:
            logger.warning("AITER allreduce fusion must be initialized")
            return

        # Aiter's fused_allreduce_rmsnorm kernel dispatches on hidden_dim.
        # Before aiter v0.1.12 the launcher was template-specialized on HIDDEN_DIM
        # and silently no-op'd for sizes outside {512, 1024, 2048, 4096}. From v0.1.12
        # hidden_dim is a runtime argument. Detect the older API via the missing
        # `_pool` attribute and skip fusion for unsupported sizes.
        # Ref (old kernel): https://github.com/ROCm/aiter/blob/6a0e7b26ccf33164785531212cc2ec2cde0b9243/csrc/include/custom_all_reduce.cuh#L2590
        aiter_ar = rocm_aiter_ops.get_aiter_allreduce()
        _AITER_OLD_FUSED_AR_RMS_HIDDEN = (512, 1024, 2048, 4096)
        if (
            aiter_ar is not None
            and not hasattr(aiter_ar, "_pool")
            and hidden_dim not in _AITER_OLD_FUSED_AR_RMS_HIDDEN
        ):
            logger.warning_once(
                "AITER allreduce-rmsnorm fusion disabled: aiter<0.1.12 "
                "only supports hidden_dim in %s; got %d. Upgrade aiter to "
                ">=0.1.12 to enable fusion for this model.",
                _AITER_OLD_FUSED_AR_RMS_HIDDEN,
                hidden_dim,
            )
            # Tear down aiter's custom-allreduce so its IPC handles don't
            # race with vllm's ca_comm on the unfused fallback path.
            with contextlib.suppress(Exception):
                rocm_aiter_ops.destroy_aiter_allreduce()
            return

        max_token_num = max_size // (hidden_dim * element_size)
        self.max_token_num = min(
            max_token_num,
            config.scheduler_config.max_num_batched_tokens,
        )

        # Only register the AR+RMS+per-group-FP8-quant patterns when the
        # running aiter exposes the kernel. Older aiter builds (pre PR #2823)
        # fall back to the AR+RMS-only fusion paired with PR #41825's
        # standalone RMS+quant fusion -- still correct, just leaves the
        # post-AR quant as a standalone kernel.
        supports_per_group_quant = (
            rocm_aiter_ops.has_fused_allreduce_rmsnorm_quant_per_group()
        )
        if not supports_per_group_quant:
            logger.warning_once(
                "AITER AR+RMS+per-group-FP8-quant fusion disabled: aiter "
                "build is missing 'fused_ar_rms_per_group_quant'. Upgrade "
                "aiter past PR #2823 to enable the trailing per-group "
                "FP8 quant fusion."
            )

        for epsilon in [1e-5, 1e-6]:
            # Quant-fused variants must register first so the pattern matcher
            # tries them before the AR+RMS-only variants. Otherwise the
            # AR+RMS-only fusion runs first and consumes the all_reduce node,
            # leaving the trailing quant op stranded as an unfused kernel.
            # Register larger subgraphs first (DeepSeek indexer fan-out, then
            # quant-only AR+RMS+quant, then AR+RMS-only).
            if supports_per_group_quant:
                self.register(
                    AiterAllreduceFusedAddRMSNormGroupQuantWithIndexerPattern(
                        epsilon,
                        self.model_dtype,
                        self.device,
                    )
                )
                self.register(
                    AiterAllreduceFusedRMSNormGroupQuantFP8Pattern(
                        epsilon,
                        self.model_dtype,
                        self.device,
                    )
                )
                self.register(
                    AiterAllreduceFusedAddRMSNormGroupQuantFP8Pattern(
                        epsilon,
                        self.model_dtype,
                        self.device,
                    )
                )

            self.register(
                AiterAllreduceFusedRMSNormPattern(
                    epsilon,
                    self.model_dtype,
                    self.device,
                )
            )
            self.register(
                AiterAllreduceFusedAddRMSNormPattern(
                    epsilon,
                    self.model_dtype,
                    self.device,
                )
            )

            # WARNING: This is a hack to clear the pattern matcher cache
            # and allow multiple values of epsilon.
            torch._inductor.pattern_matcher._seen_patterns.clear()

        self.disabled = False

        self.dump_patterns(config, self.pm_pass)

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        if self.disabled:
            logger.warning_once("AllReduce fusion pass is disabled.")
            return False
        return bool(compile_range.end <= self.max_token_num)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.pm_pass.apply(graph)
        VllmPatternMatcherPass.match_table[self.pass_name] += self.matched_count
        logger.debug(
            "%s Replaced %s patterns", self.__class__.__name__, self.matched_count
        )

    def __del__(self) -> None:
        if getattr(self, "disabled", True):
            return
        with contextlib.suppress(Exception):
            rocm_aiter_ops.destroy_aiter_allreduce()
