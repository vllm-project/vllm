# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib.util import find_spec
from typing import Optional

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op

from .vllm_inductor_pass import VllmInductorPass

FP8_DTYPE = current_platform.fp8_dtype()

if find_spec("flashinfer"):
    try:
        import flashinfer.comm as flashinfer_comm
        flashinfer_comm = (flashinfer_comm if hasattr(
            flashinfer_comm, "trtllm_allreduce_fusion") else None)
    except ImportError:
        flashinfer_comm = None
else:
    flashinfer_comm = None

logger = init_logger(__name__)

ALLREDUCE_OP = torch.ops.vllm.all_reduce.default
RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default
STATIC_FP8_QUANT_OP = torch.ops._C.static_scaled_fp8_quant.default
STATIC_FP4_QUANT_OP = torch.ops._C.scaled_fp4_quant.default


class BasePattern:

    def __init__(self, dtype: torch.dtype, device: str):
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()


class GEMMReduceScatterPattern(BasePattern):

    def get_inputs(self):
        mul = torch.empty([16, 4], device=self.device, dtype=self.dtype)
        mm_weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        return [mul, mm_weight]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(mul: torch.Tensor, mm_weight: torch.Tensor):
            mm = torch.ops.aten.mm.default(mul, mm_weight)
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            return reduce_scatter

        def replacement(mul: torch.Tensor, mm_weight: torch.Tensor):
            gemm_rs = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                mul,
                mm_weight,
                "avg",
                scatter_dim=0,
                group_name=self.tp.device_group.group_name,
            )

            return gemm_rs

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class AllGatherGEMMPattern(BasePattern):

    def get_inputs(self):
        x = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)

        return [x, weight]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

            return torch.ops.aten.mm.default(all_gather, weight)

        def replacement(
                x: torch.Tensor,
                weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_matmul(
                x,
                [weight],
                gather_dim=0,
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class ScaledMMReduceScatterPattern(BasePattern):

    def get_inputs(self):
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        mm_weight = torch.empty([16, 16], device=self.device,
                                dtype=FP8_DTYPE).contiguous().transpose(0, 1)
        scale_a = torch.empty([16, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)
        return [input, mm_weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(input: torch.Tensor, mat2: torch.Tensor,
                    scale_a: torch.Tensor,
                    scale_b: torch.Tensor) -> torch.Tensor:
            scaled_mm = torch.ops.aten._scaled_mm.default(input,
                                                          mat2=mat2,
                                                          scale_a=scale_a,
                                                          scale_b=scale_b,
                                                          bias=None,
                                                          scale_result=None,
                                                          out_dtype=self.dtype)
            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                scaled_mm,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name)
            return reduce_scatter

        def replacement(input: torch.Tensor, mat2: torch.Tensor,
                        scale_a: torch.Tensor,
                        scale_b: torch.Tensor) -> torch.Tensor:
            gemm_rs = torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
                input,
                mat2,
                scale_a,
                scale_b,
                "avg",
                scatter_dim=0,
                out_dtype=self.dtype,
                group_name=self.tp.device_group.group_name,
            )

            return gemm_rs

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class AllGatherScaledMMPattern(BasePattern):

    def get_inputs(self):
        x = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        weight = torch.empty([16, 16], device=self.device,
                             dtype=FP8_DTYPE).contiguous().transpose(0, 1)

        s1 = x.shape[0] * self.tp_size

        scale_a = torch.empty([s1, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        return [x, weight, scale_a, scale_b]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name)

            return torch.ops.aten._scaled_mm.default(all_gather,
                                                     mat2=weight,
                                                     scale_a=scale_a,
                                                     scale_b=scale_b,
                                                     bias=None,
                                                     scale_result=None,
                                                     out_dtype=self.dtype)

        def replacement(x: torch.Tensor, weight: torch.Tensor,
                        scale_a: torch.Tensor,
                        scale_b: torch.Tensor) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(  # noqa
                x,
                [weight],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[self.dtype],
                use_fast_accum=[False],
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class CutlassScaledMMReduceScatterPattern(BasePattern):

    def get_inputs(self):
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        mm_weight = torch.empty([16, 16], device=self.device,
                                dtype=FP8_DTYPE).contiguous().transpose(0, 1)
        scale_a = torch.empty([16, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        cutlass_mm_output = torch.empty([16, 16],
                                        device=self.device,
                                        dtype=self.dtype)
        return [input, mm_weight, scale_a, scale_b, cutlass_mm_output]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(input: torch.Tensor, weight: torch.Tensor,
                    scale_a: torch.Tensor, scale_b: torch.Tensor,
                    cutlass_mm_output: torch.Tensor) -> torch.Tensor:
            cutlass_scaled_mm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.cutlass_scaled_mm.default,
                out=cutlass_mm_output,
                a=input,
                b=weight,
                a_scales=scale_a,
                b_scales=scale_b,
                bias=None)

            reduce_scatter = torch.ops.vllm.reduce_scatter.default(
                cutlass_scaled_mm[1],
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name)
            return reduce_scatter

        def replacement(input: torch.Tensor, mat2: torch.Tensor,
                        scale_a: torch.Tensor, scale_b: torch.Tensor,
                        cutlass_mm_output: torch.Tensor) -> torch.Tensor:
            gemm_rs = torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
                input,
                mat2,
                scale_a,
                scale_b,
                "avg",
                scatter_dim=0,
                out_dtype=self.dtype,
                group_name=self.tp.device_group.group_name,
            )

            return gemm_rs

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class AllGatherCutlassScaledMMPattern(BasePattern):

    def get_inputs(self):
        x = torch.empty([8, 16], device=self.device, dtype=FP8_DTYPE)
        weight = torch.empty([16, 16], device=self.device,
                             dtype=FP8_DTYPE).contiguous().transpose(0, 1)

        s1 = x.shape[0] * self.tp_size

        scale_a = torch.empty([s1, 1], device=self.device, dtype=torch.float32)
        scale_b = torch.empty([1, 16], device=self.device, dtype=torch.float32)

        s2 = weight.shape[1]
        output = torch.empty([s1, s2], device=self.device, dtype=self.dtype)

        return [x, weight, scale_a, scale_b, output]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
            output: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name)

            cutlass_scaled_mm = torch.ops.higher_order.auto_functionalized(
                torch.ops._C.cutlass_scaled_mm.default,
                out=output,
                a=all_gather,
                b=weight,
                a_scales=scale_a,
                b_scales=scale_b,
                bias=None)
            return cutlass_scaled_mm[1]

        def replacement(x: torch.Tensor, weight: torch.Tensor,
                        scale_a: torch.Tensor, scale_b: torch.Tensor,
                        output: torch.Tensor) -> torch.Tensor:
            ag_output, mm_outputs = torch.ops.symm_mem.fused_all_gather_scaled_matmul(  # noqa
                x,
                [weight],
                scale_a,
                [scale_b],
                gather_dim=0,
                biases=[None],
                result_scales=[None],
                out_dtypes=[self.dtype],
                use_fast_accum=[False],
                group_name=self.tp.device_group.group_name,
            )
            return mm_outputs

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class AsyncTPPass(VllmInductorPass):

    def __init__(self, config: VllmConfig):
        super().__init__(config)

        # Enable symmetric memory for the TP process group
        enable_symm_mem_for_group(get_tp_group().device_group.group_name)
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="async_tp_pass")
        GEMMReduceScatterPattern(self.model_dtype,
                                 self.device).register(self.patterns)

        AllGatherGEMMPattern(self.model_dtype,
                             self.device).register(self.patterns)

        # These fusions are enabled only for bfloat16 models because
        # `scaled_mm` or `cutlass_scaled_mm` with per-token (row-wise) scaling
        # only supports bfloat16 as the output dtype.
        if self.model_dtype == torch.bfloat16:
            ScaledMMReduceScatterPattern(self.model_dtype,
                                         self.device).register(self.patterns)
            AllGatherScaledMMPattern(self.model_dtype,
                                     self.device).register(self.patterns)

            CutlassScaledMMReduceScatterPattern(
                self.model_dtype, self.device).register(self.patterns)
            AllGatherCutlassScaledMMPattern(
                self.model_dtype, self.device).register(self.patterns)

    def is_applicable_for_shape(self, shape: Optional[int]) -> bool:
        # only do replace for specific shapes
        tp_size = get_tensor_model_parallel_world_size()
        return shape is not None and shape % tp_size == 0

    def __call__(self, graph: fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_async_tp_pass")
        count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns with async TP pass.", count)
        self.dump_graph(graph, "after_async_tp_pass")
        self.end_and_log()


if flashinfer_comm is not None:
    _FI_WORKSPACE_TENSOR = None

    MiB = 1024 * 1024
    # Max size of the input tensor per world size
    # to use flashinfer fused allreduce
    _FI_MAX_SIZES = {
        2: 64 * MiB,  # 64MB
        4: MiB,  # 1MB
        6: MiB // 2,  # 512KB
        8: MiB // 2,  # 512KB
    }
    # opt for a more conservative default value
    # when world size is not in _FI_MAX_SIZES
    _DEFAULT_FI_MAX_SIZE = MiB // 2

    def call_trtllm_fused_allreduce_norm(
        allreduce_in: torch.Tensor,
        residual: torch.Tensor,
        rms_gamma: torch.Tensor,
        rms_eps: float,
        world_rank: int,
        world_size: int,
        launch_with_pdl: bool,
        trigger_completion_at_end: bool,
        fp32_acc: bool,
        max_token_num: int,
        pattern_code: int,
        fuse_rms_quant: bool,
        norm_out: Optional[torch.Tensor] = None,
        quant_out: Optional[torch.Tensor] = None,
        scale_out: Optional[torch.Tensor] = None,
        scale_factor: Optional[torch.Tensor] = None,
    ) -> None:
        num_tokens, hidden_size = allreduce_in.shape
        element_size = allreduce_in.element_size()
        current_tensor_size = num_tokens * hidden_size * element_size
        max_fusion_size = max_token_num * hidden_size * element_size
        use_flashinfer = current_tensor_size <= min(
            _FI_MAX_SIZES.get(world_size, _DEFAULT_FI_MAX_SIZE),
            max_fusion_size,
        )
        if use_flashinfer:
            assert (_FI_WORKSPACE_TENSOR is not None
                    ), "Flashinfer must be enabled when using flashinfer"
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
            flashinfer_comm.trtllm_allreduce_fusion(
                allreduce_in=allreduce_in,
                token_num=allreduce_in.shape[0],
                residual_in=residual,
                residual_out=residual_out,
                norm_out=norm_out,
                rms_gamma=rms_gamma,
                rms_eps=rms_eps,
                world_rank=world_rank,
                world_size=world_size,
                hidden_dim=allreduce_in.shape[-1],
                workspace_ptrs=_FI_WORKSPACE_TENSOR,
                launch_with_pdl=launch_with_pdl,
                use_oneshot=True,
                trigger_completion_at_end=trigger_completion_at_end,
                fp32_acc=fp32_acc,
                pattern_code=pattern_code,
                allreduce_out=None,
                quant_out=quant_out,
                scale_out=scale_out,
                # in vllm we only support swizzled layout
                layout_code=flashinfer_comm.FP4QuantizationSFLayout.SWIZZLED,
                scale_factor=scale_factor,
            )
        else:
            allreduce_out = tensor_model_parallel_all_reduce(allreduce_in)
            if (scale_factor is not None and scale_out is None
                    and fuse_rms_quant):
                # Do fused rms norm static fp8 quant fused op
                if norm_out is None:
                    torch.ops._C.fused_add_rms_norm_static_fp8_quant(
                        quant_out, allreduce_out, residual, rms_gamma,
                        scale_factor, rms_eps)
                else:
                    torch.ops._C.rms_norm_static_fp8_quant(
                        quant_out, allreduce_out, rms_gamma, scale_factor,
                        rms_eps)
            else:
                if norm_out is None:
                    torch.ops._C.fused_add_rms_norm(allreduce_out, residual,
                                                    rms_gamma, rms_eps)
                    norm_out = allreduce_out
                else:
                    torch.ops._C.rms_norm(norm_out, allreduce_out, rms_gamma,
                                          rms_eps)
                if scale_factor is not None:
                    if scale_out is not None:
                        torch.ops._C.scaled_fp4_quant(quant_out, norm_out,
                                                      scale_out, scale_factor)
                    else:
                        torch.ops._C.static_scaled_fp8_quant(
                            quant_out, norm_out, scale_factor)
            if scale_factor is None or norm_out is not None:
                # we need to return allreduce outpput
                # in cases of non quant fused AR + RMS norm
                # and fused AR + RMS norm + quant without fused add
                allreduce_in.copy_(allreduce_out)

    def call_trtllm_fused_allreduce_norm_fake(
            allreduce_in: torch.Tensor,
            residual: torch.Tensor,
            rms_gamma: torch.Tensor,
            rms_eps: float,
            world_rank: int,
            world_size: int,
            launch_with_pdl: bool,
            trigger_completion_at_end: bool,
            fp32_acc: bool,
            max_token_num: int,
            pattern_code: int,
            fuse_rms_quant: bool,
            norm_out: Optional[torch.Tensor] = None,
            quant_out: Optional[torch.Tensor] = None,
            scale_out: Optional[torch.Tensor] = None,
            scale_factor: Optional[torch.Tensor] = None) -> None:
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
        dispatch_key=current_platform.dispatch_key,
    )
    flashinfer_trtllm_fused_allreduce_norm = (
        torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default)


class FlashInferFusedAllReduceParams:
    """Parameters for FlashInfer fused allreduce operations."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        use_fp32_lamport: bool = False,
        max_token_num: int = 1024,
        fuse_rms_quant: bool = False,
    ):
        self.rank = rank
        self.world_size = world_size
        self.use_fp32_lamport = use_fp32_lamport
        self.trigger_completion_at_end = True
        self.launch_with_pdl = True
        self.fp32_acc = True
        self.use_oneshot = False
        self.max_token_num = max_token_num
        self.fuse_rms_quant = fuse_rms_quant

    def get_trtllm_fused_allreduce_kwargs(self):
        return {
            "world_rank": self.rank,
            "world_size": self.world_size,
            "launch_with_pdl": self.launch_with_pdl,
            "trigger_completion_at_end": self.trigger_completion_at_end,
            "fp32_acc": self.fp32_acc,
            "max_token_num": self.max_token_num,
            "fuse_rms_quant": self.fuse_rms_quant,
        }


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
        device: str,
        allreduce_params: FlashInferFusedAllReduceParams,
    ):
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.allreduce_params = allreduce_params

    def get_inputs(self):
        input = torch.empty([1, 8, 4], device=self.device, dtype=self.dtype)
        rms_result = torch.empty([1, 8, 4],
                                 device=self.device,
                                 dtype=self.dtype)
        weight = torch.empty([4], device=self.device, dtype=self.dtype)

        return [input, rms_result, weight]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(input: torch.Tensor, rms_result: torch.Tensor,
                    weight: torch.Tensor):
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms = auto_functionalized(
                RMS_OP,
                result=rms_result,
                input=allreduce_output,
                weight=weight,
                epsilon=self.epsilon,
            )
            # rms_result, allreduce_output
            return rms[1], allreduce_output

        def replacement(input: torch.Tensor, rms_result: torch.Tensor,
                        weight: torch.Tensor):
            residual = torch.zeros_like(input)
            allreduce = auto_functionalized(
                flashinfer_trtllm_fused_allreduce_norm,
                allreduce_in=input,
                residual=residual,
                norm_out=rms_result,
                quant_out=None,
                scale_out=None,
                rms_gamma=weight,
                rms_eps=self.epsilon,
                pattern_code=flashinfer_comm.AllReduceFusionPattern.
                kARResidualRMSNorm,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )
            # rms_result, allreduce_in
            return allreduce[3], allreduce[1]

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


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
        device: str,
        allreduce_params: FlashInferFusedAllReduceParams,
    ):
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.allreduce_params = allreduce_params

    def get_inputs(self):
        input = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        residual = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
        return [
            residual,
            input,
            weight,
        ]

    def register(self, pm_pass: PatternMatcherPass):

        def pattern(residual: torch.Tensor, input: torch.Tensor,
                    weight: torch.Tensor):
            allreduce_output = tensor_model_parallel_all_reduce(input)
            rms = auto_functionalized(
                RMS_ADD_OP,
                input=allreduce_output,
                residual=residual,
                weight=weight,
                epsilon=self.epsilon,
            )
            # input, residual
            return rms[1], rms[2]

        def replacement(residual: torch.Tensor, input: torch.Tensor,
                        weight: torch.Tensor):
            allreduce = auto_functionalized(
                flashinfer_trtllm_fused_allreduce_norm,
                allreduce_in=input,
                residual=residual,
                norm_out=None,
                quant_out=None,
                scale_out=None,
                rms_gamma=weight,
                rms_eps=self.epsilon,
                pattern_code=flashinfer_comm.AllReduceFusionPattern.
                kARResidualRMSNorm,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )
            # allreduce_in, residual
            return allreduce[1], allreduce[2]

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class AllReduceFusedRMSNormStaticQuantFP8Pattern(BasePattern):
    """
    This pattern replaces the allreduce + rms norm (without residual) 
    + static fp8 quant with fused flashinfer implementation.
    Applies to allreduce + rmsnorm + quant before attn 
    in the first Transformer block.
    """

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str,
                 allreduce_params: FlashInferFusedAllReduceParams):
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.allreduce_params = allreduce_params
        self.quant_dtype = torch.float8_e4m3fn

    def register(self, pm_pass: PatternMatcherPass):

        def get_inputs():
            input = torch.zeros([1, 8, 4],
                                device=self.device,
                                dtype=self.dtype)
            rmsnorm_result = torch.empty([1, 8, 4],
                                         device=self.device,
                                         dtype=self.dtype)
            quant_result = torch.empty([1, 8, 4],
                                       device=self.device,
                                       dtype=self.quant_dtype)
            weight = torch.empty([4], device=self.device, dtype=self.dtype)
            scale = torch.tensor(1.0, device=self.device, dtype=torch.float32)
            return [input, rmsnorm_result, quant_result, weight, scale]

        def pattern(
            input: torch.Tensor,
            rmsnorm_result: torch.Tensor,
            quant_result: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            all_reduce = tensor_model_parallel_all_reduce(input)
            rmsnorm_out_tuple = auto_functionalized(RMS_OP,
                                                    result=rmsnorm_result,
                                                    input=all_reduce,
                                                    weight=weight,
                                                    epsilon=self.epsilon)

            quant_out_tuple = auto_functionalized(STATIC_FP8_QUANT_OP,
                                                  result=quant_result,
                                                  input=rmsnorm_out_tuple[1],
                                                  scale=scale)

            # quant_out, allreduce_output
            return quant_out_tuple[1], all_reduce

        def replacement(input: torch.Tensor, result_rms: torch.Tensor,
                        quant_result: torch.Tensor, weight: torch.Tensor,
                        scale: torch.Tensor):
            residual = torch.zeros_like(input)
            allreduce = auto_functionalized(
                flashinfer_trtllm_fused_allreduce_norm,
                allreduce_in=input,
                residual=residual,
                norm_out=result_rms,
                quant_out=quant_result,
                scale_out=None,
                rms_gamma=weight,
                rms_eps=self.epsilon,
                pattern_code=flashinfer_comm.AllReduceFusionPattern.
                kARResidualRMSNormFP8Quant,  # we don't use norm_out afterwards
                scale_factor=scale,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )

            # quant_out, allreduce_output
            return allreduce[4], allreduce[1]

        pm.register_replacement(pattern, replacement, get_inputs(),
                                pm.fwd_only, pm_pass)


class AllReduceFusedAddRMSNormStaticQuantFP8Pattern(BasePattern):
    """
    This pattern replaces the allreduce + rms norm (with residual)
    + static fp8 quant with fused flashinfer implementation.
    Applies to o_proj + rmsnorm after attn + quant and 
    mlp + rmsnorm + quant before attn.
    """

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str,
                 allreduce_params: FlashInferFusedAllReduceParams):
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.allreduce_params = allreduce_params
        self.quant_dtype = torch.float8_e4m3fn

    def register(self, pm_pass: PatternMatcherPass):

        def get_inputs():
            input = torch.empty([4, 4], device=self.device, dtype=self.dtype)

            residual = torch.empty([4, 4],
                                   device=self.device,
                                   dtype=self.dtype)
            weight = torch.empty([4, 4], device=self.device, dtype=self.dtype)
            quant_result = torch.empty([4, 4],
                                       device=self.device,
                                       dtype=self.quant_dtype)
            scale = torch.empty([1, 1],
                                device=self.device,
                                dtype=torch.float32)

            return [
                quant_result,
                residual,
                input,
                weight,
                scale,
            ]

        def pattern(
            quant_result: torch.Tensor,
            residual: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ):
            allreduce_output = tensor_model_parallel_all_reduce(input)

            fused_add_rmsnorm_out_tuple = \
            auto_functionalized(
                RMS_ADD_OP,
                input=allreduce_output,
                residual=residual,
                weight=weight,
                epsilon=self.epsilon)
            quant_out_tuple = auto_functionalized(
                STATIC_FP8_QUANT_OP,
                result=quant_result,
                input=fused_add_rmsnorm_out_tuple[1],
                scale=scale)

            # quant_out, allreduce_output
            return quant_out_tuple[1], fused_add_rmsnorm_out_tuple[2]

        def replacement(quant_result: torch.Tensor, residual: torch.Tensor,
                        input: torch.Tensor, weight: torch.Tensor,
                        scale: torch.Tensor):
            allreduce = auto_functionalized(
                flashinfer_trtllm_fused_allreduce_norm,
                allreduce_in=input,
                residual=residual,
                norm_out=None,
                quant_out=quant_result,
                scale_out=None,
                rms_gamma=weight,
                rms_eps=self.epsilon,
                pattern_code=flashinfer_comm.AllReduceFusionPattern.
                kARResidualRMSNormFP8Quant,  # we don't use norm_out afterwards
                scale_factor=scale,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )
            # # quant_out, rms_norm_residual
            return allreduce[4], allreduce[2]

        pm.register_replacement(pattern, replacement, get_inputs(),
                                pm.fwd_only, pm_pass)


class AllReduceFusedRMSNormStaticQuantNVFP4Pattern(BasePattern):
    """
    This pattern replaces the allreduce + rms norm (without residual) 
    + static nvfp4 quant with fused flashinfer implementation.
    Applies to allreduce + rmsnorm + quant before attn 
    in the first Transformer block.
    """

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str,
                 allreduce_params: FlashInferFusedAllReduceParams):
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.allreduce_params = allreduce_params

    def register(self, pm_pass: PatternMatcherPass):

        def get_inputs():
            input = torch.empty([1, 16, 16],
                                device=self.device,
                                dtype=self.dtype)

            rmsnorm_result = torch.empty([1, 16, 16],
                                         device=self.device,
                                         dtype=self.dtype)
            quant_result = torch.empty((16, 8),
                                       device=self.device,
                                       dtype=torch.uint8)
            input_global_scale = torch.empty([1, 1],
                                             device=self.device,
                                             dtype=torch.float32)
            weight = torch.empty([16], device=self.device, dtype=self.dtype)
            output_scale = torch.empty([128, 4],
                                       device=self.device,
                                       dtype=torch.int32)

            return [
                input, rmsnorm_result, quant_result, weight,
                input_global_scale, output_scale
            ]

        def pattern(
            input: torch.Tensor,
            rmsnorm_result: torch.Tensor,
            quant_result: torch.Tensor,
            weight: torch.Tensor,
            input_global_scale: torch.Tensor,
            output_scale: torch.Tensor,
        ):
            all_reduce = tensor_model_parallel_all_reduce(input)
            rmsnorm_out_tuple = auto_functionalized(RMS_OP,
                                                    result=rmsnorm_result,
                                                    input=all_reduce,
                                                    weight=weight,
                                                    epsilon=self.epsilon)

            quant_out_tuple = auto_functionalized(
                STATIC_FP4_QUANT_OP,
                output=quant_result,
                input=rmsnorm_out_tuple[1],
                output_scale=output_scale,
                input_scale=input_global_scale)

            # quant_out, allreduce_output, output_scale
            return quant_out_tuple[1], all_reduce, quant_out_tuple[2]

        def replacement(input: torch.Tensor, result_rms: torch.Tensor,
                        quant_result: torch.Tensor, weight: torch.Tensor,
                        input_global_scale: torch.Tensor,
                        output_scale: torch.Tensor):
            residual = torch.zeros_like(input)
            allreduce = auto_functionalized(
                flashinfer_trtllm_fused_allreduce_norm,
                allreduce_in=input,
                residual=residual,
                norm_out=result_rms,
                quant_out=quant_result,
                scale_out=output_scale,
                rms_gamma=weight,
                rms_eps=self.epsilon,
                pattern_code=flashinfer_comm.AllReduceFusionPattern.
                kARResidualRMSNormFP4Quant,  # we don't use norm_out afterwards
                scale_factor=input_global_scale,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )

            # quant_out, allreduce_output, output_scale
            return allreduce[4], allreduce[1], allreduce[5]

        pm.register_replacement(pattern, replacement, get_inputs(),
                                pm.fwd_only, pm_pass)


class AllReduceFusedAddRMSNormStaticQuantNVFP4Pattern(BasePattern):
    """
    This pattern replaces the allreduce + rms norm (with residual)
    + static nvfp4 quant with fused flashinfer implementation.
    Applies to o_proj + rmsnorm after attn + quant and 
    mlp + rmsnorm + quant before attn.
    """

    def __init__(self, epsilon: float, dtype: torch.dtype, device: str,
                 allreduce_params: FlashInferFusedAllReduceParams):
        super().__init__(dtype, device)
        self.epsilon = epsilon
        self.allreduce_params = allreduce_params

    def register(self, pm_pass: PatternMatcherPass):

        def get_inputs():
            input = torch.empty([16, 16], device=self.device, dtype=self.dtype)

            residual = torch.empty([16, 16],
                                   device=self.device,
                                   dtype=self.dtype)
            weight = torch.empty([16, 16],
                                 device=self.device,
                                 dtype=self.dtype)
            quant_result = torch.empty((16, 8),
                                       device=self.device,
                                       dtype=torch.uint8)
            input_global_scale = torch.empty([1, 1],
                                             device=self.device,
                                             dtype=torch.float32)
            output_scale = torch.empty([128, 4],
                                       device=self.device,
                                       dtype=torch.int32)

            return [
                quant_result,
                residual,
                input,
                output_scale,
                weight,
                input_global_scale,
            ]

        def pattern(quant_result: torch.Tensor, residual: torch.Tensor,
                    input: torch.Tensor, output_scale: torch.Tensor,
                    weight: torch.Tensor, input_global_scale: torch.Tensor):
            allreduce_output = tensor_model_parallel_all_reduce(input)

            fused_add_rmsnorm_out_tuple = \
            auto_functionalized(
                RMS_ADD_OP,
                input=allreduce_output,
                residual=residual,
                weight=weight,
                epsilon=self.epsilon)
            quant_out_tuple = auto_functionalized(
                STATIC_FP4_QUANT_OP,
                output=quant_result,
                input=fused_add_rmsnorm_out_tuple[1],
                output_scale=output_scale,
                input_scale=input_global_scale)

            # quant_out, allreduce_output, output_scale
            return quant_out_tuple[1], fused_add_rmsnorm_out_tuple[
                2], quant_out_tuple[2]

        def replacement(quant_result: torch.Tensor, residual: torch.Tensor,
                        input: torch.Tensor, output_scale: torch.Tensor,
                        weight: torch.Tensor,
                        input_global_scale: torch.Tensor):
            allreduce = auto_functionalized(
                flashinfer_trtllm_fused_allreduce_norm,
                allreduce_in=input,
                residual=residual,
                norm_out=None,
                quant_out=quant_result,
                scale_out=output_scale,
                rms_gamma=weight,
                rms_eps=self.epsilon,
                pattern_code=flashinfer_comm.AllReduceFusionPattern.
                kARResidualRMSNormFP4Quant,  # we don't use norm_out afterwards
                scale_factor=input_global_scale,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )
            # quant_out, rms_norm_residual, output_scale
            return allreduce[4], allreduce[2], allreduce[5]

        pm.register_replacement(pattern, replacement, get_inputs(),
                                pm.fwd_only, pm_pass)


class AllReduceFusionPass(VllmInductorPass):

    def __init__(self, config: VllmConfig):
        super().__init__(config)
        self.disabled = True
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size <= 1:
            return
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="all_reduce_fusion_pass")
        if config.model_config is None:
            return
        self.hidden_dim = config.model_config.get_hidden_size()
        self.group = get_tp_group().device_group
        rank = get_tensor_model_parallel_rank()
        use_fp32_lamport = self.model_dtype == torch.float32
        if flashinfer_comm is None:
            logger.warning(
                "Flashinfer is not installed or comm module not found, "
                "skipping allreduce fusion pass")
            return
        # Check if the world size is supported
        if self.tp_size not in _FI_MAX_SIZES:
            logger.warning(
                "Flashinfer allreduce fusion is not "
                "supported for world size %s",
                self.tp_size,
            )
            return
        max_num_token = min(
            _FI_MAX_SIZES.get(self.tp_size, _DEFAULT_FI_MAX_SIZE) //
            (self.hidden_dim * self.tp_size * (4 if use_fp32_lamport else 2)),
            config.compilation_config.pass_config.
            fi_allreduce_fusion_max_token_num)
        self.ipc_handles, workspace_tensor = (
            flashinfer_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                tp_rank=rank,
                tp_size=self.tp_size,
                max_token_num=max_num_token,
                hidden_dim=self.hidden_dim,
                group=self.group,
                use_fp32_lamport=use_fp32_lamport,
            ))

        global _FI_WORKSPACE_TENSOR
        _FI_WORKSPACE_TENSOR = workspace_tensor
        self.allreduce_params = FlashInferFusedAllReduceParams(
            rank=rank,
            world_size=self.tp_size,
            use_fp32_lamport=use_fp32_lamport,
            max_token_num=max_num_token,
            # fuse rms norm static fp8 quant fused op
            # in fallback path, when we don't use flashinfer
            fuse_rms_quant=config.compilation_config.pass_config.enable_fusion)

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

    def __call__(self, graph: fx.Graph):
        if self.disabled:
            return
        self.begin()
        self.dump_graph(graph, "before_all_reduce_fusion_pass")
        count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", count)
        self.dump_graph(graph, "after_all_reduce_fusion_pass")
        self.end_and_log()

    def __del__(self):
        if self.disabled:
            return
        if flashinfer_comm is not None:
            flashinfer_comm.trtllm_destroy_ipc_workspace_for_all_reduce(
                self.ipc_handles, self.group)
