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
from vllm.utils import direct_register_custom_op

from .vllm_inductor_pass import VllmInductorPass

if find_spec("flashinfer"):
    try:
        import flashinfer.comm as flashinfer_comm
        flashinfer_comm = (flashinfer_comm if hasattr(
            flashinfer_comm, "trtllm_allreduce_fusion") else None)
    except ImportError:
        flashinfer_comm = None
else:
    flashinfer_comm = None
from vllm.platforms import current_platform

logger = init_logger(__name__)

ALLREDUCE_OP = torch.ops.vllm.all_reduce.default
RMS_OP = torch.ops._C.rms_norm.default
RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default


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

    def is_applicable_for_shape(self, shape: Optional[int]) -> bool:
        # only do replace for specific shapes
        tp_size = get_tensor_model_parallel_world_size()
        return shape is not None and shape % tp_size == 0

    def __call__(self, graph: fx.Graph):
        self.begin()
        self.dump_graph(graph, "before_async_tp_pass")
        count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", count)
        self.dump_graph(graph, "after_async_tp_pass")
        self.end_and_log()


if flashinfer_comm is not None:
    _FI_WORKSPACE_TENSOR = None

    MiB = 1024 * 1024
    # Max size of the input tensor per world size
    # to use flashinfer fused allreduce
    _FI_MAX_SIZES = {
        2: MiB,  # 1MB
        4: MiB,  # 1MB
        6: MiB // 2,  # 512KB
        8: MiB // 2,  # 512KB
    }

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
        norm_out: Optional[torch.Tensor] = None,
    ) -> None:
        use_flashinfer = allreduce_in.shape[0] * allreduce_in.shape[
            1] * allreduce_in.element_size() <= min(
                _FI_MAX_SIZES[world_size],
                max_token_num * allreduce_in.shape[0] *
                allreduce_in.element_size(),
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
                pattern_code=flashinfer_comm.AllReduceFusionPattern.
                kARResidualRMSNorm,
                allreduce_out=None,
                quant_out=None,
                scale_out=None,
                layout_code=None,
                scale_factor=None,
            )
        else:
            allreduce_out = tensor_model_parallel_all_reduce(allreduce_in)
            if norm_out is None:
                torch.ops._C.fused_add_rms_norm(allreduce_out, residual,
                                                rms_gamma, rms_eps)
            else:
                torch.ops._C.rms_norm(norm_out, allreduce_out, rms_gamma,
                                      rms_eps)
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
        norm_out: Optional[torch.Tensor] = None,
    ) -> None:
        pass

    direct_register_custom_op(
        op_name="flashinfer_trtllm_fused_allreduce_norm",
        op_func=call_trtllm_fused_allreduce_norm,
        mutates_args=[
            "allreduce_in",
            "residual",
            "norm_out",
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
    ):
        self.rank = rank
        self.world_size = world_size
        self.use_fp32_lamport = use_fp32_lamport
        self.trigger_completion_at_end = True
        self.launch_with_pdl = True
        self.fp32_acc = True
        self.use_oneshot = False
        self.max_token_num = max_token_num

    def get_trtllm_fused_allreduce_kwargs(self):
        return {
            "world_rank": self.rank,
            "world_size": self.world_size,
            "launch_with_pdl": self.launch_with_pdl,
            "trigger_completion_at_end": self.trigger_completion_at_end,
            "fp32_acc": self.fp32_acc,
            "max_token_num": self.max_token_num,
        }


class AllReduceRMSNORMPattern(BasePattern):

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
            all_reduce_output = tensor_model_parallel_all_reduce(input)
            rms = auto_functionalized(
                RMS_OP,
                result=rms_result,
                input=all_reduce_output,
                weight=weight,
                epsilon=self.epsilon,
            )
            return rms[1], all_reduce_output

        def replacement(input: torch.Tensor, rms_result: torch.Tensor,
                        weight: torch.Tensor):
            residual = torch.zeros_like(input)
            allreduce = auto_functionalized(
                torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default,
                allreduce_in=input,
                residual=residual,
                norm_out=rms_result,
                rms_gamma=weight,
                rms_eps=self.epsilon,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )

            return allreduce[3], allreduce[1]

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class AllReduceFusedAddRMSNormPattern(BasePattern):

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
            all_reduce_output = tensor_model_parallel_all_reduce(input)
            rms = auto_functionalized(
                RMS_ADD_OP,
                input=all_reduce_output,
                residual=residual,
                weight=weight,
                epsilon=self.epsilon,
            )
            return rms[1], rms[2]

        def replacement(residual: torch.Tensor, input: torch.Tensor,
                        weight: torch.Tensor):
            allreduce = auto_functionalized(
                torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default,
                allreduce_in=input,
                residual=residual,
                rms_gamma=weight,
                rms_eps=self.epsilon,
                norm_out=None,
                **self.allreduce_params.get_trtllm_fused_allreduce_kwargs(),
            )
            return allreduce[1], allreduce[2]

        pm.register_replacement(pattern, replacement, self.get_inputs(),
                                pm.fwd_only, pm_pass)


class AllReduceFusionPass(VllmInductorPass):

    def __init__(self, config: VllmConfig, max_token_num: int):
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

        self.ipc_handles, workspace_tensor = (
            flashinfer_comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
                tp_rank=rank,
                tp_size=self.tp_size,
                max_token_num=max_token_num,
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
            max_token_num=max_token_num,
        )

        for epsilon in [1e-5, 1e-6]:
            AllReduceRMSNORMPattern(
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
            flashinfer_comm.trtllm_destroy_ipc_workspace(
                self.ipc_handles, self.group)
