# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer CuTe AllReduce/RMSNorm for the Qwen TP8 shape family."""

import inspect
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator

logger = init_logger(__name__)
MAX_TOKENS = 4096
LL_MAX_TOKENS = 16
HIDDEN_SIZE = 8192
TOP_K = 10


@lru_cache(maxsize=1)
def build_policy():
    from flashinfer.comm.mnnvl_cutedsl import BT_ONLY_CONFIG, LL_ONLY_CONFIG
    from flashinfer.comm.mnnvl_cutedsl.config import (
        MNNVLCuteDSLConfig,
        MRangeDispatch,
    )

    def profile(config):
        return config.resolve(
            tp_size=8,
            hidden_size=HIDDEN_SIZE,
            top_k=TOP_K,
            dtype=torch.bfloat16,
            capacity_m=1,
        )

    ll, bt = profile(LL_ONLY_CONFIG), profile(BT_ONLY_CONFIG)
    routes = MRangeDispatch(
        upper_bounds=(LL_MAX_TOKENS, MAX_TOKENS),
        targets=(ll.all_reduce_routes.targets[0], bt.all_reduce_routes.targets[0]),
    )
    # The workspace validates both operation domains, including the unused
    # finalize path. Extend the final BT preset through this policy's capacity.
    finalize = replace(
        bt.finalize_routes,
        upper_bounds=(*bt.finalize_routes.upper_bounds[:-1], MAX_TOKENS),
    )
    return MNNVLCuteDSLConfig(
        profiles=(replace(bt, all_reduce_routes=routes, finalize_routes=finalize),)
    )


def enabled_for_config(config: "VllmConfig") -> bool:
    if not config.kernel_config.enable_cute_allreduce:
        return False
    model, parallel = config.model_config, config.parallel_config
    if model is None:
        return False
    text = model.hf_text_config
    return bool(
        getattr(text, "model_type", None) == "qwen3_5_moe_text"
        and getattr(text, "hidden_size", None) == HIDDEN_SIZE
        and getattr(text, "num_experts_per_tok", None) == TOP_K
        and model.dtype == torch.bfloat16
        and not model.enable_sleep_mode
        and parallel.tensor_parallel_size == 8
        and parallel.pipeline_parallel_size == parallel.data_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and not parallel.use_sequence_parallel_moe
        and not parallel.enable_dbo
        and not parallel.enable_fault_tolerance
        and config.lora_config is None
        and not envs.VLLM_BATCH_INVARIANT
        and config.compilation_config.pass_config.fuse_allreduce_rms
    )


class CuteAllReduce:
    """One TP group's workspaces, allocated before compilation or graph capture.

    FlashInfer bakes output dtype and residual mode into its kernels. Keep one
    workspace for each mode; sequential target and draft layers share them.
    """

    def __init__(self, tp: "GroupCoordinator", epsilon: float):
        from flashinfer.comm.mnnvl_cutedsl_ar import (
            MNNVLCuteDSLAllReduceFusionWorkspace,
        )

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CuTe workspaces must be initialized before capture")
        if tp.world_size != 8 or torch.cuda.get_device_capability(tp.device) != (10, 7):
            raise ValueError("This CuTe policy requires TP8 on SM107")
        if (
            "output_dtype"
            not in inspect.signature(MNNVLCuteDSLAllReduceFusionWorkspace).parameters
        ):
            raise RuntimeError("FlashInfer CuTe static-FP8 output support is required")
        # Symmetric-memory registration needs the PyTorch NCCL communicator.
        # vLLM collectives can have used only PyNccl before this point.
        torch.distributed.barrier(group=tp.device_group, device_ids=[tp.device.index])
        self.epsilon = epsilon
        self.workspaces = {}
        try:
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                for add_residual in (False, True):
                    self.workspaces[dtype, add_residual] = (
                        MNNVLCuteDSLAllReduceFusionWorkspace(
                            tp.world_size,
                            tp.rank_in_group,
                            MAX_TOKENS,
                            HIDDEN_SIZE,
                            torch.bfloat16,
                            group=tp.device_group,
                            top_k=TOP_K,
                            rms_eps=epsilon,
                            weight_bias=1.0,
                            include_shared_expert=True,
                            config=build_policy(),
                            output_dtype=dtype,
                            add_residual=add_residual,
                        )
                    )
        except Exception:
            self.destroy()
            raise

    def get_workspace(self, epsilon: float, dtype: torch.dtype, residual: bool):
        if epsilon != self.epsilon:
            raise ValueError("CuTe RMS epsilon differs from the initialized policy")
        return self.workspaces[dtype, residual]

    def destroy(self) -> None:
        for workspace in self.workspaces.values():
            workspace.destroy()
        self.workspaces.clear()


def initialize_for_config(config: "VllmConfig") -> None:
    if not config.kernel_config.enable_cute_allreduce:
        return
    if not enabled_for_config(config):
        raise ValueError("CuTe AllReduce is unsupported for this model/configuration")
    tp = get_tp_group()
    communicator = tp.device_communicator
    from .cuda_communicator import CudaCommunicator

    assert isinstance(communicator, CudaCommunicator)
    assert config.model_config is not None
    if communicator.cute_allreduce is None:
        epsilon = float(config.model_config.hf_text_config.rms_norm_eps)
        communicator.cute_allreduce = CuteAllReduce(tp, epsilon)
        logger.info("Initialized FlashInfer CuTe AllReduce for Qwen TP8")


def get_workspace(epsilon: float, dtype: torch.dtype, residual: bool):
    communicator = get_tp_group().device_communicator
    from .cuda_communicator import CudaCommunicator

    if (
        not isinstance(communicator, CudaCommunicator)
        or communicator.cute_allreduce is None
    ):
        raise RuntimeError("CuTe AllReduce was not initialized before execution")
    return communicator.cute_allreduce.get_workspace(epsilon, dtype, residual)


def output_dtype(scale: torch.Tensor | None) -> torch.dtype:
    return torch.bfloat16 if scale is None else torch.float8_e4m3fn


def cute_allreduce_norm(
    input: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Functional TP AR/Gemma RMSNorm with optional static FP8 output.

    Residual and norm weights must be replicated across the TP group.
    """
    if 0 < input.shape[0] <= MAX_TOKENS:
        from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion

        dtype = output_dtype(scale)
        workspace = get_workspace(epsilon, dtype, residual is not None)
        output = torch.empty_like(input, dtype=dtype)
        updated = torch.empty_like(input)
        pattern = (
            AllReduceFusionPattern.kARResidualRMSNorm
            if scale is None
            else AllReduceFusionPattern.kARResidualRMSNormFP8Quant
        )
        allreduce_fusion(
            input=input,
            workspace=workspace,
            pattern=pattern,
            residual_in=residual,
            residual_out=updated,
            rms_gamma=weight,
            rms_eps=epsilon,
            weight_bias=1.0,
            norm_out=output if scale is None else None,
            quant_out=output if scale is not None else None,
            scale_factor=scale,
            launch_with_pdl=True,
        )
        return output, updated

    import vllm.ir

    reduced = tensor_model_parallel_all_reduce(input)
    gamma = weight.float() + 1.0
    if residual is None:
        output = vllm.ir.ops.rms_norm(reduced, gamma, epsilon)
        updated = reduced
    else:
        output, updated = vllm.ir.ops.fused_add_rms_norm(
            reduced, residual.clone(), gamma, epsilon
        )
    if scale is not None:
        from vllm import _custom_ops as ops

        output, _ = ops.scaled_fp8_quant(output, scale)
    return output, updated


def _cute_allreduce_norm_fake(input, residual, weight, scale, epsilon):
    return torch.empty_like(input, dtype=output_dtype(scale)), torch.empty_like(input)


direct_register_custom_op(
    op_name="cute_allreduce_norm",
    op_func=cute_allreduce_norm,
    fake_impl=_cute_allreduce_norm_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
