# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer CuTe AllReduce/Gemma RMSNorm with communicator-owned workspaces."""

import inspect
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from .base_device_communicator import DeviceCommunicatorBase

logger = init_logger(__name__)
# vLLM allocation/dispatch policy, not a FlashInfer kernel limit. Larger
# requests use the ordinary collective without growing symmetric workspaces.
MAX_TOKENS = 4096
# Published FlashInfer BF16 profile pairs: (TP size, hidden size).
SUPPORTED_SHAPES = frozenset({(4, 5120), (8, 5120), (8, 8192), (16, 8192)})
SUPPORTED_DEVICE_CAPABILITIES = frozenset({100, 103, 107})
# Measured vLLM crossover overrides: (device capability, TP size, hidden size).
# Other shapes/devices retain FlashInfer's published protocol selection.
LL_TOKEN_THRESHOLDS = {(107, 8, 8192): 16}


@lru_cache
def build_policy(tp_size: int, hidden_size: int, device_capability: int):
    from flashinfer.comm.mnnvl_cutedsl import (
        BT_ONLY_CONFIG,
        DEFAULT_CONFIG,
        LL_ONLY_CONFIG,
    )
    from flashinfer.comm.mnnvl_cutedsl.config import (
        MNNVLCuteDSLConfig,
        MRangeDispatch,
    )

    # The workspace API keys both AR and finalize by top_k. AR does not use
    # experts, so select an existing profile independent of the model's top_k.
    profile = next(
        (
            p
            for p in DEFAULT_CONFIG.profiles
            if (p.tp_size, p.hidden_size) == (tp_size, hidden_size)
            and p.dtype == torch.bfloat16
        ),
        None,
    )
    if profile is None:
        return None
    ll_max_tokens = LL_TOKEN_THRESHOLDS.get((device_capability, tp_size, hidden_size))
    if ll_max_tokens is None:
        return MNNVLCuteDSLConfig(profiles=(profile,))

    shape = dict(
        tp_size=tp_size,
        hidden_size=hidden_size,
        top_k=profile.top_k,
        dtype=torch.bfloat16,
        capacity_m=1,
    )
    ll, bt = LL_ONLY_CONFIG.resolve(**shape), BT_ONLY_CONFIG.resolve(**shape)
    routes = MRangeDispatch(
        upper_bounds=(ll_max_tokens, MAX_TOKENS),
        targets=(ll.all_reduce_routes.targets[0], bt.all_reduce_routes.targets[0]),
    )
    # Workspace construction validates *both* route tables to MAX_TOKENS,
    # although this consumer only calls AR. Extend the last BT finalize route
    # to satisfy that API contract; it does not enable or invoke MoE finalize.
    finalize = replace(
        bt.finalize_routes,
        upper_bounds=(*bt.finalize_routes.upper_bounds[:-1], MAX_TOKENS),
    )
    return MNNVLCuteDSLConfig(
        profiles=(replace(bt, all_reduce_routes=routes, finalize_routes=finalize),)
    )


def supports_config(config: "VllmConfig") -> bool:
    model, parallel = config.model_config, config.parallel_config
    if model is None:
        return False
    text = model.hf_text_config
    spec = config.speculative_config
    draft = getattr(spec, "draft_model_config", None)
    if draft is not None and (
        draft.get_hidden_size() != model.get_hidden_size()
        or getattr(draft.hf_text_config, "rms_norm_eps", None)
        != getattr(text, "rms_norm_eps", None)
    ):
        return False
    return bool(
        (parallel.tensor_parallel_size, model.get_hidden_size()) in SUPPORTED_SHAPES
        and getattr(text, "rms_norm_eps", None) is not None
        and model.dtype == torch.bfloat16
        and not model.enable_sleep_mode
        and parallel.pipeline_parallel_size == parallel.data_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and not parallel.use_sequence_parallel_moe
        and not parallel.enable_dbo
        and not config.compilation_config.pass_config.enable_sp
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

    @classmethod
    def create(
        cls, tp: "DeviceCommunicatorBase", config: "VllmConfig | None"
    ) -> "CuteAllReduce | None":
        if config is None or not supports_config(config):
            return None
        assert config.model_config is not None
        policy = None
        capability_key = None
        if cls.is_supported(tp.device):
            capability = current_platform.get_device_capability(tp.device.index)
            assert capability is not None
            policy = build_policy(
                tp.world_size,
                config.model_config.get_hidden_size(),
                capability.to_int(),
            )
            if policy is not None:
                capability_key = capability.to_int()
        # Architecture-specific routes must agree before symmetric allocation.
        capabilities: list[int | None] = [None] * tp.world_size
        torch.distributed.all_gather_object(
            capabilities, capability_key, group=tp.cpu_group
        )
        if capability_key is None or any(c != capability_key for c in capabilities):
            logger.debug("CuTe AR/norm unavailable; retaining existing fusion backends")
            return None
        epsilon = float(config.model_config.hf_text_config.rms_norm_eps)
        backend = cls(tp, epsilon, policy)
        logger.info(
            "Initialized FlashInfer CuTe AllReduce for TP%d, hidden size %d",
            tp.world_size,
            backend.hidden_size,
        )
        return backend

    @staticmethod
    def is_supported(device: torch.device) -> bool:
        capability = current_platform.get_device_capability(device.index)
        if (
            capability is None
            or capability.to_int() not in SUPPORTED_DEVICE_CAPABILITIES
        ):
            return False
        try:
            import torch.distributed._symmetric_memory as symm_mem
            from flashinfer.comm.mnnvl import is_multicast_supported
            from flashinfer.comm.mnnvl_cutedsl_ar import (
                MNNVLCuteDSLAllReduceFusionWorkspace,
            )
        except ImportError:
            return False
        return (
            "output_dtype"
            in inspect.signature(MNNVLCuteDSLAllReduceFusionWorkspace).parameters
            and symm_mem.get_backend(device) is not None
            and is_multicast_supported(device.index)
        )

    def __init__(self, tp: "DeviceCommunicatorBase", epsilon: float, policy):
        from flashinfer.comm.mnnvl_cutedsl_ar import (
            MNNVLCuteDSLAllReduceFusionWorkspace,
        )

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("CuTe workspaces must be initialized before capture")
        # Symmetric-memory registration needs the PyTorch NCCL communicator.
        # vLLM collectives can have used only PyNccl before this point.
        torch.distributed.barrier(group=tp.device_group, device_ids=[tp.device.index])
        self.epsilon = epsilon
        self.policy = policy
        profile = policy.profiles[0]
        self.hidden_size = profile.hidden_size
        self.workspaces = {}
        try:
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                for add_residual in (False, True):
                    self.workspaces[dtype, add_residual] = (
                        MNNVLCuteDSLAllReduceFusionWorkspace(
                            tp.world_size,
                            tp.rank_in_group,
                            MAX_TOKENS,
                            self.hidden_size,
                            torch.bfloat16,
                            group=tp.device_group,
                            top_k=profile.top_k,
                            rms_eps=epsilon,
                            weight_bias=1.0,
                            include_shared_expert=True,
                            config=policy,
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


def enabled_for_config(config: "VllmConfig") -> bool:
    return supports_config(config) and get_backend() is not None


def get_backend() -> CuteAllReduce | None:
    return getattr(get_tp_group().device_communicator, "cute_allreduce", None)


def get_workspace(epsilon: float, dtype: torch.dtype, residual: bool):
    backend = get_backend()
    if backend is None:
        raise RuntimeError("CuTe AllReduce was not initialized before execution")
    return backend.get_workspace(epsilon, dtype, residual)


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
    backend = get_backend()
    if (
        backend is not None
        and 0 < input.shape[0] <= MAX_TOKENS
        and input.shape[-1] == backend.hidden_size
    ):
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
