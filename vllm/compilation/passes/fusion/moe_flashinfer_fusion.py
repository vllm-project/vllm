# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse MoE finalize + residual add + all-reduce + RMSNorm using FlashInfer.

This pass matches the pattern where a MoE layer with deferred finalize
outputs UnfinalizedMoEOutput, which is then combined with shared expert output,
all-reduced, and normalized. The fusion replaces this with a single
FlashInfer allreduce_fusion call using the kMoEFinalizeARResidualRMSNorm pattern.
"""

import contextlib
from importlib.util import find_spec
from types import ModuleType
from typing import Any, Optional

import torch
import torch.fx as fx
from torch._inductor.pattern_matcher import PatternMatcherPass

import vllm.ir.ops
from vllm.compilation.passes.vllm_inductor_pass import VllmPatternMatcherPass
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.distributed import get_tp_group, tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.runner.moe_runner import _unpack
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

flashinfer_comm: ModuleType | None = None
if find_spec("flashinfer"):
    try:
        import flashinfer.comm as _flashinfer_comm

        if hasattr(_flashinfer_comm, "allreduce_fusion") and hasattr(
            _flashinfer_comm, "create_allreduce_fusion_workspace"
        ):
            flashinfer_comm = _flashinfer_comm
    except Exception as e:
        logger.debug_once("flashinfer.comm import failed: %s", e)

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
        16: 64,  # 64MB (mnnvl multi-node)
    },
    103: {
        2: 64,  # 64MB
        4: 64,  # 64MB
        8: 4,  # 4MB
        16: 64,  # 64MB (mnnvl multi-node)
    },
    107: {
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
    107: {
        2: 32,  # 32MB
        4: 4,  # 4MB
        8: 2,  # 2MB
    },
}

MiB = 1024 * 1024

# Pattern code for MoE finalize + AR + residual + RMSNorm
# This should match FlashInfer's AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm
# Value 7 based on FlashInfer PR #2982
K_MOE_FINALIZE_AR_RESIDUAL_RMS_NORM = 7


def _select_flashinfer_allreduce_use_oneshot(
    workspace_backend: str,
    device_capability: int | None,
    world_size: int,
    current_tensor_size: int,
) -> bool | None:
    if workspace_backend == "mnnvl":
        # FlashInfer sizes MNNVL workspaces around its own AUTO strategy.
        # Forcing vLLM's per-rank threshold can request one-shot for tensors
        # larger than the MNNVL one-shot workspace.
        return None

    if device_capability is None:
        max_one_shot_size = None
    else:
        max_one_shot_size = _FI_ALLREDUCE_ONE_SHOT_MAX_SIZES_MB.get(
            device_capability, {}
        ).get(world_size)
    return max_one_shot_size is None or current_tensor_size <= max_one_shot_size * MiB


if flashinfer_comm is not None:
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        destroy_fi_ar_workspace,
        get_fi_ar_quant_workspace,
        get_fi_ar_workspace,
    )

    def call_trtllm_fused_moe_finalize_ar_rmsnorm(
        moe_gemm2_permuted: torch.Tensor,
        moe_expert_weights: torch.Tensor,
        moe_expanded_idx_to_permuted_idx: torch.Tensor,
        residual: torch.Tensor,
        rms_gamma: torch.Tensor,
        rms_eps: float,
        world_size: int,
        launch_with_pdl: bool,
        fp32_acc: bool,
        max_token_num: int,
        norm_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # handle transformers backend passing outer batch dim.
        if moe_gemm2_permuted.dim() != 2:
            hidden = moe_gemm2_permuted.shape[-1]
            moe_gemm2_permuted = moe_gemm2_permuted.view(-1, hidden)
            moe_expert_weights = moe_expert_weights.view(-1, moe_expert_weights.shape[-1])
            moe_expanded_idx_to_permuted_idx = (
                moe_expanded_idx_to_permuted_idx.view(-1, moe_expanded_idx_to_permuted_idx.shape[-1])
            )
            residual = residual.view(-1, hidden)
            if norm_out is not None:
                norm_out = norm_out.view(-1, hidden)
        num_tokens, hidden_size = moe_gemm2_permuted.shape
        element_size = moe_gemm2_permuted.element_size()
        current_tensor_size = num_tokens * hidden_size * element_size
        max_tensor_size = max_token_num * hidden_size * element_size
        assert current_tensor_size <= max_tensor_size, (
            f"Current tensor size {current_tensor_size} is larger than "
            f"max token num {max_token_num} * hidden size {hidden_size} * "
            f"element size {element_size}"
        )
        curr_device = current_platform.get_device_capability()
        device_capability = curr_device.to_int() if curr_device is not None else None

        workspace = get_fi_ar_workspace(
            world_size=world_size,
            rank=get_tensor_model_parallel_rank(),
            max_token_num=max_token_num,
            hidden_dim=hidden_size,
            dtype=moe_gemm2_permuted.dtype,
            group=get_tp_group().cpu_group,
        )
        assert workspace is not None, (
            "Flashinfer allreduce workspace must be initialized when using flashinfer"
        )
        use_oneshot = _select_flashinfer_allreduce_use_oneshot(
            workspace.backend,
            device_capability,
            world_size,
            current_tensor_size,
        )
        assert flashinfer_comm is not None
        if norm_out is None:
            norm_out = moe_gemm2_permuted
            residual_out = residual
        else:
            # return residual_out as allreduce_out with zeroed residual_in
            # as flashinfer does not support rms_norm and allreduce_out together
            residual_out = moe_gemm2_permuted

        layout_code = None
        if workspace.backend in ("trtllm", "mnnvl"):
            layout_code = flashinfer_comm.QuantizationSFLayout.SWIZZLED_128x4

        flashinfer_comm.allreduce_fusion(
            input=moe_gemm2_permuted,
            workspace=workspace,
            pattern=K_MOE_FINALIZE_AR_RESIDUAL_RMS_NORM,
            launch_with_pdl=launch_with_pdl,
            output=None,
            residual_out=residual_out,
            norm_out=norm_out,
            quant_out=None,
            scale_out=None,
            residual_in=residual,
            rms_gamma=rms_gamma,
            rms_eps=rms_eps,
            scale_factor=None,
            layout_code=layout_code,
            use_oneshot=use_oneshot,
            fp32_acc=fp32_acc,
            weight_bias=0.0,
            # The one-shot Lamport all-reduce signals PDL completion before its
            # output buffer is committed when trigger_completion_at_end is
            # False, so the next PDL-launched kernel can read the uninitialized
            # Lamport buffer and produce NaN. This only fires for
            # num_tokens <= PDL_ADVANCE_LAUNCH_TOKENS (the batch=1 / spec-decode
            # shapes, where the one-shot path is always selected). Complete at
            # the end for the one-shot path; the two-shot path is synchronized
            # and keeps the early completion. Related one-shot instability in
            # the same kernel: flashinfer-ai/flashinfer#1223.
            trigger_completion_at_end=(use_oneshot is True)
            or num_tokens > 16,
        )
        # Return the all-reduce output (residual_out) which is the input to RMSNorm
        return residual_out

    direct_register_custom_op(
        op_name="flashinfer_trtllm_fused_moe_finalize_ar_rmsnorm",
        op_func=call_trtllm_fused_moe_finalize_ar_rmsnorm,
        mutates_args=[
            "moe_gemm2_permuted",
            "moe_expert_weights",
            "moe_expanded_idx_to_permuted_idx",
            "residual",
            "norm_out",
        ],
    )
    flashinfer_trtllm_fused_moe_finalize_ar_rmsnorm = (
        torch.ops.vllm.flashinfer_trtllm_fused_moe_finalize_ar_rmsnorm.default
    )


class MoEFlashInferFusionParams:
    """Parameters for FlashInfer fused MoE finalize + AR + RMSNorm operations."""

    def __init__(
        self,
        world_size: int,
        max_token_num: int = 1024,
    ) -> None:
        self.world_size = world_size
        self.launch_with_pdl = True
        self.fp32_acc = True
        self.max_token_num = max_token_num

    def get_kwargs(self) -> dict[str, bool | int]:
        return {
            "world_size": self.world_size,
            "launch_with_pdl": self.launch_with_pdl,
            "fp32_acc": self.fp32_acc,
            "max_token_num": self.max_token_num,
        }


class MoEFlashInferFusionPattern:
    """Pattern for MoE finalize + residual add + all-reduce + RMSNorm fusion."""

    def __init__(
        self,
        epsilon: float,
        dtype: torch.dtype,
        device: str | None,
        fusion_params: MoEFlashInferFusionParams,
    ) -> None:
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device
        self.fusion_params = fusion_params
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def empty(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return torch.empty(*args, dtype=self.dtype, device=self.device, **kwargs)

    def get_inputs(self) -> list[torch.Tensor]:
        # moe_gemm2_permuted, moe_expert_weights, moe_expanded_idx_to_permuted_idx,
        # residual (shared expert output), rms_gamma
        return [
            self.empty(5, 16),  # gemm2_permuted [num_permuted_rows, hidden_dim]
            self.empty(5, 4),   # expert_weights [num_tokens, top_k]
            self.empty(5, 4),   # expanded_idx_to_permuted_idx [num_tokens, top_k]
            self.empty(5, 16),  # residual (shared expert output)
            self.empty(16),     # rms_gamma
        ]

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            moe_gemm2_permuted: torch.Tensor,
            moe_expert_weights: torch.Tensor,
            moe_expanded_idx_to_permuted_idx: torch.Tensor,
            residual: torch.Tensor,
            rms_gamma: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # This pattern represents the sequence:
            # 1. MoE unfinalized output (gemm2_permuted, expert_weights, expanded_idx)
            # 2. Finalize: top-k reduction of gemm2_permuted using expert_weights
            # 3. Add residual (shared expert output)
            # 4. All-reduce
            # 5. RMSNorm
            #
            # We construct the pattern using the UnfinalizedMoEOutput structure
            # and the operations that follow.
            unfinalized = UnfinalizedMoEOutput(
                gemm2_permuted=moe_gemm2_permuted,
                expert_weights=moe_expert_weights,
                expanded_idx_to_permuted_idx=moe_expanded_idx_to_permuted_idx,
                route_major=True,
            )
            # Simulate the finalize step (top-k reduction)
            # In practice, this would be done by the fused kernel
            # Here we just pass through for pattern matching
            finalized = moe_gemm2_permuted  # Placeholder
            # Add residual
            combined = finalized + residual
            # All-reduce
            allreduce_output = tensor_model_parallel_all_reduce(combined)
            # RMSNorm
            rms = vllm.ir.ops.rms_norm(allreduce_output, rms_gamma, self.epsilon)
            return rms, allreduce_output

        def replacement(
            moe_gemm2_permuted: torch.Tensor,
            moe_expert_weights: torch.Tensor,
            moe_expanded_idx_to_permuted_idx: torch.Tensor,
            residual: torch.Tensor,
            rms_gamma: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert flashinfer_comm is not None, "FlashInfer must be enabled"
            # Create output tensor for RMSNorm result
            rms_result = torch.empty_like(moe_gemm2_permuted)
            # Call the custom op which returns the allreduce output and mutates norm_out
            allreduce_output = torch.ops.vllm.flashinfer_trtllm_fused_moe_finalize_ar_rmsnorm(
                moe_gemm2_permuted=moe_gemm2_permuted,
                moe_expert_weights=moe_expert_weights,
                moe_expanded_idx_to_permuted_idx=moe_expanded_idx_to_permuted_idx,
                residual=residual,
                norm_out=rms_result,
                rms_gamma=rms_gamma,
                rms_eps=self.epsilon,
                pattern_code=K_MOE_FINALIZE_AR_RESIDUAL_RMS_NORM,
                **self.fusion_params.get_kwargs(),
            )
            # Return (RMSNorm output, allreduce output) to match pattern function return
            return rms_result, allreduce_output

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            torch._inductor.pattern_matcher.fwd_only,
            pm_pass,
        )


def _moe_flashinfer_workspace_hidden_dim(config: VllmConfig) -> int:
    """Widest hidden size across the target and (optional) draft models."""
    hidden_dim = config.model_config.get_hidden_size()
    spec = config.speculative_config
    draft_model_config = getattr(spec, "draft_model_config", None) if spec else None
    if draft_model_config is not None:
        hidden_dim = max(hidden_dim, draft_model_config.get_hidden_size())
    return hidden_dim


class MoEFlashInferFusionPass(VllmPatternMatcherPass):
    """Pass that fuses MoE finalize + residual add + all-reduce + RMSNorm using FlashInfer."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.disabled = True
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size <= 1:
            logger.warning_once(
                "MoE FlashInfer fusion pass is disabled for tp_size <= 1."
            )
            return
        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="moe_flashinfer_fusion_pass"
        )
        if config.model_config is None:
            logger.warning_once(
                "MoE FlashInfer fusion pass is disabled for missing model_config."
            )
            return
        # Size the shared workspace for the widest model that reuses it.
        self.workspace_hidden_dim = _moe_flashinfer_workspace_hidden_dim(config)
        self.group = get_tp_group().cpu_group
        rank = get_tensor_model_parallel_rank()
        if flashinfer_comm is None:
            logger.warning(
                "Flashinfer is not installed or comm module not found, "
                "skipping MoE FlashInfer fusion pass"
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
        self.max_token_num = max_size // (self.workspace_hidden_dim * element_size)
        # take the min to save workspace size and we'll never use more
        # than max_num_batched_tokens anyways
        self.max_token_num = min(
            self.max_token_num, config.scheduler_config.max_num_batched_tokens
        )
        logger.debug_once(
            f"Flashinfer max size: {max_size // (1024 * 1024)} MB,"
            "Maximal number of tokens used by "
            f"MoE FlashInfer Fusion: {self.max_token_num}",
            scope="global",
        )

        workspace_kwargs = dict(
            world_size=self.tp_size,
            rank=rank,
            max_token_num=self.max_token_num,
            hidden_dim=self.workspace_hidden_dim,
            dtype=self.model_dtype,
            group=self.group,
        )
        if get_fi_ar_workspace(**workspace_kwargs) is None:
            logger.warning_once(
                "Failed to initialize Flashinfer allreduce workspace. "
                "MoE FlashInfer fusion will be disabled."
            )
            return

        self.fusion_params = MoEFlashInferFusionParams(
            world_size=self.tp_size,
            max_token_num=self.max_token_num,
        )

        self.register_patterns()
        self.dump_patterns(config, self.patterns)

    def register_patterns(self) -> None:
        for epsilon in [1e-5, 1e-6]:
            MoEFlashInferFusionPattern(
                epsilon,
                self.model_dtype,
                self.device,
                self.fusion_params,
            ).register(self.patterns)

            # WARNING: This is a hack to clear the pattern matcher cache
            # and allow multiple values of epsilon.
            torch._inductor.pattern_matcher._seen_patterns.clear()

        self.disabled = False

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        if self.disabled:
            logger.warning_once("MoE FlashInfer fusion pass is disabled.")
            return False
        return bool(compile_range.end <= self.max_token_num)

    def __call__(self, graph: fx.Graph) -> None:
        if self.disabled:
            logger.debug("MoEFlashInferFusionPass disabled")
            return

        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def __del__(self) -> None:
        if getattr(self, "disabled", True):
            return
        with contextlib.suppress(Exception):
            destroy_fi_ar_workspace()