# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER MegaMoEV2 as a modular-kernel experts implementation.

MegaMoEV2 takes vLLM's top-k output and dispatches over MoRI shmem, runs the
MXFP4 GEMMs and combines in one kernel, so it is paired with a pass-through
prepare/finalize.
"""

import os
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kMxfp8Dynamic,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2, prev_power_of_2

logger = init_logger(__name__)

MAX_EP_SIZE = 8
MAX_TOPK = 16
MAX_LOCAL_EXPERTS = 256


def shuffle_aiter_mega_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shuffle per-expert MXFP4 weights and scales into the MegaMoEV2 layout.

    MegaMoEV2 takes w1 and w3 stacked, not gate/up interleaved.
    """
    from vllm._aiter_ops import rocm_aiter_ops

    num_experts = w13.shape[0]
    return (
        rocm_aiter_ops.shuffle_weight_a16w4(w13, 16, True),
        rocm_aiter_ops.shuffle_weight_a16w4(w2, 16, False),
        rocm_aiter_ops.shuffle_scale_a16w4(
            w13_scale.view(-1, w13_scale.shape[-1]), num_experts, True
        ),
        rocm_aiter_ops.shuffle_scale_a16w4(
            w2_scale.view(-1, w2_scale.shape[-1]), num_experts, False
        ),
    )


@triton.jit
def _remap_fused_shared_expert_ids_kernel(
    topk_ids_ptr,
    out_ids_ptr,
    numel,
    num_routed_experts,
    local_shared_base,
    ROUTED_PER_RANK: tl.constexpr,
    NUM_SHARED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    e = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
    routed = e + (e // ROUTED_PER_RANK) * NUM_SHARED
    shared = local_shared_base + e - num_routed_experts
    e = tl.where(e >= num_routed_experts, shared, tl.where(e >= 0, routed, e))
    tl.store(out_ids_ptr + offs, e, mask=mask)


def aiter_mega_moe_unsupported_reasons(moe_config: FusedMoEConfig) -> list[str]:
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.utils.import_utils import has_mori

    parallel = moe_config.moe_parallel_config
    reasons = []
    if not rocm_aiter_ops.is_fused_moe_enabled():
        reasons.append("AITER fused MoE disabled (set VLLM_ROCM_USE_AITER=1)")
    if not has_mori():
        reasons.append("missing MoRI")
    if not parallel.use_passthrough_all2all:
        reasons.append(f"all2all_backend={parallel.all2all_backend!r}")
    if parallel.dp_size != parallel.ep_size:
        reasons.append("tensor parallel attention (use DP attention with TP=1)")
    if parallel.ep_size > MAX_EP_SIZE:
        reasons.append(f"ep_size={parallel.ep_size} > {MAX_EP_SIZE}")
    if parallel.enable_eplb:
        reasons.append("EPLB")
    if moe_config.num_experts % parallel.ep_size:
        reasons.append(
            f"{moe_config.num_experts} experts over {parallel.ep_size} EP ranks"
        )
    if moe_config.num_local_experts > MAX_LOCAL_EXPERTS:
        reasons.append(f"{moe_config.num_local_experts} experts per rank")
    num_shared = moe_config.num_fused_shared_experts
    if moe_config.experts_per_token + num_shared > MAX_TOPK:
        reasons.append(f"top-k {moe_config.experts_per_token + num_shared}")
    if moe_config.has_bias:
        reasons.append("expert bias")
    return reasons


_mori_shmem_initialized = False


def _init_mori_shmem(default_heap_size: str) -> None:
    """Set up the MoRI symmetric heap that MegaMoEV2 places its dispatch and
    combine workspaces on.

    This must run after the worker's memory snapshot (i.e. at weight loading,
    not in the all2all manager) so the profiler charges the heap against
    gpu_memory_utilization.
    """
    global _mori_shmem_initialized
    if _mori_shmem_initialized:
        return
    from vllm.distributed import get_ep_group
    from vllm.distributed.device_communicators.all2all import init_mori_shmem

    heap_size = os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", default_heap_size)
    logger.info_once("AITER MegaMoE: MORI_SHMEM_HEAP_SIZE=%s", heap_size)
    init_mori_shmem(get_ep_group().cpu_group)
    _mori_shmem_initialized = True


class AiterMegaMoEExperts(mk.FusedMoEExpertsModular):
    # MegaMoEV2 owns large symmetric-heap buffers, so every layer with the same
    # problem shares one runtime and passes its own weights per call.
    _runtime_cache: dict[tuple, Any] = {}

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(moe_config, quant_config)
        self.num_fused_shared_experts = moe_config.num_fused_shared_experts
        # (max_tok_per_rank, runtime), ascending by capacity.
        self._runtimes: list[tuple[int, Any]] = []

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )
        if not supported:
            return False, reason
        reasons = aiter_mega_moe_unsupported_reasons(moe_config)
        if reasons:
            return False, f"kernel does not support {', '.join(reasons)}"
        return True, None

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        if not current_platform.is_rocm():
            return False
        from vllm.platforms.rocm import on_gfx950

        return on_gfx950()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return weight_key == kMxfp4Static and activation_key == kMxfp8Dynamic

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return moe_parallel_config.use_ep

    @property
    def expects_unquantized_inputs(self) -> bool:
        # The kernel quantizes its BF16 inputs to MXFP8 itself.
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (0,), (0,), (M, K)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Bind the shared MegaMoEV2 runtimes; the oracle has already shuffled
        the weights into their layout."""
        from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
            FIXED_SLOT_MAX_MTPR,
        )

        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "AITER MegaMoE does not support apply_router_weight_on_input"
            )
        # MoRI's default 2G heap is too small: MegaMoEV2 needs > 4G, or > 8G
        # with a fused shared expert.
        _init_mori_shmem("16G" if self.num_fused_shared_experts else "8G")
        max_num_tokens = next_power_of_2(self.moe_config.max_num_tokens)
        # MegaMoEV2 takes fixed-slot dispatch up to FIXED_SLOT_MAX_MTPR (255 by
        # default, raised via AITER_MEGA_FIXED_SLOT_MAX_MTPR); max_tok_per_rank
        # must also be a power of two.
        fixed_slot_max_tokens = prev_power_of_2(FIXED_SLOT_MAX_MTPR)
        capacities = [max_num_tokens]
        if max_num_tokens > fixed_slot_max_tokens and self._uses_fixed_slot(
            fixed_slot_max_tokens
        ):
            capacities.insert(0, fixed_slot_max_tokens)
        self._runtimes = [(c, self._get_runtime(layer, c)) for c in capacities]

    def _uses_fixed_slot(self, max_tok_per_rank: int) -> bool:
        from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
            build_mega_moe_bundle_plan,
        )

        moe = self.moe_config
        return build_mega_moe_bundle_plan(
            max_tok_per_rank,
            experts_per_rank=moe.num_local_experts,
            model_dim=moe.hidden_dim,
            inter_dim=moe.intermediate_size_per_partition,
            world_size=moe.moe_parallel_config.ep_size,
        ).fixed_slot_dispatch

    def _get_runtime(self, layer: torch.nn.Module, max_tok_per_rank: int) -> Any:
        from vllm.distributed import get_ep_group

        ep_group = get_ep_group()
        moe = self.moe_config
        num_experts = moe.num_local_experts * ep_group.world_size
        topk = moe.experts_per_token + self.num_fused_shared_experts
        swiglu_limit = self.quant_config.gemm1_clamp_limit or 0.0
        key = (
            id(ep_group.cpu_group),
            torch.accelerator.current_device_index(),
            num_experts,
            topk,
            moe.hidden_dim,
            moe.intermediate_size_per_partition,
            max_tok_per_rank,
            swiglu_limit,
        )
        runtime = self._runtime_cache.get(key)
        if runtime is None:
            from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

            runtime = MegaMoEV2(
                rank=ep_group.rank_in_group,
                world_size=ep_group.world_size,
                model_dim=moe.hidden_dim,
                inter_dim=moe.intermediate_size_per_partition,
                experts=num_experts,
                topk=topk,
                quant="a8w4",
                w1=layer.w13_weight,
                w1_scale=layer.w13_weight_scale,
                w2=layer.w2_weight,
                w2_scale=layer.w2_weight_scale,
                max_tok_per_rank=max_tok_per_rank,
                swiglu_limit=swiglu_limit,
            )
            self._runtime_cache[key] = runtime
        return runtime

    def _select_runtime(self, num_tokens: int) -> Any:
        # Dispatch is collective, so every EP rank must pick the same runtime.
        if is_forward_context_available():
            dp_metadata = get_forward_context().dp_metadata
            if dp_metadata is not None:
                num_tokens = max(
                    num_tokens, int(dp_metadata.num_tokens_across_dp_cpu.max())
                )
        for max_tok_per_rank, runtime in self._runtimes:
            if num_tokens <= max_tok_per_rank:
                return runtime
        raise ValueError(
            f"{num_tokens} tokens exceeds MegaMoE capacity {self._runtimes[-1][0]}"
        )

    def _remap_fused_shared_expert_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Map router ids, with the fused shared experts at
        `[num_experts, num_experts + n)`, onto the per-rank (routed + shared)
        expert layout, sending the shared routes to this rank's own copy."""
        moe = self.moe_config
        ep = moe.moe_parallel_config
        routed_per_rank = moe.num_experts // ep.ep_size
        num_shared = self.num_fused_shared_experts
        assert topk_ids.shape[1] == moe.experts_per_token + num_shared
        topk_ids = topk_ids.contiguous()
        out_ids = torch.empty_like(topk_ids, dtype=torch.int32)
        numel, block = topk_ids.numel(), 1024
        if numel:
            _remap_fused_shared_expert_ids_kernel[(triton.cdiv(numel, block),)](
                topk_ids,
                out_ids,
                numel,
                moe.num_experts,
                ep.ep_rank * moe.num_local_experts + routed_per_rank,
                ROUTED_PER_RANK=routed_per_rank,
                NUM_SHARED=num_shared,
                BLOCK=block,
            )
        return out_ids

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        if not self._runtimes:
            raise RuntimeError("AITER MegaMoE weights have not been processed")
        runtime = self._select_runtime(hidden_states.shape[0])
        if self.num_fused_shared_experts:
            topk_ids = self._remap_fused_shared_expert_ids(topk_ids)
        # MegaMoEV2 reads these per launch; rebind to this layer's experts.
        runtime._s1_w1 = w1
        runtime._s1_w1_scale = self.quant_config.w1_scale
        runtime.w2 = w2
        runtime.w2_scale = self.quant_config.w2_scale
        output.copy_(
            runtime.forward(
                hidden_states.contiguous(),
                topk_weights.to(torch.float32).contiguous(),
                topk_ids.to(torch.int32).contiguous(),
            )
        )
