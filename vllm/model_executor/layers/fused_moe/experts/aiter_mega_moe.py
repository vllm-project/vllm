# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER MegaMoEV2 as a modular-kernel experts implementation.

MegaMoEV2 takes vLLM's top-k output and dispatches over MoRI shmem, runs the
MXFP4 GEMMs and combines in one kernel, so it is paired with a pass-through
prepare/finalize.
"""

from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
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
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2

MAX_EP_SIZE = 8
MAX_TOPK = 16
MAX_LOCAL_EXPERTS = 256


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
    if moe_config.experts_per_token > MAX_TOPK:
        reasons.append(f"top-k {moe_config.experts_per_token}")
    if moe_config.has_bias:
        reasons.append("expert bias")
    if envs.VLLM_ROCM_AITER_MEGA_MOE_FUSE_SHARED_EXPERTS:
        reasons.append("a fused shared expert")
    return reasons


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
        self._runtime: Any = None

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
        return weight_key == kMxfp4Static and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return (
            moe_parallel_config.use_ep
            and not moe_parallel_config.use_batched_activation_format
        )

    @property
    def expects_unquantized_inputs(self) -> bool:
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
        """Bind the shared MegaMoEV2 runtime; the oracle has already shuffled
        the weights into its layout."""
        from vllm.distributed import get_ep_group

        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "AITER MegaMoE does not support apply_router_weight_on_input"
            )
        ep_group = get_ep_group()
        moe = self.moe_config
        swiglu_limit = self.quant_config.gemm1_clamp_limit or 0.0
        max_tok_per_rank = next_power_of_2(moe.max_num_tokens)
        key = (
            id(ep_group.cpu_group),
            torch.accelerator.current_device_index(),
            moe.num_experts,
            moe.experts_per_token,
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
                experts=moe.num_experts,
                topk=moe.experts_per_token,
                quant="a8w4",
                w1=layer.w13_weight,
                w1_scale=layer.w13_weight_scale,
                w2=layer.w2_weight,
                w2_scale=layer.w2_weight_scale,
                max_tok_per_rank=max_tok_per_rank,
                swiglu_limit=swiglu_limit,
            )
            self._runtime_cache[key] = runtime
        self._runtime = runtime

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
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("AITER MegaMoE weights have not been processed")
        # MegaMoEV2 reads these per launch; rebind to this layer's experts.
        runtime._s1_w1 = w1
        runtime._s1_w1_scale = self.quant_config.w1_scale
        runtime.w2 = w2
        runtime.w2_scale = self.quant_config.w2_scale
        output.copy_(
            runtime.forward(
                hidden_states.contiguous(),
                topk_weights.to(torch.float32),
                topk_ids.to(torch.int32),
            )
        )
