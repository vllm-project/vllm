# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER MegaMoE experts (fused dispatch, MXFP4 GEMMs, combine) for DeepSeek V4."""

from typing import Any

import torch
import torch.nn as nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    get_fp8_block_weight_scale,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import mxfp4_quantize
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import next_power_of_2


@triton.jit
def _append_shared_expert_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    out_ids_ptr,
    out_weights_ptr,
    num_tokens,
    shared_expert_id,
    TOPK: tl.constexpr,
    ROUTED_PER_RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < num_tokens
    for k in tl.static_range(TOPK):
        e = tl.load(topk_ids_ptr + rows * TOPK + k, mask=mask, other=0)
        w = tl.load(topk_weights_ptr + rows * TOPK + k, mask=mask, other=0.0)
        e = tl.where(e >= 0, e + e // ROUTED_PER_RANK, e)
        tl.store(out_ids_ptr + rows * (TOPK + 1) + k, e, mask=mask)
        tl.store(out_weights_ptr + rows * (TOPK + 1) + k, w, mask=mask)
    tl.store(
        out_ids_ptr + rows * (TOPK + 1) + TOPK,
        tl.zeros_like(rows) + shared_expert_id,
        mask=mask,
    )
    tl.store(
        out_weights_ptr + rows * (TOPK + 1) + TOPK,
        tl.full([BLOCK_M], 1.0, tl.float32),
        mask=mask,
    )


class DeepseekV4AiterMegaMoEExperts(nn.Module):
    # MegaMoEV2 owns large symmetric-heap buffers, so every layer shares one
    # instance and passes its own weights per call.
    _runtime_cache: dict[tuple, Any] = {}

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float | None,
        fuse_shared_expert: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        ep_group = get_ep_group()
        self.ep_rank = ep_group.rank_in_group
        self.ep_size = ep_group.world_size
        self.num_routed_local_experts = num_experts // self.ep_size
        self.experts_start_idx = self.ep_rank * self.num_routed_local_experts
        # A fused shared expert is one extra local expert per rank and one
        # extra route per token, to the token's own rank.
        self.fuse_shared_expert = fuse_shared_expert
        self.num_local_experts = self.num_routed_local_experts + fuse_shared_expert
        self.num_experts = self.num_local_experts * self.ep_size
        self.routed_top_k = top_k
        self.top_k = top_k + fuse_shared_expert
        self.shared_expert_id = (
            self.ep_rank * self.num_local_experts + self.num_routed_local_experts
        )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.swiglu_limit = swiglu_limit or 0.0
        self.max_num_tokens = next_power_of_2(
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.prefix = prefix

        def make_param(rows: int, cols: int) -> nn.Parameter:
            param = nn.Parameter(
                torch.zeros(self.num_local_experts, rows, cols, dtype=torch.uint8),
                requires_grad=False,
            )
            set_weight_attrs(param, {"weight_loader": self.weight_loader})
            return param

        self.w13_weight = make_param(2 * intermediate_size, hidden_size // 2)
        self.w13_weight_scale = make_param(2 * intermediate_size, hidden_size // 32)
        self.w2_weight = make_param(hidden_size, intermediate_size // 2)
        self.w2_weight_scale = make_param(hidden_size, intermediate_size // 32)
        self._runtime: Any = None

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        local_expert_id = expert_id - self.experts_start_idx
        if not 0 <= local_expert_id < self.num_routed_local_experts:
            return False if return_success else None
        expert_data = param.data[local_expert_id]
        if shard_id in ("w1", "w3"):
            shard_offset = 0 if shard_id == "w1" else self.intermediate_size
            expert_data = expert_data.narrow(0, shard_offset, self.intermediate_size)
        elif shard_id != "w2":
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        expert_data.copy_(loaded_weight.view(torch.uint8))
        return True if return_success else None

    def load_shared_expert(self, shared_experts: nn.Module) -> None:
        slot = self.num_routed_local_experts
        for linear, weight, weight_scale in (
            (shared_experts.gate_up_proj, self.w13_weight, self.w13_weight_scale),
            (shared_experts.down_proj, self.w2_weight, self.w2_weight_scale),
        ):
            block_scale = get_fp8_block_weight_scale(linear)
            assert block_scale is not None, "shared expert must be block-FP8"
            group = GroupShape(
                linear.weight.shape[0] // block_scale.shape[0],
                linear.weight.shape[1] // block_scale.shape[1],
            )
            packed, scale = mxfp4_quantize(
                scaled_dequantize(linear.weight, block_scale, group, torch.bfloat16)
            )
            weight.data[slot].copy_(packed.view(torch.uint8))
            weight_scale.data[slot].copy_(scale.view(torch.uint8))

    def finalize_weights(self) -> None:
        if self._runtime is not None:
            return
        n = self.num_local_experts
        self.w13_weight.data = rocm_aiter_ops.shuffle_weight_a16w4(
            self.w13_weight.data, 16, True
        )
        self.w13_weight_scale.data = rocm_aiter_ops.shuffle_scale_a16w4(
            self.w13_weight_scale.data.view(-1, self.hidden_size // 32), n, True
        )
        self.w2_weight.data = rocm_aiter_ops.shuffle_weight_a16w4(
            self.w2_weight.data, 16, False
        )
        self.w2_weight_scale.data = rocm_aiter_ops.shuffle_scale_a16w4(
            self.w2_weight_scale.data.view(-1, self.intermediate_size // 32), n, False
        )
        key = (
            id(get_ep_group().cpu_group),
            torch.accelerator.current_device_index(),
            self.num_experts,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.max_num_tokens,
            self.swiglu_limit,
        )
        runtime = self._runtime_cache.get(key)
        if runtime is None:
            from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

            runtime = MegaMoEV2(
                rank=self.ep_rank,
                world_size=self.ep_size,
                model_dim=self.hidden_size,
                inter_dim=self.intermediate_size,
                experts=self.num_experts,
                topk=self.top_k,
                quant="a8w4",
                w1=self.w13_weight,
                w1_scale=self.w13_weight_scale,
                w2=self.w2_weight,
                w2_scale=self.w2_weight_scale,
                max_tok_per_rank=self.max_num_tokens,
                swiglu_limit=self.swiglu_limit,
            )
            self._runtime_cache[key] = runtime
        self._runtime = runtime

    def append_shared_expert(
        self, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remap routed ids to the per-rank (routed + shared) layout and append
        a weight-1.0 route to this rank's shared slot."""
        num_tokens = topk_ids.shape[0]
        out_ids = torch.empty(
            (num_tokens, self.top_k), dtype=topk_ids.dtype, device=topk_ids.device
        )
        out_weights = torch.empty(
            (num_tokens, self.top_k), dtype=torch.float32, device=topk_ids.device
        )
        if num_tokens:
            block_m = 64
            _append_shared_expert_kernel[(triton.cdiv(num_tokens, block_m),)](
                topk_ids.contiguous(),
                topk_weights.to(torch.float32).contiguous(),
                out_ids,
                out_weights,
                num_tokens,
                self.shared_expert_id,
                TOPK=self.routed_top_k,
                ROUTED_PER_RANK=self.num_routed_local_experts,
                BLOCK_M=block_m,
            )
        return out_weights, out_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        runtime = self._runtime
        assert runtime is not None
        if self.fuse_shared_expert:
            topk_weights, topk_ids = self.append_shared_expert(topk_weights, topk_ids)
        # MegaMoEV2 reads these per launch; rebind to this layer's experts.
        runtime._s1_w1 = self.w13_weight
        runtime._s1_w1_scale = self.w13_weight_scale
        runtime.w2 = self.w2_weight
        runtime.w2_scale = self.w2_weight_scale
        return runtime.forward(hidden_states.contiguous(), topk_weights, topk_ids)


DeepseekV4AiterMegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


def finalize_mega_moe_weights(model: nn.Module) -> None:
    for module in list(model.modules()):
        experts = getattr(module, "experts", None)
        if isinstance(experts, DeepseekV4AiterMegaMoEExperts):
            if experts.fuse_shared_expert:
                experts.load_shared_expert(module.shared_experts)
                module.shared_experts = None
            experts.finalize_weights()
