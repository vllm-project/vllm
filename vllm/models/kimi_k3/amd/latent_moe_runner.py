# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import cast

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.models.kimi_k3.amd import sp

logger = init_logger(__name__)


class ROCmLatentMoERunner(MoERunner):
    """MoE runner for latent MoE with a replicated routed up-projection.

    Mirrors CUDA's LatentMoERunner, but currently only the up projection
    -sharded path is implemented. (Tier 2)

    Native path: the replicated up-proj produces the full hidden dim on every
    rank, so the base runner combines routed + shared correctly at any TP size.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        transform = self.routed_output_transform
        up_proj = getattr(transform, "up_proj", None)
        tp_size = get_tensor_model_parallel_world_size()

        self._up_proj_preshard = bool(getattr(transform, "row_sharded", False))
        self._up_proj_shard_size = 0
        self._tail_shardable = (
            up_proj is not None
            and tp_size > 1
            and (self._up_proj_preshard or up_proj.weight.shape[0] % tp_size == 0)
            and self._shared_experts is not None
            and not self.moe_config.is_sequence_parallel
            and self.routed_scaling_factor == 1.0
        )
        if self._tail_shardable:
            assert up_proj is not None
            self._up_proj_shard_size = (
                up_proj.weight.shape[0]
                if self._up_proj_preshard
                else up_proj.weight.shape[0] // tp_size
            )
        else:
            logger.warning_once(
                "K3 latent-MoE tail is not shardable under this config, "
                "falling back to the replicated up-projection.",
                scope="global",
            )
        self._logged_sharded_tail = False

    def sp_route_shard(
        self, hidden_shard: torch.Tensor, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Latent projection and top-k for this rank's token shard only."""
        latent, _ = self.apply_routed_input_transform(hidden_shard)
        topk_weights, topk_ids = self.router.select_experts(
            hidden_states=latent,
            router_logits=router_logits,
            topk_indices_dtype=self._quant_method.topk_indices_dtype,
        )
        return latent, topk_weights, topk_ids

    def _shard_up_proj_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        """Tier 2: column-parallel up-projection folded into the final reduce."""
        if not self._logged_sharded_tail:
            self._logged_sharded_tail = True
            logger.info_once(
                "Kimi-K3 latent-MoE tail: up-projecting only this rank's "
                "hidden shard into the shared output.",
                scope="global",
            )

        transform = self.routed_output_transform
        assert transform is not None

        if sp.ACTIVE:
            return self._sp_tail(fused_output, shared_output, trunc_size)

        latent = tensor_model_parallel_all_reduce(fused_output)
        if transform.norm is not None:
            latent = transform.norm(latent)

        shard_size = self._up_proj_shard_size
        shard_start = get_tensor_model_parallel_rank() * shard_size
        weight = transform.up_proj.weight
        up_proj_shard = (
            weight
            if self._up_proj_preshard
            else weight.narrow(0, shard_start, shard_size)
        )
        hidden_shard = shared_output.narrow(-1, shard_start, shard_size)

        # Not addmm_: hipBLASLt's C-accumulating bf16 GEMM faults at some row
        # counts for this shape (e.g. 23393-23405 rows at 896x3584).
        hidden_shard += latent @ up_proj_shard.t()

        return self._maybe_reduce_final_output(
            shared_output, trunc_size, output_is_reduced=False
        )

    def _sp_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        """Token-sharded tail: reduce-scatter the latent (half the bytes of an
        all-reduce) and up-project this rank's tokens with the full weight."""
        transform = self.routed_output_transform
        assert transform is not None
        latent = tensor_model_parallel_reduce_scatter(fused_output.contiguous(), dim=0)
        if transform.norm is not None:
            latent = transform.norm(latent)
        out = tensor_model_parallel_reduce_scatter(shared_output, dim=0)
        # Not addmm_: hipBLASLt's C-accumulating bf16 GEMM faults at some row
        # counts for this shape (e.g. 2914-2925 rows at 7168x3584).
        out += latent.to(out.dtype) @ transform.up_proj.weight.t()
        return out[..., :trunc_size] if trunc_size is not None else out

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._tail_shardable and not self._fused_output_is_reduced:
            return self._fused_forward(
                hidden_states, router_logits, input_ids, shared_experts_input
            )
        out = super().forward(
            hidden_states, router_logits, input_ids, shared_experts_input
        )
        if sp.ACTIVE:
            shard = out.size(0) // get_tensor_model_parallel_world_size()
            out = out.narrow(0, get_tensor_model_parallel_rank() * shard, shard)
        return out

    def _fused_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        # When the caller pre-applies the routed input transform outside the
        # runner (e.g. to overlap it on a separate stream), it passes the
        # already-transformed routed input as ``hidden_states`` and the original
        # hidden states as ``shared_experts_input``; skip the transform then.
        if shared_experts_input is None:
            hidden_states, shared_experts_input = self.apply_routed_input_transform(
                hidden_states
            )

        hidden_states, og_hidden_dim_pre_xform, og_hidden_dim_post_xform = (
            self._maybe_pad_hidden_states(
                shared_experts_input,
                hidden_states,
            )
        )

        result = self._forward_entry(
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
            self._encode_layer_name(),
            self.moe_config.hidden_dim_unpadded
            if self._quant_method.has_unpadded_output
            else 0,
        )

        shared_output, fused_output = cast(tuple[torch.Tensor, torch.Tensor], result)

        if og_hidden_dim_pre_xform is not None:
            fused_output = fused_output[..., :og_hidden_dim_pre_xform]

        result = self._shard_up_proj_tail(
            fused_output, shared_output, og_hidden_dim_post_xform
        )

        return self._maybe_add_zero_expert_output(result)
