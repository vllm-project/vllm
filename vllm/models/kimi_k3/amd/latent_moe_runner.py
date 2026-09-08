# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from typing import cast

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _layer_name_type,
    _resolve_layer_name,
    get_layer_from_name,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_KIMI_K3_ROUTED_HIDDEN = 3584
_KIMI_K3_SHARED_HIDDEN = 7168


def _kimi_k3_large_moe_core(
    hidden_states: torch.Tensor,
    layer_name: _layer_name_type,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run runtime-shape dispatch outside Dynamo."""

    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    assert isinstance(layer, ROCmLatentMoERunner)
    callback = layer._kimi_k3_large_moe_core_callback
    if callback is None:
        raise RuntimeError("Kimi-K3 large-M callback was not initialized")
    return callback(hidden_states)


def _kimi_k3_large_moe_core_fake(
    hidden_states: torch.Tensor,
    layer_name: _layer_name_type,
) -> tuple[torch.Tensor, torch.Tensor]:
    del layer_name
    return (
        hidden_states.new_empty((hidden_states.shape[0], _KIMI_K3_SHARED_HIDDEN)),
        hidden_states.new_empty((hidden_states.shape[0], _KIMI_K3_ROUTED_HIDDEN)),
    )


direct_register_custom_op(
    op_name="kimi_k3_large_moe_core",
    op_func=_kimi_k3_large_moe_core,
    mutates_args=[],
    fake_impl=_kimi_k3_large_moe_core_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


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
        tp_size = self.moe_config.tp_size

        self._up_proj_shard_size = 0
        self._tail_shardable = (
            up_proj is not None
            and tp_size > 1
            and up_proj.weight.shape[0] % tp_size == 0
            and self._shared_experts is not None
            and not self.moe_config.is_sequence_parallel
            and self.routed_scaling_factor == 1.0
        )
        if self._tail_shardable:
            assert up_proj is not None
            self._up_proj_shard_size = up_proj.weight.shape[0] // tp_size
        else:
            logger.warning_once(
                "K3 latent-MoE tail is not shardable under this config, "
                "falling back to the replicated up-projection.",
                scope="global",
            )
        self._logged_sharded_tail = False
        self._kimi_k3_large_moe_core_callback: (
            Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None
        ) = None

    def set_kimi_k3_large_moe_core_callback(
        self,
        callback: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
    ) -> None:
        self._kimi_k3_large_moe_core_callback = callback

    def kimi_k3_large_moe_core(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.kimi_k3_large_moe_core(
            hidden_states,
            self._encode_layer_name(),
        )

    def run_kimi_k3_native_moe_core(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the native MoE core without applying the latent output tail."""

        hidden_states, pre_xform_trunc_size, _ = self._maybe_pad_hidden_states(
            shared_experts_input,
            hidden_states,
        )
        result = self._forward_impl(
            hidden_states,
            router_logits,
            shared_experts_input,
        )
        shared_output, fused_output = cast(
            tuple[torch.Tensor, torch.Tensor],
            result,
        )
        if pre_xform_trunc_size is not None:
            fused_output = fused_output[..., :pre_xform_trunc_size]
        return shared_output, fused_output

    def run_kimi_k3_large_moe_core(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_intermediate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run native routed experts from the merged-front outputs."""

        self.routed_experts._ensure_moe_quant_config_init()
        if self.routed_experts.quant_method.is_monolithic:
            raise RuntimeError("Kimi-K3 large-M front requires a modular MoE backend")

        topk_weights, topk_ids = self.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_indices_dtype=self._quant_method.topk_indices_dtype,
            input_ids=None,
        )
        fused_output = self.routed_experts.forward_modular(
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )

        assert self._shared_experts is not None
        shared_layer = getattr(self._shared_experts, "_layer", None)
        assert shared_layer is not None
        shared_output, _ = shared_layer.down_proj(shared_experts_intermediate)
        return shared_output, cast(torch.Tensor, fused_output)

    def finish_kimi_k3_large_moe_core(
        self,
        shared_output: torch.Tensor,
        fused_output: torch.Tensor,
    ) -> torch.Tensor:
        result = self._shard_up_proj_tail(
            fused_output,
            shared_output,
            _KIMI_K3_SHARED_HIDDEN,
        )
        return self._maybe_add_zero_expert_output(result)

    def _shard_up_proj_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        """
        Tier 2: column-parallel up-projection folded into the final reduce.
        """
        if not torch.compiler.is_compiling() and not self._logged_sharded_tail:
            self._logged_sharded_tail = True
            logger.info_once(
                "Kimi-K3 latent-MoE tail: up-projecting only this rank's "
                "hidden shard into the shared output.",
                scope="global",
            )

        transform = self.routed_output_transform
        assert transform is not None

        latent = tensor_model_parallel_all_reduce(fused_output)
        if transform.norm is not None:
            latent = transform.norm(latent)

        shard_size = self._up_proj_shard_size
        shard_start = get_tensor_model_parallel_rank() * shard_size
        up_proj_shard = transform.up_proj.weight.narrow(0, shard_start, shard_size)
        hidden_shard = shared_output.narrow(-1, shard_start, shard_size)

        # hidden_shard += latent @ up_proj_shard.T, accumulated in the GEMM's
        # beta-add epilogue so folding in the shared partial costs no kernel.
        hidden_shard.addmm_(latent, up_proj_shard.t())

        return self._maybe_reduce_final_output(
            shared_output, trunc_size, output_is_reduced=False
        )

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
        return super().forward(
            hidden_states, router_logits, input_ids, shared_experts_input
        )

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
