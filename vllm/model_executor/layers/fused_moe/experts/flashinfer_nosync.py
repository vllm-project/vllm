# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed import get_ep_group

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig


class FlashInferNosyncMoe:
    def __init__(
        self,
        moe: FusedMoEConfig,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        if not moe.moe_parallel_config.use_ep:
            raise ValueError("FlashInfer nosync requires expert parallelism")
        if moe.moe_parallel_config.tp_size != 1:
            raise ValueError("FlashInfer nosync currently requires TP=1")
        if moe.has_bias or moe.swiglu_alpha is not None or moe.swiglu_beta is not None:
            raise ValueError(
                "FlashInfer nosync only supports standard SwiGLU without expert bias"
            )
        if moe.moe_parallel_config.enable_eplb:
            raise ValueError("FlashInfer nosync does not support EPLB")
        if not w13.is_contiguous() or not w2.is_contiguous():
            raise ValueError("FlashInfer nosync needs contiguous packed FP4 weights")
        if w13_scale.dtype != torch.int32 or w2_scale.dtype != torch.int32:
            raise ValueError("FlashInfer nosync requires packed DeepGEMM UE8M0 scales")

        from flashinfer.moe_ep import (
            BootstrapConfig,
            FleetParams,
            MegaConfig,
            MoEEpMegaLayer,
            MoEEpTensors,
            PrequantizedMoEWeights,
            Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig,
        )
        from flashinfer.moe_ep import (
            preprocess_sm120_mxfp4_mxfp8_nosync_mega_weights as preprocess_nosync,
        )

        ep_group = get_ep_group()
        self._ep_world_size = ep_group.world_size
        self._ep_process_group = ep_group.device_group
        self._max_tokens_per_rank = 128
        weights = PrequantizedMoEWeights(
            w13=w13.view(torch.float4_e2m1fn_x2),
            w2=w2.view(torch.float4_e2m1fn_x2),
            w13_scale=w13_scale,
            w2_scale=w2_scale,
        )
        transformed = preprocess_nosync(
            weights,
            hidden_size=moe.hidden_dim,
            intermediate_size=moe.intermediate_size,
            interleave_w13=False,
            scale_layout="deepgemm",
        )
        config = Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig(
            intermediate_size=moe.intermediate_size,
            top_k=moe.experts_per_token,
            swiglu_limit=(
                float("inf") if moe.swiglu_limit is None else moe.swiglu_limit
            ),
            transport="p2p",
        )
        self._mega_layer: Any | None = MoEEpMegaLayer(
            BootstrapConfig(
                world_size=ep_group.world_size,
                rank=ep_group.rank_in_group,
                device=w13.device.index,
                process_group=ep_group.device_group,
            ),
            FleetParams(
                num_experts=moe.num_experts,
                max_tokens_per_rank=self._max_tokens_per_rank,
                token_hidden_size=moe.hidden_dim,
            ),
            weights,
            MegaConfig(
                megakernel=config,
                quantize_input=True,
                preprocess_weights=False,
                transformed_weights=transformed,
            ),
        )
        self._tensors_cls = MoEEpTensors
        self._device = w13.device
        self._hidden_size = moe.hidden_dim
        self._num_experts = moe.num_experts
        self._top_k = moe.experts_per_token
        self.warmup()

    def _tensors(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Any:
        return self._tensors_cls(
            hidden_states=hidden_states,
            topk_ids=topk_ids.to(torch.int64),
            topk_weights=topk_weights.float(),
        )

    @torch.inference_mode()
    def warmup(self) -> None:
        hidden_states = torch.zeros(
            1, self._hidden_size, dtype=torch.bfloat16, device=self._device
        )
        topk_ids = (
            torch.arange(self._top_k, dtype=torch.int64, device=self._device)
            .mul_(self._num_experts // self._top_k)
            .view(1, self._top_k)
        )
        topk_weights = torch.full(
            (1, self._top_k),
            1.0 / self._top_k,
            dtype=torch.float32,
            device=self._device,
        )
        assert self._mega_layer is not None
        self._mega_layer.warmup(self._tensors(hidden_states, topk_ids, topk_weights))

    def should_use(self, num_tokens: int) -> bool:
        """Use the global DP bound to select the same kernel on every EP rank."""
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if self._ep_world_size > 1:
            dp_metadata = (
                get_forward_context().dp_metadata
                if is_forward_context_available()
                else None
            )
            if dp_metadata is not None:
                num_tokens = int(dp_metadata.num_tokens_across_dp_cpu.max())
            elif torch.cuda.is_current_stream_capturing():
                raise RuntimeError("FlashInfer nosync requires DP token metadata")
            else:
                sizes = torch.tensor(
                    [num_tokens], dtype=torch.int32, device=self._device
                )
                torch.distributed.all_reduce(
                    sizes,
                    op=torch.distributed.ReduceOp.MAX,
                    group=self._ep_process_group,
                )
                num_tokens = int(sizes.item())
        return num_tokens <= self._max_tokens_per_rank

    def __call__(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._mega_layer is None:
            raise RuntimeError("FlashInfer nosync layer was destroyed")
        if hidden_states.shape[0] > self._max_tokens_per_rank:
            raise ValueError("FlashInfer nosync supports at most 128 tokens per rank")
        return self._mega_layer.forward(
            self._tensors(hidden_states, topk_ids, topk_weights),
            return_workspace_view=True,
        )

    def destroy(self) -> None:
        if self._mega_layer is not None:
            self._mega_layer.destroy()
            self._mega_layer = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()
