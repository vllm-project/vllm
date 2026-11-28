# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import torch

from vllm import envs
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group

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
            raise ValueError("FlashInfer nosync currently requires expert TP=1")
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
        self._ep_rank = ep_group.rank_in_group
        self._ep_process_group = ep_group.device_group
        self._requires_dp_metadata = self._ep_world_size > 1 and (
            get_dp_group().world_size > 1 or moe.moe_parallel_config.sp_size > 1
        )
        self._max_tokens_per_rank = (
            envs.VLLM_FLASHINFER_MOE_EP_NOSYNC_MAX_TOKENS_PER_RANK
        )
        if not 1 <= self._max_tokens_per_rank <= 192:
            raise ValueError(
                "VLLM_FLASHINFER_MOE_EP_NOSYNC_MAX_TOKENS_PER_RANK must be in [1, 192]"
            )
        self._dedup_uniform = envs.VLLM_FLASHINFER_MOE_EP_NOSYNC_TP_DEDUP
        if self._dedup_uniform:
            if self._requires_dp_metadata:
                raise ValueError(
                    "TP-uniform dedup requires replicated TP-only inputs, not DP or SP"
                )
            custom_ar = getattr(get_tp_group().device_communicator, "ca_comm", None)
            if custom_ar is not None and not custom_ar.disabled:
                raise ValueError(
                    "TP-uniform dedup requires CustomAllreduce to be disabled"
                )
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
        config_kwargs: dict[str, Any] = {
            "intermediate_size": moe.intermediate_size,
            "top_k": moe.experts_per_token,
            "swiglu_limit": (
                float("inf") if moe.swiglu_limit is None else moe.swiglu_limit
            ),
            "transport": "p2p",
        }
        if self._dedup_uniform:
            config_kwargs["dedup_replicated_inputs"] = True
        config = Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig(**config_kwargs)
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
        """Use a uniform token bound to select the same kernel on every EP rank."""
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if self._requires_dp_metadata:
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
            raise ValueError(
                "FlashInfer nosync supports at most "
                f"{self._max_tokens_per_rank} tokens per rank"
            )
        if self._dedup_uniform:
            owned = (
                torch.arange(topk_ids.shape[0], device=topk_ids.device).remainder(
                    self._ep_world_size
                )
                == self._ep_rank
            ).unsqueeze(1)
            topk_ids = topk_ids.masked_fill(~owned, -1)
            topk_weights = topk_weights.masked_fill(~owned, 0)
        return self._mega_layer.forward(
            self._tensors(hidden_states, topk_ids, topk_weights),
            return_workspace_view=not self._dedup_uniform,
        )

    @property
    def dedup_uniform(self) -> bool:
        return self._dedup_uniform

    def destroy(self) -> None:
        if self._mega_layer is not None:
            self._mega_layer.destroy()
            self._mega_layer = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()
