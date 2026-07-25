# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlareV2Norm Qwen3 draft model for speculative decoding."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.triton_utils import tl, triton

from .qwen3_dflarev2 import (
    DFlareV2Qwen3ForCausalLM,
    DFlareV2Qwen3Model,
)
from .utils import maybe_prefix


@triton.jit
def _fused_layer_context_independent_rmsnorm_kernel(
    base_ptr,
    stacked_ptr,
    fusion_ptr,
    norm_weights_ptr,
    output_ptr,
    num_ctx,
    hidden_size: tl.constexpr,
    num_target_layers: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse target fusion, residual add, and per-draft-layer RMSNorm."""
    row = tl.program_id(0)
    layer = row // num_ctx
    ctx = row - layer * num_ctx
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size

    values = tl.load(
        base_ptr + ctx * hidden_size + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    stacked_row = ctx * num_target_layers * hidden_size
    fusion_row = layer * num_target_layers
    for target_idx in range(num_target_layers):
        target = tl.load(
            stacked_ptr
            + stacked_row
            + target_idx * hidden_size
            + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        coefficient = tl.load(fusion_ptr + fusion_row + target_idx).to(
            tl.float32
        )
        values += coefficient * target

    variance = tl.sum(values * values, axis=0) / hidden_size
    values *= tl.rsqrt(variance + eps)
    norm_weight = tl.load(
        norm_weights_ptr + layer * hidden_size + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(output_ptr + row * hidden_size + offsets, values * norm_weight, mask=mask)


class DFlareV2NormQwen3Model(DFlareV2Qwen3Model):
    """DFlareV2 with an independent context RMSNorm for every draft layer."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        self.context_norms = nn.ModuleList(
            [
                RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
                for _ in range(self.num_draft_layers)
            ]
        )
        self.register_buffer(
            "_context_norm_weights",
            torch.empty(
                self.num_draft_layers,
                self.config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            ),
            persistent=False,
        )

    def _build_fused_kv_buffers(self) -> None:
        super()._build_fused_kv_buffers()
        self._context_norm_weights.copy_(
            torch.stack([norm.weight for norm in self.context_norms])
        )

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        L, T = self.num_draft_layers, self.num_target_layers
        D = self.config.hidden_size
        if num_layers != L:
            raise ValueError(
                f"DFlareV2Norm expected {L} draft layers, got {num_layers}."
            )
        if context_states.shape[-1] != T * D:
            raise ValueError(
                "DFlareV2Norm context width mismatch: "
                f"got {context_states.shape[-1]}, expected {T * D}."
            )

        stacked = context_states.view(num_ctx, T, D)
        base_context = self.fc(context_states)
        fusion_probs = self._fusion_probs.to(dtype=context_states.dtype)
        if context_states.is_cuda:
            normed = torch.empty(
                (L, num_ctx, D),
                dtype=context_states.dtype,
                device=context_states.device,
            )
            _fused_layer_context_independent_rmsnorm_kernel[(L * num_ctx,)](
                base_context,
                stacked,
                fusion_probs,
                self._context_norm_weights,
                normed,
                num_ctx=num_ctx,
                hidden_size=D,
                num_target_layers=T,
                eps=self.config.rms_norm_eps,
                BLOCK_SIZE=triton.next_power_of_2(D),
            )
        else:
            residual_context = torch.einsum(
                "lt,ntd->lnd", fusion_probs, stacked
            )
            layer_context = residual_context + base_context.unsqueeze(0)
            variance = layer_context.float().pow(2).mean(dim=-1, keepdim=True)
            normed = (
                layer_context.float()
                * torch.rsqrt(variance + self.config.rms_norm_eps)
                * self._context_norm_weights[:, None, :].float()
            ).to(dtype=layer_context.dtype)

        kv_size_per_partition = 2 * num_kv_heads * head_dim
        w_stacked = self._fused_kv_weight.view(
            L, kv_size_per_partition, D
        )
        all_kv_flat = torch.bmm(normed, w_stacked.transpose(1, 2))
        if self._fused_kv_bias is not None:
            all_kv_flat = all_kv_flat + self._fused_kv_bias.view(
                L, 1, kv_size_per_partition
            )

        all_kv = (
            all_kv_flat.view(L, num_ctx, 2, num_kv_heads, head_dim)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )
        return all_kv[0], all_kv[1]


class DFlareV2NormQwen3ForCausalLM(DFlareV2Qwen3ForCausalLM):
    """Top-level DFlareV2Norm wrapper with training-export weight remapping."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.draft_model_config = (
            vllm_config.speculative_config.draft_model_config
        )
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(
                self.config, "vocab_size", None
            )
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = DFlareV2NormQwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        def remapped_weights():
            for name, weight in weights:
                if name == "context_proj.weight":
                    name = "fc.weight"
                elif name == "final_norm.weight":
                    name = "norm.weight"
                elif name.startswith("context_norm."):
                    name = name.replace("context_norm.", "context_norms.", 1)
                yield name, weight

        return super().load_weights(remapped_weights())


__all__ = [
    "DFlareV2NormQwen3Model",
    "DFlareV2NormQwen3ForCausalLM",
]
