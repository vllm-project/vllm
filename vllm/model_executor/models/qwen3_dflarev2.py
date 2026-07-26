# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlareV2 Qwen3 draft model for speculative decoding."""

import torch
import torch.nn.functional as F
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.triton_utils import tl, triton

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .utils import maybe_prefix


@triton.jit
def _fused_layer_context_rmsnorm_kernel(
    base_ptr,
    stacked_ptr,
    fusion_ptr,
    norm_weight_ptr,
    output_ptr,
    num_ctx,
    hidden_size: tl.constexpr,
    num_target_layers: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse target-layer mixing, residual add, and shared RMSNorm."""
    row = tl.program_id(0)
    layer = row // num_ctx
    ctx = row - layer * num_ctx
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size

    values = tl.load(base_ptr + ctx * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    stacked_row = ctx * num_target_layers * hidden_size
    fusion_row = layer * num_target_layers
    for target_idx in range(num_target_layers):
        target = tl.load(
            stacked_ptr + stacked_row + target_idx * hidden_size + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        coefficient = tl.load(fusion_ptr + fusion_row + target_idx).to(tl.float32)
        values += coefficient * target

    variance = tl.sum(values * values, axis=0) / hidden_size
    values *= tl.rsqrt(variance + eps)
    norm_weight = tl.load(norm_weight_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(output_ptr + row * hidden_size + offsets, values * norm_weight, mask=mask)


class DFlareV2Qwen3Model(DFlashQwen3Model):
    """DFlash backbone with FC plus per-layer target-fusion context."""

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

        target_hidden_size = getattr(
            self.config, "target_hidden_size", self.config.hidden_size
        )
        if target_hidden_size != self.config.hidden_size:
            raise ValueError(
                "DFlareV2 requires target_hidden_size == hidden_size, got "
                f"{target_hidden_size} and {self.config.hidden_size}."
            )

        drafter_config = {}
        drafter_config.update(getattr(self.config, "dflash_config", None) or {})
        drafter_config.update(getattr(self.config, "dflare_config", None) or {})
        target_layer_ids = (
            drafter_config.get("target_layer_ids")
            or getattr(self.config, "target_layer_ids", None)
            or getattr(self.config, "eagle_aux_hidden_state_layer_ids", None)
        )
        configured_num_target_layers = getattr(self.config, "num_target_layers", None)
        if target_layer_ids:
            num_target_layers = len(target_layer_ids)
            if (
                configured_num_target_layers is not None
                and int(configured_num_target_layers) != num_target_layers
            ):
                raise ValueError(
                    "DFlareV2 num_target_layers does not match "
                    f"target_layer_ids: {configured_num_target_layers} != "
                    f"{num_target_layers}."
                )
        elif configured_num_target_layers is not None:
            num_target_layers = int(configured_num_target_layers)
        else:
            raise ValueError("DFlareV2 requires num_target_layers or target_layer_ids.")

        expected_fc_input = num_target_layers * target_hidden_size
        if self.fc.input_size != expected_fc_input:
            raise ValueError(
                "DFlareV2 context projection width mismatch: "
                f"fc.input_size={self.fc.input_size}, "
                f"expected={expected_fc_input}."
            )

        self.num_target_layers = num_target_layers
        self.num_draft_layers = self.config.num_hidden_layers
        self.layer_fusion_weights = nn.Parameter(
            torch.empty(
                self.num_draft_layers,
                self.num_target_layers,
                dtype=vllm_config.model_config.dtype,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "_fusion_probs",
            torch.empty_like(self.layer_fusion_weights),
            persistent=False,
        )

    def _build_fused_kv_buffers(self) -> None:
        super()._build_fused_kv_buffers()
        # Fusion weights are inference-only. Avoid repeating the dtype cast and
        # softmax on every speculative step.
        self._fusion_probs.copy_(
            F.softmax(self.layer_fusion_weights.float(), dim=-1).to(
                dtype=self.layer_fusion_weights.dtype
            )
        )

    def combine_hidden_states(self, aux_hidden_states: torch.Tensor) -> torch.Tensor:
        expected_width = self.num_target_layers * self.config.hidden_size
        actual_width = aux_hidden_states.shape[-1]
        if actual_width < expected_width:
            raise ValueError(
                "DFlareV2 received too few target hidden states: "
                f"width={actual_width}, expected at least {expected_width}."
            )
        return aux_hidden_states[..., :expected_width]

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
            raise ValueError(f"DFlareV2 expected {L} draft layers, got {num_layers}.")
        if context_states.shape[-1] != T * D:
            raise ValueError(
                "DFlareV2 context width mismatch: "
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
            _fused_layer_context_rmsnorm_kernel[(L * num_ctx,)](
                base_context,
                stacked,
                fusion_probs,
                self._hidden_norm_weight,
                normed,
                num_ctx=num_ctx,
                hidden_size=D,
                num_target_layers=T,
                eps=self._rms_norm_eps,
                BLOCK_SIZE=triton.next_power_of_2(D),
            )
        else:
            residual_context = torch.einsum("lt,ntd->lnd", fusion_probs, stacked)
            layer_context = residual_context + base_context.unsqueeze(0)
            layer_context_flat = layer_context.reshape(L * num_ctx, D)
            normed_flat = torch.empty_like(layer_context_flat)
            ops.rms_norm(
                normed_flat,
                layer_context_flat,
                self._hidden_norm_weight,
                self._rms_norm_eps,
            )
            normed = normed_flat.view(L, num_ctx, D)

        kv_size_per_partition = 2 * num_kv_heads * head_dim
        w_stacked = self._fused_kv_weight.view(L, kv_size_per_partition, D)
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


class DFlareV2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    """Top-level DFlareV2 wrapper using the DFlash runtime and loader."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = DFlareV2Qwen3Model(
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

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)


__all__ = ["DFlareV2Qwen3Model", "DFlareV2Qwen3ForCausalLM"]
