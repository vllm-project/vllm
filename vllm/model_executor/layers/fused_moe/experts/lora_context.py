# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase


@dataclass
class MoELoRAContext:
    """
    Carries all LoRA state for one MoE forward pass.

    Built by FusedMoEWithLoRA.forward() and propagated explicitly through the
    modular kernel path (FusedMoEKernel -> FusedMoEExpertsModular.apply) so
    that TritonExperts.apply() can compute the LoRA contribution inline,
    replacing the decorator-based monkey-patch approach.
    """

    # LoRA weight tensors (same shapes as FusedMoEWithLoRA attributes)
    w13_lora_a_stacked: tuple[torch.Tensor, ...]
    w13_lora_b_stacked: tuple[torch.Tensor, ...]
    w2_lora_a_stacked: tuple[torch.Tensor, ...]
    w2_lora_b_stacked: tuple[torch.Tensor, ...]

    # (max_loras + 1,) int32; slot 0 is the "no-adapter" sentinel
    adapter_enabled: torch.Tensor

    # Metadata
    max_loras: int
    top_k: int
    w13_num_slices: int  # 2 = gated (gate + up), 1 = non-gated or 3D-fused
    fully_sharded: bool
    tp_rank: int
    tp_size: int
    local_num_experts: int

    punica_wrapper: PunicaWrapperBase

    # Whether VLLM_TUNED_CONFIG_FOLDER is set; selects get_lora_op_configs vs
    # try_get_optimal_moe_lora_config for Triton kernel tile configs.
    use_tuned_config: bool

    # Per-rank token→LoRA mapping after EP dispatch. Set by
    # FusedMoEPrepareAndFinalizeModular.prepare() when EP+LoRA is active, read
    # by LoRAExpertsMixin helpers in place of punica_wrapper's global mapping.
    # None means no dispatch happened (non-EP path), in which case callers
    # fall back to punica_wrapper.token_mapping_meta.
    local_token_lora_mapping: torch.Tensor | None = None
