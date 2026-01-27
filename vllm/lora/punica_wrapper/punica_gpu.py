# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

from typing import final

import torch

from vllm.lora.layers import LoRAMapping
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.lora.ops.triton_ops import (
        LoRAKernelMeta,
        fused_moe_lora,
        lora_expand,
        lora_shrink,
    )


from .punica_base import PunicaWrapperBase


@final
class PunicaWrapperGPU(PunicaWrapperBase):
    """
    PunicaWrapperGPU is designed to manage and provide metadata for the punica
    kernel. The main function is to maintain the state information for
    Multi-LoRA, and to provide the interface for the punica triton kernel.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        max_batches: int,
        device: torch.device | str,
        **kwargs,
    ):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches, device)

        self.lora_config = kwargs["lora_config"]
        self.max_loras = self.lora_config.max_loras

        self.token_mapping_meta = LoRAKernelMeta.make(
            self.max_loras, max_num_batched_tokens, device=device
        )

        # When speculative decoding is enabled, max_num_samples is
        # max_batches * (num_speculative_decoding_tokens + 1).
        # This line can be optimized by replacing max_num_batched_tokens
        # to  max_batches * (num_speculative_decoding_tokens + 1).
        self.prompt_mapping_meta = LoRAKernelMeta.make(
            self.max_loras, max_num_batched_tokens, device=device
        )

    def update_metadata(
        self,
        mapping: LoRAMapping,
        lora_index_to_id: list[int | None],
        max_loras: int,
        vocab_size: int,
        **kwargs,
    ):
        self.is_prefill = mapping.is_prefill
        self._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)

        # Prepare cuda kernel metadata tensors
        self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
        self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)

    def add_shrink(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ):
        """
        Performs GEMM  for multiple slices of lora_a.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (torch.Tensor): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        x = x.view(-1, x.shape[-1])
        lora_shrink(
            x,
            lora_a_stacked,
            y,
            *self.token_mapping_meta.meta_args(x.size(0)),
            scale,
        )

    def add_expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> None:
        """
        Performs GEMM for multiple slices of lora_b.

        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i]
                offset += slice

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensors
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight
            output_slices (tuple[int, ...]): Every slice's size
            add_inputs (bool): Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])

        assert x.ndim == 3
        assert x.size(0) == len(output_slices)
        num_tokens = x.size(1)  # first dimension is the num slices

        lora_expand(
            x,
            lora_b_stacked,
            y,
            *self.token_mapping_meta.meta_args(num_tokens),
            offset_start=offset_start,
            add_inputs=True,
        )

        y = y.view_as(y_org)

    def add_lora_embedding(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        add_inputs: bool = True,
        **kwargs,
    ) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        lora_expand(
            x.unsqueeze(dim=0),
            (lora_b_stacked,),
            y,
            *self.token_mapping_meta.meta_args(x.size(0)),
            offset_start=0,
            add_inputs=add_inputs,
        )

    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        *,
        buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """
        Applicable to linear-related lora.

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)
        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight.
            scale (float): Scaling factor.
            output_slices (tuple[int, ...]): Every slice's size.
            buffer (Optional[torch.Tensor]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)

        assert buffer is None, (
            "To minimize overhead, the buffer should be created by "
            ".add_lora_linear() instead of being passed in."
        )
        r = lora_b_stacked[0].size(-1)
        # We set the buffer to be float32 by default, refer to:
        # https://github.com/triton-lang/triton/issues/1387
        # Note: buffer is zeroed inside the shrink op
        buffer = torch.empty(
            (len(output_slices), x.size(0), r), dtype=torch.float32, device=x.device
        )

        self.add_shrink(
            buffer,  # type: ignore
            x,
            lora_a_stacked,
            scale,
            **kwargs,
        )
        self.add_expand(
            y,
            buffer,  # type: ignore
            lora_b_stacked,
            output_slices,
            add_inputs=True,
            **kwargs,
        )

    def add_lora_logits(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        scale,
        *,
        buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.

        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]): Default to None.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)

        assert buffer is None, (
            "To minimize overhead, the buffer should be created by "
            ".add_lora_linear() instead of being passed in."
        )
        # We set the buffer to be float32 by default, refer to:
        # https://github.com/triton-lang/triton/issues/1387
        # Note: buffer is zeroed inside the shrink op
        buffer = torch.empty((x.size(0), r), dtype=torch.float32, device=x.device)

        lora_shrink(
            x,
            [lora_a_stacked],
            buffer.unsqueeze(dim=0),
            *self.prompt_mapping_meta.meta_args(x.size(0)),
            scale,
        )

        lora_expand(
            buffer.unsqueeze(dim=0),
            [lora_b_stacked],
            y,
            *self.prompt_mapping_meta.meta_args(buffer.size(0)),
            add_inputs=True,
        )
        y = y.view_as(y_org)

    def moe_lora_align_block_size(
        self,
        topk_ids: torch.Tensor,
        num_tokens: int,
        block_size: int,
        num_experts: int,
        max_loras: int,
        adapter_enabled: torch.Tensor,
        expert_map: torch.Tensor | None = None,
        pad_sorted_ids: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aligns tokens and experts into block-sized chunks for LoRA-based
        mixture-of-experts (MoE) execution using combined virtual experts.

        The approach combines (lora_id, expert_id) into a single "virtual expert"
        index: virtual_expert = lora_id * num_experts + expert_id

        For tokens without LoRA (token_lora_mapping == -1), we set
        topk_ids_lora to -1 to indicate they should be skipped.
        """

        (token_lora_mapping, _, _, _, _, _) = self.token_mapping_meta.meta_args(
            num_tokens
        )

        # Number of virtual experts = num_experts * max_loras
        num_expert_lora = num_experts * max_loras
        token_lora_expanded = token_lora_mapping.unsqueeze(1)  # (num_tokens, 1)

        # Create mask for valid LoRA tokens
        has_lora = token_lora_expanded >= 0  # (num_tokens, 1)

        topk_ids_lora = token_lora_expanded * num_experts + topk_ids
        topk_ids_lora = topk_ids_lora.masked_fill(~has_lora, -1)

        # Call standard moe_align_block_size with num_expert_lora + 1 virtual
        # experts (the extra slot absorbs non-LoRA tokens harmlessly).
        sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
            topk_ids_lora,
            block_size,
            num_expert_lora,
            expert_map=None,
            pad_sorted_ids=pad_sorted_ids,
        )

        # If no tokens have active LoRA adapters, zero out num_tokens_post_pad
        # so the downstream kernel exits immediately.
        return sorted_ids, expert_ids, num_tokens_post_pad

    def add_lora_fused_moe(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        topk_weights: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        max_lora_rank: int,
        top_k_num: int,
        shrink_config,
        expand_config,
        adapter_enabled: torch.Tensor,
        mul_routed_weight=False,
        fully_sharded: bool = False,
        offset: int = 0,
    ):
        """
        Performs a fused forward computation for LoRA of Mixture-of-Experts (MoE) layer.

        Uses combined virtual expert indexing where expert_ids contains:
            expert_id_lora = lora_id * num_experts + expert_id
        """
        fused_moe_lora(
            y,
            x,
            lora_a_stacked,
            lora_b_stacked,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            max_lora_rank,
            top_k_num,
            adapter_enabled,
            shrink_config.get("BLOCK_SIZE_M", 64),
            shrink_config.get("BLOCK_SIZE_N", 64),
            shrink_config.get("BLOCK_SIZE_K", 32),
            shrink_config.get("GROUP_SIZE_M", 8),
            shrink_config.get("NUM_WARPS", 4),
            shrink_config.get("NUM_STAGES", 3),
            shrink_config.get("SPLIT_K", 1),
            expand_config.get("BLOCK_SIZE_M", 64),
            expand_config.get("BLOCK_SIZE_N", 64),
            expand_config.get("BLOCK_SIZE_K", 32),
            expand_config.get("GROUP_SIZE_M", 8),
            expand_config.get("NUM_WARPS", 4),
            expand_config.get("NUM_STAGES", 3),
            expand_config.get("SPLIT_K", 1),
            mul_routed_weight,
            fully_sharded,
            offset,
        )
