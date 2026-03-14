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

from vllm import _custom_ops as ops
from vllm.lora.layers import LoRAMapping
from vllm.lora.ops.xpu_ops import bgmv_expand, bgmv_expand_slice, bgmv_shrink
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.math_utils import round_up

if HAS_TRITON:
    from vllm.lora.ops.triton_ops import (
        LoRAKernelMeta,
        fused_moe_lora,
    )

from .punica_base import PunicaWrapperBase


@final
class PunicaWrapperXPU(PunicaWrapperBase):
    """
    PunicaWrapperXPU is designed to manage and provide metadata for the punica
    kernel. The main function is to maintain the state information for
    Multi-LoRA, and to provide the interface for the punica ipex kernel.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        max_batches: int,
        device: torch.device | str,
        **kwargs,
    ):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches, device)
        torch._dynamo.mark_dynamic(self._token_lora_indices, 0)
        torch._dynamo.mark_dynamic(self._embeddings_indices, 1)
        torch._dynamo.mark_dynamic(self._sampler_indices_padded, 0)

        self.lora_config = kwargs["lora_config"]
        self.max_loras = self.lora_config.max_loras
        self.token_mapping_meta = LoRAKernelMeta.make(
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

    def _get_token_lora_indices(self, x: torch.Tensor) -> torch.IntTensor:
        return torch.narrow(self._token_lora_indices, 0, 0, x.size(0))

    def _apply_shrink(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        bgmv_shrink(x, w_t_all, y, self._get_token_lora_indices(x), scale)

    def _apply_expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: int,
        y_slice_size: int,
        add_inputs: bool,
    ):
        token_lora_indices = self._get_token_lora_indices(x)
        bgmv_expand_slice(
            x, w_t_all, y, token_lora_indices, y_offset, y_slice_size, add_inputs
        )

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
        for slice_idx in range(len(lora_a_stacked)):
            self._apply_shrink(y[slice_idx], x, lora_a_stacked[slice_idx], scale)

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

        # TODO fuse these kernels
        for slice_idx in range(len(lora_b_stacked)):
            self._apply_expand(
                y,
                x[slice_idx],
                lora_b_stacked[slice_idx],
                offset_start,
                output_slices[slice_idx],
                add_inputs=add_inputs,
            )
            offset_start += output_slices[slice_idx]
        y.view_as(y_org)

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
        token_lora_indices = self._get_token_lora_indices(x)
        bgmv_expand(x, lora_b_stacked, y, token_lora_indices, add_inputs)

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

        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            buffer = torch.zeros(  # type: ignore
                (len(output_slices), x.size(0), r),
                dtype=x.dtype,
                device=x.device,
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

    @property
    def sampler_indices_padded(self) -> torch.Tensor:
        """
        This property provides access to padded sampler indices.
        """
        return self._sampler_indices_padded[:]

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
        if buffer is None:
            buffer = torch.zeros((x.size(0), r), dtype=x.dtype, device=x.device)
        sampler_indices = torch.narrow(self._sampler_indices, 0, 0, x.size(0))
        bgmv_shrink(x, lora_a_stacked, buffer, sampler_indices, scale)
        bgmv_expand(buffer, lora_b_stacked, y, sampler_indices, add_inputs=True)
        return y.view_as(y_org)

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
        naive_block_assignment: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aligns tokens and experts into block-sized chunks for LoRA-based
        mixture-of-experts (MoE) execution.
        """
        (token_lora_mapping, _, _, _, lora_ids, _, _) = (
            self.token_mapping_meta.meta_args(
                num_tokens, self.lora_config.specialize_active_lora
            )
        )
        if naive_block_assignment:
            expert_ids = topk_ids.reshape(-1)
            sorted_ids = None
            num_tokens_post_pad = None
        else:
            max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
            if pad_sorted_ids:
                max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
            sorted_ids = torch.empty(
                (max_loras * max_num_tokens_padded,),
                dtype=torch.int32,
                device=topk_ids.device,
            )
            max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
            # Expert ids must be set default to -1 to prevent a blank block
            expert_ids = torch.empty(
                (max_loras * max_num_m_blocks,),
                dtype=torch.int32,
                device=topk_ids.device,
            )
            num_tokens_post_pad = torch.empty(
                (max_loras), dtype=torch.int32, device=topk_ids.device
            )

            ops.moe_lora_align_block_size(
                topk_ids,
                token_lora_mapping,
                num_experts,
                block_size,
                max_loras,
                max_num_tokens_padded,
                max_num_m_blocks,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                adapter_enabled,
                lora_ids,
            )
            if expert_map is not None:
                expert_ids = expert_map[expert_ids]

        return None, sorted_ids, expert_ids, num_tokens_post_pad

    def add_lora_fused_moe(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        topk_weights: torch.Tensor,
        sorted_token_ids: torch.Tensor | None,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor | None,
        max_lora_rank: int,
        top_k_num: int,
        shrink_config,
        expand_config,
        adapter_enabled: torch.Tensor,
        mul_routed_weight=False,
        fully_sharded: bool = False,
        offset: int = 0,
        token_lora_mapping: torch.Tensor | None = None,
    ):
        """
        Performs a fused forward computation for LoRA of Mixture-of-Experts (MoE) layer.
        """
        (
            token_lora_mapping_meta,
            _,
            _,
            _,
            lora_ids,
            _,
            num_active_loras,
        ) = self.token_mapping_meta.meta_args(
            x.size(0), self.lora_config.specialize_active_lora
        )
        if token_lora_mapping is None:
            token_lora_mapping = token_lora_mapping_meta
        fused_moe_lora(
            y,
            x,
            lora_a_stacked,
            lora_b_stacked,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            token_lora_mapping,
            max_lora_rank,
            top_k_num,
            lora_ids,
            num_active_loras,
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
