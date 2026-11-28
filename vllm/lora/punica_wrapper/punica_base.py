# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.utils.torch_utils import async_tensor_h2d

from .utils import compute_meta, convert_mapping, convert_slot_mapping

if TYPE_CHECKING:
    # avoid circuit import
    from vllm.lora.layers import LoRAMapping


class PunicaWrapperABC(ABC):
    """PunicaWrapper ABC."""

    @abstractmethod
    def update_metadata(
        self,
        mapping: "LoRAMapping",
        lora_index_to_id: list[int | None],
        max_loras: int,
        vocab_size: int,
        **kwargs,
    ) -> None:
        """Update the lora-related metadata"""
        raise NotImplementedError

    @abstractmethod
    def add_shrink(
        self,
        y: tuple[torch.Tensor, ...] | torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ) -> torch.Tensor | None:
        """Performs GEMM  for multiple slices of lora_a."""
        raise NotImplementedError

    @abstractmethod
    def add_expand(
        self,
        y: torch.Tensor,
        x: tuple[torch.Tensor, ...] | torch.Tensor,
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> torch.Tensor | None:
        """Performs GEMM for multiple slices of lora_b."""
        raise NotImplementedError

    @abstractmethod
    def add_lora_embedding(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        add_inputs: bool = True,
        **kwargs,
    ) -> torch.Tensor | None:
        """Applies lora  specifically for VocabParallelEmbeddingWithLoRA,
        and this layer only requires the expand operation.
        """
        raise NotImplementedError

    @abstractmethod
    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        *,
        buffer: tuple[torch.Tensor, ...] | None = None,
        **kwargs,
    ) -> torch.Tensor | None:
        """Applicable to linear-related lora."""
        raise NotImplementedError

    @abstractmethod
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
    ) -> torch.Tensor | None:
        """Applies lora  specifically for LogitsProcessorWithLoRA."""
        raise NotImplementedError

    @abstractmethod
    def apply_lora_full_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        weight_stacked: torch.Tensor,
        bias_stacked: torch.Tensor,
        module_enabled: torch.Tensor,
    ) -> torch.Tensor | None:
        """Apply request-routed full linear weights to selected rows."""
        raise NotImplementedError


class PunicaWrapperBase(PunicaWrapperABC):
    """PunicaWrapperBase is designed to manage and provide metadata for the punica
    kernel. The main function is to maintain the state information for
    Multi-LoRA, and to provide the interface for the punica.
    """

    _supports_multi_slot_and_scales: ClassVar[bool] = False

    def __init__(
        self,
        max_num_batched_tokens: int,
        max_batches: int,
        device: torch.device | str,
        **kwargs,
    ):
        lora_config = kwargs.get("lora_config")
        max_loras_per_request = (
            lora_config.max_loras_per_request
            if lora_config is not None
            else kwargs.get("max_loras_per_request", 1)
        )
        if max_loras_per_request < 1:
            raise ValueError(
                f"max_loras_per_request must be >= 1, got {max_loras_per_request}"
            )
        self.max_loras_per_request: int = max_loras_per_request

        self._slot_token_lora_indices = torch.full(
            (max_loras_per_request, max_num_batched_tokens),
            -1,
            dtype=torch.long,
            device=device,
        )
        self._slot_sampler_indices = torch.full(
            (max_loras_per_request, max_num_batched_tokens),
            -1,
            dtype=torch.long,
            device=device,
        )
        self._slot_embeddings_indices = torch.zeros(
            (max_loras_per_request, max_num_batched_tokens),
            dtype=torch.long,
            device=device,
        )
        self._slot_token_lora_scales = torch.ones(
            (max_loras_per_request, max_num_batched_tokens),
            dtype=torch.float32,
            device=device,
        )
        self._slot_sampler_lora_scales = torch.ones(
            (max_loras_per_request, max_num_batched_tokens),
            dtype=torch.float32,
            device=device,
        )

        # Slot-0 views preserve existing attribute contracts.
        self._token_lora_indices = self._slot_token_lora_indices[0]
        self._sampler_indices = self._slot_sampler_indices[0]
        self._sampler_indices_padded = torch.empty(
            max_num_batched_tokens, dtype=torch.long, device=device
        )
        self._embeddings_indices = self._slot_embeddings_indices[0]

        # 4 is the number of indices tensors.
        # base_indices, sampler_indices, sampler_indices_padded,
        # embeddings_indices
        self.indices_len: list[int | None] = [None] * 4
        # these attributes are the information required for sgmv kernel
        self._seq_start_locs = torch.empty(max_batches, dtype=torch.long, device=device)
        self._seq_lengths = torch.empty(max_batches, dtype=torch.long, device=device)
        self._lora_indices_per_batch = torch.empty(
            max_batches, dtype=torch.long, device=device
        )
        self.device: torch.device = device
        self.max_length: int = 0
        self.token_nums: int = 0
        self.batch_size: int = -1
        self.is_prefill = False
        self.no_lora = False
        self._num_active_slots: int = 1
        self._has_custom_scales: bool = False
        self._slot_prefill_meta: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, bool]
        ] = []

    def _update_base_metadata(
        self,
        mapping: "LoRAMapping",
        lora_index_to_id: list[int | None],
        max_loras: int,
        vocab_size: int,
    ):
        num_slots = mapping.num_slots
        uses_multi_or_scales = num_slots > 1 or mapping.slot_index_scales is not None
        if uses_multi_or_scales and not self._supports_multi_slot_and_scales:
            raise NotImplementedError(
                f"{type(self).__name__} does not support per-request "
                "multi-adapter composition or static LoRA scales."
            )
        if num_slots > self.max_loras_per_request:
            raise ValueError(
                f"LoRAMapping has {num_slots} slots, exceeding "
                f"max_loras_per_request ({self.max_loras_per_request})."
            )

        (
            base_indices,
            sampler_indices,
            sampler_indices_padded,
            embeddings_indices,
            indices_len,
        ) = convert_mapping(
            mapping,
            lora_index_to_id,
            max_loras,
            vocab_size,
            self.device,
        )
        num_tokens = base_indices.shape[0]
        num_prompts = sampler_indices.shape[0]
        self._token_lora_indices[:num_tokens].copy_(base_indices)
        self._sampler_indices[:num_prompts].copy_(sampler_indices)
        self._sampler_indices_padded[: sampler_indices_padded.shape[0]].copy_(
            sampler_indices_padded
        )
        self._embeddings_indices[: embeddings_indices.shape[0]].copy_(
            embeddings_indices
        )

        self.indices_len[:] = indices_len

        if mapping.slot_index_mappings is not None:
            assert mapping.slot_prompt_mappings is not None
            for slot_idx in range(1, num_slots):
                slot_base, slot_sampler, slot_embed = convert_slot_mapping(
                    mapping.slot_index_mappings[slot_idx],
                    mapping.slot_prompt_mappings[slot_idx],
                    lora_index_to_id,
                    vocab_size,
                    self.device,
                )
                self._slot_token_lora_indices[slot_idx, :num_tokens].copy_(slot_base)
                self._slot_sampler_indices[slot_idx, :num_prompts].copy_(slot_sampler)
                self._slot_embeddings_indices[slot_idx, :num_tokens].copy_(slot_embed)

        for slot_idx in range(num_slots, self.max_loras_per_request):
            self._slot_token_lora_indices[slot_idx, :num_tokens].fill_(-1)
            self._slot_sampler_indices[slot_idx, :num_prompts].fill_(-1)
            self._slot_embeddings_indices[slot_idx, :num_tokens].fill_(0)

        if mapping.slot_index_scales is not None:
            assert mapping.slot_prompt_scales is not None
            token_scales = async_tensor_h2d(
                list(mapping.slot_index_scales),
                dtype=torch.float32,
                device=self.device,
            )
            prompt_scales = async_tensor_h2d(
                list(mapping.slot_prompt_scales),
                dtype=torch.float32,
                device=self.device,
            )
            self._slot_token_lora_scales[:num_slots, :num_tokens].copy_(token_scales)
            self._slot_sampler_lora_scales[:num_slots, :num_prompts].copy_(
                prompt_scales
            )
            if num_slots < self.max_loras_per_request:
                self._slot_token_lora_scales[num_slots:, :num_tokens].fill_(0.0)
                self._slot_sampler_lora_scales[num_slots:, :num_prompts].fill_(0.0)

            slot_index_ids = (
                mapping.slot_index_mappings
                if mapping.slot_index_mappings is not None
                else (mapping.index_mapping,)
            )
            slot_prompt_ids = (
                mapping.slot_prompt_mappings
                if mapping.slot_prompt_mappings is not None
                else (mapping.prompt_mapping,)
            )
            self._has_custom_scales = any(
                lora_id > 0 and scale != 1.0
                for id_row, scale_row in zip(slot_index_ids, mapping.slot_index_scales)
                for lora_id, scale in zip(id_row, scale_row)
            ) or any(
                lora_id > 0 and scale != 1.0
                for id_row, scale_row in zip(
                    slot_prompt_ids, mapping.slot_prompt_scales
                )
                for lora_id, scale in zip(id_row, scale_row)
            )
        else:
            if self.max_loras_per_request > 1 or self._has_custom_scales:
                self._slot_token_lora_scales[:num_slots, :num_tokens].fill_(1.0)
                self._slot_sampler_lora_scales[:num_slots, :num_prompts].fill_(1.0)
                if num_slots < self.max_loras_per_request:
                    self._slot_token_lora_scales[num_slots:, :num_tokens].fill_(0.0)
                    self._slot_sampler_lora_scales[num_slots:, :num_prompts].fill_(0.0)
            self._has_custom_scales = False

        self._num_active_slots = num_slots

    def _update_prefill_metadata(self, token_lora_tensor: torch.Tensor) -> None:
        (
            b_seq_start_tensor,
            seq_length_tensor,
            lora_indices_tensor,
            batch_size,
            max_length,
            token_nums,
            no_lora,
        ) = compute_meta(token_lora_tensor)

        self._seq_start_locs[: b_seq_start_tensor.shape[0]].copy_(b_seq_start_tensor)
        self._seq_lengths[: seq_length_tensor.shape[0]].copy_(seq_length_tensor)
        self._lora_indices_per_batch[: lora_indices_tensor.shape[0]].copy_(
            lora_indices_tensor
        )
        self.batch_size = batch_size
        self.max_length = max_length
        self.token_nums = token_nums
        self.no_lora = no_lora

        self._slot_prefill_meta = [
            (
                self._seq_start_locs[:batch_size],
                self._seq_lengths[:batch_size],
                self._lora_indices_per_batch[:batch_size],
                batch_size,
                max_length,
                token_nums,
                no_lora,
            )
        ]
        for slot_idx in range(1, self._num_active_slots):
            self._slot_prefill_meta.append(
                compute_meta(self.get_token_lora_indices(slot_idx))
            )

    def _select_full_linear_output(
        self,
        y: torch.Tensor,
        adapter_y: torch.Tensor,
        bias_stacked: torch.Tensor,
        module_enabled: torch.Tensor,
    ) -> torch.Tensor:
        if self.has_multi_slot or self.has_custom_scales:
            raise NotImplementedError(
                "Multi-adapter or scaled LoRA is not supported for full-module "
                "classification head replacement."
            )
        indices = self.sampler_indices
        assert indices.size(0) == y.size(0), (
            "Full linear rows do not match LoRA request mapping"
        )
        safe_indices = indices.clamp_min(0)
        use_full = (indices >= 0) & module_enabled[safe_indices]
        adapter_y = adapter_y.to(y.dtype) + bias_stacked[safe_indices].to(y.dtype)
        return torch.where(use_full.unsqueeze(-1), adapter_y, y)

    @property
    def num_active_slots(self) -> int:
        return self._num_active_slots

    @property
    def has_multi_slot(self) -> bool:
        return self._num_active_slots > 1

    @property
    def has_custom_scales(self) -> bool:
        return self._has_custom_scales

    def get_prefill_metadata(
        self, slot_idx: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        if slot_idx == 0:
            return self.prefill_metadata
        meta = self._slot_prefill_meta[slot_idx]
        return meta[:6]

    def is_slot_no_lora(self, slot_idx: int = 0) -> bool:
        if slot_idx == 0:
            return self.no_lora
        return self._slot_prefill_meta[slot_idx][6]

    def get_token_lora_indices(self, slot_idx: int = 0) -> torch.Tensor:
        token_lora_len = self.indices_len[0]
        return self._slot_token_lora_indices[slot_idx, :token_lora_len]

    def get_sampler_indices(self, slot_idx: int = 0) -> torch.Tensor:
        sampler_indices_len = self.indices_len[1]
        return self._slot_sampler_indices[slot_idx, :sampler_indices_len]

    def get_embeddings_indices(
        self, num_tokens: int, slot_idx: int = 0
    ) -> torch.Tensor:
        return self._slot_embeddings_indices[slot_idx, :num_tokens]

    def get_token_lora_scales(self, slot_idx: int, num_tokens: int) -> torch.Tensor:
        return self._slot_token_lora_scales[slot_idx, :num_tokens]

    def get_sampler_lora_scales(self, slot_idx: int, num_prompts: int) -> torch.Tensor:
        return self._slot_sampler_lora_scales[slot_idx, :num_prompts]

    @property
    def prefill_metadata(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        """This property provides a convenient way to access the necessary
        metadata for prefill-related  kernel computations.
            1. seq_start_locs: Tensor of sequence start positions.
            2. seq_lengths: Tensor of sequence lengths.
            3. lora_indices_per_batch: Tensor of lora indices, and an index of
                -1 means no lora should be applied.
            4. batch_size: Batch size after clustering identical lora indices.
            5. max_length: The maximum sequence length in the batch.
            6. token_nums: The token numbers in the batch.
        """
        return (
            self._seq_start_locs[: self.batch_size],
            self._seq_lengths[: self.batch_size],
            self._lora_indices_per_batch[: self.batch_size],
            self.batch_size,
            self.max_length,
            self.token_nums,
        )

    @property
    def token_lora_indices(self) -> torch.Tensor:
        """This property provides the lora indices corresponding to each token
        in the batch. An index of -1 means no lora should be applied.
        """
        token_lora_len = self.indices_len[0]
        return self._token_lora_indices[:token_lora_len]

    @property
    def sampler_indices(self) -> torch.Tensor:
        """This property is used to access the lora indices specifically for
        LogitsProcessorWithLoRA.
        """
        sampler_indices_len = self.indices_len[1]
        return self._sampler_indices[:sampler_indices_len]

    @property
    def sampler_indices_padded(self) -> torch.Tensor:
        """This property provides access to padded sampler indices."""
        indices_padded_len = self.indices_len[2]
        return self._sampler_indices_padded[:indices_padded_len]

    def update_metadata(
        self,
        mapping: "LoRAMapping",
        lora_index_to_id: list[int | None],
        max_loras: int,
        vocab_size: int,
        **kwargs,
    ):
        self._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)

        if mapping.is_prefill:
            # Update metadata required for prefill-related operators.
            self._update_prefill_metadata(self.token_lora_indices)
            self.is_prefill = True
        else:
            self.is_prefill = False

    @abstractmethod
    def add_shrink(
        self,
        y: tuple[torch.Tensor, ...] | torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ) -> torch.Tensor | None:
        """Performs GEMM  for multiple slices of lora_a.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale

        Args:
            y (Union[tuple[torch.Tensor, ...], torch.Tensor]): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation

        """
        # TODO: implement it based on torch ops
        raise NotImplementedError

    @abstractmethod
    def add_expand(
        self,
        y: torch.Tensor,
        x: tuple[torch.Tensor, ...] | torch.Tensor,
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> torch.Tensor | None:
        """Performs GEMM for multiple slices of lora_b.

        Semantics:
            offset = offset_start
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i]
                offset += slice

        Args:
            y (torch.Tensor): Output tensor.
            x (Union[tuple[torch.Tensor, ...], torch.Tensor]): Input tensors
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight
            output_slices (tuple[int, ...]): Every slice's size
            offset_start (int): The starting position of y, defaults to 0
            add_inputs (bool):  Defaults to True.

        """
        # TODO: implement it based on torch ops
        raise NotImplementedError

    @abstractmethod
    def add_lora_embedding(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        add_inputs: bool = True,
        **kwargs,
    ) -> torch.Tensor | None:
        """Applies lora  specifically for VocabParallelEmbeddingWithLoRA.
        and this layer only requires the expand operation.
        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.

        """
        # TODO: implement it based on torch ops
        raise NotImplementedError

    @abstractmethod
    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        *,
        buffer: tuple[torch.Tensor, ...] | None = None,
        **kwargs,
    ) -> torch.Tensor | None:
        """Applicable to linear-related lora.

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
            buffer (Optional[tuple[torch.Tensor, ...]]): Defaults to None.

        """
        # TODO: implement it based on torch ops
        raise NotImplementedError

    @abstractmethod
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
    ) -> torch.Tensor | None:
        """Applies lora  specifically for LogitsProcessorWithLoRA.

        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor):lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]):Default to None.

        """
        # TODO: implement it based on torch ops
        raise NotImplementedError

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
        """Aligns tokens and experts into block-sized chunks for LoRA-based
        mixture-of-experts (MoE) execution.
        """
        # TODO: implement it based on torch ops
        raise NotImplementedError

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
        """Performs a fused forward computation for LoRA of
        Mixture-of-Experts (MoE) layer.
        """
        # TODO: implement it based on torch ops
        raise NotImplementedError

    def add_lora_w13(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_map: torch.Tensor | None,
        w1: torch.Tensor,
        w2: torch.Tensor,
        num_tokens: int,
        top_k_num: int,
        max_loras: int,
        adapter_enabled: torch.Tensor,
        local_num_experts: int,
        top_k: int,
        num_slices: int,
        fully_sharded: bool,
        use_tuned_config: bool,
        add_inputs: bool = True,
        token_lora_mapping: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Apply w13 LoRA to y (intermediate_cache1) in-place before activation.

        When `token_lora_mapping` is provided it overrides the punica_wrapper's
        global mapping — used by EP+LoRA to pass the per-rank-local mapping
        after all-to-all dispatch.

        Returns (sorted_token_ids_lora, expert_ids_lora,
                 num_tokens_post_padded_lora, token_lora_mapping)
        for reuse by add_lora_w2.
        """
        raise NotImplementedError

    def add_lora_w2(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        topk_weights: torch.Tensor,
        sorted_token_ids_lora: torch.Tensor | None,
        expert_ids_lora: torch.Tensor | None,
        num_tokens_post_padded_lora: torch.Tensor | None,
        token_lora_mapping: torch.Tensor | None,
        num_tokens: int,
        w1: torch.Tensor,
        w2: torch.Tensor,
        top_k_num: int,
        max_loras: int,
        adapter_enabled: torch.Tensor,
        top_k: int,
        fully_sharded: bool,
        tp_rank: int,
        use_tuned_config: bool,
        add_inputs: bool = True,
    ) -> None:
        """Apply w2 LoRA to y (intermediate_cache3) in-place before moe_sum.

        Reuses routing tensors returned by add_lora_w13.
        """
        raise NotImplementedError
