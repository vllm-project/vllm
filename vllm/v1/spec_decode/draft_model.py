# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import torch

from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.speculative import SpeculativeConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    extend_all_queries_by_1,
    extend_flat_seqs,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import PADDING_SLOT_ID, SpecDecodeBaseProposer

logger = init_logger(__name__)


class DraftModelProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        self._raise_if_multimodal()
        self._raise_if_mrope()
        self._raise_if_padded_drafter_batch()
        self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens] or [3, num_tokens] when M-RoPE is enabled
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        This function processes the inputs first before calling the .propose()
        method of the parent class.
        """
        inputs = DraftModelInputs(
            cad=common_attn_metadata,
            token_ids=target_token_ids,
            positions=target_positions,
        )
        inputs = merge_next_token_ids_into_token_ids(
            inputs=inputs,
            next_token_ids=next_token_ids,
            block_size=self.block_size,
            max_model_len=self.max_model_len,
            arange=self.arange,
        )

        draft_token_ids = super().propose(
            target_token_ids=inputs.token_ids,
            target_positions=inputs.positions,
            common_attn_metadata=inputs.cad,
            sampling_metadata=sampling_metadata,
            # below are are not used by draft model
            target_hidden_states=None,
            next_token_ids=None,
            last_token_indices=None,
            mm_embed_inputs=None,
        )
        return draft_token_ids

    def _raise_if_multimodal(self):
        if self.supports_mm_inputs:
            raise NotImplementedError(
                "Speculative Decoding with draft models "
                "does not support multimodal models yet"
            )

    def _raise_if_mrope(self):
        if self.draft_model_config.uses_mrope:
            raise NotImplementedError(
                "Speculative Decoding with draft models does not support M-RoPE yet"
            )

    def _raise_if_padded_drafter_batch(self):
        if not self.vllm_config.speculative_config.disable_padded_drafter_batch:
            raise NotImplementedError(
                "Speculative Decoding with draft models does not support "
                "padded drafter batch yet. Please pass --disable-padded-drafter-batch "
                "in the speculative_config."
            )

    def _raise_if_vocab_size_mismatch(self):
        self.vllm_config.speculative_config.verify_equal_vocab_size_if_draft_model()

    def _raise_if_draft_tp_mismatch(self):
        # Note(Tomas Ruiz) If we run the target model with TP > 1 and
        # the draft model with TP = 1, then the different TP ranks collide.
        # Specifically when all ranks compile the draft model on rank 0
        # (because TP=1), then the torch compile cache is overwritten and corrupted.
        # We need a mechanism like this: https://github.com/vllm-project/vllm/pull/5414
        # To prevent this error, we assert that both TP sizes must be the same.
        spec_cfg: SpeculativeConfig = self.vllm_config.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size
        if draft_tp != tgt_tp:
            raise ValueError(
                f"Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' "
                f"must be the same. Got {draft_tp} and {tgt_tp}. "
                "Please pass 'draft_tensor_parallel_size' in the speculative_config."
            )

    def set_input_ids_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        num_tokens: int,
        last_token_indices: torch.Tensor,
    ) -> None:
        self.input_ids[:num_tokens] = target_token_ids

    def load_model(self, target_model: Any) -> None:
        """Takes target_model to satisfy the type checker."""

        # This must be computed before loading the draft model
        # because that mutates the forward_context of the vllm_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys()
        )

        from vllm.compilation.backends import set_model_tag

        draft_vllm_config: VllmConfig = create_vllm_config_for_draft_model(
            target_model_vllm_config=self.vllm_config
        )
        logger.info(
            "Starting to load draft model %s. TP=%d, rank=%d",
            draft_vllm_config.model_config.model,
            draft_vllm_config.parallel_config.tensor_parallel_size,
            draft_vllm_config.parallel_config.rank,
        )
        with set_model_tag("draft_model"):
            self.model = get_model(vllm_config=draft_vllm_config, prefix="draft_model")

        # This must be computed after loading the draft model
        # because that mutates the forward_context of the vllm_config
        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, Attention).keys()
            - target_attn_layer_names
        )
        self.attn_layer_names = list(draft_attn_layer_names)


def create_vllm_config_for_draft_model(
    target_model_vllm_config: VllmConfig,
) -> VllmConfig:
    """The vllm_config is configured for the target model, e.g.
    its quant_config and parallel_config. But the draft model is potentially
    quantized differently, and has potentially different tensor_parallel_size.
    This function creates a new vllm_config configured for the draft model.
    The vllm_config is useful when loading the draft model with get_model().
    """
    old = target_model_vllm_config
    new_parallel_config = old.speculative_config.draft_parallel_config.replace(
        rank=old.parallel_config.rank
    )
    new: VllmConfig = old.replace(
        quant_config=None,  # quant_config is recomputed in __init__()
        model_config=old.speculative_config.draft_model_config,
        parallel_config=new_parallel_config,
    )
    return new


@dataclass
class DraftModelInputs:
    token_ids: torch.Tensor
    positions: torch.Tensor
    cad: CommonAttentionMetadata


def merge_next_token_ids_into_token_ids(
    inputs: DraftModelInputs,
    next_token_ids: torch.Tensor,
    block_size: int,
    max_model_len: int,
    arange: torch.Tensor,
) -> DraftModelInputs:
    """
    Merges the next token ids with the existing token ids into a flat sequence.
    Does the same for the positions, computes new slot mapping,
    and updates the common_attn_metadata. The inputs are not modified in-place.
    """
    cad: CommonAttentionMetadata = inputs.cad

    # merge token_ids and next_token_ids
    query_end_locs = cad.query_start_loc[1:] - 1
    new_token_ids = extend_flat_seqs(
        seqs=inputs.token_ids, end_locs=query_end_locs, new_vals=next_token_ids
    )
    # append new positions
    positions_to_append = inputs.positions[query_end_locs] + 1
    new_positions = extend_flat_seqs(
        seqs=inputs.positions, end_locs=query_end_locs, new_vals=positions_to_append
    )

    # recompute slot mapping
    batch_size, n_blocks_per_req = cad.block_table_tensor.shape
    req_indices = torch.arange(batch_size, device=cad.query_start_loc.device)
    req_indices = torch.repeat_interleave(req_indices, cad.query_lens() + 1)
    block_table_indices = req_indices * n_blocks_per_req + new_positions // block_size
    block_nums = cad.block_table_tensor.view(-1)[block_table_indices]
    block_offsets = new_positions % block_size
    new_slot_mapping = block_nums * block_size + block_offsets
    # Mask out the position ids that exceed the max model length.
    exceeds_max_model_len = new_positions >= max_model_len
    new_slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)

    # update common_attn_metadata
    new_cad: CommonAttentionMetadata = extend_all_queries_by_1(
        cad, arange=arange, new_slot_mapping=new_slot_mapping
    )
    return DraftModelInputs(
        token_ids=new_token_ids, positions=new_positions, cad=new_cad
    )
