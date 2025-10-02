# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, replace
from typing import Any, Optional

import torch

from vllm.attention.layer import Attention
from vllm.config import ModelConfig, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata,
                                              extend_all_queries_by_1,
                                              extend_flat_seqs)
from vllm.v1.outputs import SamplerOutput
from vllm.v1.spec_decode.eagle import (PADDING_SLOT_ID, SpecDecodeBaseProposer,
                                       num_rejected_tokens)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


class DraftModelProposer(SpecDecodeBaseProposer):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config=vllm_config,
                         device=device,
                         pass_hidden_states_to_model=False,
                         pass_cudagraph_args_to_forward_ctx=True,
                         runner=runner)
        self._raise_if_multimodal()
        self._raise_if_mrope()
        self._raise_if_disabled_padded_drafter_batch()

    def update_propose_kwargs(
            self, propose_kwargs: dict, sampler_output: SamplerOutput,
            spec_decode_metadata: Optional[SpecDecodeMetadata]):
        return update_propose_kwargs(arange=self.arange,
                                     propose_kwargs=propose_kwargs,
                                     sampler_output=sampler_output,
                                     spec_decode_metadata=spec_decode_metadata,
                                     block_size=self.block_size,
                                     max_model_len=self.max_model_len)

    def _raise_if_multimodal(self):
        if self.supports_mm_inputs:
            raise NotImplementedError("Speculative Decoding with draft models "
                                      "does not support multimodal models yet")

    def _raise_if_mrope(self):
        if self.draft_model_config.uses_mrope:
            raise NotImplementedError("Speculative Decoding with draft models "
                                      "does not support M-RoPE yet")

    def _raise_if_disabled_padded_drafter_batch(self):
        if self.vllm_config.speculative_config.disable_padded_drafter_batch:
            raise NotImplementedError(
                "Speculative Decoding with draft models does not support "
                "disabled padded drafter batch yet")

    def _model_kwargs(self, num_tokens: int) -> dict[str, Any]:
        self._raise_if_multimodal()
        self._raise_if_mrope()
        return {
            "input_ids": self.input_ids[:num_tokens],
            "positions": self.positions[:num_tokens],
        }

    def dummy_run(self, num_tokens: int, forward_ctx_kwargs: dict):
        model_kwargs = self._model_kwargs(num_tokens)
        assert isinstance(self.model, torch.nn.Module)
        with set_forward_context(
                vllm_config=self.vllm_config,
                num_tokens=num_tokens,
                **forward_ctx_kwargs,
        ):
            self.model(**model_kwargs)

    def set_input_ids_first_pass(self, target_token_ids: torch.Tensor,
                                 next_token_ids: torch.Tensor, num_tokens: int,
                                 last_token_indices: torch.Tensor) -> None:
        self.input_ids[:num_tokens] = target_token_ids

    def load_model(self, target_model: Any) -> None:
        """Takes target_model to satisfy the type checker."""
        draft_model_config: ModelConfig = (
            self.vllm_config.speculative_config.draft_model_config)
        vllm_config_draft: VllmConfig = replace(
            self.vllm_config, model_config=draft_model_config)

        # This must be computed before loading the draft model
        # because that mutates the forward_context of the vllm_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())

        from vllm.compilation.backends import set_model_tag

        with set_model_tag("draft_model"):
            self.model = get_model(
                vllm_config=vllm_config_draft,
                model_config=draft_model_config,
                prefix="draft_model",
            )

        # This must be computed after loading the draft model
        # because that mutates the forward_context of the vllm_config
        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, Attention).keys() -
            target_attn_layer_names)
        self.attn_layer_names = list(draft_attn_layer_names)


logger = init_logger(__name__)


def update_propose_kwargs(arange: torch.Tensor, propose_kwargs: dict,
                          sampler_output: SamplerOutput,
                          spec_decode_metadata: Optional[SpecDecodeMetadata],
                          block_size: int, max_model_len: int) -> dict:
    """
    - Trims unnecessary tokens from the input, like those rejected by 
    the sampler, or those already processed by the draft model.
    - Merges the next_token_ids with the existing token ids into 
    a flat sequence.
    """
    cad: CommonAttentionMetadata = propose_kwargs["common_attn_metadata"]
    inputs = DraftModelInputs(cad=cad,
                              token_ids=propose_kwargs["target_token_ids"],
                              positions=propose_kwargs["target_positions"])
    inputs = trim_accepted_and_rejected_tokens(
        inputs=inputs,
        sampler_output=sampler_output,
        spec_decode_metadata=spec_decode_metadata)
    inputs = merge_next_token_ids_into_token_ids(
        inputs=inputs,
        next_token_ids=propose_kwargs["next_token_ids"],
        block_size=block_size,
        max_model_len=max_model_len,
        arange=arange)
    new_propose_kwargs = dict(
        target_token_ids=inputs.token_ids,
        target_positions=inputs.positions,
        next_token_ids=None,
        last_token_indices=None,
        common_attn_metadata=inputs.cad,
    )
    return propose_kwargs | new_propose_kwargs


@dataclass
class DraftModelInputs:
    token_ids: torch.Tensor
    positions: torch.Tensor
    cad: CommonAttentionMetadata


def trim_accepted_and_rejected_tokens(
        inputs: DraftModelInputs, sampler_output: SamplerOutput,
        spec_decode_metadata: Optional[SpecDecodeMetadata]
) -> DraftModelInputs:
    """
    Removes from the input.token_ids any tokens that have already been processed
    by the draft model, as well as tokens rejected by the sampler.
    Adjusts the positions accordingly, the slot mapping, 
    and the common_attn_metadata.
    """
    cad: CommonAttentionMetadata = inputs.cad

    # Compute the new token ids and positions
    n_accepted_tokens = sampler_output.n_sampled_tokens() - 1
    n_rejected_tokens = num_rejected_tokens(spec_decode_metadata,
                                            sampler_output.n_sampled_tokens())
    from_loc = cad.query_start_loc[:-1] + n_accepted_tokens
    to_loc = cad.query_start_loc[1:] - 1 - n_rejected_tokens
    idxs = compute_subrange_indices(from_loc, to_loc)
    new_token_ids = inputs.token_ids[idxs]
    new_positions = inputs.positions[idxs]

    # The new slot mapping is a subset of the previous one,
    # so no recomputation is needed.
    new_slot_mapping = cad.slot_mapping[idxs]

    # Update common_attn_metadata
    new_query_lens = to_loc - from_loc + 1
    new_query_start_loc = torch.zeros_like(cad.query_start_loc)
    new_query_start_loc[1:] = new_query_lens.cumsum(0)

    new_cad: CommonAttentionMetadata = cad.replace(
        query_start_loc=new_query_start_loc,
        query_start_loc_cpu=new_query_start_loc.to("cpu"),
        num_actual_tokens=new_token_ids.shape[0],
        max_query_len=new_query_lens.max().item(),
        slot_mapping=new_slot_mapping,
    )
    return DraftModelInputs(token_ids=new_token_ids,
                            positions=new_positions,
                            cad=new_cad)


def compute_subrange_indices(from_locs: torch.Tensor, to_locs: torch.Tensor):
    # Compute lengths of each subrange
    lengths = to_locs - from_locs + 1
    # Build an index for each subrange
    # torch.arange(max_len) creates [0, 1, ..., max_len-1]
    # broadcasting + masking ensures we only keep valid positions
    max_len = lengths.max()
    offsets = torch.arange(max_len, device=from_locs.device).unsqueeze(
        0)  # shape [1, max_len]
    mask = offsets < lengths.unsqueeze(1)  # shape [n, max_len]
    # Build all indices
    all_indices = from_locs.unsqueeze(1) + offsets
    all_indices = all_indices[mask]  # flatten valid indices only
    return all_indices


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
    and updates the common_attn_metadata.
    """
    cad: CommonAttentionMetadata = inputs.cad

    # merge token_ids and next_token_ids
    query_end_locs = cad.query_start_loc[1:] - 1
    new_token_ids = extend_flat_seqs(seqs=inputs.token_ids,
                                     end_locs=query_end_locs,
                                     new_vals=next_token_ids)
    # append new positions
    positions_to_append = inputs.positions[query_end_locs] + 1
    new_positions = extend_flat_seqs(seqs=inputs.positions,
                                     end_locs=query_end_locs,
                                     new_vals=positions_to_append)

    # recompute slot mapping
    batch_size, n_blocks_per_req = cad.block_table_tensor.shape
    req_indices = torch.arange(batch_size, device=cad.query_start_loc.device)
    req_indices = torch.repeat_interleave(req_indices, cad.query_lens() + 1)
    block_table_indices = (req_indices * n_blocks_per_req +
                           new_positions // block_size)
    block_nums = cad.block_table_tensor.view(-1)[block_table_indices]
    block_offsets = new_positions % block_size
    new_slot_mapping = block_nums * block_size + block_offsets
    # Mask out the position ids that exceed the max model length.
    exceeds_max_model_len = new_positions >= max_model_len
    new_slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)

    # update common_attn_metadata
    new_cad: CommonAttentionMetadata = extend_all_queries_by_1(
        cad, arange=arange, new_slot_mapping=new_slot_mapping)
    return DraftModelInputs(token_ids=new_token_ids,
                            positions=new_positions,
                            cad=new_cad)
