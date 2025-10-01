# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from typing import Any

import torch

from vllm.attention.layer import Attention
from vllm.config import ModelConfig, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata,
                                              extend_all_queries_by_1,
                                              extend_flat_seqs)
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer


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

    def update_propose_kwargs(self, propose_kwargs: dict):
        return update_propose_kwargs(arange=self.arange,
                                     propose_kwargs=propose_kwargs)

    def _raise_if_multimodal(self):
        if self.supports_mm_inputs:
            raise NotImplementedError("Speculative Decoding with draft models "
                                      "does not support multimodal models yet")

    def _raise_if_mrope(self):
        if self.draft_model_config.uses_mrope:
            raise NotImplementedError("Speculative Decoding with draft models "
                                      "does not support M-RoPE yet")

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


def update_propose_kwargs(arange: torch.Tensor, propose_kwargs: dict):
    """
    This function:
    - Merges the target_token_ids and the next_token_ids into a 
    single flat tensor.
    - Appends new positions for these next_token_ids.
    - Updates the common_attn_metadata to reflect that all query lengths are +1.
    """
    cad: CommonAttentionMetadata = propose_kwargs["common_attn_metadata"]
    target_token_ids = propose_kwargs["target_token_ids"]
    next_token_ids = propose_kwargs["next_token_ids"]
    target_positions = propose_kwargs["target_positions"]
    token_indices_to_sample = propose_kwargs["last_token_indices"]
    if token_indices_to_sample is None:
        token_indices_to_sample = cad.query_start_loc[1:] - 1

    # merge target_token_ids and next_token_ids
    end_locs = token_indices_to_sample
    new_target_token_ids = extend_flat_seqs(seqs=target_token_ids,
                                            end_locs=end_locs,
                                            new_vals=next_token_ids)
    # append new positions
    positions_to_append = target_positions[token_indices_to_sample] + 1
    new_target_positions = extend_flat_seqs(seqs=target_positions,
                                            end_locs=end_locs,
                                            new_vals=positions_to_append)

    # update common_attn_metadata
    new_cad: CommonAttentionMetadata = extend_all_queries_by_1(
        cad, arange=arange, last_token_indices=token_indices_to_sample)

    # new token indices to sample incease by [+1, +2, +3, ..., +batch_size]
    new_token_indices_to_sample = token_indices_to_sample \
        + arange_like(token_indices_to_sample) + 1

    logger.info("old last_token_indices: %s, new last_token_indices: %s.",
                token_indices_to_sample, new_token_indices_to_sample)

    new_propose_kwargs = dict(
        target_token_ids=new_target_token_ids,
        target_positions=new_target_positions,
        next_token_ids=None,
        last_token_indices=new_token_indices_to_sample,
        common_attn_metadata=new_cad,
    )
    return propose_kwargs | new_propose_kwargs


def arange_like(x: torch.Tensor) -> torch.Tensor:
    return torch.arange(x.shape[0], device=x.device)
