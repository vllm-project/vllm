# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from typing import Any

import torch

from vllm.attention.layer import Attention
from vllm.config import ModelConfig, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


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
            pass_cudagraph_args_to_forward_ctx=True,
            # The draft model runs one forward pass to prefill
            # the target_token_ids, and another forward pass for decoding
            # based on the next_token_ids. I.e. it needs 1 more forward pass.
            one_extra_forward_pass=False,
            # the first draft_token_ids are replaced by next_token_ids, so
            # they don't need to be returned as proposed tokens
            drop_first_drafted_tokens=False,
            runner=runner)
        self._raise_if_multimodal()
        self._raise_if_mrope()

    def prepare_inputs_padded(self,
                                common_attn_metadata: CommonAttentionMetadata,
                                spec_decode_metadata: SpecDecodeMetadata,
                                valid_sampled_tokens_count: torch.Tensor) -> \
                    tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        tup = super().prepare_inputs_padded(common_attn_metadata,
                                            spec_decode_metadata,
                                            valid_sampled_tokens_count)
        common_attn_metadata, token_indices, token_indices_to_sample = tup
        cad = common_attn_metadata
        batch_size = common_attn_metadata.batch_size()

        # token_indices is [0, ..., N], extend by batch_size
        new_token_indices = self.arange[:len(token_indices) + batch_size]
        # token indices to sample must be increased
        # by [+1, +2, ..., +batch_size]
        new_token_indices_to_sample = token_indices_to_sample + self.arange[
            1:batch_size + 1]

        # query start loc mus be increased by [+0, +1, +2, ..., +batch_size]
        new_query_start_loc = cad.query_start_loc + self.arange[:len(
            cad.query_start_loc)]
        # seq lens must be increased by [+1, +1, ..., +1] size batch_size
        new_seq_lens = cad.seq_lens + torch.ones_like(cad.seq_lens)
        # num requests stays unchanged
        new_num_reqs = cad.num_reqs
        # num computed tokens are increased by [+1, +1, ..., +1] size batch_size
        new_num_computed_tokens_cpu = cad.num_computed_tokens_cpu \
            + torch.ones_like(cad.num_computed_tokens_cpu)
        # num actual tokens increases by batch_size
        new_num_actual_tokens = cad.num_actual_tokens + batch_size
        # max query len and max seq len increases by 1
        new_max_query_len = cad.max_query_len + 1
        new_max_seq_len = cad.max_seq_len + 1
        # block table tensor depends on num_requests, which doesn't change
        new_block_table_tensor = cad.block_table_tensor
        # slot mapping depends on num_scheduled_tokens,
        # which increased by batch_size
        assert len(self.runner.input_batch.block_table.block_tables) == 1
        kv_cache_group_id = 0
        new_slot_mapping = self.runner.input_batch.block_table[
            kv_cache_group_id].slot_mapping.gpu[:new_num_actual_tokens]

        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            query_start_loc_cpu=new_query_start_loc.to("cpu"),
            seq_lens=new_seq_lens,
            seq_lens_cpu=new_seq_lens.to("cpu"),
            num_reqs=new_num_reqs,
            num_computed_tokens_cpu=new_num_computed_tokens_cpu,
            num_actual_tokens=new_num_actual_tokens,
            max_query_len=new_max_query_len,
            max_seq_len=new_max_seq_len,
            block_table_tensor=new_block_table_tensor,
            slot_mapping=new_slot_mapping,
        )
        return new_cad, new_token_indices, new_token_indices_to_sample

    def _raise_if_multimodal(self):
        if self.is_multimodal_model:
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
        start_locs = torch.zeros(last_token_indices.shape[0] + 1,
                                 device=last_token_indices.device,
                                 dtype=torch.int32)
        start_locs[1:] = last_token_indices + 1
        input_ids, _ = append_new_toks(toks=target_token_ids,
                                       start_locs=start_locs,
                                       new_toks=next_token_ids)
        num_tokens = input_ids.shape[0]
        self.input_ids[:num_tokens] = input_ids

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


def append_new_toks(
        toks: torch.Tensor, start_locs: torch.Tensor,
        new_toks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    long_len = toks.shape[0] + new_toks.shape[0]
    long_toks = torch.zeros(long_len, device=toks.device, dtype=toks.dtype)

    # compute indices for previous toks
    toks_idxs = torch.ones_like(toks)
    toks_idxs[start_locs[1:-1]] += 1
    toks_idxs = toks_idxs.cumsum(0) - 1

    # compute indices for new toks
    new_toks_idxs = start_locs[1:] + torch.arange(new_toks.shape[0],
                                                  device=toks.device)

    # assign toks and new toks
    long_toks[toks_idxs] = toks
    long_toks[new_toks_idxs] = new_toks

    # compute new start locs
    new_start_locs = torch.zeros_like(start_locs)
    new_start_locs[1:] = new_toks_idxs + 1

    return long_toks, new_start_locs
