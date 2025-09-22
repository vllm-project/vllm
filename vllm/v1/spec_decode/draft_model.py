# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from typing import Any, Optional

import torch

from vllm.attention.layer import Attention
from vllm.config import ModelConfig, VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import PADDING_SLOT_ID, SpecDecodeBaseProposer
from vllm.v1.worker.ubatching import dbo_current_ubatch_id


class DraftModelProposer(SpecDecodeBaseProposer):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner)
        self._raise_if_multimodal()
        self._raise_if_mrope()

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
        with set_forward_context(
                vllm_config=self.vllm_config,
                num_tokens=num_tokens,
                **forward_ctx_kwargs,
        ):
            self.model(**model_kwargs)

    # Copied and adapted from eagle.py
    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        last_token_indices: Optional[torch.Tensor],
        common_attn_metadata: CommonAttentionMetadata,
        cudagraph_runtime_mode: CUDAGraphMode,
        batch_descriptor: BatchDescriptor,
    ) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        self.input_ids[:num_tokens] = target_token_ids

        assert self.runner is not None

        # FIXME: need to consider multiple kv_cache_groups
        assert len(self.runner.attn_groups) == 1
        assert len(self.runner.attn_groups[0]) == 1
        ubatch_id = dbo_current_ubatch_id()
        attn_metadata_builder = self.runner.attn_groups[0][
            0].metadata_builders[ubatch_id]
        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0)

        # At this moment, we assume all draft model layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        if self.use_cuda_graph and num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        else:
            num_input_tokens = num_tokens
        # copy inputs to buffer for cudagraph
        self.positions[:num_tokens] = target_positions

        model_kwargs = self._model_kwargs(num_input_tokens)
        with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
        ):
            last_hidden_states = self.model(**model_kwargs)

        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        positions = target_positions[last_token_indices]

        if isinstance(attn_metadata, TreeAttentionMetadata):
            raise NotImplementedError("Speculative Decoding with draft models "
                                      "does not support tree attention yet")

        # Reuse the next_token_ids to avoid a potential rejection
        draft_token_ids = next_token_ids

        # The draft model runs one forward pass to prefill
        # the target_token_ids, and another forward pass for decoding
        # based on the next_token_ids. I.e. it needs 1 more forward pass.
        n_forward_passes = self.num_speculative_tokens + 1
        # Early exit if there is only one draft token to be generated.
        if n_forward_passes == 1:
            # [batch_size, 1]
            return draft_token_ids.view(-1, 1)

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        if self.use_cuda_graph and batch_size <= self.cudagraph_batch_sizes[-1]:
            input_batch_size = self.vllm_config.pad_for_cudagraph(batch_size)
        else:
            input_batch_size = batch_size

        attn_metadata.num_actual_tokens = batch_size
        attn_metadata.max_query_len = 1
        attn_metadata.query_start_loc = self.arange[:batch_size + 1]
        for _ in range(n_forward_passes - 1):
            # Update the inputs.
            # cast to int32 is crucial when draft model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()
            positions += 1

            # NOTE(woosuk): We should handle the case where the draft model
            # generates tokens beyond the max model length. Since it is complex
            # to remove such requests from the batch, we keep them in the batch
            # but adjust the position ids and slot mappings to avoid the
            # out-of-range access during the model execution. The draft tokens
            # generated with this adjustment should be ignored.
            exceeds_max_model_len = positions >= self.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            clamped_positions = torch.where(exceeds_max_model_len, 0,
                                            positions)

            # Increment the sequence lengths.
            attn_metadata.max_seq_len += 1
            attn_metadata.seq_lens += 1
            # Consider max model length.
            attn_metadata.max_seq_len = min(attn_metadata.max_seq_len,
                                            self.max_model_len)
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Compute the slot mapping.
            block_numbers = clamped_positions // self.block_size
            block_ids = attn_metadata.block_table.gather(
                dim=1, index=block_numbers.view(-1, 1))
            block_ids = block_ids.view(-1)
            attn_metadata.slot_mapping = (block_ids * self.block_size +
                                          clamped_positions % self.block_size)
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            attn_metadata.slot_mapping.masked_fill_(exceeds_max_model_len,
                                                    PADDING_SLOT_ID)

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self.positions[:batch_size] = clamped_positions

            model_kwargs = self._model_kwargs(input_batch_size)
            batch_descriptor = BatchDescriptor(num_tokens=input_batch_size,
                                               uniform_decode=True)
            cudagraph_runtime_mode, batch_descriptor = (
                self.runner.cudagraph_dispatcher.dispatch(batch_descriptor))

            # Run the model.
            with set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=input_batch_size,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
            ):
                last_hidden_states = self.model(**model_kwargs)

            logits = self.model.compute_logits(last_hidden_states[:batch_size])
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # the first draft_token_ids are identical to next_token_ids, so
        # they don't need to be returned as proposed tokens
        draft_token_ids_list = draft_token_ids_list[1:]

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def load_model(self) -> None:
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
