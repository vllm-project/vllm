# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionMetadataBuilder, CommonAttentionMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin

PADDING_SLOT_ID = -1


class ExtractHiddenStatesProposer:
    def __init__(self, vllm_config: VllmConfig, device):
        assert vllm_config.speculative_config is not None

        assert vllm_config.speculative_config.num_speculative_tokens == 1
        self.vllm_config = vllm_config
        self.device = device
        self.dtype = vllm_config.model_config.dtype
        # Maximum length of the model.
        self.max_model_len = vllm_config.model_config.max_model_len

        # Model and attention layer tracking (initialized in load_model)
        self.model: nn.Module | None = None
        self.attn_layer_names: list[str] = []
        self.attn_metadata_builder: AttentionMetadataBuilder | None = None

        # Maximum number of tokens for buffers
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens + max_batch_size
        )

        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.num_hidden_states = len(
            getattr(
                self.config, "eagle_aux_hidden_state_layer_ids", [0, 0, 0]
            )  # fallback to 3
        )

        # Get hidden size from draft model config (will be available after load_model)
        # For now, we'll set it based on target model and update in load_model if needed
        self.hidden_size = vllm_config.model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size * self.num_hidden_states),
            dtype=self.dtype,
            device=device,
        )

        # Initialize buffers and attributes needed for slot mapping
        self.indexer_layer_names: list[str] = []
        self._slot_mapping_buffer = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

    def propose(
        self,
        sampled_token_ids: torch.Tensor,
        target_hidden_states: list[torch.Tensor],
        common_attn_metadata: CommonAttentionMetadata,
        scheduler_output: "SchedulerOutput",
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor:
        """Propose draft tokens by calling the ExtractHiddenStatesModel model.

        The ExtractHiddenStatesModel caches the hidden states in the KV cache
        without performing actual attention computation. This allows us to
        extract and store hidden states for later use (e.g., KV transfer).

        This proposer doesn't actually perform speculation - it returns the
        sampled tokens as "draft" tokens, ensuring they always verify (match).
        The main purpose is to cache hidden states, not to speculate.

        Args:
            sampled_token_ids: Sampled token IDs from the target model
            target_hidden_states: List of hidden state tensors from target model
                                (one per aux hidden state layer)
            common_attn_metadata: Attention metadata
            scheduler_output: Scheduler output for KV connector
            slot_mappings: Slot mappings for KV cache (unused, provided for
                          interface compatibility)

        Returns:
            Draft tokens that match the sampled tokens, shape [batch_size, 1]
        """
        # Call the ExtractHiddenStatesModel model to cache hidden states
        # This triggers the KV cache storage and KV connector API
        assert self.model is not None and isinstance(target_hidden_states, list)

        # target_hidden_states is a list of tensors (one per layer)
        # Stack them to create the input for ExtractHiddenStatesModel
        # Shape: [num_tokens, num_layers * hidden_size]
        stacked_hidden_states = torch.cat(target_hidden_states, dim=-1)
        num_tokens = stacked_hidden_states.shape[0]

        # Copy hidden states to buffer
        self.hidden_states[:num_tokens] = stacked_hidden_states

        # Build attention metadata for drafting
        if self.attn_metadata_builder is None:
            self.attn_metadata_builder = self._get_attention_metadata_builder()

        attn_metadata = self.attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0
        )

        # Build per-layer attention metadata
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        # Call model with proper forward context
        with (
            set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                slot_mapping=self._get_slot_mapping(
                    num_tokens, common_attn_metadata.slot_mapping
                ),
            ),
            (
                KVConnectorModelRunnerMixin._get_kv_connector_output(scheduler_output)
                if has_kv_transfer_group()
                else nullcontext()
            ) as kv_connecter_output,
        ):
            # Forward pass: caches hidden states in KV cache
            # Output is ignored - we only care about the KV cache side effects
            self.model(
                hidden_states=self.hidden_states[:num_tokens],
            )

        # Return the sampled tokens as "draft" tokens
        # This ensures they will always verify (match) since they're identical
        # We're not actually doing speculation - just caching hidden states
        # Shape: [batch_size, 1] to match num_speculative_tokens=1
        return sampled_token_ids.unsqueeze(-1)

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return slot_mapping dict for EAGLE layers.

        If slot_mapping is provided, copies it into the buffer first.
        """
        if slot_mapping is not None:
            num_actual = slot_mapping.shape[0]
            self._slot_mapping_buffer[:num_actual].copy_(slot_mapping)
            if num_tokens > num_actual:
                self._slot_mapping_buffer[num_actual:num_tokens].fill_(PADDING_SLOT_ID)

        view = self._slot_mapping_buffer[:num_tokens]
        return {name: view for name in self.attn_layer_names + self.indexer_layer_names}

    def _get_attention_metadata_builder(self) -> AttentionMetadataBuilder:
        """Get the attention metadata builder for the draft model."""
        # Get the first attention layer to determine the backend
        if not self.attn_layer_names:
            raise ValueError("No attention layers found for ExtractHiddenStatesModel")

        # Get the attention layer from the model
        # Layer names include the prefix (e.g., "drafter.cache_only_layers.32")
        # We need to skip the prefix part since self.model is already the drafter
        layer_name = self.attn_layer_names[0]
        parts = layer_name.split(".")
        # Skip the first part if it's the prefix (e.g., "drafter")
        if parts[0] == "drafter":
            parts = parts[1:]

        layer = self.model
        for part in parts:
            layer = getattr(layer, part)

        # Get the attention backend and its metadata builder
        attn_backend = layer.get_attn_backend()
        return attn_backend.get_builder_cls()(
            layer.get_kv_cache_spec(self.vllm_config),
            self.attn_layer_names,
            self.vllm_config,
            self.device,
        )

    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead. This is denoted
        the "backup" token id. It also counts rejected tokens via `sampled_token_ids`.
        """

        batch_size, num_tokens = sampled_token_ids.shape
        device = sampled_token_ids.device
        num_reqs = gpu_input_batch.num_reqs

        # 1. Compute backup tokens for discarded requests
        backup_tokens = []
        for i in range(num_reqs):
            req_id = gpu_input_batch.req_ids[i]
            seq_len = common_attn_metadata.seq_lens_cpu[i].item()
            token_id = requests[req_id].get_token_id(seq_len)
            backup_tokens.append(token_id)

        backup_tokens_gpu = torch.tensor(
            backup_tokens, dtype=torch.int32, device=device
        )

        # 2. Extract next token IDs
        # For valid requests: find the LAST VALID token (skipping rejected -1 tokens)
        # For discarded requests: use backup tokens

        # Create mask for valid tokens (>= 0 and < vocab_size)
        vocab_size = gpu_input_batch.vocab_size
        is_valid = (sampled_token_ids >= 0) & (sampled_token_ids < vocab_size)

        # Count valid tokens per request
        valid_sampled_tokens_count = is_valid.sum(dim=1).to(torch.int32)

        # Find the last valid token for each request
        next_token_ids = torch.empty(batch_size, dtype=torch.int32, device=device)
        for i in range(batch_size):
            if discard_request_mask[i]:
                # Use backup token for discarded requests
                next_token_ids[i] = backup_tokens_gpu[i]
            elif valid_sampled_tokens_count[i] > 0:
                # Find last valid token index
                valid_indices = torch.where(is_valid[i])[0]
                last_valid_idx = valid_indices[-1]
                next_token_ids[i] = sampled_token_ids[i, last_valid_idx].to(torch.int32)
            else:
                # No valid tokens, use backup
                next_token_ids[i] = backup_tokens_gpu[i]

        return next_token_ids, valid_sampled_tokens_count

    def load_model(self, target_model: nn.Module) -> None:
        """Load the ExtractHiddenStatesModel model.

        This method instantiates the ExtractHiddenStatesModel model which is used
        to cache hidden states during speculative decoding. The model uses
        cache-only attention (no computation, just caching KV states).

        Args:
            target_model: The target model (passed for compatibility with
                         EagleProposer interface, but not used here)
        """
        # Get the target model's attention layers before loading draft model
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
        )

        # Instantiate ExtractHiddenStatesModel directly
        # This model only needs to know KV cache shapes from the target model config
        from vllm.model_executor.models.extract_hidden_states import (
            ExtractHiddenStatesModel,
        )

        self.drafter = ExtractHiddenStatesModel(
            vllm_config=self.vllm_config, prefix="drafter"
        )
        self.model = self.drafter

        # Identify draft model's attention layers (difference from target)
        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
            - target_attn_layer_names
        )
        self.attn_layer_names = list(draft_attn_layer_names)
