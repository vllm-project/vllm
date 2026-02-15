# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backend import AttentionMetadataBuilder, CommonAttentionMetadata
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig

PADDING_SLOT_ID = -1


class ExtractHiddenStatesProposer:
    def __init__(self, vllm_config: VllmConfig, device):
        assert vllm_config.speculative_config is not None

        assert vllm_config.speculative_config.num_speculative_tokens == 1
        if vllm_config.speculative_config.disable_padded_drafter_batch:
            raise ValueError(
                "disable_padded_drafter_batch is not supported with "
                "extract_hidden_states method"
            )
        self.vllm_config = vllm_config
        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        # Model and attention layer tracking (initialized in load_model)
        self.model: nn.Module | None = None
        self.attn_layer_names: list[str] = []
        self.attn_metadata_builder: AttentionMetadataBuilder | None = None

        # Maximum number of tokens for buffers
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens + max_batch_size
        )

        self.hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        layer_ids = getattr(self.hf_config, "eagle_aux_hidden_state_layer_ids", None)
        if not layer_ids:
            raise ValueError(
                "eagle_aux_hidden_state_layer_ids must be set in the draft "
                "model config for extract_hidden_states method"
            )
        self.num_hidden_states = len(layer_ids)
        self.hidden_size = vllm_config.model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.num_hidden_states, self.hidden_size),
            dtype=self.dtype,
            device=device,
        )
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        self._slot_mapping_buffer = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

    def propose(
        self,
        sampled_token_ids: torch.Tensor,
        target_hidden_states: list[torch.Tensor],
        common_attn_metadata: CommonAttentionMetadata,
        scheduler_output: SchedulerOutput,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> tuple[torch.Tensor, KVConnectorOutput | None]:
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
            Tuple of:
                - Draft tokens matching sampled tokens, shape [batch_size, 1]
                - KV connector output (if KV transfer is active), else None
        """
        assert self.model is not None and isinstance(target_hidden_states, list)

        # target_hidden_states is a list of tensors (one per layer)
        # Each tensor has shape [num_tokens, hidden_size]
        # Stack to shape: [num_tokens, num_hidden_states, hidden_size]
        stacked_hidden_states = torch.stack(target_hidden_states, dim=1)
        num_tokens = stacked_hidden_states.shape[0]

        # Copy hidden states to buffer
        self.hidden_states[:num_tokens] = stacked_hidden_states

        assert self.attn_metadata_builder is not None
        attn_metadata = self.attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0
        )

        # We assume all cache-only layers belong to the same KV cache group,
        # thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=num_tokens,
            num_tokens_padded=num_tokens,
        )

        cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens_dp_padded
        )
        num_input_tokens = batch_desc.num_tokens
        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        with (
            set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(
                    num_input_tokens, common_attn_metadata.slot_mapping
                ),
            ),
            (
                KVConnectorModelRunnerMixin._get_kv_connector_output(scheduler_output)
                if has_kv_transfer_group()
                else nullcontext()
            ) as kv_connector_output,
        ):
            self.model(
                hidden_states=self.hidden_states[:num_input_tokens],
            )

        # Return the sampled tokens as "draft" tokens
        # Shape: [batch_size, 1] to match num_speculative_tokens=1
        return sampled_token_ids.unsqueeze(-1), kv_connector_output

    def _get_slot_mapping(
        self,
        num_tokens: int,
        slot_mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return slot_mapping dict for cache-only attention layers.

        If slot_mapping is provided, copies it into the buffer first.
        """
        if slot_mapping is not None:
            num_actual = slot_mapping.shape[0]
            self._slot_mapping_buffer[:num_actual].copy_(slot_mapping)
            if num_tokens > num_actual:
                self._slot_mapping_buffer[num_actual:num_tokens].fill_(PADDING_SLOT_ID)

        view = self._slot_mapping_buffer[:num_tokens]
        return {name: view for name in self.attn_layer_names}

    def _pad_batch_across_dp(
        self,
        num_tokens_unpadded: int,
        num_tokens_padded: int,
    ) -> tuple[int, torch.Tensor | None]:
        should_ubatch, num_toks_across_dp, _ = coordinate_batch_across_dp(
            num_tokens_unpadded=num_tokens_unpadded,
            parallel_config=self.vllm_config.parallel_config,
            allow_microbatching=False,
            allow_dp_padding=self.cudagraph_dispatcher.cudagraph_mode
            != CUDAGraphMode.NONE,
            num_tokens_padded=num_tokens_padded,
            uniform_decode=None,
            num_scheduled_tokens_per_request=None,
        )
        assert not should_ubatch, (
            "DBO ubatching not implemented for extract_hidden_states"
        )

        num_tokens_dp_padded = num_tokens_padded
        if num_toks_across_dp is not None:
            num_tokens_dp_padded = int(num_toks_across_dp[self.dp_rank].item())
        return num_tokens_dp_padded, num_toks_across_dp

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Initialize cudagraph dispatcher keys.

        Only supports PIECEWISE cudagraphs (via mixed_mode).
        Should be called after adjust_cudagraph_sizes_for_spec_decode.
        """
        assert self.vllm_config.speculative_config is not None
        if (
            not self.vllm_config.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            proposer_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            proposer_cudagraph_mode = CUDAGraphMode.NONE

        self.cudagraph_dispatcher.initialize_cudagraph_keys(proposer_cudagraph_mode)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        assert self.model is not None, "Model must be initialized before dummy_run"
        num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
            num_tokens_unpadded=num_tokens,
            num_tokens_padded=num_tokens,
        )

        if use_cudagraphs:
            cudagraph_runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
                num_tokens_dp_padded
            )
            num_input_tokens = batch_desc.num_tokens
        else:
            cudagraph_runtime_mode = CUDAGraphMode.NONE
            num_input_tokens = num_tokens_dp_padded

        if num_tokens_across_dp is not None:
            num_tokens_across_dp[self.dp_rank] = num_input_tokens

        # Use our own slot mapping buffer during cudagraph capture.
        if (
            self.attn_layer_names
            and slot_mappings is not None
            and self.attn_layer_names[0] in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping_dict,
        ):
            self.model(
                hidden_states=self.hidden_states[:num_input_tokens],
            )

    def _build_attn_metadata_builder(
        self, draft_attn_layers: dict[str, AttentionLayerBase]
    ) -> AttentionMetadataBuilder:
        """Build the attention metadata builder from draft attention layers."""
        if not draft_attn_layers:
            raise ValueError("No attention layers found for ExtractHiddenStatesModel")
        layer = next(iter(draft_attn_layers.values()))
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
        Prepare next token IDs for speculative decoding.

        Since num_speculative_tokens == 1, sampled_token_ids has shape
        (batch_size, 1). For each request we either use the sampled token
        (if valid and not discarded) or a backup token from the request state.
        """
        num_reqs = gpu_input_batch.num_reqs
        device = sampled_token_ids.device

        # Compute backup tokens for discarded / invalid requests
        backup_tokens_gpu = torch.tensor(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(
                    common_attn_metadata.seq_lens_cpu[i].item()
                )
                for i in range(num_reqs)
            ],
            dtype=torch.int32,
            device=device,
        )

        assert discard_request_mask.dtype == torch.bool

        # With num_speculative_tokens == 1, there is exactly one token
        sampled = sampled_token_ids[:, 0]
        is_valid = (sampled >= 0) & (sampled < gpu_input_batch.vocab_size)
        valid_sampled_tokens_count = is_valid.to(torch.int32)

        use_sampled = is_valid & ~discard_request_mask[:num_reqs]
        next_token_ids = torch.where(
            use_sampled, sampled.to(torch.int32), backup_tokens_gpu
        )

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
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()  # type: ignore[type-abstract]
        )

        assert self.vllm_config.speculative_config is not None
        draft_model_config = self.vllm_config.speculative_config.draft_model_config
        from vllm.compilation.backends import set_model_tag

        with set_model_tag("extract_hidden_states"):
            self.model = get_model(
                vllm_config=self.vllm_config, model_config=draft_model_config
            )

        # Identify draft model's attention layers (difference from target)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        draft_attn_layers = {
            name: layer
            for name, layer in all_attn_layers.items()
            if name not in target_attn_layer_names
        }
        self.attn_layer_names = list(draft_attn_layers.keys())
        assert len(draft_attn_layers) == 1, (
            "ExtractHiddenStatesModel should have exactly one "
            f"attention layer, found {len(draft_attn_layers)}"
        )
        self.attn_metadata_builder = self._build_attn_metadata_builder(
            draft_attn_layers
        )

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """Validate all drafting layers belong to the same KV cache group.

        With exactly one attention layer (asserted in load_model), this is
        trivially satisfied.
        """
        assert len(self.attn_layer_names) == 1
