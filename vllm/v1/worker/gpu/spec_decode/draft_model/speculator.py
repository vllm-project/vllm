# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.config.utils import replace
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
    init_attn_backend,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CapturedAttentionState,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState

logger = init_logger(__name__)


class DraftModelSpeculator:
    """Speculative decoding using a separate smaller draft LM.

    Unlike Eagle, the draft model is fully independent: it does not consume
    the target model's hidden states.  The proposal loop is:
      - Step 0 (prefill): run the draft model on the same token ids / positions
        as the target model to populate the draft KV cache and produce the
        first draft token from each request's last position.
      - Steps 1..k-1 (decode): feed the previous draft token back, advance
        positions, and sample the next draft token.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.draft_model_config = self.speculative_config.draft_model_config
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens

        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dtype = vllm_config.model_config.dtype

        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )
        # [max_num_reqs, num_speculative_steps]
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        # Index of the last token in input_buffers for each request.
        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        # CPU arange used for query_start_loc in 1-token-per-request decode.
        self.arange = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device="cpu"
        )

        # Draft model does not handle multimodal inputs.
        self.supports_mm_inputs = False
        # Greedy sampling for draft tokens (no probabilistic draft logits).
        self.draft_logits: torch.Tensor | None = None

        # Set in load_model / set_attn.
        self.model: nn.Module | None = None
        self.draft_attn_layer_names: set[str] = set()
        self.attn_groups: list[list[Any]] = []
        self.kv_cache_config: KVCacheConfig | None = None
        self.block_tables: BlockTables | None = None
        self.model_state: ModelState | None = None

    # ------------------------------------------------------------------
    # Lifecycle hooks called by ModelRunner
    # ------------------------------------------------------------------

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # CUDA graph capture for draft model is not yet supported.
        pass

    def load_model(self, target_model: nn.Module) -> None:
        # Snapshot existing attention layer names BEFORE loading the draft model
        # so we can identify which new layers belong to the draft model.
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        from vllm.compilation.backends import set_model_tag

        # Build a draft-specific VllmConfig that:
        #  - uses draft_model_config (correct hidden_size, architecture)
        #  - shares the same compilation_config (shallow copy via replace), so
        #    draft attention layers are registered in the shared
        #    static_forward_context and appear in get_kv_cache_spec().
        spec = self.speculative_config
        draft_vllm_config = replace(
            self.vllm_config,
            model_config=self.draft_model_config,
            quant_config=None,
            parallel_config=replace(
                spec.draft_parallel_config,
                rank=self.vllm_config.parallel_config.rank,
            ),
        )

        with set_model_tag("draft_model"):
            self.model = get_model(
                vllm_config=draft_vllm_config,
                prefix="draft_model",
            )

        all_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )
        self.draft_attn_layer_names = all_attn_layer_names - target_attn_layer_names
        logger.info(
            "Draft model has %d attention layers: %s",
            len(self.draft_attn_layer_names),
            sorted(self.draft_attn_layer_names),
        )

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        _, self.attn_groups, _ = init_attn_backend(
            kv_cache_config,
            self.vllm_config,
            self.device,
            active_layer_names=self.draft_attn_layer_names,
        )
        self.block_tables = block_tables

    def capture(
        self,
        attn_states: dict[BatchExecutionDescriptor, CapturedAttentionState],
    ) -> None:
        # No CUDA graph capture for draft model speculator yet.
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            hidden_states = self.model(  # type: ignore[misc]
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
            )
        # Standard LMs return a plain tensor; handle tuple defensively.
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return hidden_states

    def _build_decode_attn_metadata(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any] | None:
        """Build attention metadata for decode steps (1 token per request)."""
        assert self.kv_cache_config is not None
        assert self.block_tables is not None

        if not self.draft_attn_layer_names:
            return None

        query_start_loc_cpu = torch.clamp(self.arange[: num_reqs + 1], max=num_reqs)
        block_tables = [x[:num_reqs] for x in self.block_tables.input_block_tables]
        slot_mappings = self.block_tables.slot_mappings[:, :num_tokens]
        return build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=self.input_buffers.query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=1,
            seq_lens=self.input_buffers.seq_lens[:num_reqs],
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )

    # ------------------------------------------------------------------
    # propose: main speculator entry point
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size] — not used by draft model
        last_hidden_states: torch.Tensor,
        # not used by draft model
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        assert self.model is not None

        num_tokens = input_batch.num_tokens_after_padding
        num_reqs = input_batch.num_reqs

        # Choose whether to skip attention (profile / dummy warmup).
        # When skip_attn=True, block_tables and kv_cache_config are never used.
        skip_attn = dummy_run and skip_attn_for_dummy_run

        if not skip_attn:
            assert self.block_tables is not None
            assert self.kv_cache_config is not None

        # ----------------------------------------------------------
        # Step 0 (prefill-equivalent)
        # ----------------------------------------------------------
        # Copy the target model's input_ids and positions verbatim —
        # no shift is needed because the draft model is a plain causal LM.
        src_tokens = input_batch.num_tokens
        self.input_buffers.input_ids[:src_tokens].copy_(
            input_batch.input_ids[:src_tokens]
        )
        self.input_buffers.positions[:src_tokens].copy_(
            input_batch.positions[:src_tokens]
        )

        # Compute the index of the last valid token for each request,
        # accounting for rejected speculative tokens from the previous step.
        query_start_loc = input_batch.query_start_loc  # GPU [num_reqs+1]
        query_lens = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
        adjusted_lens = query_lens - num_rejected
        self.last_token_indices[:num_reqs] = (
            query_start_loc[:num_reqs] + adjusted_lens - 1
        )

        attn_md_0 = None if skip_attn else attn_metadata
        slot_maps_0 = None if skip_attn else slot_mappings

        hidden_states = self.run_model(
            num_tokens, attn_md_0, slot_maps_0, num_tokens_across_dp
        )

        # Sample draft token[0] at the last position of each request.
        last_indices = self.last_token_indices[:num_reqs]
        logits = self.model.compute_logits(hidden_states[last_indices])
        self.draft_tokens[:num_reqs, 0] = logits.argmax(dim=-1)

        if self.num_speculative_steps == 1:
            return self.draft_tokens[:num_reqs, :1]

        # ----------------------------------------------------------
        # Prepare state for decode steps
        # ----------------------------------------------------------
        # Save the position of each request's last token; subsequent steps
        # will increment this by one each time.
        last_positions = self.input_buffers.positions[last_indices]
        self.input_buffers.positions[:num_reqs].copy_(last_positions)

        # Adjust seq_lens: remove rejected tokens and add the first draft token.
        target_seq_lens = input_batch.seq_lens[:num_reqs]  # GPU [num_reqs]
        adjusted_seq_lens = torch.clamp(
            target_seq_lens - num_rejected + 1, max=self.max_model_len
        )
        self.input_buffers.seq_lens[:num_reqs].copy_(adjusted_seq_lens)

        # Decode steps use exactly 1 token per request.
        self.input_buffers.query_start_loc[: num_reqs + 1].copy_(
            self.arange[: num_reqs + 1].to(self.device)
        )

        idx_mapping = input_batch.idx_mapping

        # ----------------------------------------------------------
        # Steps 1 .. num_speculative_steps - 1
        # ----------------------------------------------------------
        for step in range(1, self.num_speculative_steps):
            # Feed the previous draft token as input.
            self.input_buffers.input_ids[:num_reqs].copy_(
                self.draft_tokens[:num_reqs, step - 1].int()
            )
            # Advance positions (seq_lens is incremented AFTER the model run).
            torch.clamp(
                self.input_buffers.positions[:num_reqs] + 1,
                max=self.max_model_len - 1,
                out=self.input_buffers.positions[:num_reqs],
            )

            if skip_attn:
                hidden_states = self.run_model(
                    num_reqs, None, None, num_tokens_across_dp
                )
            else:
                q_start = self.input_buffers.query_start_loc[: num_reqs + 1]
                positions = self.input_buffers.positions[:num_reqs]
                step_slot_mappings = self.block_tables.compute_slot_mappings(
                    idx_mapping, q_start, positions, num_reqs
                )
                step_slot_maps_by_layer = build_slot_mappings_by_layer(
                    step_slot_mappings, self.kv_cache_config
                )
                decode_attn_md = self._build_decode_attn_metadata(num_reqs, num_reqs)
                hidden_states = self.run_model(
                    num_reqs,
                    decode_attn_md,
                    step_slot_maps_by_layer,
                    num_tokens_across_dp,
                )

            logits = self.model.compute_logits(hidden_states[:num_reqs])
            self.draft_tokens[:num_reqs, step] = logits.argmax(dim=-1)

            # Advance seq_lens after the model run so that the current step
            # sees the correct KV count (matching Eagle's update_eagle_draft_inputs).
            torch.clamp(
                self.input_buffers.seq_lens[:num_reqs] + 1,
                max=self.max_model_len,
                out=self.input_buffers.seq_lens[:num_reqs],
            )

        return self.draft_tokens[:num_reqs]
