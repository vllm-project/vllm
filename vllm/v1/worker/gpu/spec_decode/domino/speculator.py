# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.dflash.utils import (
    load_dflash_model,
)
from vllm.v1.worker.gpu.spec_decode.domino.cudagraph import DominoCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.domino.utils import (
    get_domino_causal,
    prepare_domino_inputs,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id

logger = init_logger(__name__)


class DominoSpeculator(DraftModelSpeculator):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # --- BEGIN: copied from DFlashSpeculator.__init__ ---
        super().__init__(vllm_config, device)

        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )

        # Multimodal inputs not currently supported.
        self.supports_mm_inputs = False

        # Each request emits exactly (bonus + N mask) query tokens per step.
        self.num_query_per_req = 1 + self.num_speculative_steps

        self.parallel_drafting_token_id = get_parallel_drafting_token_id(
            self.draft_model_config.hf_config
        )

        self.dflash_causal = get_domino_causal(self.draft_model_config)

        # Whether the anchor query position is itself a prediction.
        self.sample_from_anchor = False

        # Context positions for the K/V precompute.
        self.context_positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        # Per-mask-token sampling buffers.
        max_num_sampled_tokens = self.max_num_reqs * self.num_speculative_steps
        self.sample_indices = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_pos = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_idx_mapping = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int32, device=device
        )
        self.sample_col = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        ).repeat(self.max_num_reqs)

        self.query_cudagraph_manager: DominoCudaGraphManager | None = None
        self.draft_kv_cache_group_id: int = -1
        # --- END: copied from DFlashSpeculator.__init__ ---

        # --- Domino-specific ---
        self.is_domino = True

        eagle_cfg = getattr(self.draft_model_config.hf_config, "eagle_config", {})
        dflash_cfg = getattr(self.draft_model_config.hf_config, "dflash_config", {})
        gru_hidden_dim = eagle_cfg.get("gru_hidden_dim") or dflash_cfg.get(
            "gru_hidden_dim"
        )
        if gru_hidden_dim is None:
            raise ValueError(
                "gru_hidden_dim must be set in eagle_config or dflash_config "
                "for domino speculator"
            )
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_hidden_buffer = torch.zeros(
            self.max_num_reqs, gru_hidden_dim, dtype=self.dtype, device=device
        )
        self._domino_cat_buf = torch.zeros(
            self.max_num_reqs,
            1,
            self.hidden_size + gru_hidden_dim,
            dtype=self.dtype,
            device=device,
        )

    # --- BEGIN: methods copied from DFlashSpeculator ---

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        self.query_cudagraph_manager = DominoCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.num_query_per_req,
            causal=self.dflash_causal,
        )

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        super().set_attn(model_state, kv_cache_config, block_tables)

        self.draft_kv_cache_group_ids = [
            gid for gid, g in enumerate(self.attn_groups) if g
        ]
        assert self.draft_kv_cache_group_ids, "No draft attention groups found."
        self.draft_kv_cache_group_id = self.draft_kv_cache_group_ids[0]
        self.draft_block_size = self.block_tables.block_sizes[
            self.draft_kv_cache_group_id
        ]

        # Per-group context slot buffers for the precompute.
        self._context_slot_mappings = torch.zeros(
            len(self.draft_kv_cache_group_ids),
            self.max_num_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        self._layer_group_idx: list[int] | None = None
        if hasattr(self.model, "get_draft_kv_cache_layer_names"):
            name_to_gid = {
                ln: gid
                for gid, group in enumerate(kv_cache_config.kv_cache_groups)
                for ln in group.layer_names
            }
            gid_to_idx = {gid: i for i, gid in enumerate(self.draft_kv_cache_group_ids)}
            self._layer_group_idx = [
                gid_to_idx[name_to_gid[name]]
                for name in self.model.get_draft_kv_cache_layer_names()
            ]

    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        num_query_per_req: int | None = None,
        causal: bool = False,
    ) -> dict[str, Any] | None:
        if not self.draft_attn_layer_names:
            return None
        assert num_query_per_req is None
        return super()._build_draft_attn_metadata(
            num_reqs,
            num_reqs_padded,
            num_tokens_padded,
            num_query_per_req=self.num_query_per_req,
            causal=causal,
        )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(
            max_seq_len + self.num_query_per_req, self.max_model_len
        )

        if aux_hidden_states:
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states
        self.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        if dummy_run and skip_attn_for_dummy_run:
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:num_target_tokens],
                self.context_positions[:num_target_tokens],
            )
            self._prepare_eplb_forward(num_query_tokens)
            self._generate_draft(
                num_reqs,
                num_query_tokens,
                attn_metadata=None,
                slot_mappings=None,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            return self.draft_tokens[:num_reqs]

        assert self.draft_kv_cache_group_id >= 0
        for i, gid in enumerate(self.draft_kv_cache_group_ids):
            prepare_domino_inputs(
                self.input_buffers,
                self.block_tables.slot_mappings[gid],
                self.context_positions,
                self._context_slot_mappings[i],
                self.sample_indices,
                self.sample_pos,
                self.sample_idx_mapping,
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                self.block_tables.input_block_tables[gid],
                self.block_tables.block_sizes[gid],
                self.parallel_drafting_token_id,
                self.num_query_per_req,
                self.num_speculative_steps,
                self.max_num_reqs,
                self.max_num_tokens,
                self.max_model_len,
                self.sample_from_anchor,
            )

        if dummy_run:
            context_slots: torch.Tensor | list[torch.Tensor | None] | None = None
        elif self._layer_group_idx is not None:
            context_slots = [
                self._context_slot_mappings[gidx][:num_target_tokens]
                for gidx in self._layer_group_idx
            ]
        else:
            context_slots = self._context_slot_mappings[0][:num_target_tokens]
        self.model.precompute_and_store_context_kv(
            self.hidden_states[:num_target_tokens],
            self.context_positions[:num_target_tokens],
            context_slots,
        )

        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            num_reqs,
            num_query_tokens,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )

        num_reqs_padded = batch_desc.num_reqs or num_reqs
        num_tokens_padded = batch_desc.num_tokens

        draft_attn_metadata = self._build_draft_attn_metadata(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens_padded=num_tokens_padded,
            causal=self.dflash_causal,
        )
        draft_slot_mappings_by_layer = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :num_tokens_padded],
            self.kv_cache_config,
        )

        self._prepare_eplb_forward(num_query_tokens)

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.query_cudagraph_manager is not None
            self.query_cudagraph_manager.run_fullgraph(batch_desc)
        else:
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                draft_attn_metadata,
                draft_slot_mappings_by_layer,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=batch_desc.cg_mode,
            )

        return self.draft_tokens[:num_reqs]

    # --- END: methods copied from DFlashSpeculator ---

    # --- Domino-specific overrides ---

    def capture(self, attn_states: dict | None = None) -> None:
        logger.info("Capturing model for Domino speculator...")
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.zero_()
        self.gru_hidden_buffer.zero_()
        assert self.query_cudagraph_manager is not None
        self.query_cudagraph_manager.capture(
            self._generate_draft,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            progress_bar_desc="Capturing domino CUDA graphs",
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_dflash_model(target_model, self.vllm_config)

    @torch.inference_mode()
    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
        ):
            last_hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
            )
        return last_hidden_states

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        draft_tokens = self.sample_draft(
            last_hidden_states,
            num_reqs,
        )
        self.draft_tokens[:num_reqs] = draft_tokens.view(
            num_reqs, self.num_speculative_steps
        )

    @torch.inference_mode()
    def sample_draft(
        self,
        last_hidden_states: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        assert self.gru_hidden_buffer is not None and self.is_domino

        K = self.num_speculative_steps
        draft_tokens = torch.zeros(
            [num_reqs, self.num_speculative_steps],
            dtype=torch.int64,
            device=self.device,
        )
        base_logits = self.model.compute_logits(last_hidden_states)
        begin_reqs = torch.arange(num_reqs, device=self.device) * self.num_query_per_req
        base_draft = self._sample_single_token(
            base_logits[begin_reqs],
            0,
        )
        draft_tokens[:, 0] = base_draft.squeeze(-1)
        anchor_tokens = self.input_buffers.input_ids[begin_reqs]
        combined = torch.stack([anchor_tokens, base_draft], dim=1)
        realized_prefix_embed = self.model.embed_input_ids(combined)
        gru_hidden = self.model.gru_forward(realized_prefix_embed[:, 0, :], None)
        gru_hidden = self.model.gru_forward(realized_prefix_embed[:, 1, :], gru_hidden)
        hidden_3d = last_hidden_states.view(num_reqs, self.num_query_per_req, -1)
        logits_3d = base_logits.view(num_reqs, self.num_query_per_req, -1)

        for i in range(1, K):
            self._domino_cat_buf[:num_reqs, :, : self.hidden_size] = hidden_3d[
                :, i : i + 1, :
            ]
            self._domino_cat_buf[:num_reqs, :, self.hidden_size :] = (
                gru_hidden.unsqueeze(1)
            )
            bias = self.model.domino_mlp_forward(self._domino_cat_buf[:num_reqs])
            current_token_id = self._sample_single_token(
                logits_3d[:, i : i + 1, :] + bias,
                0,
            )
            draft_tokens[:, i : i + 1] = current_token_id

            if i + 1 < K:
                new_embed = self.model.embed_input_ids(draft_tokens[:, i])
                gru_hidden = self.model.gru_forward(new_embed, gru_hidden)

        return draft_tokens

    def _sample_single_token(self, logits: torch.Tensor, temperature: float = 0.0):
        if temperature <= 1e-6:
            return torch.argmax(logits, dim=-1)

        else:
            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)

            return torch.multinomial(probs, num_samples=1)
