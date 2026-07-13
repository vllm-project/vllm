# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlare speculator.

Extends :class:`DFlashSpeculator` with per-layer adaptive fusion and
heterogeneous KV projections.  The draft model forward itself is
identical to DFlash (query-only pass with pre-filled context KV cache).
"""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import (
    DFlashCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.dflare.utils import load_dflare_model

from vllm.v1.worker.gpu.spec_decode.utils import (
    get_parallel_drafting_token_id,
)
from vllm.v1.worker.gpu.spec_decode.dflash.utils import get_dflash_causal


class DFlareSpeculator(DFlashSpeculator):
    _speculator_name = "DFlare"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # ★ Per-layer fused context states [max_num_tokens, D, H].
        # DFlash has a single shared hidden_states [max_num_tokens, H];
        # DFlare needs a distinct fused representation per draft layer.
        D = self.draft_model_config.hf_config.num_hidden_layers
        self.per_layer_hidden_states = torch.zeros(
            self.max_num_tokens,
            D,
            self.hidden_size,
            dtype=self.dtype,
            device=device,
        )

    def load_draft_model(self, target_model, target_attn_layer_names):
        return load_dflare_model(target_model, self.vllm_config)

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
        max_seq_len = (
            input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        )
        self.draft_max_seq_len = min(
            max_seq_len + self.num_query_per_req, self.max_model_len
        )

        # ★ Per-layer fusion: [N, T*H] → [N, D, H]
        if aux_hidden_states:
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            # Fallback: replicate last_hidden_states across D draft layers
            D = self.draft_model_config.hf_config.num_hidden_layers
            hidden_states = last_hidden_states.unsqueeze(1).expand(
                -1, D, -1
            )
        self.per_layer_hidden_states[:num_target_tokens].copy_(
            hidden_states[:num_target_tokens]
        )

        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        if dummy_run and skip_attn_for_dummy_run:
            self.model.precompute_and_store_context_kv(
                self.per_layer_hidden_states[:num_target_tokens],
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
            from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
                prepare_dflash_inputs,
            )

            prepare_dflash_inputs(
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
                self.block_tables.kernel_block_sizes[gid],
                self.parallel_drafting_token_id,
                self.num_query_per_req,
                self.num_speculative_steps,
                self.max_num_reqs,
                self.max_num_tokens,
                self.max_model_len,
                self.sample_from_anchor,
            )

        if dummy_run:
            context_slots: (
                torch.Tensor | list[torch.Tensor | None] | None
            ) = None
        elif self._layer_group_idx is not None:
            context_slots = [
                self._context_slot_mappings[gidx][:num_target_tokens]
                for gidx in self._layer_group_idx
            ]
        else:
            context_slots = self._context_slot_mappings[0][
                :num_target_tokens
            ]

        # ★ Precompute with per-layer fused states [N, D, H]
        self.model.precompute_and_store_context_kv(
            self.per_layer_hidden_states[:num_target_tokens],
            self.context_positions[:num_target_tokens],
            context_slots,
        )

        from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp

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
            causal=self._group_causal,
        )

        from vllm.v1.worker.gpu.attn_utils import (
            build_slot_mappings_by_layer,
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
