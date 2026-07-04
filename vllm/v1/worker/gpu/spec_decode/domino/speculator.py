# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash.utils import (
    load_dflash_model,
)
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id

logger = init_logger(__name__)


class DominoSpeculator(DFlashSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

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

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # PIECEWISE cudagraphs are not supported for dflash
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            cudagraph_mode = CUDAGraphMode.NONE

        self.query_cudagraph_manager = DFlashCudaGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            decode_query_len=self.num_query_per_req,
            causal=self.dflash_causal,
        )

    def capture(self, attn_states: dict | None = None) -> None:
        logger.info("Capturing model for Domino speculator...")
        # Reset sampling indices to zero to prevent stale values from prior
        # dummy runs from being baked into the captured graph.
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
            progress_bar_desc="Capturing dflash CUDA graphs",
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_dflash_model(target_model, self.vllm_config)

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
    ) -> None:
        super().set_attn(model_state, kv_cache_config, block_tables)

        # DFlash precomputes context K/V with a single block_size; mixing
        # kv-cache groups would silently corrupt the cache for the non-matching group.
        draft_groups = [gid for gid, g in enumerate(self.attn_groups) if g]
        assert len(draft_groups) == 1, (
            "DFlash currently requires all draft attention layers to share "
            "a single kv-cache group."
        )
        self.draft_kv_cache_group_id = draft_groups[0]
        self.draft_block_size = self.block_tables.block_sizes[
            self.draft_kv_cache_group_id
        ]

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
        hidden_3d = last_hidden_states.view(
            num_reqs, self.num_query_per_req, -1
        )
        logits_3d = base_logits.view(num_reqs, self.num_query_per_req, -1)

        for i in range(1, K):
            self._domino_cat_buf[:num_reqs, :, : self.hidden_size] = hidden_3d[
                :, i : i + 1, :
            ]
            self._domino_cat_buf[
                :num_reqs, :, self.hidden_size :
            ] = gru_hidden.unsqueeze(1)
            bias = self.model.domino_mlp_forward(self._domino_cat_buf[:num_reqs])
            current_token_id = self._sample_single_token(
                logits_3d[:, i : i + 1, :] + bias,
                0,
            )
            draft_tokens[:, i: i + 1] = current_token_id

            if i + 1 < K:
                new_embed = self.model.embed_input_ids(draft_tokens[:, i])
                gru_hidden = self.model.gru_forward(new_embed, gru_hidden)

        return draft_tokens

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
        assert num_query_per_req is None  # Omitted for DFlash, read from self instead
        return super()._build_draft_attn_metadata(
            num_reqs,
            num_reqs_padded,
            num_tokens_padded,
            num_query_per_req=self.num_query_per_req,
            causal=causal,
        )

    def _sample_single_token(self, logits: torch.Tensor, temperature: float = 0.0):
        if temperature <= 1e-6:
            return torch.argmax(logits, dim=-1)

        else:
            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)

            return torch.multinomial(probs, num_samples=1)

