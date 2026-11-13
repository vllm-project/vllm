# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    compute_mm_prefix_ranges,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupRole
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.rope import get_rope_state
from vllm.v1.worker.gpu.model_states.interface import (
    ModelSpecificAttnMetadata,
    ModelState,
)
from vllm.v1.worker.gpu.model_states.mm_pruning import maybe_create_mm_pruner
from vllm.v1.worker.gpu.model_states.prompt_embeds import PromptEmbedsState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


@triton.jit
def _protect_draft_recovery_main_slots_kernel(
    slot_mappings_ptr,
    group_stride,
    main_groups_ptr,
    query_start_loc_ptr,
    positions_ptr,
    main_cached_end_ptr,
    pad_slot_id,
    NUM_MAIN_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    tile_idx = tl.program_id(1)
    main_cached_end = tl.load(main_cached_end_ptr + req_idx)
    begin = tl.load(query_start_loc_ptr + req_idx)
    end = tl.load(query_start_loc_ptr + req_idx + 1)
    offsets = begin + tile_idx * BLOCK + tl.arange(0, BLOCK)
    positions = tl.load(positions_ptr + offsets, mask=offsets < end, other=0)
    protected = (offsets < end) & (positions < main_cached_end)
    for group_offset in tl.static_range(NUM_MAIN_GROUPS):
        group_idx = tl.load(main_groups_ptr + group_offset)
        tl.store(
            slot_mappings_ptr + group_idx * group_stride + offsets,
            pad_slot_id,
            mask=protected,
        )


def _protect_draft_recovery_main_slots_cpu(
    slot_mappings: torch.Tensor,
    query_start_loc: tuple[int, ...],
    positions: torch.Tensor,
    main_cached_ends: tuple[int, ...],
    main_group_ids: tuple[int, ...],
    pad_slot_id: int = PAD_SLOT_ID,
) -> None:
    """Prevent replay from overwriting KV blocks already cached by the target."""
    for req_idx, main_cached_end in enumerate(main_cached_ends):
        if main_cached_end == 0:
            continue
        begin = query_start_loc[req_idx]
        end = query_start_loc[req_idx + 1]
        protected = positions[begin:end] < main_cached_end
        for group_idx in main_group_ids:
            slot_mappings[group_idx, begin:end].masked_fill_(protected, pad_slot_id)


class DefaultModelState(ModelState):
    supports_prompt_embeds = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)

        self.prompt_embeds_state: PromptEmbedsState | None = None
        if self.model_config.enable_prompt_embeds:
            self.prompt_embeds_state = PromptEmbedsState(
                self.max_num_reqs, self.inputs_embeds_size, self.dtype, self.device
            )
            if not self.supports_mm_inputs:
                # Persistent buffer analogous to encoder_runner.inputs_embeds.
                embeds_buffer_size = (self.max_num_tokens, self.inputs_embeds_size)
                self.inputs_embeds = torch.zeros(
                    embeds_buffer_size, dtype=self.dtype, device=self.device
                )

        self.rope_state = get_rope_state(
            self.model_config,
            model,
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len,
            device=self.device,
        )

        # Pruner is used for multimodal embedding pruning (EVS).
        self.mm_pruner = maybe_create_mm_pruner(
            self.model_config, model, self.rope_state, encoder_cache
        )

        # Per-request bidirectional MM ranges. Not kept in the encoder cache:
        # only the first PP rank has one, but every rank builds attn metadata.
        self.mm_prefix_ranges: dict[str, list[tuple[int, int]]] | None = None
        if (
            self.model_config.supports_multimodal_inputs
            and self.model_config.is_mm_prefix_lm
        ):
            self.mm_prefix_ranges = {}

        self._draft_recovery_main_cached_end_np: np.ndarray | None = None
        self._draft_recovery_main_cached_end: torch.Tensor | None = None
        self._draft_recovery_staging: UvaBufferPool | None = None
        self._draft_recovery_main_groups: torch.Tensor | None = None

    def _init_draft_window_recovery_state(self) -> None:
        self._draft_recovery_main_cached_end_np = np.zeros(
            self.max_num_reqs, dtype=np.int32
        )
        self._draft_recovery_main_cached_end = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self._draft_recovery_staging = (
            UvaBufferPool(self.max_num_reqs, torch.int32)
            if self.device.type != "cpu"
            else None
        )

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, block_tables: BlockTables
    ) -> None:
        del block_tables
        main_group_ids = tuple(
            group_idx
            for group_idx, group in enumerate(kv_cache_config.kv_cache_groups)
            if group.role is not KVCacheGroupRole.DRAFT_WINDOW
        )
        if len(main_group_ids) == len(kv_cache_config.kv_cache_groups):
            return
        self._init_draft_window_recovery_state()
        self._draft_recovery_main_groups = torch.tensor(
            main_group_ids, dtype=torch.int32, device=self.device
        )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        if self.mm_prefix_ranges is not None:
            hf_text_config = self.model_config.hf_text_config
            self.mm_prefix_ranges[new_req_data.req_id] = compute_mm_prefix_ranges(
                [new_req_data.mm_features],
                sliding_window=self.model_config.get_sliding_window(),
                clamp_sliding_window=getattr(
                    self.model, "mm_prefix_clamp_sliding_window", False
                )
                or getattr(hf_text_config, "mm_prefix_clamp_sliding_window", False),
                span_pad=getattr(self.model, "mm_prefix_span_leading_pad_modulus", 0),
            )[0]
        main_cached_end = new_req_data.draft_recovery_main_cached_end
        if self._draft_recovery_main_cached_end_np is not None:
            self._draft_recovery_main_cached_end_np[req_index] = main_cached_end
        if self.rope_state is not None:
            # `prompt_embeds` is a passthrough modality with no grid info, but
            # M-RoPE assumes per-feature grids. Filter it out.
            mm_features = [
                f for f in new_req_data.mm_features if f.modality != "prompt_embeds"
            ]
            self.rope_state.init_prefill_positions(
                req_index,
                self.model,
                new_req_data.prefill_token_ids,
                mm_features=mm_features,
            )
        if self.prompt_embeds_state is not None:
            self.prompt_embeds_state.add_request(req_index, new_req_data)

    def remove_request(self, req_id: str) -> None:
        if self.mm_prefix_ranges is not None:
            self.mm_prefix_ranges.pop(req_id, None)
        if self.prompt_embeds_state is not None:
            self.prompt_embeds_state.remove_request(req_id)

    def apply_staged_writes(self) -> None:
        if self.rope_state is not None:
            self.rope_state.apply_staged_writes()
        if self.prompt_embeds_state is not None:
            self.prompt_embeds_state.apply_staged_writes()

    def dummy_inputs_embeds(self, num_tokens: int) -> torch.Tensor:
        """Pre-allocated inputs_embeds buffer for dummy runs (contents unused)."""
        if self.supports_mm_inputs:
            return self.encoder_runner.inputs_embeds[:num_tokens]
        return self.inputs_embeds[:num_tokens]

    def prepare_inputs_embeds(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        # Use unpadded input_ids to match is_mm_embed size (num_tokens).
        # input_batch.input_ids may be padded for CUDA graphs.
        input_ids_unpadded = input_batch.input_ids[: input_batch.num_tokens]

        if self.supports_mm_inputs:
            self.execute_mm_encoder(scheduled_encoder_inputs)

            mm_embeds, is_mm_embed = super().gather_mm_embeddings(input_batch)
            if self.mm_pruner is not None and mm_embeds:
                # EVS: recompute mrope positions for pruned media.
                mm_embeds = self.mm_pruner.recompute(mm_embeds, input_batch, req_states)
                # We must flush the staged rope updates for prepare_inputs() to pick up.
                self.apply_staged_writes()

            inputs_embeds = self.encoder_runner.get_inputs_embeds(
                input_ids_unpadded, mm_embeds, is_mm_embed
            )
        else:
            input_embeddings = self.model.embed_input_ids(input_ids_unpadded)
            self.inputs_embeds[: input_embeddings.shape[0]] = input_embeddings
            inputs_embeds = self.inputs_embeds

        if self.prompt_embeds_state is not None:
            self.prompt_embeds_state.apply(
                input_batch, req_states.num_computed_tokens.gpu, inputs_embeds
            )

        return inputs_embeds[: input_batch.num_tokens_after_padding]

    def gather_mm_embeddings(
        self, input_batch: InputBatch, draft_lookahead: int = 0
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        mm_embeds, is_mm_embed = super().gather_mm_embeddings(
            input_batch, draft_lookahead
        )
        if self.mm_pruner is not None:
            # EVS: strip the appended mrope-position channels.
            mm_embeds = self.mm_pruner.strip(mm_embeds)
        return mm_embeds, is_mm_embed

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        if self.rope_state is None:
            return {}  # Common case (1D positions).

        self.rope_state.prepare_positions(
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            req_states.prefill_len.gpu,
            req_states.num_computed_tokens.gpu,
        )
        positions = self.rope_state.get_positions(input_batch.num_tokens_after_padding)
        return {"positions": positions}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = {}
        if self.supports_mm_inputs or self.prompt_embeds_state is not None:
            model_inputs["inputs_embeds"] = self.dummy_inputs_embeds(num_tokens)
        if self.rope_state is not None:
            model_inputs["positions"] = self.rope_state.get_positions(num_tokens)
        return model_inputs

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            # Use padded sizes - padding is handled by model_runner.prepare_attn.
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            # For piecewise cudagraphs and eager, use unpadded sizes.
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        self._protect_draft_recovery_main_slots(input_batch, slot_mappings)
        query_start_loc_cpu = torch.from_numpy(
            input_batch.query_start_loc_np[: num_reqs + 1]
        )
        query_start_loc_gpu = input_batch.query_start_loc[: num_reqs + 1]
        max_query_len = input_batch.max_query_len
        if max_query_len is None:
            max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            # Capture with worst-case max_seq_len so the graph is valid at any replay.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
        req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
        if self.mm_prefix_ranges is not None:
            req_doc_ranges = {
                i: self.mm_prefix_ranges.get(req_id, [])
                for i, req_id in enumerate(input_batch.req_ids)
            }
        dcp_local_seq_lens_cpu_upper_bound = None
        if input_batch.pcp is not None:
            dcp_local_seq_lens_cpu_upper_bound = (
                input_batch.pcp.dcp_local_seq_lens_cpu_upper_bound
            )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            dcp_local_seq_lens_cpu_upper_bound=dcp_local_seq_lens_cpu_upper_bound,
            positions=input_batch.positions,
            is_prefilling=torch.from_numpy(input_batch.is_prefilling_np),
            mm_req_doc_ranges=req_doc_ranges,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
            req_idx=input_batch.idx_mapping_np,
            ubatch_idx=ubatch_idx,
            fast_prefill=input_batch.fast_prefill,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
        return attn_metadata

    def _protect_draft_recovery_main_slots(
        self,
        input_batch: InputBatch,
        slot_mappings: torch.Tensor,
    ) -> None:
        cached_ends = self._draft_recovery_main_cached_end_np
        if cached_ends is None:
            return
        num_reqs = input_batch.num_reqs
        batch_cached_ends = np.where(
            input_batch.is_prefilling_np[:num_reqs],
            cached_ends[input_batch.idx_mapping_np[:num_reqs]],
            0,
        ).astype(np.int32)
        if not batch_cached_ends.any():
            return

        if self._draft_recovery_main_groups is None:
            raise RuntimeError("draft recovery state was not initialized")
        main_groups = self._draft_recovery_main_groups

        if slot_mappings.device.type == "cpu":
            _protect_draft_recovery_main_slots_cpu(
                slot_mappings=slot_mappings,
                query_start_loc=tuple(input_batch.query_start_loc_np[: num_reqs + 1]),
                positions=input_batch.positions,
                main_cached_ends=tuple(batch_cached_ends),
                main_group_ids=tuple(main_groups.tolist()),
            )
            return

        assert self._draft_recovery_main_cached_end is not None
        assert self._draft_recovery_staging is not None
        main_cached_end = self._draft_recovery_staging.copy_to_gpu(
            batch_cached_ends,
            out=self._draft_recovery_main_cached_end[:num_reqs],
        )
        max_query_len = int(
            np.diff(input_batch.query_start_loc_np[: num_reqs + 1]).max()
        )
        _protect_draft_recovery_main_slots_kernel[
            (num_reqs, triton.cdiv(max_query_len, 1024))
        ](
            slot_mappings,
            slot_mappings.stride(0),
            main_groups,
            input_batch.query_start_loc,
            input_batch.positions,
            main_cached_end,
            PAD_SLOT_ID,
            NUM_MAIN_GROUPS=main_groups.numel(),
            BLOCK=1024,
        )
