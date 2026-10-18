# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import get_dp_group, get_pp_group
from vllm.forward_context import BatchDescriptor, DPMetadata, create_forward_context
from vllm.models.deepseek_v41.decoder_replay_layers import (
    DecoderReplayLayers,
    ReplayBatch,
    ReplayStage,
)
from vllm.models.deepseek_v41.nvidia.decoder_replay_cudagraph import (
    DecoderReplayCudaGraphManager,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadataBuilder
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.dp_utils import should_skip_dp_coordination
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import UvaBufferPool
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


@triton.jit
def _gather_lookback_kernel(
    lookback_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    positions_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    num_reqs,
    DEPTH: tl.constexpr,
    BLOCK_DEPTH: tl.constexpr,
):
    # One program per lookback row; rows past the batch are filled with -1.
    batch_idx = tl.program_id(0)
    in_batch = batch_idx < num_reqs
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx, mask=in_batch, other=0)
    begin = tl.load(query_start_loc_ptr + batch_idx, mask=in_batch, other=0)
    end = tl.load(query_start_loc_ptr + batch_idx + 1, mask=in_batch, other=0)
    in_batch = in_batch & (end > begin)
    num_computed = tl.load(positions_ptr + begin, mask=in_batch, other=0)

    offs = tl.arange(0, BLOCK_DEPTH)
    pos = num_computed - 1 - offs
    valid = in_batch & (offs < DEPTH) & (pos >= 0)
    ids = tl.load(
        all_token_ids_ptr + req_state_idx.to(tl.int64) * all_token_ids_stride + pos,
        mask=valid,
        other=-1,
    )
    tl.store(lookback_ptr + batch_idx * DEPTH + offs, ids, mask=offs < DEPTH)


@triton.jit
def _pad_replayed_slots_kernel(
    slot_mappings_ptr,  # [num_groups, num_tokens_padded]
    group_stride,
    cacheable_groups_ptr,  # [NUM_GROUPS] indices of the prefix-cacheable groups
    query_start_loc_ptr,  # [num_reqs + 1]
    positions_ptr,  # [num_tokens]
    replay_start_ptr,  # [num_reqs]
    window,
    pad_slot_id,
    NUM_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(replay_start_ptr + req)
    if start == 0:
        return
    begin = tl.load(query_start_loc_ptr + req)
    end = tl.load(query_start_loc_ptr + req + 1)
    for tok in range(begin, end, BLOCK):
        offs = tok + tl.arange(0, BLOCK)
        pos = tl.load(positions_ptr + offs, mask=offs < end, other=0)
        replayed = (offs < end) & (pos >= start) & (pos < start + window)
        for i in tl.static_range(NUM_GROUPS):
            group = tl.load(cacheable_groups_ptr + i)
            tl.store(
                slot_mappings_ptr + group * group_stride + offs,
                pad_slot_id,
                mask=replayed,
            )


@triton.jit
def _gather_replay_batch_kernel(
    query_start_loc_ptr,  # [num_reqs + 1]
    dropped_before_ptr,  # [num_reqs + 1] rows trimmed before each boundary
    seq_lens_ptr,  # [num_reqs]
    replay_start_ptr,  # [num_reqs] the encoder-side replay start
    positions_ptr,
    slot_mappings_ptr,  # [num_groups, num_tokens]
    slot_mappings_stride,
    rows_ptr,  # out: [num_kept] batch row of every kept row
    kept_query_start_loc_ptr,  # out: [num_reqs + 1]
    kept_positions_ptr,  # out: [num_kept]
    kept_slot_mappings_ptr,  # out: [num_groups, num_kept]
    kept_slot_mappings_stride,
    kept_kv_start_ptr,  # out: [num_reqs]
    window,
    NUM_GROUPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per request: its kept rows are the last ones before its
    boundary, and land at that boundary minus the rows trimmed before it."""
    req = tl.program_id(0)
    begin = tl.load(query_start_loc_ptr + req)
    end = tl.load(query_start_loc_ptr + req + 1)
    kept_begin = begin - tl.load(dropped_before_ptr + req)
    kept_end = end - tl.load(dropped_before_ptr + req + 1)
    if req == 0:
        tl.store(kept_query_start_loc_ptr, kept_begin)
    tl.store(kept_query_start_loc_ptr + req + 1, kept_end)

    # A trimmed request holds no replay-layer window KV below its kept rows.
    kv_start = tl.load(replay_start_ptr + req)
    if kept_end - kept_begin < end - begin:
        kv_start = tl.maximum(kv_start, tl.load(seq_lens_ptr + req) - window)
    tl.store(kept_kv_start_ptr + req, kv_start)

    first_row = end - (kept_end - kept_begin)
    for tok in range(kept_begin, kept_end, BLOCK):
        offs = tok + tl.arange(0, BLOCK)
        mask = offs < kept_end
        row = first_row + (offs - kept_begin)
        tl.store(rows_ptr + offs, row.to(tl.int64), mask=mask)
        tl.store(
            kept_positions_ptr + offs,
            tl.load(positions_ptr + row, mask=mask),
            mask=mask,
        )
        for g in tl.static_range(NUM_GROUPS):
            tl.store(
                kept_slot_mappings_ptr + g * kept_slot_mappings_stride + offs,
                tl.load(slot_mappings_ptr + g * slot_mappings_stride + row, mask=mask),
                mask=mask,
            )


class _ReplayStageBuffers:
    """A later exact-replay stage's inputs, like the first stage's ``_kept_*``
    buffers, plus each row's index among the previous stage's rows."""

    def __init__(self, max_num_reqs: int, max_num_tokens: int, device) -> None:
        self.dropped_before = UvaBufferPool(max_num_reqs + 1, torch.int32)
        self.rows = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.select = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.query_start_loc = torch.zeros(
            max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        self.slot_mappings: torch.Tensor | None = None
        self.kv_start = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.attn_groups: list[list[AttentionGroup]] | None = None


class ReplayAttnMetadata(ModelSpecificAttnMetadata):
    """Hands the batch's replay starts to the sliding-window builders."""

    def __init__(self, replay_start: torch.Tensor) -> None:
        self.replay_start = replay_start

    def get_extra_attn_kwargs(
        self, attn_metadata_builder: Any, num_reqs: int
    ) -> dict[str, Any]:
        if isinstance(attn_metadata_builder, DeepseekSparseSWAMetadataBuilder):
            return {"replay_start": self.replay_start[:num_reqs]}
        return {}


class DeepseekV41ModelState(DefaultModelState):
    """DefaultModelState plus the engram lookback window and SWA bounded replay.

    The engram n-gram hash needs the ids of the ``depth`` tokens preceding
    each request's chunk start (see ``common/engram.py``). The runner keeps
    the full token history on device, so the window is gathered there every
    step: exact for prompt and generated tokens alike, whatever instance
    produced their KV.

    SWA bounded replay (``CacheConfig.swa_bounded_replay``) keeps the
    sliding-window KV out of prefix caching and rebuilds it after a prefix hit
    by recomputing the hit's last window; the scheduler tells each request the
    position from which it holds window KV (``NewRequestData.replay_start``).
    ``prepare_attn`` gathers those per batch, pads the replayed tokens' slots
    in the prefix-cacheable groups so the cached KV stays as is, and hands the
    starts to the sliding-window metadata builders, whose kernels read no
    window KV below them. With the decoder side on
    (``model.decoder_replay_layers``) it also prepares the replay layers'
    batch in eager steps: each prefill's last ``window`` rows as a sub-batch
    with metadata and a forward context of its own, like a microbatch.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        depth = model.token_lookback_depth
        self.lookback_token_ids: torch.Tensor | None = None
        if depth > 0:
            # Persistent so a captured graph can read it on replay.
            max_num_reqs = self.max_num_reqs
            if vllm_config.parallel_config.prefill_context_parallel_size > 1:
                max_num_reqs *= 2
            self.lookback_token_ids = torch.full(
                (max_num_reqs, depth), -1, dtype=torch.int32, device=device
            )

        # Per request state index; batches gather from it (see prepare_attn).
        self._replay_start_np = np.zeros(self.max_num_reqs, dtype=np.int32)
        self._replay_start = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self._replay_start_staging = UvaBufferPool(self.max_num_reqs, torch.int32)
        # The replay window and the indices of the prefix-cacheable groups, from
        # the KV cache config on first use; None until then.
        self._replay: tuple[int, torch.Tensor] | None = None

        self.decoder_replay_layers: DecoderReplayLayers | None = getattr(
            model, "decoder_replay_layers", None
        )
        if self.decoder_replay_layers is not None:
            # Requests whose every prompt row is read (prompt logprobs) never trim.
            self._keeps_rows_np = np.zeros(self.max_num_reqs, dtype=np.bool_)
            self._dropped_before = UvaBufferPool(self.max_num_reqs + 1, torch.int32)
            # The replay batch's inputs: its rows of the batch, query offsets,
            # positions, slot mappings (sized with the KV cache groups on first
            # use) and the position from which each request holds replay-layer
            # window KV.
            self._kept_rows = torch.zeros(
                self.max_num_tokens, dtype=torch.int64, device=device
            )
            self._kept_query_start_loc = torch.zeros(
                self.max_num_reqs + 1, dtype=torch.int32, device=device
            )
            self._kept_positions = torch.zeros(
                self.max_num_tokens, dtype=torch.int64, device=device
            )
            self._kept_slot_mappings: torch.Tensor | None = None
            self._kept_kv_start = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=device
            )
            # The replay layers' and the first one's SWA attention groups.
            self._replay_attn_groups: (
                tuple[list[list[AttentionGroup]], list[list[AttentionGroup]]] | None
            ) = None
            # Exact replay in stages: the later stages' buffers.
            self._replay_stages = [
                _ReplayStageBuffers(self.max_num_reqs, self.max_num_tokens, device)
                for _ in self.decoder_replay_layers.stage_rows[1:]
            ]
            # Per stage, the rows padding its graph past its tokens: MoE must
            # not dispatch them (their routes are garbage and can overflow an
            # expert's receive slots on the other ranks). Persistent so the
            # graphs read each step's mask.
            self._replay_is_padding = torch.ones(
                len(self._replay_stages) + 1,
                self.max_num_tokens,
                dtype=torch.bool,
                device=device,
            )
        layers = self.decoder_replay_layers
        compilation_config = vllm_config.compilation_config
        self.replay_cudagraphs: DecoderReplayCudaGraphManager | None = None
        if (
            layers is not None
            and compilation_config.cudagraph_decoder_replay
            and compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
            and is_breakable_cudagraph_enabled()
            and get_pp_group().is_first_rank
            # The graphs' fixed inputs hold no LoRA mapping and no None input ids.
            and vllm_config.lora_config is None
            and not vllm_config.model_config.enable_prompt_embeds
        ):
            threshold = compilation_config.decoder_replay_trim_threshold
            if layers.kept_rows > layers.window:
                # Exact replay trims to as many rows as the full forward needs;
                # a smaller graph has nothing to trim.
                threshold = max(threshold, layers.kept_rows + 1)
            if threshold <= layers.kept_rows:
                raise ValueError(
                    "decoder_replay_trim_threshold must exceed the replay layers' "
                    f"{layers.kept_rows} kept rows: smaller graphs have nothing to "
                    "trim"
                )
            self.replay_cudagraphs = DecoderReplayCudaGraphManager(
                vllm_config, device, layers
            )
            layers.trim_threshold = threshold

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        self._replay_start_np[req_index] = new_req_data.replay_start
        if self.decoder_replay_layers is not None:
            params = new_req_data.sampling_params
            self._keeps_rows_np[req_index] = (
                params is not None and params.prompt_logprobs is not None
            ) or self.vllm_config.aux_output_config.enable_return_routed_experts

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, torch.Tensor | None]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        window = self.lookback_token_ids
        if window is None:
            return model_inputs
        all_token_ids = req_states.all_token_ids.gpu
        depth = window.shape[1]
        _gather_lookback_kernel[(window.shape[0],)](
            window,
            input_batch.idx_mapping,
            input_batch.query_start_loc,
            input_batch.positions,
            all_token_ids,
            all_token_ids.stride(0),
            input_batch.idx_mapping.shape[0],
            DEPTH=depth,
            BLOCK_DEPTH=triton.next_power_of_2(depth),
        )
        model_inputs["lookback_token_ids"] = window
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self.lookback_token_ids is not None:
            # The captured graph reads this buffer; replays refill it in place.
            self.lookback_token_ids.fill_(-1)
            model_inputs["lookback_token_ids"] = self.lookback_token_ids
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
        if self._replay is None:
            specs = [group.kv_cache_spec for group in kv_cache_config.kv_cache_groups]
            self._replay = (
                max(spec.prefix_replay_tokens for spec in specs),
                torch.tensor(
                    [i for i, spec in enumerate(specs) if spec.prefix_cacheable],
                    dtype=torch.int32,
                    device=self.device,
                ),
            )
            if self.decoder_replay_layers is not None:
                self._kept_slot_mappings = torch.zeros(
                    len(specs),
                    self.max_num_tokens,
                    dtype=torch.int64,
                    device=self.device,
                )
                for stage in self._replay_stages:
                    stage.slot_mappings = torch.zeros_like(self._kept_slot_mappings)
            self._warm_up_replay_kernels(input_batch, slot_mappings)
        window, cacheable_groups = self._replay
        replay_start: torch.Tensor | None = None
        if window or self.decoder_replay_layers is not None:
            num_reqs = input_batch.num_reqs
            # Decode rows sit above the hit, so only prefills carry a replay
            # start; dummy batches (captures, profiling) carry none.
            replay_start_np = np.where(
                input_batch.is_prefilling_np[:num_reqs],
                self._replay_start_np[input_batch.idx_mapping_np[:num_reqs]],
                0,
            ).astype(np.int32)
            replay_start = self._replay_start_staging.copy_to_gpu(
                replay_start_np, out=self._replay_start[:num_reqs]
            )
            if window and replay_start_np.any():
                # The replayed tokens rebuild window KV only: their slots in the
                # prefix-cacheable groups are padded so the cached KV stays as is.
                _pad_replayed_slots_kernel[(num_reqs,)](
                    slot_mappings,
                    slot_mappings.stride(0),
                    cacheable_groups,
                    input_batch.query_start_loc,
                    input_batch.positions,
                    replay_start,
                    window,
                    PAD_SLOT_ID,
                    NUM_GROUPS=cacheable_groups.numel(),
                    BLOCK=1024,
                )
            assert model_specific_attn_metadata is None
            model_specific_attn_metadata = ReplayAttnMetadata(replay_start)
        attn_metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=ubatch_idx,
            model_specific_attn_metadata=model_specific_attn_metadata,
        )
        if self.decoder_replay_layers is not None:
            assert replay_start is not None
            self._prepare_replay_batch(
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                replay_start,
            )
        return attn_metadata

    def _warm_up_replay_kernels(
        self, input_batch: InputBatch, slot_mappings: torch.Tensor
    ) -> None:
        """Compile the replay kernels on the buffers they run on: the startup
        dummy batches never replay or trim, so they would not launch them."""
        assert self._replay is not None
        window, cacheable_groups = self._replay
        # Pad warmup is prefix-cache SWA. A zero window must not skip the gather warmup.
        if window:
            _pad_replayed_slots_kernel.warmup(
                slot_mappings,
                slot_mappings.stride(0),
                cacheable_groups,
                input_batch.query_start_loc,
                input_batch.positions,
                self._replay_start,
                window,
                PAD_SLOT_ID,
                NUM_GROUPS=cacheable_groups.numel(),
                BLOCK=1024,
                grid=(1,),
            )
        if self.decoder_replay_layers is None:
            return
        if self._bi_replay_gather():
            return  # the batch-invariant CUDA gather needs no warmup
        assert self._kept_slot_mappings is not None
        _gather_replay_batch_kernel.warmup(
            input_batch.query_start_loc,
            self._dropped_before.copy_to_uva(np.zeros(1, dtype=np.int32)),
            input_batch.seq_lens,
            self._replay_start,
            input_batch.positions,
            slot_mappings,
            slot_mappings.stride(0),
            self._kept_rows,
            self._kept_query_start_loc,
            self._kept_positions,
            self._kept_slot_mappings,
            self._kept_slot_mappings.stride(0),
            self._kept_kv_start,
            self.decoder_replay_layers.window,
            NUM_GROUPS=slot_mappings.shape[0],
            BLOCK=1024,
            grid=(1,),
        )

    def _bi_replay_gather(self) -> bool:
        """Exact replay (more kept rows than the window, or stages) is batch
        invariance's, so its gather runs the batch-invariant CUDA op."""
        layers = self.decoder_replay_layers
        assert layers is not None
        return (
            envs.VLLM_BATCH_INVARIANT
            or layers.kept_rows != layers.window
            or len(layers.stage_rows) > 1
        )

    def _prepare_replay_batch(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        replay_start: torch.Tensor,
        capture_desc: BatchExecutionDescriptor | None = None,
    ) -> None:
        """Set the replay layers' batch for this forward, or none. Eager steps trim;
        captures and PIECEWISE graphs at the trim threshold or above break out to a
        replay batch, trimmed or not; FULL and smaller PIECEWISE graphs keep the
        whole batch. Under DP every rank replays if any does."""
        layers = self.decoder_replay_layers
        assert layers is not None
        layers.replay_batch = None
        force_replay_batch = capture_desc is not None or (
            cudagraph_mode == CUDAGraphMode.PIECEWISE
            and layers.graph_uses_replay_batch(input_batch.num_tokens_after_padding)
        )
        if cudagraph_mode != CUDAGraphMode.NONE and not force_replay_batch:
            return

        num_reqs = input_batch.num_reqs
        query_lens = np.diff(input_batch.query_start_loc_np[: num_reqs + 1])
        keeps_rows = self._keeps_rows_np[input_batch.idx_mapping_np[:num_reqs]]
        stage_rows = layers.stage_rows
        if capture_desc is not None:
            # A capture keeps the dummy batch's rows at every stage: its graphs
            # are keyed by the one size.
            stage_lens = [query_lens.copy() for _ in stage_rows]
        else:
            stage_lens = [
                np.where(keeps_rows, query_lens, np.minimum(query_lens, rows))
                for rows in stage_rows
            ]
        kept_lens = stage_lens[0]
        # A batch of context-free non-prefill rows is a dummy (an idle DP rank's):
        # nothing reads it and its idx_mapping hits stale slots, so keep one row.
        seq_lens = input_batch.seq_lens_cpu_upper_bound[:num_reqs].numpy()
        is_dummy = not input_batch.is_prefilling_np[:num_reqs].any() and bool(
            (seq_lens == query_lens).all()
        )
        if is_dummy and capture_desc is None:
            for lens in stage_lens:
                lens[:] = 0
                lens[0] = 1
        # Dummy batches (captures, an idle DP rank's) are not prefills: they
        # keep their rows unless a rank trims.
        needs_replay_batch = force_replay_batch or bool(
            (kept_lens < query_lens)[input_batch.is_prefilling_np[:num_reqs]].any()
        )
        stage_tokens = [int(lens.sum()) for lens in stage_lens]
        dp_tokens = None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            needs_replay_batch, dp_tokens = self._agree_across_dp(
                needs_replay_batch, stage_tokens
            )
        if not needs_replay_batch:
            return

        if self._replay_attn_groups is None:
            self._replay_attn_groups = (
                self._copy_attn_groups(
                    attn_groups,
                    layers.stage_metadata_prefixes[0]
                    if len(stage_rows) > 1
                    else layers.metadata_prefixes,
                ),
                self._copy_attn_groups(attn_groups, {layers.first_swa_prefix}),
            )
            for i, stage in enumerate(self._replay_stages, start=1):
                stage.attn_groups = self._copy_attn_groups(
                    attn_groups, layers.stage_metadata_prefixes[i]
                )
        replay_groups, first_swa_groups = self._replay_attn_groups
        stages: list[ReplayStage] = []
        prev_dropped = None
        replay_batch: ReplayBatch | None = None
        for i, lens in enumerate(stage_lens):
            cudagraphs = self.replay_cudagraphs
            num_tokens = stage_tokens[i]
            rank_tokens = None if dp_tokens is None else dp_tokens[i].clone()
            run_graph, num_padded, cg_mode = None, num_tokens, CUDAGraphMode.NONE
            if cudagraphs is not None:
                # Every rank runs the graph that fits the largest rank's batch.
                dp_max_tokens = (
                    num_tokens if rank_tokens is None else int(rank_tokens.max())
                )
                desc = capture_desc or cudagraphs.dispatch(
                    num_reqs, dp_max_tokens, None, 0
                )
                if desc.cg_mode == CUDAGraphMode.PIECEWISE:
                    num_padded, cg_mode = desc.num_tokens, desc.cg_mode
                    run_graph = partial(cudagraphs.run, stage=i)
                    if rank_tokens is not None:
                        rank_tokens.fill_(num_padded)
            dp_metadata = None
            if rank_tokens is not None:
                dp_metadata = DPMetadata.make(
                    self.vllm_config.parallel_config, num_padded, rank_tokens
                )
            kept_batch, kept_slot_mappings, rows, select, kv_start, prev_dropped = (
                self._kept_input_batch(
                    input_batch,
                    slot_mappings,
                    replay_start,
                    lens,
                    num_padded,
                    stage=i,
                    prev_dropped=prev_dropped,
                )
            )
            groups = replay_groups if i == 0 else self._replay_stages[i - 1].attn_groups
            assert groups is not None
            attn_metadata = super().prepare_attn(
                kept_batch,
                CUDAGraphMode.NONE,
                block_tables,
                kept_slot_mappings,
                groups,
                kv_cache_config,
                model_specific_attn_metadata=ReplayAttnMetadata(kv_start[:num_reqs]),
            )
            if i == 0:
                # The first replay layer wrote every row's window KV: its window
                # starts are the batch's.
                first_swa = layers.first_swa_prefix
                attn_metadata[first_swa] = super().prepare_attn(
                    kept_batch,
                    CUDAGraphMode.NONE,
                    block_tables,
                    kept_slot_mappings,
                    first_swa_groups,
                    kv_cache_config,
                    model_specific_attn_metadata=ReplayAttnMetadata(replay_start),
                )[first_swa]
            is_padding = None
            if run_graph is not None and envs.VLLM_MOE_SKIP_PADDING:
                is_padding = self._replay_is_padding[i, :num_padded]
                is_padding[:num_tokens].fill_(False)
                is_padding[num_tokens:].fill_(True)
            forward_context = create_forward_context(
                attn_metadata,
                self.vllm_config,
                dp_metadata=dp_metadata,
                cudagraph_runtime_mode=cg_mode,
                batch_descriptor=BatchDescriptor(num_tokens=num_padded),
                slot_mapping=build_slot_mappings_by_layer(
                    kept_slot_mappings, kv_cache_config
                ),
                is_padding=is_padding,
            )
            if i == 0:
                replay_batch = ReplayBatch(
                    rows[:num_tokens], forward_context, run_graph, stages
                )
            else:
                stages.append(
                    ReplayStage(
                        rows[:num_tokens],
                        select[:num_tokens],
                        forward_context,
                        run_graph,
                    )
                )
        layers.replay_batch = replay_batch

    def capture_inner_cudagraphs(
        self,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        def prepare(desc: BatchExecutionDescriptor) -> None:
            # Capture reuses the runtime path on a dummy batch, so the graph reads
            # the buffers a real step fills.
            num_reqs = min(desc.num_tokens, self.max_num_reqs)
            self._prepare_replay_batch(
                InputBatch.make_dummy(num_reqs, desc.num_tokens, input_buffers),
                CUDAGraphMode.NONE,
                block_tables.get_dummy_block_tables(num_reqs),
                block_tables.get_dummy_slot_mappings(desc.num_tokens),
                attn_groups,
                kv_cache_config,
                self._replay_start[:num_reqs].zero_(),
                capture_desc=desc,
            )

        if self.replay_cudagraphs is not None:
            # profile_only captures use pre-KV-cache groups: drop the cache.
            self._replay_attn_groups = None
            self.replay_cudagraphs.capture_replay_graphs(prepare)

    def _kept_input_batch(
        self,
        input_batch: InputBatch,
        slot_mappings: torch.Tensor,
        replay_start: torch.Tensor,
        kept_lens: np.ndarray,
        num_padded: int,
        stage: int = 0,
        prev_dropped: torch.Tensor | None = None,
    ) -> tuple[
        InputBatch,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build the sub-``InputBatch`` of each request's last ``kept_lens``
        rows, and its slot mappings, batch rows, rows among the previous stage's
        (``prev_dropped``: its rows trimmed before each boundary), window KV
        starts and rows trimmed before each boundary.

        The kept rows follow the device boundaries minus the rows trimmed before
        them: adaptive verification resizes the decodes on the GPU alone, and
        decodes never trim. Like a microbatch's, the sub-batch describes the
        forward only; its sampling fields are carried over and must not be read.
        """
        num_reqs = input_batch.num_reqs
        num_tokens = int(kept_lens.sum())
        kept_query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(kept_lens, out=kept_query_start_loc_np[1:])
        if stage == 0:
            pool, rows = self._dropped_before, self._kept_rows
            select = rows
            query_start_loc, positions = (
                self._kept_query_start_loc,
                self._kept_positions,
            )
            all_slot_mappings, kv_start = self._kept_slot_mappings, self._kept_kv_start
        else:
            buffers = self._replay_stages[stage - 1]
            pool, rows, select = buffers.dropped_before, buffers.rows, buffers.select
            query_start_loc, positions = buffers.query_start_loc, buffers.positions
            all_slot_mappings, kv_start = buffers.slot_mappings, buffers.kv_start
            assert prev_dropped is not None
        dropped_before = pool.copy_to_uva(
            input_batch.query_start_loc_np[: num_reqs + 1] - kept_query_start_loc_np
        )
        assert all_slot_mappings is not None
        kept_slot_mappings = all_slot_mappings[:, :num_padded]
        kept_slot_mappings[:, num_tokens:].fill_(PAD_SLOT_ID)
        layers = self.decoder_replay_layers
        assert layers is not None
        if self._bi_replay_gather():
            from vllm.model_executor.kernels.batch_invariant import (
                gather_replay_batch,
            )

            gather_replay_batch(
                input_batch.query_start_loc,
                dropped_before,
                input_batch.seq_lens,
                replay_start,
                input_batch.positions,
                slot_mappings,
                num_reqs,
                layers.stage_rows[stage],
                rows,
                query_start_loc,
                positions,
                kept_slot_mappings,
                kv_start,
                prev_dropped_before=prev_dropped,
                select=None if prev_dropped is None else select,
            )
        else:
            _gather_replay_batch_kernel[(num_reqs,)](
                input_batch.query_start_loc,
                dropped_before,
                input_batch.seq_lens,
                replay_start,
                input_batch.positions,
                slot_mappings,
                slot_mappings.stride(0),
                rows,
                query_start_loc,
                positions,
                kept_slot_mappings,
                kept_slot_mappings.stride(0),
                kv_start,
                layers.window,
                NUM_GROUPS=slot_mappings.shape[0],
                BLOCK=1024,
            )
        # Adaptive verification may lengthen the decodes on the device, up to
        # the batch's bound.
        max_query_len = max(int(kept_lens.max()), input_batch.max_query_len or 0)
        kept_batch = replace(
            input_batch,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_padded,
            query_start_loc=query_start_loc[: num_reqs + 1],
            query_start_loc_np=kept_query_start_loc_np,
            max_query_len=max_query_len,
            positions=positions[:num_padded],
            fast_prefill=None,
        )
        return kept_batch, kept_slot_mappings, rows, select, kv_start, dropped_before

    def _agree_across_dp(
        self, needs_replay_batch: bool, stage_tokens: list[int]
    ) -> tuple[bool, torch.Tensor | None]:
        """Whether any rank needs a replay batch and, if so, each rank's token
        count at every replay stage (one row per stage)."""
        parallel_config = self.vllm_config.parallel_config
        agreed = torch.zeros(
            1 + len(stage_tokens), parallel_config.data_parallel_size, dtype=torch.int32
        )
        agreed[:, parallel_config.data_parallel_rank] = torch.tensor(
            [needs_replay_batch, *stage_tokens]
        )
        if not should_skip_dp_coordination():
            dist.all_reduce(agreed, group=get_dp_group().cpu_group)
        if not agreed[0].any():
            return False, None
        return True, agreed[1:]

    def _copy_attn_groups(
        self, attn_groups: list[list[AttentionGroup]], prefixes: set[str]
    ) -> list[list[AttentionGroup]]:
        """Copies of the groups holding any of ``prefixes``, with builders of their
        own like a microbatch's: a builder keeps the metadata it built."""
        own_groups = []
        for groups in attn_groups:
            copies = []
            for group in groups:
                if prefixes.isdisjoint(group.layer_names):
                    continue
                copy = replace(group, metadata_builders=[])
                copy.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    group.kernel_block_size,
                )
                copies.append(copy)
            own_groups.append(copies)
        return own_groups
