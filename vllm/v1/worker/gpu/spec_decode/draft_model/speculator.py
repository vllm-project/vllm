# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.triton_utils import tl, triton
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.utils import next_power_of_2
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.cudagraph_utils import SpeculatorCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import AttentionGroup, get_uniform_decode_token_count

logger = init_logger(__name__)


class PlainDraftModelSpeculator(DraftModelSpeculator):
    """Speculative decoding using a separate smaller draft LM.

    Unlike Eagle, the draft model runs fully independently of the target model.
    Step 0 builds an expanded buffer (accepted + correction token + rejected
    slots) via a Triton kernel; steps 1..k-1 are single-token decode steps.
    """

    # Plain draft model needs one extra slot per request for correction.
    num_extra_query_per_req = 1

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # draft_max_seq_len is read by the parent's attention metadata builder.
        # Plain draft model doesn't do per-batch adjustment; cap at max.
        self.draft_max_seq_len = self.max_model_len

        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        # GPU-side arange for decode query_start_loc initialisation.
        self.arange_gpu = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device=device
        )
        # Scalar step counter used as column index into draft_logits.
        self.current_draft_step = torch.tensor(0, dtype=torch.int64, device=device)
        self.decode_output = torch.empty(
            self.max_num_reqs, dtype=torch.int64, device=device
        )

        _expanded_max = self.max_num_tokens + self.max_num_reqs
        self.expanded_input_ids = torch.zeros(
            _expanded_max, dtype=torch.int32, device=device
        )
        self.expanded_positions = torch.zeros(
            _expanded_max, dtype=torch.int64, device=device
        )
        self.expanded_slot_mappings: torch.Tensor
        self.supports_mm_inputs = False

    def capture(self) -> None:
        logger.info("Capturing model for plain draft-model speculator...")
        self.last_token_indices.zero_()
        self.idx_mapping.zero_()
        self.current_draft_step.zero_()
        assert self.prefill_cudagraph_manager is not None
        if self.prefill_cudagraph_manager.use_breakable_cg:
            self.prefill_cudagraph_manager.init_breakable_cg_runner(self.model)
        self.prefill_cudagraph_manager.capture(
            self._generate_prefill_drafts,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing plain draft prefill CUDA graphs",
        )

        if self.num_speculative_steps == 1:
            return

        assert self.decode_cudagraph_manager is not None
        self.decode_cudagraph_manager.capture(
            self._generate_fused_drafts,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc="Capturing plain draft decode CUDA graphs",
        )

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self.expanded_slot_mappings = torch.empty(
            block_tables.num_kv_cache_groups,
            self.max_num_tokens + self.max_num_reqs,
            dtype=torch.int64,
            device=self.device,
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        spec = self.speculative_config
        assert spec is not None
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
            return get_model(
                vllm_config=draft_vllm_config,
                prefix="draft_model",
            )

    @torch.inference_mode()
    def _run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        cudagraph_manager: SpeculatorCudaGraphManager | None = None,
    ) -> torch.Tensor:
        num_tokens = input_ids.shape[0]
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
            model_inputs = dict(input_ids=input_ids, positions=positions)
            if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                assert cudagraph_manager is not None
                hidden_states = cudagraph_manager.run_pw_graph(self.model, model_inputs)
            else:
                hidden_states = self.model(**model_inputs)  # type: ignore[misc]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return hidden_states

    def _accepted_last_indices(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        qsl = input_batch.query_start_loc
        adjusted_lens = qsl[1 : num_reqs + 1] - qsl[:num_reqs] - num_rejected[:num_reqs]
        return qsl[:num_reqs] + adjusted_lens - 1

    def _sample_prefill_drafts(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, num_reqs: int
    ) -> None:
        self.draft_tokens[:num_reqs, 0] = self.sample_draft(
            hidden_states[self.last_token_indices[:num_reqs]],
            positions[self.last_token_indices[:num_reqs]],
            self.idx_mapping[:num_reqs],
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )

    def _generate_prefill_drafts(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode,
    ) -> None:
        assert self.prefill_cudagraph_manager is not None
        hidden_states = self._run_model(
            self.expanded_input_ids[:num_tokens],
            self.expanded_positions[:num_tokens],
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            self.prefill_cudagraph_manager,
        )
        self._sample_prefill_drafts(hidden_states, self.expanded_positions, num_reqs)

    def _prepare_decode_inputs(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
    ) -> torch.Tensor:
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        if skip_attn:
            last_positions = input_batch.positions[
                self._accepted_last_indices(input_batch, num_rejected, num_reqs)
            ]
        else:
            last_positions = self.expanded_positions[self.last_token_indices[:num_reqs]]

        self.input_buffers.positions[:num_reqs].copy_(last_positions)
        # The decode loop increments seq_lens BEFORE each forward.
        # Initial seq_lens = target_seq_lens - num_rejected + 1.
        # Step 1 forward uses      target_seq_lens - num_rejected + 2.
        # Step 2 forward uses      target_seq_lens - num_rejected + 3.
        self.input_buffers.seq_lens[:num_reqs].copy_(
            torch.clamp(
                input_batch.seq_lens[:num_reqs] - num_rejected[:num_reqs].int() + 1,
                max=self.max_model_len,
            )
        )
        if num_reqs_padded > num_reqs:
            self.input_buffers.seq_lens[num_reqs:num_reqs_padded].zero_()
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs_padded + 1]
        query_start_loc[: num_reqs + 1].copy_(self.arange_gpu[: num_reqs + 1])
        if num_reqs_padded > num_reqs:
            query_start_loc[num_reqs + 1 :].fill_(num_reqs)

        seq_lens_cpu_upper_bound = torch.zeros_like(
            input_batch.seq_lens_cpu_upper_bound
        )
        seq_lens_cpu_upper_bound[:num_reqs].copy_(
            input_batch.seq_lens_cpu_upper_bound[:num_reqs]
        )
        return seq_lens_cpu_upper_bound

    def _multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
    ) -> None:
        """Prepare eager decode metadata, then run the fused draft loop."""
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        attn_metadata = None
        slot_mappings_by_layer = None
        if not skip_attn:
            assert self.block_tables is not None
            assert self.kv_cache_config is not None
            slot_mappings = self.block_tables.slot_mappings[:, : batch_desc.num_tokens]
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            attn_metadata = self._build_uniform_attn_metadata(
                batch_desc=batch_desc,
                num_reqs=num_reqs,
                num_query_per_req=1,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                # The loop increments seq_lens before its first forward.
                step=2,
            )

        self._generate_fused_drafts(
            num_reqs_padded,
            batch_desc.num_tokens,
            attn_metadata,
            slot_mappings_by_layer,
            num_tokens_across_dp,
            batch_desc.cg_mode,
        )

    def _generate_fused_drafts(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        attn_groups = (
            [group for groups in self.attn_groups for group in groups]
            if attn_metadata is not None
            else []
        )

        for step in range(1, self.num_speculative_steps):
            self.input_buffers.input_ids[:num_reqs].copy_(
                self.draft_tokens[:num_reqs, step - 1].int()
            )
            torch.clamp(
                self.input_buffers.positions[:num_reqs] + 1,
                max=self.max_model_len - 1,
                out=self.input_buffers.positions[:num_reqs],
            )
            torch.clamp(
                self.input_buffers.seq_lens[:num_reqs] + 1,
                max=self.max_model_len,
                out=self.input_buffers.seq_lens[:num_reqs],
            )

            if attn_metadata is not None:
                assert self.block_tables is not None
                self.block_tables.compute_slot_mappings(
                    self.idx_mapping[:num_reqs],
                    self.input_buffers.query_start_loc[: num_reqs + 1],
                    self.input_buffers.positions[:num_reqs],
                    num_tokens_padded,
                )
                for attn_group in attn_groups:
                    attn_group.update_draft_decode_metadata(attn_metadata)

            self.current_draft_step.fill_(step)
            self._prepare_eplb_forward(num_reqs)
            hidden_states = self._run_model(
                self.input_buffers.input_ids[:num_tokens_padded],
                self.input_buffers.positions[:num_tokens_padded],
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                self.decode_cudagraph_manager,
            )
            self.decode_output[:num_reqs] = self.sample_draft(
                hidden_states[:num_reqs],
                self.input_buffers.positions[:num_reqs],
                self.idx_mapping[:num_reqs],
                self.temperature,
                self.seeds,
                self.current_draft_step,
                self.draft_logits,
            )
            self.draft_tokens[:num_reqs, step].copy_(self.decode_output[:num_reqs])

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,  # unused
        aux_hidden_states: list[torch.Tensor] | None,  # unused
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        assert self.model is not None

        num_reqs = input_batch.num_reqs
        skip_attn = dummy_run and skip_attn_for_dummy_run
        num_tokens_across_dp = (
            dp_sync.num_tokens_across_dp if dp_sync is not None else None
        )
        # Plain prefill adds one correction-token slot per request, so its
        # batch shape cannot reuse the target's DP sync. Preserve the target's
        # eager decision when dispatching the expanded draft batch.
        need_eager = is_profile or (dp_sync is not None and dp_sync.eager)

        # Copy per-request temperature/seeds/idx_mapping into pre-allocated
        # buffers so sample_draft can read them without extra slicing.
        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
            dummy_run=dummy_run,
        )

        if skip_attn:
            src_tokens = input_batch.num_tokens
            self.last_token_indices[:num_reqs] = self._accepted_last_indices(
                input_batch, num_rejected, num_reqs
            )
            hidden_states = self._run_model(
                input_batch.input_ids[:src_tokens],
                input_batch.positions[:src_tokens],
                None,
                None,
                num_tokens_across_dp,
            )
            self._sample_prefill_drafts(hidden_states, input_batch.positions, num_reqs)
        else:
            total_expanded = prepare_prefill_inputs(
                input_buffers=self.input_buffers,
                input_batch=input_batch,
                last_sampled=last_sampled,
                num_rejected=num_rejected,
                expanded_input_ids=self.expanded_input_ids,
                expanded_positions=self.expanded_positions,
                last_token_indices=self.last_token_indices,
                max_num_reqs=self.max_num_reqs,
                max_model_len=self.max_model_len,
            )
            max_query_len = int(input_batch.num_scheduled_tokens.max())
            uniform_token_count = get_uniform_decode_token_count(
                num_reqs,
                total_expanded,
                max_query_len + self.num_extra_query_per_req,
                input_batch.decode_graph_eligible,
            )
            prefill_batch_desc, prefill_batch_sync = dispatch_cg_and_sync_dp(
                self.prefill_cudagraph_manager,
                num_reqs,
                total_expanded,
                uniform_token_count=uniform_token_count,
                dp_size=self.dp_size,
                dp_rank=self.dp_rank,
                need_eager=need_eager,
            )
            num_tokens_across_dp = (
                prefill_batch_sync.num_tokens_across_dp
                if prefill_batch_sync is not None
                else None
            )

            block_tables = self.block_tables
            assert block_tables is not None
            num_tokens = prefill_batch_desc.num_tokens
            num_reqs_padded = prefill_batch_desc.num_reqs or num_reqs
            # num_tokens_padding = num_tokens - total_expanded
            # assert num_tokens_padding >= 0
            # if num_tokens_padding:
            #     self.expanded_input_ids[total_expanded:num_tokens].zero_()
            #     self.expanded_positions[total_expanded:num_tokens].zero_()

            query_start_loc_cpu = (
                input_batch.query_start_loc_np[: num_reqs + 1]
                + self.arange_np[: num_reqs + 1]
            )
            slot_mappings = block_tables.compute_slot_mappings(
                self.idx_mapping[:num_reqs_padded],
                self.input_buffers.query_start_loc[: num_reqs_padded + 1],
                self.expanded_positions[:num_tokens],
                num_tokens,
                out=self.expanded_slot_mappings,
            )
            prefill_attn_md = self._build_attn_metadata(
                num_reqs=num_reqs,
                batch_desc=prefill_batch_desc,
                query_start_loc_np=query_start_loc_cpu,
                seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
                step=1,
                slot_mappings=slot_mappings,
            )
            kv_cache_config = self.kv_cache_config
            assert kv_cache_config is not None
            prefill_slot_maps_by_layer = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            self.current_draft_step.zero_()
            self._prepare_eplb_forward(total_expanded)
            if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.prefill_cudagraph_manager is not None
                self.prefill_cudagraph_manager.run_fullgraph(prefill_batch_desc)
            else:
                self._generate_prefill_drafts(
                    prefill_batch_desc.num_reqs or num_reqs,
                    prefill_batch_desc.num_tokens,
                    prefill_attn_md,
                    prefill_slot_maps_by_layer,
                    num_tokens_across_dp,
                    prefill_batch_desc.cg_mode,
                )

        if self.num_speculative_steps == 1:
            return self.draft_tokens[:num_reqs, :1]

        decode_sync, num_batch_tokens = (
            self._build_uniform_batch_dp_sync(dp_sync, num_reqs)
            if dp_sync is not None
            else (None, num_reqs)
        )
        decode_batch_desc, decode_sync = dispatch_cg_and_sync_dp(
            self.decode_cudagraph_manager,
            num_reqs,
            num_batch_tokens,
            uniform_token_count=1,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=need_eager or skip_attn,
            dp_sync=decode_sync,
        )
        decode_num_tokens_across_dp = (
            decode_sync.num_tokens_across_dp if decode_sync is not None else None
        )

        seq_lens_cpu_upper_bound = self._prepare_decode_inputs(
            input_batch,
            num_rejected,
            num_reqs,
            skip_attn,
            decode_batch_desc,
        )
        if decode_batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.decode_cudagraph_manager is not None
            self.decode_cudagraph_manager.run_fullgraph(decode_batch_desc)
        else:
            self._multi_step_decode(
                num_reqs,
                skip_attn,
                decode_batch_desc,
                decode_num_tokens_across_dp,
                seq_lens_cpu_upper_bound,
            )

        return self.draft_tokens[:num_reqs]


@triton.jit
def _prepare_prefill_inputs_kernel(
    target_input_ids_ptr,  # [src_tokens] int32
    target_positions_ptr,  # [src_tokens] int64
    last_sampled_ptr,  # [max_num_reqs] int32
    idx_mapping_ptr,  # [num_reqs] int32
    out_input_ids_ptr,  # [src_tokens + num_reqs] int32
    out_positions_ptr,  # [src_tokens + num_reqs] int64
    last_token_indices_ptr,  # [max_num_reqs] int64
    out_query_start_loc_ptr,  # [max_num_reqs + 1] int32
    out_seq_lens_ptr,  # [max_num_reqs] int32
    query_start_loc_ptr,  # [num_reqs + 1] int32
    seq_lens_ptr,  # [num_reqs] int32
    num_rejected_ptr,  # [num_reqs] int32
    max_num_reqs,
    max_model_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-request step-0 input preparation for the plain draft-model speculator.

    Output layout for request i (out_start = query_start_loc[i] + i):
        [out_start,              out_start + num_valid)      accepted tokens
        [out_start + num_valid]                              correction token
        (out_start + num_valid, out_start + total_out)      rejected slots

    where num_valid  = query_lens[i] - num_rejected[i]
          total_out  = query_lens[i] + 1
    """
    req_idx = tl.program_id(0)
    num_reqs = tl.num_programs(0)

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    q_start = tl.load(query_start_loc_ptr + req_idx)
    q_next = tl.load(query_start_loc_ptr + req_idx + 1)
    seq_len = tl.load(seq_lens_ptr + req_idx)
    num_rejected = tl.load(num_rejected_ptr + req_idx)

    num_valid = q_next - q_start - num_rejected
    correction_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    start_pos = tl.load(target_positions_ptr + q_start)
    out_start = q_start + req_idx
    total_out = q_next - q_start + 1

    for i in range(0, total_out, BLOCK_SIZE):
        j = i + tl.arange(0, BLOCK_SIZE)
        in_bounds = j < total_out

        is_valid = j < num_valid
        is_correction = j == num_valid
        src_idx = q_start + j
        token_ids = tl.load(target_input_ids_ptr + src_idx, mask=is_valid, other=0)
        positions = tl.minimum(start_pos + j, max_model_len - 1)
        token_ids = tl.where(is_correction, correction_token, token_ids)

        out_idx = out_start + j
        tl.store(out_input_ids_ptr + out_idx, token_ids, mask=in_bounds)
        tl.store(out_positions_ptr + out_idx, positions, mask=in_bounds)

    tl.store(last_token_indices_ptr + req_idx, out_start + num_valid)
    tl.store(out_query_start_loc_ptr + req_idx, out_start)
    # seqlen_k for the expanded prefill attention:
    #   pre_existing = seq_len - query_len (draft KV before this round)
    #   expanded_query_len = query_len + 1 (adds correction token)
    #   seqlen_k = pre_existing + expanded_query_len = seq_len + 1
    # num_rejected does NOT change seqlen_k — the expanded query always has
    # query_len+1 slots. Rejected positions write temporary KV entries that
    # are overwritten with valid tokens in a later round.
    new_seq_len = tl.minimum(seq_len + 1, max_model_len)
    tl.store(out_seq_lens_ptr + req_idx, new_seq_len)

    if req_idx == num_reqs - 1:
        total_expanded = out_start + total_out
        tl.store(out_query_start_loc_ptr + num_reqs, total_expanded)
        for i in range(num_reqs + 1, max_num_reqs + 1, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block <= max_num_reqs
            tl.store(out_query_start_loc_ptr + block, total_expanded, mask=mask)
        for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
            block = i + tl.arange(0, BLOCK_SIZE)
            mask = block < max_num_reqs
            tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            tl.store(last_token_indices_ptr + block, 0, mask=mask)


def prepare_prefill_inputs(
    input_buffers: InputBuffers,
    input_batch: InputBatch,
    last_sampled: torch.Tensor,  # [max_num_reqs] int64
    num_rejected: torch.Tensor,  # [num_reqs]     int64
    expanded_input_ids: torch.Tensor,  # [max_tokens + max_reqs] int32
    expanded_positions: torch.Tensor,  # [max_tokens + max_reqs] int64
    last_token_indices: torch.Tensor,  # [max_num_reqs] int64
    max_num_reqs: int,
    max_model_len: int,
) -> int:
    """Call _prepare_prefill_inputs_kernel and return total_expanded tokens.

    Side-effects (kernel writes):
      - expanded_input_ids, expanded_positions
      - last_token_indices
      - input_buffers.query_start_loc  (expanded: original[i] + i)
      - input_buffers.seq_lens         (= target_seq_lens - num_rejected + 1)
    """
    num_reqs = input_batch.num_reqs
    src_tokens = input_batch.num_tokens
    qsl_np = input_batch.query_start_loc_np
    query_lens = qsl_np[1 : num_reqs + 1] - qsl_np[:num_reqs]
    max_total_out = int(query_lens.max()) + 2  # +1 correction, +1 alignment
    BLOCK_SIZE = min(512, next_power_of_2(max_total_out))

    _prepare_prefill_inputs_kernel[(num_reqs,)](
        input_batch.input_ids,
        input_batch.positions,
        last_sampled.int(),
        input_batch.idx_mapping,
        expanded_input_ids,
        expanded_positions,
        last_token_indices,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        input_batch.query_start_loc,
        input_batch.seq_lens,
        num_rejected.int(),
        max_num_reqs,
        max_model_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return src_tokens + num_reqs
