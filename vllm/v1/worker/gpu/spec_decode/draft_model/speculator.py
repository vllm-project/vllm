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
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.spec_decode.utils import next_power_of_2
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionStatePair,
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

logger = init_logger(__name__)


class PlainDraftModelSpeculator(DraftModelSpeculator):
    """Speculative decoding using a separate smaller draft LM.

    Unlike Eagle, the draft model runs fully independently of the target model.
    Step 0 builds an expanded buffer (accepted + correction token + rejected
    slots masked with PAD_SLOT_ID) via a Triton kernel; steps 1..k-1 are
    single-token decode steps.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)

        # draft_max_seq_len is read by parent's _build_draft_attn_metadata.
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

        _expanded_max = self.max_num_tokens + self.max_num_reqs
        self.expanded_input_ids = torch.zeros(
            _expanded_max, dtype=torch.int32, device=device
        )
        self.expanded_positions = torch.zeros(
            _expanded_max, dtype=torch.int64, device=device
        )
        self.is_rejected_mask = torch.zeros(
            _expanded_max, dtype=torch.bool, device=device
        )

        self.supports_mm_inputs = False

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass  # CUDA graph not yet supported for plain draft model speculator.

    def capture(
        self,
        attn_states: dict[BatchExecutionDescriptor, AttentionStatePair],
    ) -> None:
        pass  # CUDA graph not yet supported for plain draft model speculator.

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
    ) -> torch.Tensor:
        num_tokens = input_ids.shape[0]
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
                input_ids=input_ids,
                positions=positions,
            )
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

    def _prefill(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        num_reqs: int,
        skip_attn: bool,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
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
            positions_buf = input_batch.positions
        else:
            block_tables = self.block_tables
            kv_cache_config = self.kv_cache_config
            assert block_tables is not None
            assert kv_cache_config is not None

            total_expanded = prepare_prefill_inputs(
                input_buffers=self.input_buffers,
                input_batch=input_batch,
                last_sampled=last_sampled,
                num_rejected=num_rejected,
                expanded_input_ids=self.expanded_input_ids,
                expanded_positions=self.expanded_positions,
                is_rejected_mask=self.is_rejected_mask,
                last_token_indices=self.last_token_indices,
                max_num_reqs=self.max_num_reqs,
                max_model_len=self.max_model_len,
            )

            prefill_slot_mappings = block_tables.compute_slot_mappings(
                input_batch.idx_mapping,
                self.input_buffers.query_start_loc[: num_reqs + 1],
                self.expanded_positions[:total_expanded],
                total_expanded,
            )
            # Mask rejected positions so they do not pollute the KV cache.
            is_rejected = self.is_rejected_mask[:total_expanded]
            prefill_slot_mappings[:, is_rejected] = PAD_SLOT_ID

            prefill_slot_maps_by_layer = build_slot_mappings_by_layer(
                prefill_slot_mappings, kv_cache_config
            )

            qsl_np = input_batch.query_start_loc_np
            query_start_loc_cpu_expanded = (
                torch.from_numpy(qsl_np[: num_reqs + 1]).int()
                + self.arange[: num_reqs + 1]
            )
            max_query_len = (
                int((qsl_np[1 : num_reqs + 1] - qsl_np[:num_reqs]).max()) + 1
            )

            prefill_attn_md = build_attn_metadata(
                attn_groups=self.attn_groups,
                num_reqs=num_reqs,
                num_tokens=total_expanded,
                query_start_loc_gpu=self.input_buffers.query_start_loc[: num_reqs + 1],
                query_start_loc_cpu=query_start_loc_cpu_expanded,
                max_query_len=max_query_len,
                seq_lens=self.input_buffers.seq_lens[:num_reqs],
                max_seq_len=self.max_model_len,
                block_tables=[x[:num_reqs] for x in block_tables.input_block_tables],
                slot_mappings=prefill_slot_mappings,
                kv_cache_config=kv_cache_config,
            )
            hidden_states = self._run_model(
                self.expanded_input_ids[:total_expanded],
                self.expanded_positions[:total_expanded],
                prefill_attn_md,
                prefill_slot_maps_by_layer,
                num_tokens_across_dp,
            )
            positions_buf = self.expanded_positions

        last_indices = self.last_token_indices[:num_reqs]
        self.current_draft_step.fill_(0)
        self.draft_tokens[:num_reqs, 0] = self.sample_draft(
            hidden_states[last_indices],
            positions_buf[last_indices],
            self.idx_mapping[:num_reqs],
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )

    def _multi_step_decode(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        num_reqs: int,
        skip_attn: bool,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
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
        self.input_buffers.query_start_loc[: num_reqs + 1].copy_(
            self.arange_gpu[: num_reqs + 1]
        )

        idx_mapping = input_batch.idx_mapping

        for step in range(1, self.num_speculative_steps):
            self.input_buffers.input_ids[:num_reqs].copy_(
                self.draft_tokens[:num_reqs, step - 1].int()
            )
            torch.clamp(
                self.input_buffers.positions[:num_reqs] + 1,
                max=self.max_model_len - 1,
                out=self.input_buffers.positions[:num_reqs],
            )
            # Increment seq_lens BEFORE the forward so the attention reads from
            # the KV slot written by the previous step.
            torch.clamp(
                self.input_buffers.seq_lens[:num_reqs] + 1,
                max=self.max_model_len,
                out=self.input_buffers.seq_lens[:num_reqs],
            )

            decode_attn_md = None
            step_slot_maps_by_layer = None
            if not skip_attn:
                block_tables = self.block_tables
                kv_cache_config = self.kv_cache_config
                assert block_tables is not None
                assert kv_cache_config is not None
                q_start = self.input_buffers.query_start_loc[: num_reqs + 1]
                positions = self.input_buffers.positions[:num_reqs]
                step_slot_mappings = block_tables.compute_slot_mappings(
                    idx_mapping, q_start, positions, num_reqs
                )
                step_slot_maps_by_layer = build_slot_mappings_by_layer(
                    step_slot_mappings, kv_cache_config
                )
                decode_attn_md = self._build_draft_attn_metadata(
                    num_reqs, num_reqs, num_reqs
                )
            hidden_states = self._run_model(
                self.input_buffers.input_ids[:num_reqs],
                self.input_buffers.positions[:num_reqs],
                decode_attn_md,
                step_slot_maps_by_layer,
                num_tokens_across_dp,
            )

            self.current_draft_step.fill_(step)
            self.draft_tokens[:num_reqs, step] = self.sample_draft(
                hidden_states[:num_reqs],
                self.input_buffers.positions[:num_reqs],
                self.idx_mapping[:num_reqs],
                self.temperature,
                self.seeds,
                self.current_draft_step,
                self.draft_logits,
            )

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
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        assert self.model is not None

        num_reqs = input_batch.num_reqs
        skip_attn = dummy_run and skip_attn_for_dummy_run

        # Copy per-request temperature/seeds/idx_mapping into pre-allocated
        # buffers so sample_draft can read them without extra slicing.
        self._copy_request_inputs(
            num_reqs,
            input_batch.idx_mapping,
            temperature,
            seeds,
        )

        self._prefill(
            input_batch,
            num_rejected,
            last_sampled,
            num_reqs,
            skip_attn,
            num_tokens_across_dp,
        )

        if self.num_speculative_steps == 1:
            return self.draft_tokens[:num_reqs, :1]

        self._multi_step_decode(
            input_batch,
            num_rejected,
            num_reqs,
            skip_attn,
            num_tokens_across_dp,
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
    out_is_rejected_ptr,  # [src_tokens + num_reqs] bool
    last_token_indices_ptr,  # [max_num_reqs] int64
    out_query_start_loc_ptr,  # [max_num_reqs + 1] int32
    out_seq_lens_ptr,  # [max_num_reqs] int32
    query_start_loc_ptr,  # [num_reqs + 1] int32
    seq_lens_ptr,  # [num_reqs] int32
    num_rejected_ptr,  # [num_reqs] int32
    src_tokens,
    max_num_reqs,
    max_model_len,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Per-request step-0 input preparation for the plain draft-model speculator.

    Output layout for request i (out_start = query_start_loc[i] + i):
        [out_start,              out_start + num_valid)      accepted tokens
        [out_start + num_valid]                              correction token
        (out_start + num_valid, out_start + total_out)      rejected (masked)

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
    correction_pos = start_pos + num_valid
    out_start = q_start + req_idx
    total_out = q_next - q_start + 1

    for i in range(0, total_out, BLOCK_SIZE):
        j = i + tl.arange(0, BLOCK_SIZE)
        in_bounds = j < total_out

        is_valid = j < num_valid
        is_correction = j == num_valid
        is_rejected = (j > num_valid) & in_bounds

        src_idx = tl.minimum(q_start + j, src_tokens - 1)
        token_ids = tl.load(target_input_ids_ptr + src_idx, mask=is_valid, other=0)
        positions = tl.load(target_positions_ptr + src_idx, mask=is_valid, other=0)
        token_ids = tl.where(is_correction, correction_token, token_ids)
        positions = tl.where(is_correction, correction_pos, positions)
        token_ids = tl.where(is_rejected, 0, token_ids)
        positions = tl.where(is_rejected, 0, positions)

        out_idx = out_start + j
        tl.store(out_input_ids_ptr + out_idx, token_ids, mask=in_bounds)
        tl.store(out_positions_ptr + out_idx, positions, mask=in_bounds)
        tl.store(out_is_rejected_ptr + out_idx, is_rejected, mask=in_bounds)

    tl.store(last_token_indices_ptr + req_idx, out_start + num_valid)
    tl.store(out_query_start_loc_ptr + req_idx, out_start)
    # seqlen_k for the expanded prefill attention:
    #   pre_existing = seq_len - query_len (draft KV before this round)
    #   expanded_query_len = query_len + 1 (adds correction token)
    #   seqlen_k = pre_existing + expanded_query_len = seq_len + 1
    # num_rejected does NOT change seqlen_k — the expanded query always has
    # query_len+1 slots (rejected ones are masked with PAD_SLOT_ID, so they
    # don't write KV but still occupy query positions).
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
    is_rejected_mask: torch.Tensor,  # [max_tokens + max_reqs] bool
    last_token_indices: torch.Tensor,  # [max_num_reqs] int64
    max_num_reqs: int,
    max_model_len: int,
) -> int:
    """Call _prepare_prefill_inputs_kernel and return total_expanded tokens.

    Side-effects (kernel writes):
      - expanded_input_ids, expanded_positions, is_rejected_mask
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
        is_rejected_mask,
        last_token_indices,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        input_batch.query_start_loc,
        input_batch.seq_lens,
        num_rejected.int(),
        src_tokens,
        max_num_reqs,
        max_model_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return src_tokens + num_reqs
