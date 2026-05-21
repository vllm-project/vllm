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
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.spec_decode.utils import next_power_of_2
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


@triton.jit
def _prepare_draft_model_step0_kernel(
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


def prepare_draft_model_step0_inputs(
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
    """Call _prepare_draft_model_step0_kernel and return total_expanded tokens.

    Side-effects (kernel writes):
      - expanded_input_ids, expanded_positions, is_rejected_mask
      - last_token_indices
      - input_buffers.query_start_loc  (expanded: original[i] + i)
      - input_buffers.seq_lens         (= target_seq_lens + 1)
    """
    num_reqs = input_batch.num_reqs
    src_tokens = input_batch.num_tokens
    qsl_np = input_batch.query_start_loc_np
    query_lens = qsl_np[1 : num_reqs + 1] - qsl_np[:num_reqs]
    max_total_out = int(query_lens.max()) + 2  # +1 correction, +1 alignment
    BLOCK_SIZE = min(512, next_power_of_2(max_total_out))

    _prepare_draft_model_step0_kernel[(num_reqs,)](
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


# ---------------------------------------------------------------------------
# DraftModelSpeculator
# ---------------------------------------------------------------------------


class DraftModelSpeculator:
    """Speculative decoding using a separate smaller draft LM.

    Unlike Eagle, the draft model runs fully independently of the target model.
    Step 0 builds an expanded buffer (accepted + correction token + rejected
    slots masked with PAD_SLOT_ID) via a Triton kernel; steps 1..k-1 are
    single-token decode steps.
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

        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        self.last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.arange = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device="cpu"
        )
        self.arange_gpu = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device=device
        )

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
        self.draft_logits: torch.Tensor | None = None

        # Set in load_model / set_attn.
        self.model: nn.Module | None = None
        self.draft_attn_layer_names: set[str] = set()
        self.attn_groups: list[list[Any]] = []
        self.kv_cache_config: KVCacheConfig | None = None
        self.block_tables: BlockTables | None = None
        self.model_state: ModelState | None = None

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        pass  # CUDA graph not yet supported for draft model speculator.

    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )

        from vllm.compilation.backends import set_model_tag

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
        _, self.attn_groups, *_ = init_attn_backend(
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
        pass  # CUDA graph not yet supported for draft model speculator.

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _SAMPLING_EPS: float = 1e-5

    def _sample_tokens(
        self,
        logits: torch.Tensor,  # [n, vocab_size] — modified in-place
        temperature: torch.Tensor,  # [n]
    ) -> torch.Tensor:
        """Greedy (argmax) or Gumbel-max sampling, matching V1 convention."""
        is_greedy = temperature < self._SAMPLING_EPS
        if is_greedy.all():
            return logits.argmax(dim=-1)

        safe_temp = torch.where(is_greedy, torch.ones_like(temperature), temperature)
        logits.div_(safe_temp.unsqueeze(1))
        probs = logits.softmax(dim=-1, dtype=torch.float32)

        q = torch.empty_like(probs)
        q.exponential_()
        sampled = probs.div(q).argmax(dim=-1)
        greedy = probs.argmax(dim=-1)
        return torch.where(is_greedy, greedy, sampled)

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
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return hidden_states

    def _build_decode_attn_metadata(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any] | None:
        """Attention metadata for decode steps (1 token per request)."""
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
    # propose
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,  # unused by draft model
        aux_hidden_states: list[torch.Tensor] | None,  # unused
        num_sampled: torch.Tensor,  # [num_reqs]
        num_rejected: torch.Tensor,  # [num_reqs]
        last_sampled: torch.Tensor,  # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,  # [max_num_reqs]
        temperature: torch.Tensor,  # [max_num_reqs]
        seeds: torch.Tensor,  # [max_num_reqs]
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        assert self.model is not None

        num_reqs = input_batch.num_reqs
        skip_attn = dummy_run and skip_attn_for_dummy_run

        if not skip_attn:
            assert self.block_tables is not None
            assert self.kv_cache_config is not None

        # ----------------------------------------------------------
        # Step 0 — expanded-buffer prefill
        # ----------------------------------------------------------
        if skip_attn:
            # Dummy/profile path: simple copy without correction-token injection.
            src_tokens = input_batch.num_tokens
            self.input_buffers.input_ids[:src_tokens].copy_(
                input_batch.input_ids[:src_tokens]
            )
            self.input_buffers.positions[:src_tokens].copy_(
                input_batch.positions[:src_tokens]
            )
            qsl = input_batch.query_start_loc
            query_lens = qsl[1 : num_reqs + 1] - qsl[:num_reqs]
            adjusted_lens = query_lens - num_rejected[:num_reqs]
            self.last_token_indices[:num_reqs] = qsl[:num_reqs] + adjusted_lens - 1
            hidden_states = self.run_model(src_tokens, None, None, num_tokens_across_dp)
        else:
            block_tables = self.block_tables
            kv_cache_config = self.kv_cache_config
            assert block_tables is not None
            assert kv_cache_config is not None

            total_expanded = prepare_draft_model_step0_inputs(
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

            step0_slot_mappings = block_tables.compute_slot_mappings(
                input_batch.idx_mapping,
                self.input_buffers.query_start_loc[: num_reqs + 1],
                self.expanded_positions[:total_expanded],
                total_expanded,
            )
            # Mask rejected positions so they do not pollute the KV cache.
            is_rejected = self.is_rejected_mask[:total_expanded]
            step0_slot_mappings[:, is_rejected] = PAD_SLOT_ID

            step0_slot_maps_by_layer = build_slot_mappings_by_layer(
                step0_slot_mappings, kv_cache_config
            )

            qsl_np = input_batch.query_start_loc_np
            query_start_loc_cpu_expanded = (
                torch.from_numpy(qsl_np[: num_reqs + 1]).int()
                + self.arange[: num_reqs + 1]
            )

            qsl_np_reqs = qsl_np[: num_reqs + 1]
            max_query_len = int((qsl_np_reqs[1:] - qsl_np_reqs[:-1]).max()) + 1

            block_tables_step0 = [x[:num_reqs] for x in block_tables.input_block_tables]
            attn_md_0 = build_attn_metadata(
                attn_groups=self.attn_groups,
                num_reqs=num_reqs,
                num_tokens=total_expanded,
                query_start_loc_gpu=self.input_buffers.query_start_loc[: num_reqs + 1],
                query_start_loc_cpu=query_start_loc_cpu_expanded,
                max_query_len=max_query_len,
                seq_lens=self.input_buffers.seq_lens[:num_reqs],
                max_seq_len=self.max_model_len,
                block_tables=block_tables_step0,
                slot_mappings=step0_slot_mappings,
                kv_cache_config=kv_cache_config,
            )

            batch_descriptor = BatchDescriptor(num_tokens=total_expanded)
            with set_forward_context(
                attn_md_0,
                self.vllm_config,
                num_tokens=total_expanded,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                num_tokens_across_dp=num_tokens_across_dp,
                slot_mapping=step0_slot_maps_by_layer,
                batch_descriptor=batch_descriptor,
            ):
                hidden_states = self.model(  # type: ignore[misc]
                    input_ids=self.expanded_input_ids[:total_expanded],
                    positions=self.expanded_positions[:total_expanded],
                )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        last_indices = self.last_token_indices[:num_reqs]
        logits = self.model.compute_logits(hidden_states[last_indices])
        self.draft_tokens[:num_reqs, 0] = self._sample_tokens(
            logits, temperature[:num_reqs]
        )

        if self.num_speculative_steps == 1:
            return self.draft_tokens[:num_reqs, :1]

        # ----------------------------------------------------------
        # Prepare state for decode steps
        # ----------------------------------------------------------
        if skip_attn:
            qsl = input_batch.query_start_loc
            query_lens = qsl[1 : num_reqs + 1] - qsl[:num_reqs]
            adjusted_lens = query_lens - num_rejected[:num_reqs]
            last_positions = self.input_buffers.positions[
                qsl[:num_reqs] + adjusted_lens - 1
            ]
        else:
            last_positions = self.expanded_positions[last_indices]

        self.input_buffers.positions[:num_reqs].copy_(last_positions)

        # seq_lens initial value = target_seq_lens - num_rejected + 1.
        # The decode loop increments this BEFORE each forward, so step 1
        # effectively uses target - rejected + 2 = (past KV count) + 1.
        target_seq_lens = input_batch.seq_lens[:num_reqs]
        self.input_buffers.seq_lens[:num_reqs].copy_(
            torch.clamp(
                target_seq_lens - num_rejected[:num_reqs].int() + 1,
                max=self.max_model_len,
            )
        )

        self.input_buffers.query_start_loc[: num_reqs + 1].copy_(
            self.arange_gpu[: num_reqs + 1]
        )

        idx_mapping = input_batch.idx_mapping

        # ----------------------------------------------------------
        # Steps 1 .. num_speculative_steps - 1
        # ----------------------------------------------------------
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
            # the KV slot written by the previous step.  flash_attn computes
            # context_kv_lens = seq_lens - query_len, so without this
            # pre-increment the freshly written correction/draft token is missed.
            torch.clamp(
                self.input_buffers.seq_lens[:num_reqs] + 1,
                max=self.max_model_len,
                out=self.input_buffers.seq_lens[:num_reqs],
            )

            if skip_attn:
                hidden_states = self.run_model(
                    num_reqs, None, None, num_tokens_across_dp
                )
            else:
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
                decode_attn_md = self._build_decode_attn_metadata(num_reqs, num_reqs)
                hidden_states = self.run_model(
                    num_reqs,
                    decode_attn_md,
                    step_slot_maps_by_layer,
                    num_tokens_across_dp,
                )

            logits = self.model.compute_logits(hidden_states[:num_reqs])
            self.draft_tokens[:num_reqs, step] = self._sample_tokens(
                logits, temperature[:num_reqs]
            )

        return self.draft_tokens[:num_reqs]
