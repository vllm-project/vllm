# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P/D hidden-state handoff (prototype).

The prefiller (P) computes every prompt token and stores what the decoder needs
from the last prompt positions in a per-request "record" (``RecordLayout``),
kept in the request's own KV blocks: in the token slots just past the prompt of
one full-attention cache group (see ``vllm.v1.core.hidden_state_record``). The
KV connector transfers it with the blocks. The decoder (D) loads all prompt
tokens plus the record and samples the first output token from it, without
running a forward pass for any prompt position.

The scheduler still accounts for the last prompt token as a one-token step
(``SchedulerOutput.hidden_state_record_req_ids``), so the request's computed
token count, sampling position (seeds) and token budget match a recompute. The
model runner drops those rows from the forward batch and appends them, one
token each, to the batch it samples from, with their hidden states read from
the record.

The drafter runs on a batch where each sample-only row covers the drafter's
``window`` trailing prompt positions, so it rewrites its own KV for them with
D's sampled token exactly as a local final prefill chunk would (P wrote them
with its own sampled token and drafts). With a one-position window and an
otherwise uniform decode batch, each sample-only row is padded with
placeholder positions to a full decode row, rejected like the placeholder
drafts of a recomputed row, so the drafter keeps its full CUDA graph.

The record's slots are free while it lives there: P writes it at the end of the
step that completes the prompt, after its drafter has written its own KV past
the prompt, and the request then finishes; D reads it at the start of its
sampling step, before that step's drafter or the next step's forward writes
those slots.

The sampling batch reuses the forward batch's tokens and appends one token per
sample-only row. Prompt logprobs are assumed never to be requested on D.

Records are gathered, written and read by Triton kernels that find the token
slots from the block table, and the appended rows are written by one kernel
into the forward batch's input buffers, so a step costs a handful of launches
and one host-to-device copy.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d, get_dtype_size
from vllm.v1.core.hidden_state_record import (
    RECORD_ENCODING_EXPANSION,
    RecordLayout,
    get_record_carrier_group,
    get_record_layers,
    get_record_layout,
    get_record_tail_tokens,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.utils import get_drafter_hidden_states
from vllm.v1.worker.utils import AttentionGroup, get_uniform_decode_token_count

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)

# Fields per carrier layer in HiddenStateHandoff.layer_info.
_LAYER_INFO_FIELDS = 7


def _check_record_layout(
    vllm_config: VllmConfig, model: nn.Module, layout: RecordLayout
) -> None:
    """The scheduler sizes the record from the config; check it against the
    model's actual auxiliary hidden-state layers."""
    if not layout.num_aux:
        return
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        get_eagle3_aux_layers_from_config,
    )

    spec_config = vllm_config.speculative_config
    assert spec_config is not None
    aux_layers = get_eagle3_aux_layers_from_config(spec_config)
    if not aux_layers:
        aux_layers = model.get_eagle3_default_aux_hidden_state_layers()
    if len(aux_layers) != layout.num_aux:
        raise NotImplementedError(
            f"P/D hidden-state handoff assumed {layout.num_aux} auxiliary "
            f"hidden-state layers, but the model uses {len(aux_layers)}."
        )


def _aligned_offsets(sizes: list[int], element_size: int) -> list[int]:
    """Offsets of consecutive segments of ``sizes`` elements, each starting
    16-byte aligned, then the total length. Triton specializes kernels on
    pointer alignment, so unaligned segments would compile new kernel variants
    while serving, as batch sizes vary."""
    step = 16 // element_size
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + -(-size // step) * step)
    return offsets


def _h2d_packed(parts: list[np.ndarray], device: torch.device) -> list[torch.Tensor]:
    """``parts`` as int64 device tensors, copied in one, each 16-byte aligned
    (see ``_aligned_offsets``)."""
    offsets = _aligned_offsets([len(p) for p in parts], 8)
    packed = np.zeros(offsets[-1], dtype=np.int64)
    for part, offset in zip(parts, offsets):
        packed[offset : offset + len(part)] = part
    packed_gpu = async_tensor_h2d(packed, device, torch.int64)
    return [packed_gpu[o : o + len(p)] for p, o in zip(parts, offsets)]


def _cat_rows(fwd: InputBatch | None, name: str, rows_value: Any) -> np.ndarray:
    """The forward batch's per-request field ``name`` (zeros if unset), then
    the sample-only rows' ``rows_value``."""
    if fwd is None:
        return rows_value
    value = getattr(fwd, name)
    if value is None:
        value = np.zeros(fwd.num_reqs, dtype=rows_value.dtype)
    return np.concatenate([np.asarray(value[: fwd.num_reqs]), rows_value])


class HiddenStateHandoff:
    """Writes records on P and samples (and drafts) from them on D."""

    def __init__(self, runner: "GPUModelRunner", kv_caches: dict[str, torch.Tensor]):
        vllm_config = runner.vllm_config
        self.model = runner.model
        self.kv_cache_config = kv_cache_config = runner.kv_cache_config
        self.block_tables = runner.block_tables
        self.req_states = runner.req_states
        self.prompt_logprobs_worker = runner.prompt_logprobs_worker
        self.model_state = runner.model_state
        self.speculator = speculator = runner.speculator
        self.input_buffers = runner.input_buffers
        self.decode_query_len = runner.decode_query_len
        self.max_model_len = runner.max_model_len
        self.device = runner.device
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        # Only a prefiller hands records over.
        self.saves_records = kv_transfer_config.is_kv_producer

        self.layout = layout = get_record_layout(vllm_config)
        _check_record_layout(vllm_config, self.model, layout)
        if (
            isinstance(speculator, DraftModelSpeculator)
            and speculator.supports_mm_inputs
            and layout.window > 1
        ):
            # A one-token drafting window starts past the prompt, so it needs
            # no encoder outputs; a longer one may cover media placeholders.
            raise NotImplementedError(
                "Multimodal multi-module MTP drafters are not supported with "
                "the P/D hidden-state handoff yet."
            )
        self.dtype = vllm_config.model_config.dtype
        self.record_bytes = (
            layout.num_slots * layout.hidden_size * get_dtype_size(self.dtype)
        )

        # The layers holding the record, as byte views [blocks, heads, tokens,
        # bytes per head]. A token slot holds the record bytes, split over the
        # layers, in each head. They are on the prefiller's last pipeline
        # stage, which has the hidden states; earlier stages never touch them.
        self.group_id = group_id = get_record_carrier_group(kv_cache_config)
        self.tail_tokens = get_record_tail_tokens(vllm_config, kv_cache_config)
        record_layers = get_record_layers(kv_cache_config)
        layer_views = (
            [kv_caches[name].view(torch.uint8) for name, _ in record_layers]
            if runner.is_last_pp_rank
            else []
        )
        self.head_bytes = sum(size for _, size in record_layers)
        assert not layer_views or self.head_bytes == sum(
            v.shape[-1] for v in layer_views
        )
        assert (
            self.tail_tokens * self.head_bytes
            >= self.record_bytes * RECORD_ENCODING_EXPANSION
        )
        # The views are in kernel blocks, which split the block table's.
        block_size = self.block_tables.block_sizes[group_id]
        kernel_block_size = runner.kernel_block_sizes[group_id]
        if block_size % kernel_block_size or any(
            v.shape[2] != kernel_block_size for v in layer_views
        ):
            raise NotImplementedError(
                "P/D hidden-state handoff needs one token per KV cache slot."
            )
        self._init_record_slots(
            layer_views,
            self.block_tables.block_tables[group_id].gpu,
            block_size,
            kernel_block_size,
        )
        logger.info(
            "P/D hidden-state records: %d bytes in %d token slot(s) past the "
            "prompt of KV cache group %d",
            self.record_bytes,
            self.tail_tokens,
            group_id,
        )

        # Attention groups of the drafter's own layers, whose metadata the
        # drafter's first step reuses from the target's forward.
        self.draft_attn_groups: list[list[AttentionGroup]] | None = None
        if isinstance(speculator, DraftModelSpeculator):
            draft_layers = speculator.draft_attn_layer_names
            self.draft_attn_groups = [
                [g for g in groups if draft_layers.intersection(g.layer_names)]
                for groups in runner.attn_groups
            ]
        # Whether sample-only rows may be padded to full decode rows for the
        # drafter's full CUDA graph (see _plan_draft): a one-position window,
        # and decode rows the scheduler pads alike (fixed draft counts).
        spec_config = vllm_config.speculative_config
        self.pads_draft_rows = (
            isinstance(speculator, TargetDependentARSpeculator)
            and layout.window == 1
            and runner.dp_size == 1
            and runner.adaptive_verification is None
            and spec_config is not None
            and not spec_config.uses_dynamic_speculative_decoding()
        )
        # Requests sampled from their record this step, in batch order, and
        # their rows (with the loaded records) once read.
        self.sample_only_req_ids: list[str] = []
        self._rows: _SampleOnlyRows | None = None
        # P: records of prompts this step completed, written into the KV cache
        # at the end of the step: (request indices, prompt lengths, records).
        self._pending: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._no_padding = torch.zeros(
            self.input_buffers.max_num_tokens, dtype=torch.bool, device=self.device
        )
        # Zero drafter inputs for the padding tokens of a CUDA graph.
        self._zero_rows = torch.zeros(
            0, layout.hidden_size, dtype=self.dtype, device=self.device
        )

    def _init_record_slots(
        self,
        layer_views: list[torch.Tensor],
        block_table: torch.Tensor,
        block_size: int,
        kernel_block_size: int | None = None,
    ) -> None:
        """What the record kernels need to find a record's token slots: the
        carrier group's block table ``[max_num_reqs, max_num_blocks]``, its
        block size and the layer views' (kernel) block size, and, per layer,
        its byte view's address, block, head and token strides, number of
        heads, bytes per head and their offset in a slot's record bytes."""
        self.layer_views = layer_views
        self.block_table = block_table
        self.block_size = block_size
        self.kernel_block_size = kernel_block_size or block_size
        info, start = [], 0
        for view in layer_views:
            assert view.dtype == torch.uint8 and view.dim() == 4
            assert view.stride(-1) == 1 and view.shape[-1] % 2 == 0
            size = view.shape[-1]
            info.append(
                [view.data_ptr(), *view.stride()[:3], view.shape[1], size, start]
            )
            start += size
        self.layer_info = torch.tensor(
            info, dtype=torch.int64, device=self.device
        ).reshape(-1, _LAYER_INFO_FIELDS)
        self.max_layer_bytes = max((v.shape[-1] for v in layer_views), default=0)

    def _write_records(
        self, req_idx: torch.Tensor, prompt_len: torch.Tensor, records: torch.Tensor
    ) -> None:
        """Store ``records`` ``[num_reqs, record_bytes]`` (bytes) past the
        prompts of the requests at ``req_idx``, in every head of each layer."""
        num_reqs = records.shape[0]
        _write_records_kernel[(num_reqs, self.tail_tokens, len(self.layer_views))](
            records,
            req_idx,
            prompt_len,
            self.block_table,
            self.block_table.stride(0),
            self.layer_info,
            self.record_bytes,
            self.head_bytes,
            self.block_size,
            self.kernel_block_size,
            BLOCK=triton.next_power_of_2(self.max_layer_bytes),
        )

    def _read_records(
        self, req_idx: torch.Tensor, prompt_len: torch.Tensor
    ) -> torch.Tensor:
        """The records ``[num_reqs, num_slots, hidden]`` past the prompts (from
        head 0, which any TP rank holds a copy in)."""
        num_reqs = req_idx.shape[0]
        records = torch.empty(
            num_reqs, self.record_bytes, dtype=torch.uint8, device=self.device
        )
        _read_records_kernel[(num_reqs, self.tail_tokens, len(self.layer_views))](
            records,
            req_idx,
            prompt_len,
            self.block_table,
            self.block_table.stride(0),
            self.layer_info,
            self.record_bytes,
            self.head_bytes,
            self.block_size,
            self.kernel_block_size,
            BLOCK=triton.next_power_of_2(max(self.max_layer_bytes // 2, 1)),
        )
        return records.view(self.dtype).view(
            num_reqs, self.layout.num_slots, self.layout.hidden_size
        )

    def begin_step(self, scheduler_output: SchedulerOutput) -> SchedulerOutput:
        """Take the sample-only rows out of the batch the model runs."""
        req_ids = scheduler_output.hidden_state_record_req_ids
        self._rows = None
        if not req_ids:
            self.sample_only_req_ids = []
            return scheduler_output
        self.sample_only_req_ids = sorted(req_ids)
        num_scheduled_tokens = {
            req_id: n
            for req_id, n in scheduler_output.num_scheduled_tokens.items()
            if req_id not in req_ids
        }
        return replace(
            scheduler_output,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        )

    @property
    def has_sample_only_reqs(self) -> bool:
        return bool(self.sample_only_req_ids)

    def prepare_sampling(
        self,
        input_batch: InputBatch | None,
        hidden_states: torch.Tensor | None,
        aux_hidden_states: list[torch.Tensor] | None,
        has_structured_output_reqs: bool,
    ) -> tuple[InputBatch | None, torch.Tensor | None]:
        """Save the records of prompts this step's forward completed (P), and
        add the rows sampled from transferred records (D). ``input_batch`` is
        None when no forward ran because every row is sample-only."""
        if input_batch is not None and self.saves_records:
            assert hidden_states is not None
            drafter_hidden_states = None
            if self.layout.window:
                drafter_hidden_states = get_drafter_hidden_states(
                    self.model, hidden_states
                )
            self._save(
                input_batch, hidden_states, drafter_hidden_states, aux_hidden_states
            )
        if not self.has_sample_only_reqs:
            return input_batch, hidden_states
        rows = self._load_sample_only_rows(input_batch)
        batch = self._sampling_batch(input_batch, rows, has_structured_output_reqs)
        records = rows.records[:, 0]
        if input_batch is None:
            return batch, records
        assert hidden_states is not None
        return batch, torch.cat([hidden_states[: input_batch.num_tokens], records])

    def prepare_drafting(
        self,
        fwd_batch: InputBatch | None,
        fwd_hidden_states: torch.Tensor | None,
        aux_hidden_states: list[torch.Tensor] | None,
        sampling_batch: InputBatch,
        num_rejected: torch.Tensor,
    ) -> tuple[
        InputBatch,
        torch.Tensor,
        list[torch.Tensor] | None,
        dict[str, Any] | None,
        dict[str, torch.Tensor] | None,
        torch.Tensor,
    ]:
        """The batch, drafter inputs, draft-layer attention metadata (and slot
        mappings) and rejected-token counts to propose from, including the
        sample-only rows, which ran no target forward."""
        layout = self.layout
        if not layout.window:
            # No hidden-state drafter (ngram, draft model): drafting reads only
            # tokens.
            dummy = torch.zeros(
                sampling_batch.num_tokens,
                layout.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
            attn_metadata = slot_mappings_by_layer = None
            if self.draft_attn_groups is not None:
                attn_metadata, slot_mappings_by_layer = self._draft_attn(
                    sampling_batch, CUDAGraphMode.NONE
                )
            return (
                sampling_batch,
                dummy,
                None,
                attn_metadata,
                slot_mappings_by_layer,
                num_rejected,
            )

        rows = self._rows
        assert rows is not None and rows.draft is not None
        draft_batch, num_rejected = self._draft_batch(
            fwd_batch, rows, sampling_batch, num_rejected
        )

        # Per kind of drafter input: the forward's real tokens, the sample-only
        # rows' from their records, then zeros for a CUDA graph's padding.
        plan = rows.draft
        num_fwd_tokens = fwd_batch.num_tokens if fwd_batch is not None else 0
        num_pad = draft_batch.num_tokens_after_padding - draft_batch.num_tokens
        records = rows.records[plan.token_row, plan.token_slot]

        def drafter_inputs(kind: int, fwd: torch.Tensor | None) -> torch.Tensor:
            parts = [records[kind]]
            if fwd is not None:
                parts.insert(0, fwd[:num_fwd_tokens])
            if num_pad:
                parts.append(self._zeros(num_pad))
            return torch.cat(parts)

        drafter_hidden_states = None
        if fwd_hidden_states is not None:
            drafter_hidden_states = get_drafter_hidden_states(
                self.model, fwd_hidden_states
            )
        hidden_states = drafter_inputs(0, drafter_hidden_states)
        aux = None
        if layout.num_aux:
            assert fwd_batch is None or aux_hidden_states is not None
            aux = [
                drafter_inputs(
                    1 + i, aux_hidden_states[i] if aux_hidden_states else None
                )
                for i in range(layout.num_aux)
            ]
        attn_metadata = slot_mappings_by_layer = None
        if self.draft_attn_groups is not None:
            # The drafter's first step reuses the target's attention metadata,
            # so rebuild it for the draft layers over the extended batch.
            attn_metadata, slot_mappings_by_layer = self._draft_attn(
                draft_batch,
                CUDAGraphMode.FULL if plan.full_cudagraph else CUDAGraphMode.NONE,
            )
        return (
            draft_batch,
            hidden_states,
            aux,
            attn_metadata,
            slot_mappings_by_layer,
            num_rejected,
        )

    def _zeros(self, num_rows: int) -> torch.Tensor:
        if self._zero_rows.shape[0] < num_rows:
            self._zero_rows = torch.zeros(
                triton.next_power_of_2(num_rows),
                self.layout.hidden_size,
                dtype=self.dtype,
                device=self.device,
            )
        return self._zero_rows[:num_rows]

    def _draft_attn(
        self, draft_batch: InputBatch, cg_mode: CUDAGraphMode
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        assert self.draft_attn_groups is not None
        block_tables = self.block_tables.gather_block_tables(
            draft_batch.idx_mapping,
            num_reqs_padded=draft_batch.num_reqs_after_padding,
        )
        slot_mappings = self.block_tables.compute_slot_mappings(
            draft_batch.idx_mapping,
            draft_batch.query_start_loc[: draft_batch.num_reqs + 1],
            draft_batch.positions,
            num_tokens_padded=draft_batch.num_tokens_after_padding,
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, self.kv_cache_config
        )
        attn_metadata = self.model_state.prepare_attn(
            draft_batch,
            cg_mode,
            block_tables,
            slot_mappings,
            self.draft_attn_groups,
            self.kv_cache_config,
        )
        return attn_metadata, slot_mappings_by_layer

    # ---------------------------------------------------------------- P side

    def _save(
        self,
        input_batch: InputBatch,
        hidden_states: torch.Tensor,
        drafter_hidden_states: torch.Tensor | None,
        aux_hidden_states: list[torch.Tensor] | None,
    ) -> None:
        """Stage the records of the prompts this step's forward completed."""
        completes_prompt = input_batch.is_prefilling_np & (
            input_batch.num_computed_prefill_tokens_np
            + input_batch.num_scheduled_tokens
            >= input_batch.prefill_len_np
        )
        # Only P/D prefills reserve the record's slots past the prompt.
        num_reqs = input_batch.num_reqs
        num_slots = (
            self.block_tables.num_blocks.np[
                self.group_id, input_batch.idx_mapping_np[:num_reqs]
            ]
            * self.block_size
        )
        completes_prompt[:num_reqs] &= (
            num_slots >= input_batch.prefill_len_np[:num_reqs] + self.tail_tokens
        )
        rows = np.flatnonzero(completes_prompt)
        if rows.size == 0:
            return
        logger.info_once("Saving P/D hidden-state records for completed prompts")
        layout = self.layout
        prompt_len = input_batch.prefill_len_np[rows]
        # Token index of each completed prompt's last position.
        query_start = input_batch.query_start_loc_np[rows]
        last_token = (
            query_start
            + prompt_len
            - 1
            - input_batch.num_computed_prefill_tokens_np[rows]
        )
        # Slot 0's source, then (with a hidden-state drafter) each kind of
        # drafter input, for the trailing window right-aligned in its slots.
        sources = [hidden_states]
        if layout.window:
            # The scheduler keeps the final prefill chunk at least as long as
            # the drafter's lookahead, and prefix-cache hits drop the trailing
            # block.
            first_token = last_token + 1 - np.minimum(prompt_len, layout.window)
            assert (first_token >= query_start).all(), (
                "The final prefill chunk does not cover the drafter's window."
            )
            assert drafter_hidden_states is not None
            sources += [drafter_hidden_states, *(aux_hidden_states or ())]
            assert len(sources) == 2 + layout.num_aux
        n = len(rows)
        # One copy: request indices, prompt lengths, last tokens, and each
        # source's address and row stride in bytes.
        source_info = np.empty((len(sources), 2), dtype=np.int64)
        for i, source in enumerate(sources):
            assert source.dtype == self.dtype
            assert source.shape[-1] == layout.hidden_size and source.stride(-1) == 1
            source_info[i] = source.data_ptr(), source.stride(0) * source.element_size()
        req_idx, prompt_len_gpu, last_token_gpu, sources_gpu = _h2d_packed(
            [
                input_batch.idx_mapping_np[rows],
                prompt_len,
                last_token,
                source_info.reshape(-1),
            ],
            self.device,
        )
        records = torch.empty(
            n, self.record_bytes, dtype=torch.uint8, device=self.device
        )
        slot_bytes = layout.hidden_size * get_dtype_size(self.dtype)
        block = 1024
        _gather_records_kernel[(n, layout.num_slots, triton.cdiv(slot_bytes, block))](
            records,
            prompt_len_gpu,
            last_token_gpu,
            sources_gpu,
            self.record_bytes,
            slot_bytes,
            WINDOW=max(layout.window, 1),
            BLOCK=block,
        )
        # Written into the KV cache at the end of the step (finish_step).
        self._pending = (req_idx, prompt_len_gpu, records)

    def finish_step(self) -> None:
        """P: store the records staged this step, now that the drafter has
        written its own KV past the prompts."""
        if self._pending is not None:
            self._write_records(*self._pending)
            self._pending = None

    # ---------------------------------------------------------------- D side

    def _load_sample_only_rows(self, fwd: InputBatch | None) -> "_SampleOnlyRows":
        """The sample-only rows, with their records (read once per step,
        before the step's drafter writes past the prompts)."""
        req_ids = self.sample_only_req_ids
        req_states = self.req_states
        num_rows = len(req_ids)
        idx_mapping_np = np.fromiter(
            map(req_states.req_id_to_index.__getitem__, req_ids),
            dtype=np.intp,
            count=num_rows,
        )
        # The scheduled token is the last prompt token, already computed by P.
        num_computed_np = req_states.num_computed_tokens_np[idx_mapping_np]
        prefill_len_np = req_states.prefill_len.np[idx_mapping_np]
        assert (num_computed_np + 1 == prefill_len_np).all()
        if self.prompt_logprobs_worker is not None:
            uses = self.prompt_logprobs_worker.uses_prompt_logprobs
            assert not uses[idx_mapping_np].any(), (
                "Prompt logprobs are not supported on a P/D decoder with the "
                "hidden-state handoff."
            )
        logger.info_once("Sampling from transferred P/D hidden-state records")
        plan = self._plan_draft(fwd, prefill_len_np)
        # One copy: request indices and prompt lengths, then the draft plan's
        # per-row token offsets and real (prompt) tokens, and each drafter
        # token's record row and (per kind of input) slot.
        parts = [idx_mapping_np, prefill_len_np]
        if plan is not None:
            parts += [
                plan.cu_num_tokens_np,
                plan.num_real_np,
                plan.token_row_np,
                plan.token_slot_np.reshape(-1),
            ]
        req_idx, prompt_len, *draft_parts = _h2d_packed(parts, self.device)
        if plan is not None:
            plan.cu_num_tokens, plan.num_real, plan.token_row, token_slot = draft_parts
            plan.token_slot = token_slot.view(-1, len(plan.token_row_np))
        self._rows = _SampleOnlyRows(
            req_ids=list(req_ids),
            idx_mapping_np=idx_mapping_np,
            prefill_len_np=prefill_len_np,
            req_idx=req_idx,
            prompt_len=prompt_len,
            records=self._read_records(req_idx, prompt_len),
            draft=plan,
        )
        return self._rows

    def _plan_draft(
        self, fwd: InputBatch | None, prefill_len_np: np.ndarray
    ) -> "_DraftPlan | None":
        """How the drafter's batch covers the sample-only rows: each row's
        ``window`` trailing prompt positions (fewer for shorter prompts), or,
        when that makes the batch a uniform decode with a full CUDA graph, a
        full decode row of its last prompt position and placeholders after it
        (as for a recomputed row padded with placeholder drafts)."""
        layout = self.layout
        if not layout.window or self.speculator is None:
            return None
        num_rows = len(prefill_len_np)
        num_real = np.minimum(prefill_len_np, layout.window)
        num_tokens = num_real
        num_fwd_reqs = fwd.num_reqs if fwd is not None else 0
        num_fwd_tokens = fwd.num_tokens if fwd is not None else 0
        num_reqs = num_fwd_reqs + num_rows
        full_cudagraph = False
        num_tokens_padded = num_reqs_padded = None
        query_len = self.decode_query_len
        cudagraph_manager = getattr(self.speculator, "prefill_cudagraph_manager", None)
        if (
            self.pads_draft_rows
            and query_len > 1
            and cudagraph_manager is not None
            and (
                fwd is None
                or get_uniform_decode_token_count(
                    num_fwd_reqs,
                    num_fwd_tokens,
                    int(fwd.num_scheduled_tokens[:num_fwd_reqs].max()),
                    fwd.decode_graph_eligible,
                )
                == query_len
            )
            # Room for the placeholders and a sampled token, as the scheduler
            # requires to pad a row.
            and int(prefill_len_np.max()) + query_len <= self.max_model_len
        ):
            desc = cudagraph_manager.dispatch(
                num_reqs, num_reqs * query_len, query_len, num_active_loras=0
            )
            if desc.cg_mode == CUDAGraphMode.FULL:
                full_cudagraph = True
                num_tokens = np.full(num_rows, query_len)
                num_tokens_padded = desc.num_tokens
                num_reqs_padded = desc.num_reqs
        cu_num_tokens_np = np.zeros(num_rows + 1, dtype=np.int64)
        np.cumsum(num_tokens, out=cu_num_tokens_np[1:])
        total = int(cu_num_tokens_np[-1])
        # Each drafter token's row and window position: the real tokens are the
        # window's last ones, and placeholders repeat the last.
        token_row_np = np.repeat(np.arange(num_rows), num_tokens)
        local = np.arange(total) - cu_num_tokens_np[token_row_np]
        real = num_real[token_row_np]
        window_pos = layout.window - real + np.minimum(local, real - 1)
        kinds = np.arange(1 + layout.num_aux)
        token_slot_np = 1 + kinds[:, None] * layout.window + window_pos[None, :]
        num_tokens_total = num_fwd_tokens + total
        return _DraftPlan(
            num_tokens_np=num_tokens.astype(np.int32),
            num_real_np=num_real.astype(np.int64),
            cu_num_tokens_np=cu_num_tokens_np,
            token_row_np=token_row_np,
            token_slot_np=token_slot_np,
            full_cudagraph=full_cudagraph,
            num_tokens_padded=num_tokens_padded or num_tokens_total,
            num_reqs_padded=num_reqs_padded or num_reqs,
        )

    def _sampling_batch(
        self,
        fwd: InputBatch | None,
        rows: "_SampleOnlyRows",
        has_structured_output_reqs: bool,
    ) -> InputBatch:
        """The forward batch with the sample-only rows appended, one token and
        logit each, its hidden state from the record."""
        nb = len(rows.req_ids)
        na = fwd.num_reqs if fwd is not None else 0
        num_fwd_tokens = fwd.num_tokens if fwd is not None else 0
        num_fwd_logits = fwd.logits_indices.shape[0] if fwd is not None else 0
        num_reqs = na + nb
        num_tokens = num_fwd_tokens + nb
        num_logits = num_fwd_logits + nb

        # Per-request and per-logit fields: [idx_mapping, query_start_loc,
        # seq_lens, cu_num_logits, expanded_idx_mapping, expanded_local_pos].
        sizes = [num_reqs, num_reqs + 1, num_reqs, num_reqs + 1, num_logits, num_logits]
        offsets = _aligned_offsets(sizes, 4)
        ints = torch.empty(offsets[-1], dtype=torch.int32, device=self.device)
        (
            idx_mapping,
            query_start_loc,
            seq_lens,
            cu_num_logits,
            expanded_idx_mapping,
            expanded_local_pos,
        ) = (ints[o : o + n] for o, n in zip(offsets, sizes))
        logits_indices = torch.empty(num_logits, dtype=torch.int64, device=self.device)
        self._append_rows(
            fwd,
            rows,
            query_start_loc,
            seq_lens,
            num_reqs,
            sampling_outputs=(
                idx_mapping,
                cu_num_logits,
                logits_indices,
                expanded_idx_mapping,
                expanded_local_pos,
            ),
        )

        # Under adaptive verification the forward batch's per-request
        # boundaries are exact only on the GPU (query_start_loc,
        # cu_num_logits); its CPU arrays are upper bounds, and stay so here.
        steps = np.arange(1, nb + 1, dtype=np.int32)
        trues = np.ones(nb, dtype=np.bool_)
        num_computed_np = rows.prefill_len_np - 1
        zero = np.zeros(1, dtype=np.int32)
        query_start_loc_np = np.concatenate(
            [fwd.query_start_loc_np[: na + 1] if fwd is not None else zero, steps]
        )
        query_start_loc_np[na + 1 :] += num_fwd_tokens
        cu_num_logits_np = np.concatenate(
            [fwd.cu_num_logits_np[: na + 1] if fwd is not None else zero, steps]
        )
        cu_num_logits_np[na + 1 :] += cu_num_logits_np[na]
        num_draft_tokens_per_req = None
        if fwd is not None and fwd.num_draft_tokens_per_req is not None:
            num_draft_tokens_per_req = _cat_rows(
                fwd, "num_draft_tokens_per_req", np.zeros(nb, dtype=np.int32)
            )
        max_query_len = None
        if fwd is not None and fwd.max_query_len is not None:
            max_query_len = max(fwd.max_query_len, 1)
        return InputBatch(
            req_ids=(fwd.req_ids[:na] if fwd is not None else []) + rows.req_ids,
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=_cat_rows(fwd, "idx_mapping_np", rows.idx_mapping_np),
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=_cat_rows(
                fwd, "num_scheduled_tokens", np.ones(nb, dtype=np.int32)
            ),
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens,
            num_draft_tokens=fwd.num_draft_tokens if fwd is not None else 0,
            num_draft_tokens_per_req=num_draft_tokens_per_req,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=torch.from_numpy(
                _cat_rows(
                    fwd,
                    "seq_lens_cpu_upper_bound",
                    rows.prefill_len_np.astype(np.int32),
                )
            ),
            dcp_local_seq_lens=None,
            num_computed_tokens_np=_cat_rows(
                fwd, "num_computed_tokens_np", num_computed_np
            ),
            prefill_len_np=_cat_rows(fwd, "prefill_len_np", rows.prefill_len_np),
            num_computed_prefill_tokens_np=_cat_rows(
                fwd, "num_computed_prefill_tokens_np", num_computed_np
            ),
            is_prefilling_np=_cat_rows(fwd, "is_prefilling_np", trues),
            has_prefill=True,
            decode_graph_eligible=False,
            input_ids=self.input_buffers.input_ids[:num_tokens],
            positions=self.input_buffers.positions[:num_tokens],
            is_padding=self._no_padding[:num_tokens],
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=(
                has_structured_output_reqs
                or (fwd is not None and fwd.has_structured_output_reqs)
            ),
            prompt_lens=None,
            prefill_runs_as_decode_np=_cat_rows(
                fwd, "prefill_runs_as_decode_np", trues
            ),
            max_query_len=max_query_len,
        )

    def _draft_batch(
        self,
        fwd: InputBatch | None,
        rows: "_SampleOnlyRows",
        sampling_batch: InputBatch,
        num_rejected: torch.Tensor,
    ) -> tuple[InputBatch, torch.Tensor]:
        """The batch the drafter proposes from: the forward batch plus the
        sample-only rows as planned (``_plan_draft``), laid out in the input
        buffers (which a full CUDA graph reads), and its rejected-token
        counts, which count each row's placeholders."""
        plan = rows.draft
        assert plan is not None
        nb = len(rows.req_ids)
        na = fwd.num_reqs if fwd is not None else 0
        num_fwd_tokens = fwd.num_tokens if fwd is not None else 0
        num_reqs = na + nb
        num_reqs_padded = plan.num_reqs_padded
        num_tokens = num_fwd_tokens + int(plan.cu_num_tokens_np[-1])
        num_tokens_padded = plan.num_tokens_padded
        buffers = self.input_buffers
        query_start_loc = buffers.query_start_loc[: num_reqs_padded + 1]
        seq_lens = buffers.seq_lens[:num_reqs_padded]
        draft_num_rejected = torch.empty_like(num_rejected)
        self._append_rows(
            fwd,
            rows,
            query_start_loc,
            seq_lens,
            num_reqs_padded,
            rejected=(num_rejected, draft_num_rejected),
        )

        num_real_np = plan.num_real_np
        so_num_computed = rows.prefill_len_np - num_real_np
        query_start_loc_np = np.full(num_reqs_padded + 1, num_tokens, dtype=np.int32)
        query_start_loc_np[: na + 1] = (
            fwd.query_start_loc_np[: na + 1] if fwd is not None else 0
        )
        query_start_loc_np[na + 1 : num_reqs + 1] = (
            num_fwd_tokens + plan.cu_num_tokens_np[1:]
        )
        seq_lens_np = np.zeros(num_reqs_padded, dtype=np.int32)
        seq_lens_np[na:num_reqs] = so_num_computed + plan.num_tokens_np
        if fwd is not None:
            seq_lens_np[:na] = fwd.seq_lens_cpu_upper_bound[:na].numpy()
        num_placeholders = plan.num_tokens_np - num_real_np.astype(np.int32)
        so_runs_as_decode = (num_real_np == 1) & (so_num_computed > 0)
        draft_batch = replace(
            sampling_batch,
            num_reqs_after_padding=num_reqs_padded,
            num_scheduled_tokens=_cat_rows(
                fwd, "num_scheduled_tokens", plan.num_tokens_np
            ),
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_padded,
            num_draft_tokens=(fwd.num_draft_tokens if fwd is not None else 0)
            + int(num_placeholders.sum()),
            num_draft_tokens_per_req=_cat_rows(
                fwd, "num_draft_tokens_per_req", num_placeholders
            ),
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_np),
            num_computed_tokens_np=_cat_rows(
                fwd, "num_computed_tokens_np", so_num_computed
            ),
            num_computed_prefill_tokens_np=_cat_rows(
                fwd, "num_computed_prefill_tokens_np", so_num_computed
            ),
            decode_graph_eligible=plan.full_cudagraph,
            input_ids=buffers.input_ids[:num_tokens_padded],
            positions=buffers.positions[:num_tokens_padded],
            is_padding=self._no_padding[:num_tokens_padded],
            prefill_runs_as_decode_np=_cat_rows(
                fwd, "prefill_runs_as_decode_np", so_runs_as_decode
            ),
            max_query_len=self.decode_query_len if plan.full_cudagraph else None,
        )
        return draft_batch, draft_num_rejected

    def _append_rows(
        self,
        fwd: InputBatch | None,
        rows: "_SampleOnlyRows",
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        num_reqs_padded: int,
        sampling_outputs: tuple[torch.Tensor, ...] | None = None,
        rejected: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Write the sample-only rows after the forward batch's: their tokens
        into the input buffers after its real tokens, and the per-request (and
        for sampling, per-logit) fields into the given tensors after a copy of
        its own. One token per row for sampling, else as the draft plan says.
        The rows past them up to ``num_reqs_padded`` are a CUDA graph's
        padding."""
        num_rows = len(rows.req_ids)
        sampling = sampling_outputs is not None
        draft = rows.draft
        # Placeholder tensors for the inputs a launch does not read.
        unused = rows.req_idx
        if fwd is None:
            num_fwd_reqs = num_fwd_tokens = num_fwd_logits = 0
            fwd_inputs: tuple[torch.Tensor, ...] = (unused,) * 7
        else:
            num_fwd_reqs, num_fwd_tokens = fwd.num_reqs, fwd.num_tokens
            num_fwd_logits = fwd.logits_indices.shape[0]
            fwd_inputs = (
                fwd.query_start_loc,
                fwd.seq_lens,
                fwd.idx_mapping,
                fwd.cu_num_logits,
                fwd.logits_indices,
                fwd.expanded_idx_mapping,
                fwd.expanded_local_pos,
            )
        if sampling:
            max_tokens = 1
            cu_num_tokens = num_real = unused
        else:
            assert draft is not None and rejected is not None
            max_tokens = int(draft.num_tokens_np.max())
            cu_num_tokens, num_real = draft.cu_num_tokens, draft.num_real
        block = 1024
        num_copy_programs = triton.cdiv(max(num_fwd_reqs + 1, num_fwd_logits), block)
        all_token_ids = self.req_states.all_token_ids.gpu
        _append_rows_kernel[(num_copy_programs + num_rows,)](
            rows.req_idx,
            rows.prompt_len,
            cu_num_tokens,
            num_real,
            all_token_ids,
            all_token_ids.stride(0),
            self.input_buffers.input_ids,
            self.input_buffers.positions,
            num_rows,
            num_fwd_reqs,
            num_fwd_tokens,
            num_fwd_logits,
            num_reqs_padded,
            num_copy_programs,
            *fwd_inputs,
            query_start_loc,
            seq_lens,
            *(sampling_outputs or (unused,) * 5),
            *(rejected or (unused, unused)),
            HAS_FWD=fwd is not None,
            SAMPLING=sampling,
            BLOCK=block,
            TOKEN_BLOCK=triton.next_power_of_2(max(max_tokens, 2)),
        )


@dataclass
class _DraftPlan:
    """How the drafter's batch covers the sample-only rows (``_plan_draft``):
    per row, its tokens and how many of them are real (the trailing prompt
    positions, then placeholders), and per drafter token its record row and,
    per kind of drafter input, record slot. The tensors are the arrays' device
    copies."""

    num_tokens_np: np.ndarray
    num_real_np: np.ndarray
    cu_num_tokens_np: np.ndarray
    token_row_np: np.ndarray
    # [1 + num_aux, num_tokens]
    token_slot_np: np.ndarray
    # A uniform decode batch replaying the drafter's full CUDA graph.
    full_cudagraph: bool
    num_tokens_padded: int
    num_reqs_padded: int
    cu_num_tokens: torch.Tensor | None = None
    num_real: torch.Tensor | None = None
    token_row: torch.Tensor | None = None
    token_slot: torch.Tensor | None = None


@dataclass
class _SampleOnlyRows:
    req_ids: list[str]
    idx_mapping_np: np.ndarray
    prefill_len_np: np.ndarray
    # Device copies (int64) of the request indices and prompt lengths.
    req_idx: torch.Tensor
    prompt_len: torch.Tensor
    # [num_reqs, num_slots, hidden_size]
    records: torch.Tensor
    # None without a hidden-state drafter.
    draft: _DraftPlan | None


@triton.jit
def _record_slot(
    req_idx_ptr,
    prompt_len_ptr,
    block_table_ptr,
    block_table_stride,
    row,
    token,
    block_size,
    kernel_block_size,
):
    """The kernel block and in-block offset of a record's token slot: kernel
    block j of block b is b * (block_size // kernel_block_size) + j."""
    req = tl.load(req_idx_ptr + row)
    pos = tl.load(prompt_len_ptr + row) + token
    block = tl.load(block_table_ptr + req * block_table_stride + pos // block_size)
    kernel_block = (
        block.to(tl.int64) * (block_size // kernel_block_size)
        + (pos % block_size) // kernel_block_size
    )
    return kernel_block, pos % kernel_block_size


@triton.jit
def _gather_records_kernel(
    records_ptr,  # uint8 [num_reqs, record_bytes]
    prompt_len_ptr,  # [num_reqs]
    last_token_ptr,  # [num_reqs] token index of the last prompt position
    sources_ptr,  # [num_sources, 2] address and row stride in bytes
    record_bytes,
    slot_bytes,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy each record slot's hidden state from its source: slot 0 from the
    model output at the last prompt position, then each kind of drafter input
    at the window's positions (zeros before the prompt's start)."""
    row = tl.program_id(0)
    slot = tl.program_id(1)
    chunk = tl.program_id(2)
    source = tl.where(slot == 0, 0, 1 + (slot - 1) // WINDOW)
    token_offset = tl.where(slot == 0, 0, (slot - 1) % WINDOW - (WINDOW - 1))
    valid = tl.load(prompt_len_ptr + row) - 1 + token_offset >= 0
    token = tl.load(last_token_ptr + row) + token_offset
    src = tl.cast(tl.load(sources_ptr + 2 * source), tl.pointer_type(tl.uint8))
    src_stride = tl.load(sources_ptr + 2 * source + 1)
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < slot_bytes
    data = tl.load(src + token * src_stride + offsets, mask=mask & valid, other=0)
    tl.store(
        records_ptr + row * record_bytes + slot * slot_bytes + offsets, data, mask=mask
    )


@triton.jit
def _write_records_kernel(
    records_ptr,  # uint8 [num_reqs, record_bytes]
    req_idx_ptr,  # [num_reqs]
    prompt_len_ptr,  # [num_reqs]
    block_table_ptr,
    block_table_stride,
    layer_info_ptr,  # [num_layers, 7]
    record_bytes,
    head_bytes,
    block_size,
    kernel_block_size,
    BLOCK: tl.constexpr,
):
    """Store one layer's share of one token slot of a record, one nibble per
    byte, in every head."""
    row = tl.program_id(0)
    token = tl.program_id(1)
    layer = tl.program_id(2)
    block, offset = _record_slot(
        req_idx_ptr,
        prompt_len_ptr,
        block_table_ptr,
        block_table_stride,
        row,
        token,
        block_size,
        kernel_block_size,
    )
    info = layer_info_ptr + layer * 7
    base = tl.cast(tl.load(info), tl.pointer_type(tl.uint8))
    block_stride = tl.load(info + 1)
    head_stride = tl.load(info + 2)
    token_stride = tl.load(info + 3)
    num_heads = tl.load(info + 4)
    size = tl.load(info + 5)
    start = tl.load(info + 6)
    cols = tl.arange(0, BLOCK)
    mask = cols < size
    # The stored byte's index in the record's encoding, and its source byte.
    encoded = token * head_bytes + start + cols
    src = encoded // 2
    data = tl.load(
        records_ptr + row * record_bytes + src,
        mask=mask & (src < record_bytes),
        other=0,
    )
    nibble = tl.where(encoded % 2 == 0, data & 0xF, data >> 4)
    dst = base + block * block_stride + offset * token_stride + cols
    for head in range(num_heads):
        tl.store(dst + head * head_stride, nibble, mask=mask)


@triton.jit
def _read_records_kernel(
    records_ptr,  # uint8 [num_reqs, record_bytes]
    req_idx_ptr,  # [num_reqs]
    prompt_len_ptr,  # [num_reqs]
    block_table_ptr,
    block_table_stride,
    layer_info_ptr,  # [num_layers, 7]
    record_bytes,
    head_bytes,
    block_size,
    kernel_block_size,
    BLOCK: tl.constexpr,
):
    """Decode one layer's share of one token slot of a record, from head 0."""
    row = tl.program_id(0)
    token = tl.program_id(1)
    layer = tl.program_id(2)
    block, offset = _record_slot(
        req_idx_ptr,
        prompt_len_ptr,
        block_table_ptr,
        block_table_stride,
        row,
        token,
        block_size,
        kernel_block_size,
    )
    info = layer_info_ptr + layer * 7
    base = tl.cast(tl.load(info), tl.pointer_type(tl.uint8))
    block_stride = tl.load(info + 1)
    token_stride = tl.load(info + 3)
    size = tl.load(info + 5)
    start = tl.load(info + 6)
    # Pairs of stored bytes (low nibble, high nibble) of one record byte; the
    # layers' shares start at even offsets.
    pairs = tl.arange(0, BLOCK)
    dst = (token * head_bytes + start) // 2 + pairs
    mask = (pairs < size // 2) & (dst < record_bytes)
    src = base + block * block_stride + offset * token_stride + 2 * pairs
    low = tl.load(src, mask=mask, other=0)
    high = tl.load(src + 1, mask=mask, other=0)
    tl.store(
        records_ptr + row * record_bytes + dst, (low & 0xF) | (high << 4), mask=mask
    )


# Batch sizes vary step to step: specializing on them would compile new
# variants during serving.
@triton.jit(
    do_not_specialize=[
        "num_rows",
        "num_fwd_reqs",
        "num_fwd_tokens",
        "num_fwd_logits",
        "num_reqs_padded",
        "num_copy_programs",
    ]
)
def _append_rows_kernel(
    req_idx_ptr,  # [num_rows]
    prompt_len_ptr,  # [num_rows]
    cu_num_tokens_ptr,  # [num_rows + 1] (unless SAMPLING)
    num_real_ptr,  # [num_rows] (unless SAMPLING)
    all_token_ids_ptr,
    all_token_ids_stride,
    input_ids_ptr,
    positions_ptr,
    num_rows,
    num_fwd_reqs,
    num_fwd_tokens,
    num_fwd_logits,
    num_reqs_padded,
    num_copy_programs,
    fwd_query_start_loc_ptr,
    fwd_seq_lens_ptr,
    fwd_idx_mapping_ptr,
    fwd_cu_num_logits_ptr,
    fwd_logits_indices_ptr,
    fwd_expanded_idx_ptr,
    fwd_expanded_pos_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    idx_mapping_ptr,  # (SAMPLING, as are the next four)
    cu_num_logits_ptr,
    logits_indices_ptr,
    expanded_idx_ptr,
    expanded_pos_ptr,
    num_rejected_ptr,  # (unless SAMPLING)
    draft_num_rejected_ptr,
    HAS_FWD: tl.constexpr,
    # Sampling batch (one token per row), else draft batch.
    SAMPLING: tl.constexpr,
    BLOCK: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < num_copy_programs:
        # Copy the forward batch's fields.
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        reqs = offsets < num_fwd_reqs
        cum = offsets <= num_fwd_reqs
        logits = offsets < num_fwd_logits
        if HAS_FWD:
            x = tl.load(fwd_query_start_loc_ptr + offsets, mask=cum)
            tl.store(query_start_loc_ptr + offsets, x, mask=cum)
            x = tl.load(fwd_seq_lens_ptr + offsets, mask=reqs)
            tl.store(seq_lens_ptr + offsets, x, mask=reqs)
            if SAMPLING:
                x = tl.load(fwd_idx_mapping_ptr + offsets, mask=reqs)
                tl.store(idx_mapping_ptr + offsets, x, mask=reqs)
                x = tl.load(fwd_cu_num_logits_ptr + offsets, mask=cum)
                tl.store(cu_num_logits_ptr + offsets, x, mask=cum)
                x = tl.load(fwd_logits_indices_ptr + offsets, mask=logits)
                tl.store(logits_indices_ptr + offsets, x, mask=logits)
                x = tl.load(fwd_expanded_idx_ptr + offsets, mask=logits)
                tl.store(expanded_idx_ptr + offsets, x, mask=logits)
                x = tl.load(fwd_expanded_pos_ptr + offsets, mask=logits)
                tl.store(expanded_pos_ptr + offsets, x, mask=logits)
            else:
                x = tl.load(num_rejected_ptr + offsets, mask=reqs)
                tl.store(draft_num_rejected_ptr + offsets, x, mask=reqs)
        elif pid == 0:
            tl.store(query_start_loc_ptr, 0)
            if SAMPLING:
                tl.store(cu_num_logits_ptr, 0)
        if pid == 0:
            # A full CUDA graph's padding requests: empty, past the tokens.
            if SAMPLING:
                num_tokens = num_fwd_tokens + num_rows
            else:
                num_tokens = num_fwd_tokens + tl.load(cu_num_tokens_ptr + num_rows)
            for i in range(num_fwd_reqs + num_rows, num_reqs_padded, BLOCK):
                pad = i + tl.arange(0, BLOCK)
                mask = pad < num_reqs_padded
                tl.store(query_start_loc_ptr + pad + 1, num_tokens, mask=mask)
                tl.store(seq_lens_ptr + pad, 0, mask=mask)
        return

    # A sample-only row: its tokens are the trailing num_real prompt
    # positions, then placeholders (past the prompt, token id 0).
    i = pid - num_copy_programs
    req = tl.load(req_idx_ptr + i)
    prompt_len = tl.load(prompt_len_ptr + i)
    if SAMPLING:
        start = i
        end = i + 1
        num_real = 1
    else:
        start = tl.load(cu_num_tokens_ptr + i)
        end = tl.load(cu_num_tokens_ptr + i + 1)
        num_real = tl.load(num_real_ptr + i)
    j = tl.arange(0, TOKEN_BLOCK)
    mask = j < end - start
    pos = prompt_len - num_real + j
    token_ids = tl.load(
        all_token_ids_ptr + req * all_token_ids_stride + pos,
        mask=mask & (pos < prompt_len),
        other=0,
    )
    tokens = num_fwd_tokens + start + j
    tl.store(input_ids_ptr + tokens, token_ids, mask=mask)
    tl.store(positions_ptr + tokens, pos, mask=mask)
    r = num_fwd_reqs + i
    tl.store(query_start_loc_ptr + r + 1, num_fwd_tokens + end)
    tl.store(seq_lens_ptr + r, prompt_len - num_real + end - start)
    if SAMPLING:
        tl.store(idx_mapping_ptr + r, req)
        tl.store(cu_num_logits_ptr + r + 1, num_fwd_logits + i + 1)
        logit = num_fwd_logits + i
        tl.store(logits_indices_ptr + logit, num_fwd_tokens + start + num_real - 1)
        tl.store(expanded_idx_ptr + logit, req)
        tl.store(expanded_pos_ptr + logit, 0)
    else:
        # Placeholders are rejected, like a recomputed row's placeholder
        # drafts.
        rejected = tl.load(num_rejected_ptr + r)
        tl.store(draft_num_rejected_ptr + r, rejected + end - start - num_real)
