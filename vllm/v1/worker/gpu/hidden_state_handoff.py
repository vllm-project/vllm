# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P/D hidden-state handoff (prototype).

The prefiller (P) computes every prompt token and stores what the decoder needs
from the last prompt positions in a per-request "record": a one-block cache
group (``HiddenStateRecordSpec``) that the KV connector transfers with the
request's other blocks. The decoder (D) loads all prompt tokens plus the record
and samples the first output token from it, without running a forward pass for
any prompt position.

The scheduler still accounts for the last prompt token as a one-token step
(``SchedulerOutput.hidden_state_record_req_ids``), so the request's computed
token count, sampling position (seeds) and token budget match a recompute. The
model runner drops those rows from the forward batch and appends them, one
token each, to the batch it samples from, with their hidden states read from
the record.

Record layout, ``[num_slots, hidden_size]`` (see ``RecordLayout``):

* slot 0: the model output at the last prompt position (sampling).
* With a hidden-state drafter (EAGLE, EAGLE3, MTP, DFlash, DSpark), the
  drafter's inputs for the last ``window`` prompt positions: the target hidden
  states handed to the drafter, then each auxiliary layer's hidden states.
  ``window`` is the drafter's prefill lookahead: 1, or k for k-module MTP.
  Token-only drafters (ngram, draft model) need no drafter inputs.

The drafter runs on a batch where each sample-only row covers those ``window``
positions, so it rewrites its own KV for them with D's sampled token exactly as
a local final prefill chunk would (P wrote them with its own sampled token and
drafts).

The sampling batch reuses the forward batch's tokens and appends one token per
sample-only row. Prompt logprobs are assumed never to be requested on D.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import HiddenStateRecordSpec
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.gpu.spec_decode.utils import get_drafter_hidden_states
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)

# Name of the pseudo layer that owns the record. It is not a module in the
# forward context; the model runner reads and writes the cache directly.
HIDDEN_STATE_RECORD_LAYER = "hidden_state_record.0"


@dataclass(frozen=True)
class RecordLayout:
    hidden_size: int
    # Prompt positions of drafter inputs stored (0: no hidden-state drafter).
    window: int = 0
    # EAGLE3 auxiliary hidden states per position.
    num_aux: int = 0

    @property
    def num_slots(self) -> int:
        return 1 + self.window * (1 + self.num_aux)

    def drafter_slots(self, kind: int) -> slice:
        """Slots of the drafter input ``kind`` (0: target hidden states,
        1 + i: auxiliary layer i), ordered by position, last position last."""
        start = 1 + kind * self.window
        return slice(start, start + self.window)


def get_record_layout(vllm_config: VllmConfig, model: nn.Module) -> RecordLayout:
    hidden_size = vllm_config.model_config.get_hidden_size()
    spec_config = vllm_config.speculative_config
    if spec_config is None or spec_config.use_ngram() or spec_config.uses_draft_model():
        # The drafter reads only token ids (its own KV comes with the transfer).
        return RecordLayout(hidden_size)
    # Hidden-state drafters: EAGLE, EAGLE3, MTP (incl. multi-module and Gemma4),
    # DFlash (incl. DFlash2/LiLiCorr) and DSpark.
    if not spec_config.use_eagle():
        raise NotImplementedError(
            "P/D hidden-state handoff does not support "
            f"{spec_config.method!r} speculative decoding."
        )
    num_aux = 0
    if spec_config.method in ("eagle3", "dflash", "dspark"):
        # The target model outputs auxiliary hidden states for these drafters
        # (see GPUModelRunner.use_aux_hidden_state_outputs).
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            get_eagle3_aux_layers_from_config,
        )

        aux_layers = get_eagle3_aux_layers_from_config(spec_config)
        if not aux_layers:
            aux_layers = model.get_eagle3_default_aux_hidden_state_layers()
        num_aux = len(aux_layers)
    window = max(1, vllm_config.num_prefill_lookahead_tokens)
    return RecordLayout(hidden_size, window, num_aux)


def get_hidden_state_record_spec(
    vllm_config: VllmConfig, model: nn.Module
) -> HiddenStateRecordSpec | None:
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None or not kv_transfer_config.hidden_state_handoff:
        return None
    parallel_config = vllm_config.parallel_config
    unsupported = {
        "pipeline parallelism": parallel_config.pipeline_parallel_size > 1,
        "context parallelism": (
            parallel_config.decode_context_parallel_size > 1
            or parallel_config.prefill_context_parallel_size > 1
        ),
        "batch-sharded sampling": parallel_config.enable_batch_sharded_sampling,
        "pooling models": vllm_config.model_config.runner_type == "pooling",
    }
    if names := [name for name, on in unsupported.items() if on]:
        raise NotImplementedError(
            f"P/D hidden-state handoff does not support {', '.join(names)} yet."
        )
    layout = get_record_layout(vllm_config, model)
    return HiddenStateRecordSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=layout.num_slots * layout.hidden_size,
        head_size_v=0,
        dtype=vllm_config.model_config.dtype,
    )


class HiddenStateHandoff:
    """Writes records on P and samples (and drafts) from them on D."""

    def __init__(self, runner: "GPUModelRunner", kv_caches: dict[str, torch.Tensor]):
        self.model = runner.model
        self.kv_cache_config = runner.kv_cache_config
        self.block_tables = runner.block_tables
        self.req_states = runner.req_states
        self.prompt_logprobs_worker = runner.prompt_logprobs_worker
        self.model_state = runner.model_state
        self.speculator = speculator = runner.speculator
        self.device = runner.device

        self.layout = get_record_layout(runner.vllm_config, self.model)
        if (
            isinstance(speculator, DraftModelSpeculator)
            and speculator.supports_mm_inputs
            and self.layout.window > 1
        ):
            # A one-token drafting window starts past the prompt, so it needs
            # no encoder outputs; a longer one may cover media placeholders.
            raise NotImplementedError(
                "Multimodal multi-module MTP drafters are not supported with "
                "the P/D hidden-state handoff yet."
            )
        group_id = next(
            i
            for i, group in enumerate(self.kv_cache_config.kv_cache_groups)
            if HIDDEN_STATE_RECORD_LAYER in group.layer_names
        )
        assert self.block_tables.blocks_per_kv_block[group_id] == 1
        # [num_blocks, 1, 1, num_slots * hidden_size]
        # -> [num_blocks, num_slots, hidden_size]
        self.records = kv_caches[HIDDEN_STATE_RECORD_LAYER][:, 0, 0].unflatten(
            -1, (self.layout.num_slots, self.layout.hidden_size)
        )
        # [max_num_reqs, 1]: the record's block id per request slot.
        self.block_table = self.block_tables.block_tables[group_id].gpu
        # Attention groups of the drafter's own layers, whose metadata the
        # drafter's first step reuses from the target's forward.
        self.draft_attn_groups: list[list[AttentionGroup]] | None = None
        if isinstance(speculator, DraftModelSpeculator):
            draft_layers = speculator.draft_attn_layer_names
            self.draft_attn_groups = [
                [g for g in groups if draft_layers.intersection(g.layer_names)]
                for groups in runner.attn_groups
            ]
        # Requests sampled from their record this step, in batch order.
        self.sample_only_req_ids: list[str] = []

    def _h2d(self, x: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return async_tensor_h2d(x, self.device, dtype)

    def _record_block_ids(self, idx_mapping: torch.Tensor) -> torch.Tensor:
        return self.block_table[idx_mapping.long(), 0].long()

    def begin_step(self, scheduler_output: SchedulerOutput) -> SchedulerOutput:
        """Take the sample-only rows out of the batch the model runs."""
        req_ids = scheduler_output.hidden_state_record_req_ids
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
        if input_batch is not None:
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
        return self._merge_for_sampling(
            input_batch, hidden_states, has_structured_output_reqs
        )

    def prepare_drafting(
        self,
        fwd_batch: InputBatch | None,
        fwd_hidden_states: torch.Tensor | None,
        aux_hidden_states: list[torch.Tensor] | None,
        sampling_batch: InputBatch,
    ) -> tuple[
        InputBatch,
        torch.Tensor,
        list[torch.Tensor] | None,
        dict[str, Any] | None,
        dict[str, torch.Tensor] | None,
    ]:
        """The batch, drafter inputs and draft-layer attention metadata (and
        slot mappings) to propose from, including the sample-only rows, which
        ran no target forward."""
        drafter_hidden_states = None
        if fwd_hidden_states is not None:
            drafter_hidden_states = get_drafter_hidden_states(
                self.model, fwd_hidden_states
            )
        draft_batch, hidden_states, aux_hidden_states = self._merge_for_drafting(
            fwd_batch, drafter_hidden_states, aux_hidden_states, sampling_batch
        )
        attn_metadata = slot_mappings_by_layer = None
        if self.draft_attn_groups is not None:
            # The drafter's first step reuses the target's attention metadata,
            # so rebuild it for the draft layers over the extended batch.
            num_reqs = draft_batch.num_reqs
            block_tables = self.block_tables.gather_block_tables(
                draft_batch.idx_mapping, num_reqs_padded=num_reqs
            )
            slot_mappings = self.block_tables.compute_slot_mappings(
                draft_batch.idx_mapping,
                draft_batch.query_start_loc,
                draft_batch.positions,
                num_tokens_padded=draft_batch.num_tokens,
            )
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            attn_metadata = self.model_state.prepare_attn(
                draft_batch,
                CUDAGraphMode.NONE,
                block_tables,
                slot_mappings,
                self.draft_attn_groups,
                self.kv_cache_config,
            )
        return (
            draft_batch,
            hidden_states,
            aux_hidden_states,
            attn_metadata,
            slot_mappings_by_layer,
        )

    # ---------------------------------------------------------------- P side

    def _save(
        self,
        input_batch: InputBatch,
        hidden_states: torch.Tensor,
        drafter_hidden_states: torch.Tensor | None,
        aux_hidden_states: list[torch.Tensor] | None,
    ) -> None:
        """Store the records of the prompts this step's forward completed."""
        completes_prompt = input_batch.is_prefilling_np & (
            input_batch.num_computed_prefill_tokens_np
            + input_batch.num_scheduled_tokens
            >= input_batch.prefill_len_np
        )
        rows = np.flatnonzero(completes_prompt)
        if rows.size == 0:
            return
        logger.info_once("Saving P/D hidden-state records for completed prompts")
        layout = self.layout
        # Token index of each completed prompt's last position.
        query_start = input_batch.query_start_loc_np[rows]
        last_token = (
            query_start
            + input_batch.prefill_len_np[rows]
            - 1
            - input_batch.num_computed_prefill_tokens_np[rows]
        )
        block_ids = self._record_block_ids(
            self._h2d(input_batch.idx_mapping_np[rows], torch.int64)
        )
        self.records[block_ids, 0] = hidden_states[self._h2d(last_token, torch.int64)]
        if not layout.window:
            return

        # Drafter inputs for the trailing window, right-aligned in the slots.
        # Shorter prompts leave the leading slots unused.
        offsets = np.arange(layout.window) - (layout.window - 1)
        token_idx = last_token[:, None] + offsets[None, :]
        prompt_pos = input_batch.prefill_len_np[rows, None] - 1 + offsets[None, :]
        valid = prompt_pos >= 0
        # The scheduler keeps the final prefill chunk at least as long as the
        # drafter's lookahead, and prefix-cache hits drop the trailing block.
        assert (token_idx[valid] >= np.repeat(query_start, valid.sum(1))).all(), (
            "The final prefill chunk does not cover the drafter's window."
        )
        row_sel, win_sel = np.nonzero(valid)
        src = self._h2d(token_idx[row_sel, win_sel], torch.int64)
        dst_block = block_ids[self._h2d(row_sel, torch.int64)]
        win_sel_gpu = self._h2d(win_sel, torch.int64)
        assert drafter_hidden_states is not None
        sources = [drafter_hidden_states, *(aux_hidden_states or ())]
        assert len(sources) == 1 + layout.num_aux
        for kind, source in enumerate(sources):
            assert source.shape[-1] == layout.hidden_size
            first_slot = layout.drafter_slots(kind).start
            self.records[dst_block, first_slot + win_sel_gpu] = source[src]

    # ---------------------------------------------------------------- D side

    def _merge_for_sampling(
        self,
        input_batch: InputBatch | None,
        hidden_states: torch.Tensor | None,
        has_structured_output_reqs: bool,
    ) -> tuple[InputBatch, torch.Tensor]:
        """Append the sample-only rows, one token each with the hidden state
        from the record, to the forward batch (None when every scheduled row is
        sample-only)."""
        rows = self._sample_only_rows()
        records = self.records[rows.block_ids, 0]
        batch = _make_batch(
            rows, window=1, has_structured_output_reqs=has_structured_output_reqs
        )
        if self.prompt_logprobs_worker is not None:
            uses = self.prompt_logprobs_worker.uses_prompt_logprobs
            assert not uses[rows.idx_mapping_np].any(), (
                "Prompt logprobs are not supported on a P/D decoder with the "
                "hidden-state handoff."
            )
        if input_batch is None:
            return batch, records
        assert hidden_states is not None
        return _concat(input_batch, batch), _concat_tokens(
            input_batch, hidden_states, records
        )

    def _merge_for_drafting(
        self,
        input_batch: InputBatch | None,
        drafter_hidden_states: torch.Tensor | None,
        aux_hidden_states: list[torch.Tensor] | None,
        sampling_batch: InputBatch,
    ) -> tuple[InputBatch, torch.Tensor, list[torch.Tensor] | None]:
        """The batch the drafter proposes from: the forward batch plus each
        sample-only row's trailing ``window`` prompt positions, with the
        drafter inputs from the record."""
        layout = self.layout
        if not layout.window:
            # No hidden-state drafter (ngram): drafting reads only tokens.
            assert drafter_hidden_states is None or input_batch is not None
            dummy = torch.zeros(
                sampling_batch.num_tokens,
                layout.hidden_size,
                dtype=self.records.dtype,
                device=self.device,
            )
            return sampling_batch, dummy, None
        rows = self._sample_only_rows()
        batch = _make_batch(
            rows,
            window=layout.window,
            has_structured_output_reqs=sampling_batch.has_structured_output_reqs,
        )
        # Per sample-only row, its window's slots (last position last).
        num_tokens_np = batch.num_scheduled_tokens
        slot_offsets = np.concatenate(
            [np.arange(layout.window - n, layout.window) for n in num_tokens_np]
        )
        row_idx = np.repeat(np.arange(len(num_tokens_np)), num_tokens_np)
        block_ids = rows.block_ids[self._h2d(row_idx, torch.int64)]
        slot_offsets_gpu = self._h2d(slot_offsets, torch.int64)

        def gather(kind: int) -> torch.Tensor:
            first = layout.drafter_slots(kind).start
            return self.records[block_ids, first + slot_offsets_gpu]

        hidden = gather(0)
        aux = [gather(1 + i) for i in range(layout.num_aux)] or None
        if input_batch is None:
            return batch, hidden, aux
        assert drafter_hidden_states is not None
        hidden = _concat_tokens(input_batch, drafter_hidden_states, hidden)
        if aux is not None:
            assert aux_hidden_states is not None
            aux = [
                _concat_tokens(input_batch, a, b)
                for a, b in zip(aux_hidden_states, aux)
            ]
        return _concat(input_batch, batch), hidden, aux

    def _sample_only_rows(self) -> "_SampleOnlyRows":
        req_ids = self.sample_only_req_ids
        req_states = self.req_states
        idx_mapping_np = np.fromiter(
            map(req_states.req_id_to_index.__getitem__, req_ids),
            dtype=np.intp,
            count=len(req_ids),
        )
        # The scheduled token is the last prompt token, already computed by P.
        num_computed_np = req_states.num_computed_tokens_np[idx_mapping_np]
        prefill_len_np = req_states.prefill_len.np[idx_mapping_np]
        assert (num_computed_np + 1 == prefill_len_np).all()
        idx_mapping = self._h2d(idx_mapping_np, torch.int32)
        logger.info_once("Sampling from transferred P/D hidden-state records")
        return _SampleOnlyRows(
            req_ids=list(req_ids),
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            prefill_len_np=prefill_len_np,
            block_ids=self._record_block_ids(idx_mapping),
            all_token_ids=req_states.all_token_ids.gpu,
        )


@dataclass
class _SampleOnlyRows:
    req_ids: list[str]
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray
    prefill_len_np: np.ndarray
    block_ids: torch.Tensor
    all_token_ids: torch.Tensor


def _make_batch(
    rows: _SampleOnlyRows, window: int, has_structured_output_reqs: bool
) -> InputBatch:
    """A batch of the sample-only rows, each covering its last ``window``
    prompt positions (fewer for shorter prompts) and producing one logit."""
    num_reqs = len(rows.req_ids)
    device = rows.idx_mapping.device
    seq_lens_np = rows.prefill_len_np.astype(np.int32)
    num_tokens_np = np.minimum(seq_lens_np, window).astype(np.int32)
    query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(num_tokens_np, out=query_start_loc_np[1:])
    num_tokens = int(query_start_loc_np[-1])
    row_idx = np.repeat(np.arange(num_reqs), num_tokens_np)
    positions_np = (
        np.arange(num_tokens)
        - query_start_loc_np[row_idx]
        + (seq_lens_np - num_tokens_np)[row_idx]
    )
    positions = async_tensor_h2d(positions_np, device, torch.int64)
    token_req_idx = async_tensor_h2d(rows.idx_mapping_np[row_idx], device, torch.int64)
    input_ids = rows.all_token_ids[token_req_idx, positions]
    cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
    num_computed_np = seq_lens_np - num_tokens_np
    return InputBatch(
        req_ids=rows.req_ids,
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=rows.idx_mapping,
        idx_mapping_np=rows.idx_mapping_np,
        expanded_idx_mapping=rows.idx_mapping,
        expanded_local_pos=torch.zeros(num_reqs, dtype=torch.int32, device=device),
        num_scheduled_tokens=num_tokens_np,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_draft_tokens=0,
        num_draft_tokens_per_req=None,
        query_start_loc=async_tensor_h2d(query_start_loc_np, device, torch.int32),
        query_start_loc_np=query_start_loc_np,
        seq_lens=async_tensor_h2d(seq_lens_np, device, torch.int32),
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_np),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=num_computed_np,
        prefill_len_np=rows.prefill_len_np,
        num_computed_prefill_tokens_np=num_computed_np,
        is_prefilling_np=np.ones(num_reqs, dtype=np.bool_),
        has_prefill=True,
        decode_graph_eligible=False,
        input_ids=input_ids,
        positions=positions,
        is_padding=torch.zeros(num_tokens, dtype=torch.bool, device=device),
        logits_indices=async_tensor_h2d(
            query_start_loc_np[1:] - 1, device, torch.int64
        ),
        cu_num_logits=async_tensor_h2d(cu_num_logits_np, device, torch.int32),
        cu_num_logits_np=cu_num_logits_np,
        has_structured_output_reqs=has_structured_output_reqs,
        prompt_lens=None,
        prefill_runs_as_decode_np=num_tokens_np == 1,
    )


def _concat_tokens(
    a: InputBatch, a_values: torch.Tensor, b_values: torch.Tensor
) -> torch.Tensor:
    """Per-token values of ``a``'s real (unpadded) tokens, then ``b``'s."""
    return torch.cat([a_values[: a.num_tokens], b_values])


def _concat(a: InputBatch, b: InputBatch) -> InputBatch:
    """Concatenate two batches, ``b``'s tokens placed after ``a``'s real
    tokens. Every per-token, per-request and per-logit field is real, so the
    result serves both sampling and the drafter's attention metadata.

    Under adaptive verification ``a``'s per-request boundaries are exact only
    on the GPU (``query_start_loc``, ``cu_num_logits``); its CPU arrays are
    upper bounds, and stay so in the result."""
    assert b.num_draft_tokens == 0
    na, nb = a.num_reqs, b.num_reqs
    num_tokens_a = a.num_tokens
    num_logits_a = a.logits_indices.shape[0]
    device = a.idx_mapping.device

    def cat_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.concatenate([x[:na], y[:nb]])

    def cat_cum(x: np.ndarray, y: np.ndarray, offset: int) -> np.ndarray:
        return np.concatenate([x[: na + 1], offset + y[1 : nb + 1]])

    query_start_loc_np = cat_cum(
        a.query_start_loc_np, b.query_start_loc_np, num_tokens_a
    )
    cu_num_logits_np = cat_cum(
        a.cu_num_logits_np, b.cu_num_logits_np, int(a.cu_num_logits_np[na])
    )
    num_tokens = num_tokens_a + b.num_tokens
    num_draft_tokens_per_req = None
    if a.num_draft_tokens_per_req is not None:
        num_draft_tokens_per_req = np.concatenate(
            [a.num_draft_tokens_per_req[:na], np.zeros(nb, dtype=np.int32)]
        )
    prefill_runs_as_decode_np = None
    if b.prefill_runs_as_decode_np is not None:
        a_decode = (
            a.prefill_runs_as_decode_np
            if a.prefill_runs_as_decode_np is not None
            else np.zeros(na, dtype=np.bool_)
        )
        prefill_runs_as_decode_np = cat_np(a_decode, b.prefill_runs_as_decode_np)
    return InputBatch(
        req_ids=a.req_ids[:na] + b.req_ids,
        num_reqs=na + nb,
        num_reqs_after_padding=na + nb,
        idx_mapping=torch.cat([a.idx_mapping[:na], b.idx_mapping]),
        idx_mapping_np=cat_np(a.idx_mapping_np, b.idx_mapping_np),
        expanded_idx_mapping=torch.cat(
            [a.expanded_idx_mapping[:num_logits_a], b.expanded_idx_mapping]
        ),
        expanded_local_pos=torch.cat(
            [a.expanded_local_pos[:num_logits_a], b.expanded_local_pos]
        ),
        num_scheduled_tokens=cat_np(a.num_scheduled_tokens, b.num_scheduled_tokens),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_draft_tokens=a.num_draft_tokens,
        num_draft_tokens_per_req=num_draft_tokens_per_req,
        query_start_loc=torch.cat(
            [a.query_start_loc[: na + 1], num_tokens_a + b.query_start_loc[1:]]
        ),
        query_start_loc_np=query_start_loc_np,
        seq_lens=torch.cat([a.seq_lens[:na], b.seq_lens]),
        seq_lens_cpu_upper_bound=torch.cat(
            [a.seq_lens_cpu_upper_bound[:na], b.seq_lens_cpu_upper_bound]
        ),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=cat_np(
            a.num_computed_tokens_np, b.num_computed_tokens_np
        ),
        prefill_len_np=cat_np(a.prefill_len_np, b.prefill_len_np),
        num_computed_prefill_tokens_np=cat_np(
            a.num_computed_prefill_tokens_np, b.num_computed_prefill_tokens_np
        ),
        is_prefilling_np=cat_np(a.is_prefilling_np, b.is_prefilling_np),
        has_prefill=True,
        decode_graph_eligible=False,
        input_ids=_concat_tokens(a, a.input_ids, b.input_ids),
        positions=_concat_tokens(a, a.positions, b.positions),
        is_padding=torch.zeros(num_tokens, dtype=torch.bool, device=device),
        logits_indices=torch.cat(
            [a.logits_indices[:num_logits_a], num_tokens_a + b.logits_indices]
        ),
        cu_num_logits=torch.cat(
            [a.cu_num_logits[: na + 1], num_logits_a + b.cu_num_logits[1:]]
        ),
        cu_num_logits_np=cu_num_logits_np,
        has_structured_output_reqs=(
            a.has_structured_output_reqs or b.has_structured_output_reqs
        ),
        prompt_lens=None,
        prefill_runs_as_decode_np=prefill_runs_as_decode_np,
        max_query_len=(
            max(a.max_query_len, int(b.num_scheduled_tokens.max()))
            if a.max_query_len is not None
            else None
        ),
    )
