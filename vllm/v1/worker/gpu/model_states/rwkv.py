# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.tasks import GenerationTask
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class RWKV7ModelState(ModelState):
    """Dense batched recurrent state for RWKV7."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = model
        self.device = device
        self.max_num_reqs = self.scheduler_config.max_num_seqs

        cfg = self.model_config.hf_config
        total_num_layers = int(
            getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))
        )
        self.layer_offset = int(getattr(model, "start_layer", 0))
        self.num_layers = (
            int(getattr(model, "end_layer", total_num_layers)) - self.layer_offset
        )
        self.hidden_size = int(cfg.hidden_size)
        self.head_size = int(getattr(cfg, "head_size", 64))
        total_num_heads = int(
            getattr(
                cfg,
                "num_attention_heads",
                self.hidden_size // self.head_size,
            )
        )
        self.num_heads = int(getattr(model, "tp_num_heads", total_num_heads))
        wkv_dtype = getattr(model, "wkv_state_dtype", None)
        if wkv_dtype is None:
            wkv_dtype = (
                torch.float32
                if getattr(model, "wkv_mode", "fp16") == "fp32io16"
                else torch.float16
            )

        self.shift_state = torch.zeros(
            (self.num_layers, 2, self.max_num_reqs, self.hidden_size),
            dtype=torch.float16,
            device=device,
        )
        self.wkv_state = torch.zeros(
            (
                self.num_layers,
                self.max_num_reqs,
                self.num_heads,
                self.head_size,
                self.head_size,
            ),
            dtype=wkv_dtype,
            device=device,
        )
        self.elapsed = torch.zeros(
            (self.max_num_reqs,), dtype=torch.int32, device=device
        )
        self.execution_idx_mapping = torch.arange(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.decode_slot_indices = torch.empty(
            (self.max_num_reqs,), dtype=torch.int32, device=device
        )
        self.decode_token_positions = torch.empty(
            (self.max_num_reqs,), dtype=torch.long, device=device
        )
        # Maps request ids to stable RWKV state slots. A vLLM request index can
        # change after request metadata is condensed, but this slot must not.
        self.req_id_to_index: dict[str, int] = {}
        self.req_slot_to_row = [-1] * self.max_num_reqs
        self.row_to_req_slot = [-1] * self.max_num_reqs
        self.free_rows = set(range(self.max_num_reqs))
        self.decode_req_slots: set[int] = set()
        self._prefill_req_slots: list[int] = []
        self._prefill_becomes_decode: list[bool] = []

    def _reset_mappings(self) -> None:
        self.req_slot_to_row = [-1] * self.max_num_reqs
        self.row_to_req_slot = [-1] * self.max_num_reqs
        self.free_rows = set(range(self.max_num_reqs))
        self.decode_req_slots = set()
        self._prefill_req_slots = []
        self._prefill_becomes_decode = []

    def _state_slot_for_batch_entry(
        self,
        input_batch: InputBatch,
        batch_idx: int,
    ) -> int:
        req_ids = getattr(input_batch, "req_ids", None)
        if req_ids is None or batch_idx >= len(req_ids):
            raise RuntimeError("RWKV7 requires request ids for state lookup")
        req_id = req_ids[batch_idx]
        if req_id is None:
            raise RuntimeError("RWKV7 request id cannot be None")
        req_slot = self.req_id_to_index.get(req_id)
        if req_slot is None:
            raise RuntimeError(f"RWKV state for request id {req_id!r} missing")
        return req_slot

    @staticmethod
    def _is_contiguous_decode_context(
        decode_rows: list[int],
        decode_token_positions: list[int],
    ) -> bool:
        if not decode_rows:
            return False
        decode_len = len(decode_rows)
        if decode_rows != list(range(decode_len)):
            return False
        start = decode_token_positions[0]
        return decode_token_positions == list(range(start, start + decode_len))

    def _new_dummy_state_tensors(self, num_reqs: int) -> dict[str, torch.Tensor]:
        return {
            "shift_state": torch.zeros(
                (self.num_layers, 2, num_reqs, self.hidden_size),
                dtype=self.shift_state.dtype,
                device=self.device,
            ),
            "wkv_state": torch.zeros(
                (
                    self.num_layers,
                    num_reqs,
                    self.num_heads,
                    self.head_size,
                    self.head_size,
                ),
                dtype=self.wkv_state.dtype,
                device=self.device,
            ),
            "elapsed": torch.zeros(
                (num_reqs,),
                dtype=self.elapsed.dtype,
                device=self.device,
            ),
        }

    @staticmethod
    def _set_sampling_logits_fast_path(
        input_batch: InputBatch,
        enabled: bool,
    ) -> None:
        try:
            input_batch.rwkv_sampling_logits_contiguous = (  # type: ignore[attr-defined]
                enabled
            )
        except Exception:
            return

    @staticmethod
    def _build_prefill_groups(
        prefill_ranges: list[tuple[int, int, int]],
        prefill_rows: list[int],
    ) -> list[tuple[int, int, int, int, int, int]]:
        groups: list[tuple[int, int, int, int, int, int]] = []
        group_start = 0
        while group_start < len(prefill_ranges):
            first_batch_idx, first_start, first_end = prefill_ranges[group_start]
            query_len = first_end - first_start
            row_start = prefill_rows[group_start]
            group_end = group_start + 1
            token_end = first_end
            while group_end < len(prefill_ranges):
                batch_idx, start, end = prefill_ranges[group_end]
                row = prefill_rows[group_end]
                if (
                    batch_idx != first_batch_idx + (group_end - group_start)
                    or start != token_end
                    or end - start != query_len
                    or row != row_start + (group_end - group_start)
                ):
                    break
                token_end = end
                group_end += 1
            groups.append(
                (
                    first_batch_idx,
                    first_batch_idx + (group_end - group_start),
                    query_len,
                    first_start,
                    token_end,
                    row_start,
                )
            )
            group_start = group_end
        return groups

    def get_supported_generation_tasks(self) -> tuple[GenerationTask, ...]:
        return ("generate",)

    def custom_sampler(self, sampler: Any) -> tuple[Any, None]:
        if not sampler.use_rapid:
            raise RuntimeError("RWKV7 requires rapid-sampling on CUDA.")
        sampler.require_rapid = True
        return sampler, None

    def sort_scheduled_req_ids(
        self,
        req_ids: list[str],
        num_scheduled_tokens: dict[str, int],
        req_states: RequestState,
    ) -> list[str]:
        def key(item: tuple[int, str]) -> tuple[int, int, int]:
            order, req_id = item
            current_req_index = req_states.req_id_to_index.get(req_id)
            req_slot = self.req_id_to_index.get(req_id)
            if current_req_index is None or req_slot is None:
                return (num_scheduled_tokens[req_id], 1, order)
            is_prefilling = (
                req_states.num_computed_prefill_tokens[current_req_index]
                < req_states.prefill_len.np[current_req_index]
            )
            row = self.req_slot_to_row[req_slot]
            if (
                num_scheduled_tokens[req_id] == 1
                and not bool(is_prefilling)
                and row >= 0
            ):
                return (1, 0, row)
            return (num_scheduled_tokens[req_id], 1, order)

        return [req_id for _order, req_id in sorted(enumerate(req_ids), key=key)]

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        self.req_id_to_index[new_req_data.req_id] = req_index
        if not self.free_rows:
            raise RuntimeError("RWKV7 state pool is full")
        row = min(self.free_rows)
        self.free_rows.remove(row)
        self.req_slot_to_row[req_index] = row
        self.row_to_req_slot[row] = req_index
        self._zero_row(row)

    def remove_request(self, req_id: str) -> None:
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return
        row = self.req_slot_to_row[req_index]
        if row == -1:
            return
        if req_index in self.decode_req_slots:
            self._remove_decode_row(req_index, row)
        else:
            self.req_slot_to_row[req_index] = -1
            self.row_to_req_slot[row] = -1
            self.free_rows.add(row)
            self._zero_row(row)

    def _remove_decode_row(self, req_index: int, row: int) -> None:
        self.decode_req_slots.remove(req_index)
        self.req_slot_to_row[req_index] = -1
        self.row_to_req_slot[row] = -1
        self.free_rows.add(row)
        self._zero_row(row)

    def _mark_resident_row_decode(self, req_slot: int) -> int:
        current_row = self.req_slot_to_row[req_slot]
        if current_row == -1:
            raise RuntimeError(f"RWKV state for request slot {req_slot} missing")
        self.decode_req_slots.add(req_slot)
        return current_row

    def _validate_decode_membership(self) -> None:
        for req_slot in self.decode_req_slots:
            row = self.req_slot_to_row[req_slot]
            if row < 0 or row >= self.max_num_reqs:
                raise RuntimeError("RWKV7 live decode resident row is out of range")
            if self.row_to_req_slot[row] != req_slot:
                raise RuntimeError("RWKV7 decode resident mapping is inconsistent")

    def _zero_row(self, row: int) -> None:
        self.shift_state[:, :, row].zero_()
        self.wkv_state[:, row].zero_()
        self.elapsed[row].zero_()

    def reset_after_weight_update(self) -> None:
        active_rows = self.max_num_reqs - len(self.free_rows)
        if active_rows:
            logger.warning(
                "Resetting RWKV7 state after weight update with %d active rows. "
                "The trainer should quiesce requests before updating weights.",
                active_rows,
            )
        self.shift_state.zero_()
        self.wkv_state.zero_()
        self.elapsed.zero_()

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor | None:
        return None

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        self._set_sampling_logits_fast_path(input_batch, False)
        self._prefill_req_slots = []
        self._prefill_becomes_decode = []
        req_ids = getattr(input_batch, "req_ids", None)
        is_dummy_batch = bool(req_ids) and all(
            req_id not in self.req_id_to_index for req_id in req_ids
        )
        if is_dummy_batch:
            if not self.req_id_to_index:
                self._reset_mappings()
            return {
                "query_start_loc": input_batch.query_start_loc,
                "idx_mapping": self.execution_idx_mapping[: input_batch.num_reqs],
                **self._new_dummy_state_tensors(input_batch.num_reqs),
            }

        query_start_loc_np = getattr(input_batch, "query_start_loc_np", None)
        if query_start_loc_np is None:
            raise RuntimeError("RWKV7 requires CPU query_start_loc metadata")
        is_prefilling_np = getattr(input_batch, "is_prefilling_np", None)
        if is_prefilling_np is None:
            raise RuntimeError("RWKV7 requires CPU prefill metadata")

        batch_entries: list[tuple[int, int, bool, int, int]] = []
        for batch_idx in range(len(input_batch.idx_mapping_np)):
            req_slot = self._state_slot_for_batch_entry(input_batch, batch_idx)
            current_row = self.req_slot_to_row[req_slot]
            if current_row == -1:
                raise RuntimeError(f"RWKV state for request slot {req_slot} missing")
            start = int(query_start_loc_np[batch_idx])
            end = int(query_start_loc_np[batch_idx + 1])
            query_len = end - start
            is_prefill = bool(is_prefilling_np[batch_idx]) or query_len > 1
            batch_entries.append((batch_idx, req_slot, is_prefill, start, end))

        decode_entries: list[tuple[int, int, int, int]] = []
        live_decode_req_slots = set(self.decode_req_slots)
        scheduled_decode_req_slots: set[int] = set()
        prefill_entries: list[tuple[int, int, int, bool]] = []
        for batch_idx, req_slot, is_prefill, start, end in batch_entries:
            current_row = self.req_slot_to_row[req_slot]
            if current_row == -1:
                raise RuntimeError(f"RWKV state for request slot {req_slot} missing")
            if is_prefill:
                num_computed_prefill = getattr(
                    input_batch, "num_computed_prefill_tokens_np", None
                )
                prefill_len = getattr(input_batch, "prefill_len_np", None)
                scheduled_tokens = getattr(input_batch, "num_scheduled_tokens", None)
                becomes_decode = False
                if (
                    num_computed_prefill is not None
                    and prefill_len is not None
                    and scheduled_tokens is not None
                ):
                    becomes_decode = int(num_computed_prefill[batch_idx]) + int(
                        scheduled_tokens[batch_idx]
                    ) >= int(prefill_len[batch_idx])
                prefill_entries.append(
                    (batch_idx, req_slot, current_row, becomes_decode)
                )
            else:
                decode_row = self._mark_resident_row_decode(req_slot)
                scheduled_decode_req_slots.add(req_slot)
                decode_entries.append((batch_idx, req_slot, decode_row, start))
        if scheduled_decode_req_slots:
            missing_decode_req_slots = (
                live_decode_req_slots - scheduled_decode_req_slots
            )
            if missing_decode_req_slots:
                raise RuntimeError(
                    "RWKV7 native decode requires scheduling all live decode "
                    "rows; missing request slots "
                    f"{sorted(missing_decode_req_slots)}"
                )
        if decode_entries:
            self._validate_decode_membership()
        scheduled_rows = [
            self.req_slot_to_row[req_slot]
            for _batch_idx, req_slot, _is_prefill, _start, _end in batch_entries
        ]
        idx_mapping = torch.tensor(
            scheduled_rows,
            dtype=torch.int32,
            device=self.device,
        )
        source_decode_rows = [
            row for _batch_idx, _req_slot, row, _start in decode_entries
        ]
        decode_token_positions = [
            start for _batch_idx, _req_slot, _row, start in decode_entries
        ]
        use_contiguous_decode = self._is_contiguous_decode_context(
            source_decode_rows,
            decode_token_positions,
        )
        if decode_entries and not use_contiguous_decode:
            decode_len = len(decode_entries)
            self.decode_slot_indices[:decode_len].copy_(
                torch.tensor(
                    source_decode_rows,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            self.decode_token_positions[:decode_len].copy_(
                torch.tensor(
                    decode_token_positions,
                    dtype=torch.long,
                    device=self.device,
                )
            )
            slot_indices = self.decode_slot_indices[:decode_len]
            decode_token_position_tensor = self.decode_token_positions[:decode_len]
        elif decode_entries:
            slot_indices = None
            decode_token_position_tensor = decode_token_positions
        else:
            slot_indices = None
            decode_token_position_tensor = None
        decode_rows = source_decode_rows if decode_entries else []
        decode_context_size = len(decode_rows)
        decode_state_tensors = {
            "shift_state": self.shift_state,
            "wkv_state": self.wkv_state,
            "elapsed": self.elapsed,
        }

        if not prefill_entries:
            self._set_sampling_logits_fast_path(
                input_batch,
                bool(decode_entries and len(decode_entries) == input_batch.num_reqs),
            )
            return {
                "query_start_loc": input_batch.query_start_loc,
                "idx_mapping": idx_mapping,
                **decode_state_tensors,
                "rwkv_decode_batch_size": decode_context_size,
                "rwkv_decode_rows": decode_rows,
                "rwkv_decode_token_positions": decode_token_position_tensor,
                "slot_indices": slot_indices,
            }

        prefill_rows: list[int] = []
        for (
            _batch_idx,
            req_slot,
            _decode_row,
            becomes_decode,
        ) in prefill_entries:
            prefill_rows.append(self.req_slot_to_row[req_slot])
            self._prefill_req_slots.append(req_slot)
            self._prefill_becomes_decode.append(becomes_decode)

        prefill_ranges = [
            (
                batch_idx,
                int(query_start_loc_np[batch_idx]),
                int(query_start_loc_np[batch_idx + 1]),
            )
            for batch_idx, _req_slot, _row, _becomes_decode in prefill_entries
        ]
        prefill_groups = self._build_prefill_groups(prefill_ranges, prefill_rows)
        prefill_lengths = [end - start for _batch_idx, start, end in prefill_ranges]
        has_positive_prefill_lengths = all(length > 0 for length in prefill_lengths)
        grouped_ranges = sum(
            batch_end - batch_start for batch_start, batch_end, *_ in prefill_groups
        )
        can_use_grouped_prefill = (
            has_positive_prefill_lengths
            and len(prefill_groups) == 1
            and grouped_ranges == len(prefill_ranges)
        )
        can_use_varlen_prefill = (
            not can_use_grouped_prefill and has_positive_prefill_lengths
        )
        prefill_varlen_inputs: dict[str, Any] = {}
        if can_use_varlen_prefill:
            query_offsets = [0]
            token_positions: list[int] = []
            req_id: list[int] = []
            for local_req, ((_batch_idx, start, end), length) in enumerate(
                zip(prefill_ranges, prefill_lengths)
            ):
                token_positions.extend(range(start, end))
                req_id.extend([local_req] * length)
                query_offsets.append(query_offsets[-1] + length)
            prefill_groups = []
            prefill_varlen_inputs = {
                "rwkv_prefill_query_start_loc": torch.tensor(
                    query_offsets,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "rwkv_prefill_slot_indices": torch.tensor(
                    prefill_rows,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "rwkv_prefill_token_positions": torch.tensor(
                    token_positions,
                    dtype=torch.long,
                    device=self.device,
                ),
                "rwkv_prefill_req_id": torch.tensor(
                    req_id,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "rwkv_prefill_max_t": max(prefill_lengths),
            }
        elif not can_use_grouped_prefill:
            raise RuntimeError(
                "RWKV7 fast prefill requires one contiguous equal-length "
                "group or positive-length varlen metadata."
            )
        if len(prefill_entries) == input_batch.num_reqs:
            return {
                "query_start_loc": input_batch.query_start_loc,
                "idx_mapping": idx_mapping,
                "rwkv_prefill_token_ranges": prefill_ranges,
                "rwkv_prefill_rows": prefill_rows,
                "rwkv_prefill_groups": prefill_groups,
                **prefill_varlen_inputs,
                "shift_state": self.shift_state,
                "wkv_state": self.wkv_state,
                "elapsed": self.elapsed,
            }
        mixed_inputs = {
            "query_start_loc": input_batch.query_start_loc,
            "idx_mapping": idx_mapping,
            **decode_state_tensors,
            "rwkv_decode_batch_size": decode_context_size,
            "rwkv_decode_rows": decode_rows,
            "rwkv_decode_token_positions": decode_token_position_tensor,
            "slot_indices": slot_indices,
            "rwkv_prefill_token_ranges": prefill_ranges,
            "rwkv_prefill_rows": prefill_rows,
            "rwkv_prefill_groups": prefill_groups,
            **prefill_varlen_inputs,
        }
        if decode_entries:
            mixed_inputs.update(
                {
                    "prefill_shift_state": self.shift_state,
                    "prefill_wkv_state": self.wkv_state,
                    "prefill_elapsed": self.elapsed,
                }
            )
        return mixed_inputs

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        if not self._prefill_req_slots:
            return
        for scratch_row, req_slot in enumerate(self._prefill_req_slots):
            if self._prefill_becomes_decode[scratch_row]:
                self._mark_resident_row_decode(req_slot)
        self._validate_decode_membership()
        self._prefill_req_slots = []
        self._prefill_becomes_decode = []

    def has_pending_postprocess_state(self) -> bool:
        return bool(self._prefill_req_slots)

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        lengths = torch.full(
            (num_reqs,),
            num_tokens // num_reqs,
            dtype=torch.int32,
            device="cpu",
        )
        lengths[: num_tokens % num_reqs] += 1
        query_start_loc = torch.empty((num_reqs + 1,), dtype=torch.int32)
        query_start_loc[0] = 0
        query_start_loc[1:] = lengths.cumsum(dim=0)
        idx_mapping = self.execution_idx_mapping[:num_reqs]
        # Full CUDAGraph replay binds captured state pointers, so decode capture
        # uses resident buffers. Request-level dummy profiling uses scratch.
        state_tensors = {
            "shift_state": self.shift_state,
            "wkv_state": self.wkv_state,
            "elapsed": self.elapsed,
        }
        if num_tokens == num_reqs:
            return {
                "query_start_loc": query_start_loc,
                "idx_mapping": idx_mapping,
                **state_tensors,
                "rwkv_decode_batch_size": num_reqs,
                "rwkv_decode_rows": list(range(num_reqs)),
                "rwkv_decode_token_positions": list(range(num_reqs)),
                "slot_indices": None,
            }
        return {
            "query_start_loc": query_start_loc,
            "idx_mapping": idx_mapping,
            **state_tensors,
        }

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        return {}
