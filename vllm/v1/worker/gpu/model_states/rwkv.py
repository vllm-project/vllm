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
        self.req_id_to_index: dict[str, int] = {}
        self.req_slot_to_row = [-1] * self.max_num_reqs
        self.row_to_req_slot = [-1] * self.max_num_reqs
        self.free_rows = set(range(self.max_num_reqs))
        self.decode_req_slots: set[int] = set()
        self.num_decode_rows = 0
        self._prefill_decode_rows: list[int] = []
        self._prefill_req_slots: list[int] = []
        self._prefill_becomes_decode: list[bool] = []
        self.reset_state_movement_stats()

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
                (num_reqs,), dtype=self.elapsed.dtype, device=self.device
            ),
        }

    def _reset_mappings(self) -> None:
        self.req_slot_to_row = [-1] * self.max_num_reqs
        self.row_to_req_slot = [-1] * self.max_num_reqs
        self.free_rows = set(range(self.max_num_reqs))
        self.decode_req_slots = set()
        self.num_decode_rows = 0
        self._prefill_decode_rows = []
        self._prefill_req_slots = []
        self._prefill_becomes_decode = []

    def _active_rows(self) -> int:
        return self.max_num_reqs - len(self.free_rows)

    def _decode_context_size(self, decode_rows: list[int]) -> int:
        if not decode_rows:
            return 0
        return len(decode_rows)

    def reset_state_movement_stats(self) -> None:
        self._state_movement_stats = {
            "resident_to_decode_copies": 0,
            "decode_compactions": 0,
            "decode_compaction_rows": 0,
        }

    def get_state_movement_stats(self) -> dict[str, int]:
        return dict(self._state_movement_stats)

    def _sync_decode_rows_to_resident_rows(
        self,
        decode_entries: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        """Expose scheduled decode as compact resident rows."""
        if not decode_entries:
            return decode_entries

        req_slots = [req_slot for _batch_idx, req_slot, _row, _start in decode_entries]
        resident_rows = [row for _batch_idx, _req_slot, row, _start in decode_entries]
        compact_rows = list(range(len(resident_rows)))
        sorted_resident_rows = sorted(resident_rows)
        if resident_rows != sorted_resident_rows:
            raise RuntimeError(
                "RWKV7 native decode requires scheduling decode rows in "
                "resident-row order; scheduled decode rows "
                f"{resident_rows}"
            )
        has_holes = sorted_resident_rows != compact_rows

        if has_holes:
            scheduled_req_slots = set(req_slots)
            blocked_rows: list[tuple[int, int]] = []
            for row in compact_rows:
                owner = self.row_to_req_slot[row]
                if owner != -1 and owner not in scheduled_req_slots:
                    blocked_rows.append((row, owner))
            if blocked_rows:
                raise RuntimeError(
                    "RWKV7 cannot compact decode rows over live "
                    "prefill/non-decode resident rows "
                    f"{blocked_rows}; scheduled decode rows {resident_rows}"
                )

            req_slot_by_row = {
                row: req_slot for _batch_idx, req_slot, row, _start in decode_entries
            }
            compact_row_by_row = {
                row: compact_row for compact_row, row in enumerate(sorted_resident_rows)
            }
            batch_shift_state = self.shift_state[:, :, sorted_resident_rows, :].clone()
            batch_wkv_state = self.wkv_state[:, sorted_resident_rows, :, :, :].clone()
            batch_elapsed = self.elapsed[sorted_resident_rows].clone()
            self.shift_state[:, :, : len(resident_rows), :].copy_(batch_shift_state)
            self.wkv_state[:, : len(resident_rows), :, :, :].copy_(batch_wkv_state)
            self.elapsed[: len(resident_rows)].copy_(batch_elapsed)
            self._state_movement_stats["decode_compactions"] += 1
            self._state_movement_stats["decode_compaction_rows"] += len(resident_rows)

            affected_rows = set(resident_rows) | set(compact_rows)
            for row in affected_rows:
                self.row_to_req_slot[row] = -1
                self.free_rows.add(row)
            for source_row in sorted_resident_rows:
                compact_row = compact_row_by_row[source_row]
                req_slot = req_slot_by_row[source_row]
                self.req_slot_to_row[req_slot] = compact_row
                self.row_to_req_slot[compact_row] = req_slot
                self.free_rows.discard(compact_row)
            for row in affected_rows - set(compact_rows):
                self._zero_row(row)

        self.num_decode_rows = len(self.decode_req_slots)

        return [
            (
                batch_idx,
                req_slot,
                compact_row_by_row[row] if has_holes else row,
                start,
            )
            for (
                batch_idx,
                req_slot,
                row,
                start,
            ) in decode_entries
        ]

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

    def sort_scheduled_req_ids(
        self,
        req_ids: list[str],
        num_scheduled_tokens: dict[str, int],
        req_states: RequestState,
    ) -> list[str]:
        def key(item: tuple[int, str]) -> tuple[int, int, int]:
            order, req_id = item
            req_slot = req_states.req_id_to_index.get(req_id)
            if req_slot is None:
                return (num_scheduled_tokens[req_id], 1, order)
            is_prefilling = (
                req_states.num_computed_prefill_tokens[req_slot]
                < req_states.prefill_len.np[req_slot]
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
        if self.num_decode_rows <= 0:
            raise RuntimeError(
                f"RWKV7 decode row invariant violated for request slot {req_index}"
            )
        self.decode_req_slots.remove(req_index)
        self.req_slot_to_row[req_index] = -1
        self.row_to_req_slot[row] = -1
        self.num_decode_rows = len(self.decode_req_slots)
        self.free_rows.add(row)
        self._zero_row(row)
        self._fill_decode_row_hole(row)

    def _fill_decode_row_hole(self, row: int) -> None:
        if not self.decode_req_slots:
            return
        source_req_slot = -1
        source_row = -1
        for req_slot in self.decode_req_slots:
            candidate_row = self.req_slot_to_row[req_slot]
            if candidate_row > row and candidate_row > source_row:
                source_req_slot = req_slot
                source_row = candidate_row
        if source_req_slot == -1:
            return

        self.shift_state[:, :, row, :].copy_(self.shift_state[:, :, source_row, :])
        self.wkv_state[:, row, :, :, :].copy_(self.wkv_state[:, source_row, :, :, :])
        self.elapsed[row].copy_(self.elapsed[source_row])
        self._zero_row(source_row)

        self.req_slot_to_row[source_req_slot] = row
        self.row_to_req_slot[row] = source_req_slot
        self.row_to_req_slot[source_row] = -1
        self.free_rows.discard(row)
        self.free_rows.add(source_row)

    def _compact_decode_rows_into_free_prefix(self) -> None:
        target_width = len(self.decode_req_slots)
        for row in range(target_width):
            if self.row_to_req_slot[row] == -1:
                self._fill_decode_row_hole(row)

    def _mark_resident_row_decode(self, req_slot: int) -> int:
        current_row = self.req_slot_to_row[req_slot]
        if current_row == -1:
            raise RuntimeError(f"RWKV state for request slot {req_slot} missing")
        self.decode_req_slots.add(req_slot)
        self.num_decode_rows = len(self.decode_req_slots)
        return current_row

    def _validate_decode_membership(self) -> None:
        if self.num_decode_rows != len(self.decode_req_slots):
            raise RuntimeError("RWKV7 decode row accounting is inconsistent")
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
        self._prefill_decode_rows = []
        self._prefill_req_slots = []
        self._prefill_becomes_decode = []
        req_ids = getattr(input_batch, "req_ids", None)
        is_dummy_batch = req_ids and all(
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
            query_start_loc_np = input_batch.query_start_loc.detach().cpu().numpy()
        is_prefilling_np = getattr(input_batch, "is_prefilling_np", None)
        if is_prefilling_np is None:
            is_prefilling_np = [False] * input_batch.num_reqs

        batch_entries: list[tuple[int, int, bool, int, int]] = []
        for batch_idx, req_slot_np in enumerate(input_batch.idx_mapping_np):
            req_slot = int(req_slot_np)
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
            decode_entries = self._sync_decode_rows_to_resident_rows(decode_entries)
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
        if not decode_entries:
            decode_rows = []
            decode_context_size = 0
            decode_state_tensors = {
                "shift_state": self.shift_state,
                "wkv_state": self.wkv_state,
                "elapsed": self.elapsed,
            }
        else:
            decode_rows = source_decode_rows
            decode_context_size = self._decode_context_size(decode_rows)
            decode_state_tensors = {
                "shift_state": self.shift_state,
                "wkv_state": self.wkv_state,
                "elapsed": self.elapsed,
            }

        if not prefill_entries:
            return {
                "query_start_loc": input_batch.query_start_loc,
                "idx_mapping": idx_mapping,
                **decode_state_tensors,
                "rwkv_decode_batch_size": decode_context_size,
                "rwkv_decode_rows": decode_rows,
                "rwkv_decode_token_positions": decode_token_positions,
            }

        prefill_idx_mapping = torch.full(
            (input_batch.num_reqs,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        prefill_rows: list[int] = []
        for (
            _batch_idx,
            req_slot,
            _decode_row,
            becomes_decode,
        ) in prefill_entries:
            decode_row = self.req_slot_to_row[req_slot]
            prefill_rows.append(decode_row)
            self._prefill_decode_rows.append(decode_row)
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

        if len(prefill_entries) == input_batch.num_reqs:
            return {
                "query_start_loc": input_batch.query_start_loc,
                "idx_mapping": idx_mapping,
                "prefill_idx_mapping": prefill_idx_mapping,
                "rwkv_prefill_token_ranges": prefill_ranges,
                "rwkv_prefill_rows": prefill_rows,
                "rwkv_prefill_groups": prefill_groups,
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
            "rwkv_decode_token_positions": decode_token_positions,
            "rwkv_prefill_token_ranges": prefill_ranges,
            "rwkv_prefill_rows": prefill_rows,
            "rwkv_prefill_groups": prefill_groups,
            "prefill_idx_mapping": prefill_idx_mapping,
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
        self, idx_mapping: torch.Tensor, num_sampled: torch.Tensor | int
    ) -> None:
        if not self._prefill_decode_rows:
            return
        for scratch_row, req_slot in enumerate(self._prefill_req_slots):
            if self._prefill_becomes_decode[scratch_row]:
                self._mark_resident_row_decode(req_slot)
        self.num_decode_rows = len(self.decode_req_slots)
        self._compact_decode_rows_into_free_prefix()
        self._validate_decode_membership()
        self._prefill_decode_rows = []
        self._prefill_req_slots = []
        self._prefill_becomes_decode = []

    def has_pending_postprocess_state(self) -> bool:
        return bool(self._prefill_decode_rows)

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
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32)
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
