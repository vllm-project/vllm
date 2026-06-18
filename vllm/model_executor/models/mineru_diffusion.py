# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental MinerU-Diffusion entry point.

This module provides the config/model registry surface, MinerU's remasking
primitives, and the native SDAR modules used by the compatibility benchmark.
The v2 ModelState integration below is the entry point for replacing the
compatibility path with vLLM KV-cache driven diffusion serving.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import partial
import json
import os
import re
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BatchFeature
from transformers.activations import ACT2FN
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)

from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config_or_none
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.models.utils import (
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.utils import record_function_or_nullcontext
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)

from .interfaces import MultiModalEmbeddings, SupportsMultiModal
from .qwen2_vl import (
    Qwen2VLProcessingInfo,
    _create_qwen2vl_field_factory,
)

_DEBUG_TRACE_ENV = "VLLM_MINERU_DIFFUSION_DEBUG_TRACE"
_DEBUG_TRACE_LIMIT_ENV = "VLLM_MINERU_DIFFUSION_DEBUG_TRACE_LIMIT"
_MASK_ONLY_SAMPLING_ENV = "VLLM_MINERU_DIFFUSION_MASK_ONLY_SAMPLING"
_debug_trace_counts: dict[str, int] = {}


def _mineru_debug_trace_enabled() -> bool:
    return bool(os.environ.get(_DEBUG_TRACE_ENV))


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.lower() not in {"", "0", "false", "no"}


def _mineru_mask_only_logits_row_indices(
    input_batch: Any,
    states: "MinerUDiffusionRequestStates",
    mask_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    sample_mask = torch.zeros(
        int(input_batch.cu_num_logits_np[input_batch.num_reqs]),
        dtype=torch.bool,
        device=device,
    )
    slots_np = input_batch.idx_mapping_np[: input_batch.num_reqs]
    per_req_nlogits_np = np.diff(
        input_batch.cu_num_logits_np[: input_batch.num_reqs + 1]
    )
    for req_idx in range(input_batch.num_reqs):
        valid_len = int(per_req_nlogits_np[req_idx])
        if valid_len <= 0:
            continue
        slot = int(slots_np[req_idx])
        if bool(states.is_encoder_phase[slot].item()):
            continue
        start = int(input_batch.cu_num_logits_np[req_idx])
        end = start + valid_len
        sample_mask[start:end] = (
            states.canvas[slot, :valid_len] == mask_token_id
        ).to(device)
    return torch.nonzero(sample_mask, as_tuple=False).flatten()


def _mineru_trace_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.numel() > 256:
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }
        return tensor.cpu().tolist()
    if isinstance(value, Mapping):
        return {str(k): _mineru_trace_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mineru_trace_value(item) for item in value]
    return value


def _mineru_batch_trace_payload(input_batch: Any) -> dict[str, Any]:
    fields = (
        "req_ids",
        "num_reqs",
        "num_tokens",
        "num_tokens_after_padding",
        "num_draft_tokens",
        "num_draft_tokens_per_req",
        "num_scheduled_tokens",
        "query_start_loc_np",
        "cu_num_logits_np",
        "num_computed_tokens_np",
        "prefill_len_np",
        "num_computed_prefill_tokens_np",
        "is_prefilling_np",
    )
    payload = {
        field: _mineru_trace_value(getattr(input_batch, field))
        for field in fields
        if hasattr(input_batch, field)
    }
    if hasattr(input_batch, "positions") and hasattr(input_batch, "query_start_loc_np"):
        positions = input_batch.positions.detach().cpu()
        ranges: list[dict[str, int | None]] = []
        query_start_loc_np = input_batch.query_start_loc_np
        num_reqs = int(getattr(input_batch, "num_reqs", len(query_start_loc_np) - 1))
        for req_idx in range(num_reqs):
            start = int(query_start_loc_np[req_idx])
            end = int(query_start_loc_np[req_idx + 1])
            if end <= start:
                ranges.append({"len": 0, "min": None, "max": None})
                continue
            req_positions = positions[start:end]
            ranges.append(
                {
                    "len": end - start,
                    "min": int(req_positions.min().item()),
                    "max": int(req_positions.max().item()),
                }
            )
        payload["position_ranges"] = ranges
    return payload


def _mineru_debug_trace(event: str, **payload: Any) -> None:
    path = os.environ.get(_DEBUG_TRACE_ENV)
    if not path:
        return
    limit = int(os.environ.get(_DEBUG_TRACE_LIMIT_ENV, "10000"))
    count = _debug_trace_counts.get(path, 0)
    if limit >= 0 and count >= limit:
        return
    _debug_trace_counts[path] = count + 1

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    record = {
        "event": event,
        "pid": os.getpid(),
        "time": time.time(),
        **{key: _mineru_trace_value(value) for key, value in payload.items()},
    }
    with open(path, "a", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps(record, sort_keys=True) + "\n")


class MinerUDiffusionSampler:
    """Block diffusion sampler for MinerU's v2 runtime path."""

    def __init__(
        self,
        sampler: Any,
        *,
        diffusion_states: "MinerUDiffusionRequestStates" | None = None,
        canvas_length: int | None = None,
        mask_token_id: int | None = None,
        denoising_steps: int | None = None,
        dynamic_threshold: float = 0.85,
    ):
        self.sampler = sampler
        self.sampling_states = sampler.sampling_states
        self.req_states = sampler.req_states
        self.diffusion_states = diffusion_states
        self.canvas_length = canvas_length or (
            diffusion_states.canvas_length if diffusion_states is not None else 32
        )
        self.mask_token_id = mask_token_id if mask_token_id is not None else (
            diffusion_states.mask_token_id if diffusion_states is not None else 0
        )
        self.denoising_steps = denoising_steps or self.canvas_length
        self.dynamic_threshold = dynamic_threshold
        self.num_transfer_tokens = get_num_transfer_tokens(
            self.canvas_length,
            self.denoising_steps,
        )
        self._transfer_schedule_cache = {
            self.denoising_steps: self.num_transfer_tokens
        }
        if diffusion_states is not None:
            self.dynamic_threshold_by_slot = torch.full(
                (diffusion_states.max_num_reqs,),
                float(dynamic_threshold),
                dtype=torch.float32,
                device=diffusion_states.device,
            )
            self.denoising_steps_by_slot = torch.full(
                (diffusion_states.max_num_reqs,),
                int(self.denoising_steps),
                dtype=torch.int32,
                device=diffusion_states.device,
            )
        else:
            self.dynamic_threshold_by_slot = None
            self.denoising_steps_by_slot = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.sampler, name)

    def add_request(self, req_idx: int, prompt_len: int, sampling_params: Any) -> None:
        extra_args = getattr(sampling_params, "extra_args", None) or {}
        requested_block_size = extra_args.get("block_size")
        if requested_block_size is not None:
            requested_block_size = int(requested_block_size)
            if requested_block_size != self.canvas_length:
                raise ValueError(
                    "MinerU-Diffusion vLLM serving currently requires request "
                    f"block_size={requested_block_size} to match engine "
                    f"canvas_length={self.canvas_length}."
                )

        self.sampler.add_request(req_idx, prompt_len, sampling_params)
        if self.dynamic_threshold_by_slot is None:
            return
        threshold = extra_args.get("dynamic_threshold", self.dynamic_threshold)
        self.dynamic_threshold_by_slot[req_idx] = float(threshold)
        requested_steps = extra_args.get(
            "max_denoising_steps",
            extra_args.get("denoising_steps", self.denoising_steps),
        )
        requested_steps = int(requested_steps)
        if requested_steps <= 0 or requested_steps > self.canvas_length:
            raise ValueError(
                "MinerU-Diffusion max_denoising_steps must be in "
                f"[1, {self.canvas_length}], got {requested_steps}."
            )
        if self.denoising_steps_by_slot is not None:
            self.denoising_steps_by_slot[req_idx] = requested_steps

    def apply_staged_writes(self) -> None:
        self.sampler.apply_staged_writes()

    def _empty_output(
        self,
        *,
        num_reqs: int,
        device: torch.device,
        width: int = 1,
    ) -> SamplerOutput:
        sampled = torch.zeros(num_reqs, width, dtype=torch.int64, device=device)
        num_sampled = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_sampled,
        )

    def _finish_prefills(
        self,
        input_batch: Any,
        prefill_indices: np.ndarray | None = None,
    ) -> None:
        states = self.diffusion_states
        if states is None:
            return
        if prefill_indices is None:
            prefill_indices = np.arange(input_batch.num_reqs)
        done_prefill = (
            input_batch.num_computed_prefill_tokens_np[prefill_indices]
            + input_batch.num_scheduled_tokens[prefill_indices]
            >= input_batch.prefill_len_np[prefill_indices]
        )
        slots_np = input_batch.idx_mapping_np[prefill_indices[done_prefill]]
        if len(slots_np) == 0:
            return
        slots = torch.as_tensor(slots_np, dtype=torch.int64, device=states.device)
        states.init_canvas(slots)
        self.req_states.draft_tokens[slots, : self.canvas_length] = states.canvas[
            slots
        ]
        states.is_encoder_phase[slots] = False

    def _sample_logits(
        self,
        logits: torch.Tensor,
        input_batch: Any,
        selected_indices: torch.Tensor | None = None,
        *,
        logits_are_compacted: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_logits = logits
        if selected_indices is not None and not logits_are_compacted:
            sample_logits = logits[selected_indices]
        if hasattr(input_batch, "expanded_idx_mapping"):
            expanded_idx_mapping = input_batch.expanded_idx_mapping
            pos = input_batch.positions[input_batch.logits_indices]
            input_ids = input_batch.input_ids[input_batch.logits_indices]
            expanded_local_pos = input_batch.expanded_local_pos
            if selected_indices is not None:
                expanded_idx_mapping = expanded_idx_mapping[selected_indices]
                pos = pos[selected_indices]
                input_ids = input_ids[selected_indices]
                expanded_local_pos = expanded_local_pos[selected_indices]
            return self.sampler.sample(
                sample_logits,
                expanded_idx_mapping,
                input_batch.idx_mapping_np,
                pos,
                input_ids,
                expanded_local_pos,
                return_logprobs=False,
            )
        return sample_logits.argmax(dim=-1), sample_logits.float()

    def _denoising_steps_for_slot(self, slot: int) -> int:
        if self.denoising_steps_by_slot is None:
            return self.denoising_steps
        return int(self.denoising_steps_by_slot[slot].item())

    def _transfer_schedule_for_steps(self, denoising_steps: int) -> list[int]:
        schedule = self._transfer_schedule_cache.get(denoising_steps)
        if schedule is None:
            schedule = get_num_transfer_tokens(self.canvas_length, denoising_steps)
            self._transfer_schedule_cache[denoising_steps] = schedule
        return schedule

    def _mask_only_sample_indices(
        self,
        input_batch: Any,
        per_req_nlogits_np: np.ndarray,
        states: "MinerUDiffusionRequestStates",
        device: torch.device,
    ) -> torch.Tensor:
        return _mineru_mask_only_logits_row_indices(
            input_batch,
            states,
            self.mask_token_id,
            device,
        )

    def _build_output(
        self,
        input_batch: Any,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        per_req_nlogits_np: np.ndarray,
    ) -> SamplerOutput:
        device = sampled.device
        num_reqs = input_batch.num_reqs
        num_logits = torch.as_tensor(
            per_req_nlogits_np, dtype=torch.int32, device=device
        )
        if hasattr(input_batch, "query_start_loc_np"):
            query_lens = torch.as_tensor(
                np.diff(input_batch.query_start_loc_np[: num_reqs + 1]),
                dtype=torch.int32,
                device=device,
            )
        else:
            query_lens = num_logits
        num_rejected = torch.where(
            (num_logits > 0) & (num_sampled == 0),
            query_lens,
            num_logits - num_sampled,
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: Any,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        del draft_logits
        if self.diffusion_states is None:
            return self.sampler(logits, input_batch)

        num_reqs = input_batch.num_reqs
        device = logits.device
        if input_batch.num_draft_tokens == 0:
            self._finish_prefills(input_batch)
            output = self._empty_output(num_reqs=num_reqs, device=device)
            _mineru_debug_trace(
                "mineru_diffusion_sampler_prefill",
                batch=_mineru_batch_trace_payload(input_batch),
                num_sampled=output.num_sampled,
                num_rejected=output.num_rejected,
            )
            return output

        states = self.diffusion_states
        per_req_nlogits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])
        prefill_indices = np.where(per_req_nlogits_np == 0)[0]
        if len(prefill_indices) > 0:
            self._finish_prefills(input_batch, prefill_indices)
        sampled = torch.zeros(
            num_reqs, self.canvas_length, dtype=torch.int64, device=device
        )
        num_sampled = torch.zeros(num_reqs, dtype=torch.int32, device=device)

        runner_logits_row_indices = getattr(input_batch, "logits_row_indices", None)
        logits_are_compacted = runner_logits_row_indices is not None
        mask_only_sampling = (
            _env_flag_enabled(_MASK_ONLY_SAMPLING_ENV) or logits_are_compacted
        )
        sample_indices: torch.Tensor | None = None
        if runner_logits_row_indices is not None:
            sample_indices = runner_logits_row_indices.to(device=device)
        elif mask_only_sampling:
            sample_indices = self._mask_only_sample_indices(
                input_batch,
                per_req_nlogits_np,
                states,
                device,
            )

        if sample_indices is not None and sample_indices.numel() == 0:
            sampled_tokens = torch.zeros(
                logits.shape[0], dtype=torch.int64, device=device
            )
            confidence = torch.full(
                (logits.shape[0],), -torch.inf, dtype=torch.float32, device=device
            )
        else:
            sampled_subset, processed_logits = self._sample_logits(
                logits,
                input_batch,
                sample_indices,
                logits_are_compacted=logits_are_compacted,
            )
            probs = torch.softmax(processed_logits.float(), dim=-1)
            confidence_subset = probs.gather(
                -1, sampled_subset.view(-1, 1)
            ).squeeze(-1)
            if sample_indices is None:
                sampled_tokens = sampled_subset
                confidence = confidence_subset
            else:
                num_logits = int(input_batch.cu_num_logits_np[num_reqs])
                sampled_tokens = torch.zeros(
                    num_logits, dtype=sampled_subset.dtype, device=device
                )
                confidence = torch.full(
                    (num_logits,),
                    -torch.inf,
                    dtype=confidence_subset.dtype,
                    device=device,
                )
                sampled_tokens[sample_indices] = sampled_subset
                confidence[sample_indices] = confidence_subset

        slots_np = input_batch.idx_mapping_np[:num_reqs]
        trace_requests: list[dict[str, Any]] | None = (
            [] if _mineru_debug_trace_enabled() else None
        )
        for req_idx in range(num_reqs):
            valid_len = int(per_req_nlogits_np[req_idx])
            if valid_len <= 0:
                continue
            slot = int(slots_np[req_idx])
            start = int(input_batch.cu_num_logits_np[req_idx])
            end = start + valid_len
            is_commit = bool(states.is_encoder_phase[slot].item())
            step_before = int(states.step[slot].item())
            mask_count_before = int(
                (states.canvas[slot, :valid_len] == self.mask_token_id).sum().item()
            )

            if is_commit:
                sampled[req_idx, :valid_len] = states.canvas[slot, :valid_len]
                num_sampled[req_idx] = valid_len
                slot_tensor = torch.tensor(
                    [slot], dtype=torch.int64, device=states.device
                )
                states.init_canvas(slot_tensor)
                self.req_states.draft_tokens[
                    slot, : self.canvas_length
                ] = states.canvas[slot]
                states.is_encoder_phase[slot] = False
                if trace_requests is not None:
                    trace_requests.append(
                        {
                            "req_idx": req_idx,
                            "slot": slot,
                            "valid_len": valid_len,
                            "is_commit": True,
                            "step_before": step_before,
                            "step_after": int(states.step[slot].item()),
                            "mask_count_before": mask_count_before,
                            "mask_count_after": int(
                                (
                                    states.canvas[slot, :valid_len]
                                    == self.mask_token_id
                                )
                                .sum()
                                .item()
                            ),
                            "transferred": 0,
                        }
                    )
                continue

            current = states.canvas[slot, :valid_len]
            mask_index = current == self.mask_token_id
            reached_max_steps = False
            transfer_count = 0
            transferred = 0
            if bool(mask_index.any().item()):
                row_confidence = torch.where(
                    mask_index,
                    confidence[start:end].to(current.device),
                    torch.full(
                        (valid_len,),
                        -torch.inf,
                        dtype=confidence.dtype,
                        device=current.device,
                    ),
                )
                step = step_before
                denoising_steps = self._denoising_steps_for_slot(slot)
                transfer_schedule = self._transfer_schedule_for_steps(denoising_steps)
                transfer_count = transfer_schedule[min(step, denoising_steps - 1)]
                transfer_index = select_transfer_indices(
                    row_confidence.unsqueeze(0),
                    threshold=float(self.dynamic_threshold_by_slot[slot].item())
                    if self.dynamic_threshold_by_slot is not None
                    else self.dynamic_threshold,
                    transfer_count=transfer_count,
                ).squeeze(0)
                if step + 1 >= denoising_steps:
                    transfer_index = mask_index
                sampled_row = sampled_tokens[start:end].to(current.device)
                current[transfer_index] = sampled_row[transfer_index]
                states.canvas[slot, :valid_len] = current
                states.step[slot] += 1
                reached_max_steps = step + 1 >= denoising_steps
                transferred = int(transfer_index.sum().item())

            done = reached_max_steps or not bool(
                (states.canvas[slot, :valid_len] == self.mask_token_id).any()
            )
            states.is_encoder_phase[slot] = done
            self.req_states.draft_tokens[slot, : self.canvas_length] = states.canvas[
                slot
            ]
            if trace_requests is not None:
                trace_requests.append(
                    {
                        "req_idx": req_idx,
                        "slot": slot,
                        "valid_len": valid_len,
                        "is_commit": False,
                        "step_before": step_before,
                        "step_after": int(states.step[slot].item()),
                        "mask_count_before": mask_count_before,
                        "mask_count_after": int(
                            (
                                states.canvas[slot, :valid_len]
                                == self.mask_token_id
                            )
                            .sum()
                            .item()
                        ),
                        "transfer_count": int(transfer_count),
                        "transferred": transferred,
                        "done": done,
                    }
                )

        output = self._build_output(
            input_batch,
            sampled,
            num_sampled,
            per_req_nlogits_np,
        )
        if trace_requests is not None:
            for request in trace_requests:
                req_idx = int(request["req_idx"])
                request["num_sampled"] = int(output.num_sampled[req_idx].item())
                request["num_rejected"] = int(output.num_rejected[req_idx].item())
            _mineru_debug_trace(
                "mineru_diffusion_sampler_denoise",
                batch=_mineru_batch_trace_payload(input_batch),
                per_req_num_logits=per_req_nlogits_np,
                requests=trace_requests,
            )
        return output


class MinerUDiffusionRequestStates:
    """Per-request runtime state for MinerU block diffusion scheduling."""

    def __init__(
        self,
        *,
        max_num_reqs: int,
        canvas_length: int,
        mask_token_id: int,
        device: torch.device,
    ) -> None:
        self.max_num_reqs = max_num_reqs
        self.canvas_length = canvas_length
        self.mask_token_id = mask_token_id
        self.device = device
        self.is_encoder_phase = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )
        self.canvas = torch.full(
            (max_num_reqs, canvas_length),
            mask_token_id,
            dtype=torch.int64,
            device=device,
        )
        self.step = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.prompt_len = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    def init_canvas(self, slots: torch.Tensor) -> None:
        self.canvas[slots, :] = self.mask_token_id
        self.step[slots] = 0

    def add_request(self, slot: int) -> None:
        slot_tensor = torch.tensor([slot], dtype=torch.int64, device=self.device)
        self.is_encoder_phase[slot] = True
        self.init_canvas(slot_tensor)

    def remove_request(self, slot: int) -> None:
        self.is_encoder_phase[slot] = False
        self.canvas[slot, :] = self.mask_token_id
        self.step[slot] = 0
        self.prompt_len[slot] = 0


class MinerUDiffusionModelState(DefaultModelState):
    """v2 ModelState registration point for MinerU block diffusion.

    This first stage switches MinerU off the default AR ModelState and gives the
    model a dedicated place to implement canvas state, draft-token scheduling,
    and mixed prefill/denoise attention.
    """

    num_new_sampled_tokens_per_step: int = 0

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: Any,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        diffusion_config = getattr(vllm_config, "diffusion_config", None)
        canvas_length = (
            diffusion_config.canvas_length
            if diffusion_config is not None and diffusion_config.canvas_length
            else 32
        )
        mask_token_id = getattr(self.model_config.hf_config, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = getattr(self.model_config.hf_text_config, "mask_token_id")
        self.diffusion_states = MinerUDiffusionRequestStates(
            max_num_reqs=self.max_num_reqs,
            canvas_length=canvas_length,
            mask_token_id=int(mask_token_id),
            device=device,
        )
        self._req_id_to_index: dict[str, int] = {}
        self._causal_buf = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        super().add_request(req_index, new_req_data)
        self._req_id_to_index[new_req_data.req_id] = req_index
        self.diffusion_states.add_request(req_index)
        prompt_ids = getattr(new_req_data, "prompt_token_ids", None)
        if prompt_ids is None:
            prompt_ids = getattr(new_req_data, "prefill_token_ids", [])
        self.diffusion_states.prompt_len[req_index] = len(prompt_ids)

    def remove_request(self, req_id: str) -> None:
        idx = self._req_id_to_index.pop(req_id, None)
        if idx is not None:
            self.diffusion_states.remove_request(idx)

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        diffusion_config = getattr(self.vllm_config, "diffusion_config", None)
        gen_config: Mapping[str, Any] = {}
        if hasattr(self.model_config, "try_get_generation_config"):
            gen_config = self.model_config.try_get_generation_config()
        denoising_steps = (
            diffusion_config.max_denoising_steps
            if diffusion_config is not None
            else None
        ) or self.diffusion_states.canvas_length
        return MinerUDiffusionSampler(
            sampler,
            diffusion_states=self.diffusion_states,
            canvas_length=self.diffusion_states.canvas_length,
            mask_token_id=self.diffusion_states.mask_token_id,
            denoising_steps=denoising_steps,
            dynamic_threshold=float(gen_config.get("dynamic_threshold", 0.85)),
        ), None

    def prepare_sample_hidden_states(
        self,
        hidden_states: torch.Tensor,
        input_batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not _env_flag_enabled(_MASK_ONLY_SAMPLING_ENV):
            return super().prepare_sample_hidden_states(hidden_states, input_batch)

        logits_row_indices = _mineru_mask_only_logits_row_indices(
            input_batch,
            self.diffusion_states,
            self.diffusion_states.mask_token_id,
            hidden_states.device,
        )
        return hidden_states[input_batch.logits_indices[logits_row_indices]], (
            logits_row_indices
        )

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: Any,
    ) -> torch.Tensor:
        with record_function_or_nullcontext("mineru_mm_embeddings: prepare"):
            mm_hashes, mm_kwargs = self.encoder_runner.prepare_mm_inputs(
                scheduled_encoder_inputs
            )

        with record_function_or_nullcontext("mineru_mm_embeddings: encoder"):
            if mm_kwargs:
                encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
                self.encoder_cache.encoder_outputs.update(
                    zip(mm_hashes, encoder_outputs)
                )

        with record_function_or_nullcontext("mineru_mm_embeddings: gather"):
            mm_embeds, is_mm_embed = self.encoder_runner.gather_mm_embeddings(
                input_batch.req_ids,
                input_batch.num_tokens,
                input_batch.num_scheduled_tokens,
                input_batch.query_start_loc_np,
                input_batch.prefill_len_np,
                input_batch.num_computed_prefill_tokens_np,
            )

        with record_function_or_nullcontext("mineru_mm_embeddings: input_embeds"):
            input_ids_unpadded = input_batch.input_ids[: input_batch.num_tokens]
            inputs_embeds = self.encoder_runner.get_inputs_embeds(
                input_ids_unpadded,
                mm_embeds,
                is_mm_embed,
            )
            return inputs_embeds[: input_batch.num_tokens_after_padding]

    def prepare_attn(
        self,
        input_batch: Any,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[Any]],
        kv_cache_config: Any,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = getattr(
            input_batch, "seq_lens_cpu_upper_bound", None
        )
        if for_capture:
            max_seq_len = self.max_model_len
        elif seq_lens_cpu_upper_bound is not None:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()
        else:
            max_seq_len = self.max_model_len

        actual_num_reqs = input_batch.num_reqs
        draft_counts = getattr(input_batch, "num_draft_tokens_per_req", None)
        if draft_counts is None:
            self._causal_buf[:actual_num_reqs] = True
        else:
            scheduled_canvas = torch.as_tensor(
                draft_counts[:actual_num_reqs],
                dtype=torch.int32,
                device=self.device,
            ) > 0
            self._causal_buf[:actual_num_reqs] = ~scheduled_canvas
        if actual_num_reqs < num_reqs:
            self._causal_buf[actual_num_reqs:num_reqs] = False

        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=getattr(input_batch, "dcp_local_seq_lens", None),
            positions=getattr(input_batch, "positions", None),
            for_cudagraph_capture=for_capture,
            causal=self._causal_buf[:num_reqs],
        )


def get_num_transfer_tokens(block_length: int, denoising_steps: int) -> list[int]:
    """Return MinerU's uniform per-step transfer budget."""
    if block_length <= 0:
        raise ValueError(f"block_length must be positive, got {block_length}")
    if denoising_steps <= 0:
        raise ValueError(f"denoising_steps must be positive, got {denoising_steps}")

    base = block_length // denoising_steps
    remainder = block_length % denoising_steps
    return [base + (1 if step < remainder else 0) for step in range(denoising_steps)]


def sample_with_temperature_topk_topp(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample token ids and return sampled-token probability as confidence."""
    if temperature <= 0:
        token_ids = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits.float(), dim=-1)
        confidence = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        return token_ids, confidence

    filtered = logits.float() / temperature
    if top_k > 0 and top_k < filtered.shape[-1]:
        kth = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(sorted_remove).scatter(
            -1, sorted_indices, sorted_remove
        )
        filtered = filtered.masked_fill(remove, float("-inf"))

    probs = torch.softmax(filtered, dim=-1)
    flat_probs = probs.reshape(-1, probs.shape[-1])
    token_ids = torch.multinomial(flat_probs, num_samples=1).reshape(probs.shape[:-1])
    confidence = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return token_ids, confidence


def select_transfer_indices(
    confidence: torch.Tensor,
    *,
    threshold: float,
    transfer_count: int,
) -> torch.Tensor:
    """Select positions transferred from sampled block output this step."""
    if transfer_count < 0:
        raise ValueError(f"transfer_count must be non-negative, got {transfer_count}")

    selected = torch.zeros_like(confidence, dtype=torch.bool)
    if transfer_count == 0:
        return selected

    batch = confidence.reshape(-1, confidence.shape[-1])
    valid = torch.isfinite(batch)
    high = ((confidence > threshold).reshape_as(batch)) & valid
    out = selected.reshape_as(batch)

    for row_idx in range(batch.shape[0]):
        row_high = high[row_idx]
        if int(row_high.sum().item()) >= transfer_count:
            out[row_idx] = row_high
        else:
            valid_idx = torch.nonzero(valid[row_idx], as_tuple=False).flatten()
            k = min(transfer_count, valid_idx.numel())
            if k == 0:
                continue
            top_idx = torch.topk(batch[row_idx, valid_idx], k=k).indices
            out[row_idx, valid_idx[top_idx]] = True
    return selected


class PatchMerger(nn.Module):

    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2):
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x).view(-1, self.hidden_size)
        return self.mlp(x)


def build_projection(projection_type: str, in_dim: int, out_dim: int) -> nn.Module:
    pm_match = re.match(r"(?:patch_merger|pm)(\d+)x$", projection_type)
    if pm_match:
        return PatchMerger(
            out_dim,
            in_dim,
            spatial_merge_size=int(pm_match.group(1)),
        )
    raise ValueError(
        "Only patch_merger-style MinerU projectors are supported, "
        f"got: {projection_type}"
    )


class PerceiverProjection(nn.Module):

    def __init__(self, projection_type: str, in_dim: int, out_dim: int):
        super().__init__()
        self.projection = build_projection(projection_type, in_dim, out_dim)

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        return self.projection(input_embeds)


class SDARRMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)


class SDARMLP(nn.Module):

    def __init__(self, config: Any):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class SDARRotaryEmbedding(nn.Module):

    def __init__(self, config: Any):
        super().__init__()
        dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        base = float(getattr(config, "rope_theta", 10000.0))
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        inv_freq = self.inv_freq.to(x.device)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded @ position_ids_expanded
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class SDARAttention(nn.Module):

    def __init__(
        self,
        config: Any,
        *,
        cache_config: Any | None = None,
        quant_config: Any | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = SDARRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = SDARRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        if cache_config is not None:
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_key_value_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim == 2 and hasattr(self, "attn"):
            num_tokens = hidden_states.shape[0]
            query_states = self.q_norm(
                self.q_proj(hidden_states).view(
                    num_tokens, self.num_heads, self.head_dim
                )
            )
            key_states = self.k_norm(
                self.k_proj(hidden_states).view(
                    num_tokens, self.num_key_value_heads, self.head_dim
                )
            )
            value_states = self.v_proj(hidden_states).view(
                num_tokens, self.num_key_value_heads, self.head_dim
            )
            cos, sin = position_embeddings
            if cos.ndim == 3 and cos.shape[0] == 1:
                cos = cos.squeeze(0)
                sin = sin.squeeze(0)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            query_states = (query_states * cos) + (rotate_half(query_states) * sin)
            key_states = (key_states * cos) + (rotate_half(key_states) * sin)
            attn_output = self.attn(
                query_states.reshape(num_tokens, -1),
                key_states.reshape(num_tokens, -1),
                value_states.reshape(num_tokens, -1),
            )
            return self.o_proj(attn_output.contiguous())

        input_shape = hidden_states.shape[:-1]
        query_shape = (*input_shape, self.num_heads, self.head_dim)
        kv_shape = (*input_shape, self.num_key_value_heads, self.head_dim)
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(query_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(kv_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attention_mask.bool() if attention_mask is not None else None,
            is_causal=False,
            scale=self.scaling,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
        return self.o_proj(attn_output.contiguous())


class SDARDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Any,
        *,
        cache_config: Any | None = None,
        quant_config: Any | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = SDARAttention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "self_attn"),
        )
        self.mlp = SDARMLP(config)
        self.input_layernorm = SDARRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = SDARRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class SDARModel(nn.Module):

    def __init__(
        self,
        config: Any,
        *,
        cache_config: Any | None = None,
        quant_config: Any | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.uses_vllm_attention = cache_config is not None
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                SDARDecoderLayer(
                    config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx}"),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = SDARRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = SDARRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.ndim == 2 and not self.uses_vllm_attention:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        if position_ids is None:
            if inputs_embeds.ndim == 2:
                position_ids = torch.arange(
                    inputs_embeds.shape[0], device=inputs_embeds.device
                )
            else:
                position_ids = torch.arange(
                    inputs_embeds.shape[1], device=inputs_embeds.device
                ).unsqueeze(0)
        elif inputs_embeds.ndim == 3 and position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
        return self.norm(hidden_states)


class SDARForCausalLM(nn.Module):

    def __init__(
        self,
        config: Any,
        *,
        cache_config: Any | None = None,
        quant_config: Any | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.uses_vllm_attention = cache_config is not None
        self.model = SDARModel(
            config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @staticmethod
    def _build_block_attention_mask(
        num_blocks: int,
        block_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        block_mask = torch.tril(
            torch.ones(num_blocks, num_blocks, device=device, dtype=torch.bool)
        )
        return block_mask.repeat_interleave(
            block_length, dim=0
        ).repeat_interleave(block_length, dim=1)

    def _build_full_attention_mask(
        self,
        prompt_length: int,
        gen_length: int,
        block_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        prompt_blocks = (prompt_length + block_length - 1) // block_length
        prompt_mask = self._build_block_attention_mask(
            prompt_blocks, block_length, device
        )
        prompt_mask = prompt_mask[-prompt_length:, -prompt_length:]

        gen_blocks = gen_length // block_length
        gen_mask = self._build_block_attention_mask(
            gen_blocks, block_length, device
        )

        full_mask = torch.zeros(
            prompt_length + gen_length,
            prompt_length + gen_length,
            device=device,
            dtype=torch.bool,
        )
        full_mask[:prompt_length, :prompt_length] = prompt_mask
        full_mask[prompt_length:, :prompt_length] = True
        full_mask[prompt_length:, prompt_length:] = gen_mask
        return full_mask

    def _initialize_generation_buffers(
        self,
        inputs_embeds: torch.Tensor,
        gen_length: int,
        mask_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, prompt_length, _ = inputs_embeds.shape
        device = inputs_embeds.device
        mask_token = torch.tensor([mask_token_id], device=device)
        mask_embeds = self.get_input_embeddings()(mask_token)
        generation_embeds = mask_embeds.unsqueeze(0).expand(
            batch_size, gen_length, -1
        )
        x_embeds = torch.cat([inputs_embeds, generation_embeds], dim=1)
        tokens = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_token_id,
            dtype=torch.long,
            device=device,
        )
        step_map = torch.zeros_like(tokens, dtype=torch.int64)
        step_time = torch.zeros_like(tokens, dtype=torch.float32)
        return x_embeds, tokens, step_map, step_time

    @staticmethod
    def _prepare_stop_tokens(
        stopping_criteria: Sequence[str] | None,
        tokenizer: Any,
        device: torch.device,
    ) -> list[torch.Tensor]:
        if stopping_criteria is None:
            return []
        if tokenizer is None:
            raise ValueError("tokenizer is required when stopping_criteria is set")
        return [
            torch.tensor(
                tokenizer.encode(stop, add_special_tokens=False),
                device=device,
            )
            for stop in stopping_criteria
        ]

    @staticmethod
    def _find_stop_position(
        generated_tokens: torch.Tensor,
        stop_tokens: Sequence[torch.Tensor],
    ) -> int | None:
        for stop_token in stop_tokens:
            stop_len = stop_token.numel()
            if stop_len == 0 or generated_tokens.numel() < stop_len:
                continue
            for end_idx in range(stop_len, generated_tokens.numel() + 1):
                if torch.equal(
                    generated_tokens[end_idx - stop_len:end_idx],
                    stop_token,
                ):
                    return end_idx - stop_len
        return None

    @torch.no_grad()
    def generate_with_embeds(
        self,
        inputs_embeds: torch.Tensor,
        gen_length: int,
        block_length: int,
        mask_token_id: int,
        denoising_steps: int = 8,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        remasking_strategy: str = "low_confidence_dynamic",
        dynamic_threshold: float = 0.85,
        stopping_criteria: Sequence[str] | None = None,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if gen_length % block_length != 0:
            raise ValueError(
                f"gen_length({gen_length}) must be multiple of "
                f"block_length({block_length})"
            )
        if remasking_strategy != "low_confidence_dynamic":
            raise ValueError(
                "Only remasking_strategy='low_confidence_dynamic' is supported."
            )
        if inputs_embeds.ndim != 3:
            raise ValueError("inputs_embeds must have shape [batch, seq, hidden]")

        prompt_length = inputs_embeds.shape[1]
        gen_blocks = gen_length // block_length
        full_attn_mask = self._build_full_attention_mask(
            prompt_length,
            gen_length,
            block_length,
            inputs_embeds.device,
        )
        position_ids = torch.arange(
            prompt_length + gen_length,
            device=inputs_embeds.device,
        ).unsqueeze(0)
        x_embeds, x, step_map, step_time = self._initialize_generation_buffers(
            inputs_embeds,
            gen_length,
            mask_token_id,
        )
        num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps)
        stop_tokens = self._prepare_stop_tokens(
            stopping_criteria, tokenizer, inputs_embeds.device
        )
        global_step = 0
        found_stop = False
        stop_pos = -1
        stop_chunk_end = -1
        start_time = time.perf_counter()

        for block_idx in range(gen_blocks):
            block_start = prompt_length + block_idx * block_length
            block_end = prompt_length + (block_idx + 1) * block_length
            cur_x = x[:, block_start:block_end]
            cur_x_embeds = x_embeds[:, block_start:block_end, :]
            cur_step_map = step_map[:, block_start:block_end]
            cur_step_time = step_time[:, block_start:block_end]

            for step in range(denoising_steps + 1):
                mask_index = cur_x == mask_token_id
                if int(mask_index.sum().item()) == 0:
                    break

                context_embeds = x_embeds[:, :block_end, :]
                context_attn_mask = full_attn_mask[
                    :block_end, :block_end
                ].unsqueeze(0).unsqueeze(0)
                context_position_ids = position_ids[:, :block_end]
                hidden_states = self(
                    input_ids=None,
                    inputs_embeds=context_embeds,
                    attention_mask=context_attn_mask,
                    position_ids=context_position_ids,
                )
                logits = self.lm_head(hidden_states)[:, block_start:block_end, :]
                sampled_ids, confidence = sample_with_temperature_topk_topp(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                confidence = torch.where(mask_index, confidence, -torch.inf)
                transfer_index = select_transfer_indices(
                    confidence,
                    threshold=dynamic_threshold,
                    transfer_count=num_transfer_tokens[
                        min(step, denoising_steps - 1)
                    ],
                )
                if int(transfer_index.sum().item()) == 0:
                    break

                global_step += 1
                sampled_embeds = self.get_input_embeddings()(sampled_ids)
                cur_x_embeds[transfer_index] = sampled_embeds[transfer_index]
                cur_x[transfer_index] = sampled_ids[transfer_index]
                cur_step_map[transfer_index] = global_step
                cur_step_time[transfer_index] = time.perf_counter() - start_time

            if stop_tokens:
                generated = x[0, prompt_length:block_end]
                stop_offset = self._find_stop_position(generated, stop_tokens)
                if stop_offset is not None:
                    found_stop = True
                    stop_pos = prompt_length + stop_offset
                    stop_chunk_end = block_end
            if found_stop:
                break

        if found_stop:
            x = x[:, :stop_pos]
            step_map = step_map[:, :stop_chunk_end]
            step_time = step_time[:, :stop_chunk_end]
        return (
            x[:, prompt_length:],
            step_map[:, prompt_length:],
            step_time[:, prompt_length:],
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None:
            raise ValueError("is_multimodal is required with multimodal_embeddings")
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )


class MinerUDiffusionProcessingInfo(Qwen2VLProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}


class MinerUDiffusionDummyInputsBuilder(
    BaseDummyInputsBuilder[MinerUDiffusionProcessingInfo]
):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        hf_processor = self.info.get_hf_processor()
        return hf_processor.image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        target_width, target_height = self.info.get_image_size_with_most_features()
        image_overrides = mm_options.get("image")
        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
        }


class MinerUDiffusionMultiModalProcessor(
    BaseMultiModalProcessor[MinerUDiffusionProcessingInfo]
):

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        image_token_id = tokenizer.get_vocab()[hf_processor.image_token]
        merge_length = image_processor.merge_size**2

        def get_replacement_mineru(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            num_tokens = int(grid_thw.prod()) // merge_length
            return [image_token_id] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=partial(get_replacement_mineru),
            )
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _create_qwen2vl_field_factory(
            self.info.get_hf_config().vision_config.spatial_merge_size
        )(hf_inputs)


@MULTIMODAL_REGISTRY.register_processor(
    MinerUDiffusionMultiModalProcessor,
    info=MinerUDiffusionProcessingInfo,
    dummy_inputs=MinerUDiffusionDummyInputsBuilder,
)
class MinerUDiffusionForConditionalGeneration(nn.Module, SupportsMultiModal):
    """Registration anchor for native MinerU-Diffusion support."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.",
            "language_model.": "language_model.",
            "model.visual.": "visual.",
            "visual.": "visual.",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.text_config = vllm_config.model_config.hf_text_config
        self.prefix = prefix
        self.vision_model = Qwen2VisionTransformerPretrainedModel._from_config(
            self.config.vision_config
        )
        self.vision_model.merger = nn.Identity()
        self.vision_abstractor = PerceiverProjection(
            projection_type=self.config.vision_projector_type,
            in_dim=self.config.vision_config.embed_dim,
            out_dim=self.text_config.hidden_size,
        )
        in_runtime_config = get_current_vllm_config_or_none() is not None
        cache_config = getattr(vllm_config, "cache_config", None)
        quant_config = getattr(vllm_config, "quant_config", None)
        self.language_model = SDARForCausalLM(
            self.text_config,
            cache_config=cache_config if in_runtime_config else None,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

    @staticmethod
    def get_model_state_cls():
        return MinerUDiffusionModelState

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if is_multimodal is None:
            raise ValueError("is_multimodal is required with multimodal_embeddings")
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        self.vision_model.to(device=pixel_values.device, dtype=pixel_values.dtype)
        image_grid_thw = image_grid_thw.to(device=pixel_values.device)
        vision_outputs = self.vision_model(pixel_values, image_grid_thw)
        if hasattr(vision_outputs, "last_hidden_state"):
            vision_hidden_states = vision_outputs.last_hidden_state
        elif isinstance(vision_outputs, (tuple, list)):
            vision_hidden_states = vision_outputs[0]
        else:
            vision_hidden_states = vision_outputs
        return self.vision_abstractor(vision_hidden_states)

    def _merge_input_and_image_features(
        self,
        input_ids: torch.Tensor,
        image_features: torch.Tensor | None,
    ) -> torch.Tensor:
        inputs_embeds = self.embed_input_ids(input_ids)
        if image_features is None:
            return inputs_embeds
        vision_mask = input_ids == self.config.image_token_id
        num_image_tokens = torch.count_nonzero(vision_mask).item()
        num_image_features = image_features.shape[:-1].numel()
        if num_image_tokens != num_image_features:
            raise ValueError(
                "vision token count mismatch: "
                f"{num_image_tokens} vs {num_image_features}"
            )
        return torch.masked_scatter(
            inputs_embeds,
            vision_mask.unsqueeze(-1),
            image_features.to(inputs_embeds.dtype).view(-1, image_features.size(-1)),
        )

    def embed_multimodal(self, **kwargs: object):
        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        if pixel_values is None:
            return []
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required with pixel_values")
        assert isinstance(pixel_values, torch.Tensor)
        assert isinstance(image_grid_thw, torch.Tensor)

        image_features = self.get_image_features(pixel_values, image_grid_thw)
        merge_size = self.config.vision_config.spatial_merge_size
        sizes = (image_grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return tuple(image_features.split(sizes))

    @torch.no_grad()
    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        **generate_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        image_features = None
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required with pixel_values")
            image_features = self.get_image_features(pixel_values, image_grid_thw)
        inputs_embeds = self._merge_input_and_image_features(
            input_ids,
            image_features,
        )
        return self.language_model.generate_with_embeds(
            inputs_embeds=inputs_embeds,
            **generate_kwargs,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            if input_ids.ndim == 1 and self.language_model.uses_vllm_attention:
                model_input_ids = input_ids
            elif input_ids.ndim == 1:
                model_input_ids = input_ids.unsqueeze(0)
            else:
                model_input_ids = input_ids
            image_features = None
            if pixel_values is not None:
                if image_grid_thw is None:
                    raise ValueError("image_grid_thw is required with pixel_values")
                image_features = self.get_image_features(pixel_values, image_grid_thw)
            inputs_embeds = self._merge_input_and_image_features(
                model_input_ids, image_features
            )
        elif inputs_embeds.ndim == 2 and not self.language_model.uses_vllm_attention:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        if positions.ndim == 1 and not self.language_model.uses_vllm_attention:
            position_ids = positions.unsqueeze(0)
        else:
            position_ids = positions

        hidden_states = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        return hidden_states.reshape(-1, hidden_states.shape[-1])

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.lm_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        loaded: set[str] = set()
        for name, weight in weights:
            target = params.get(name)
            if target is None:
                target = buffers.get(name)
            if target is None:
                continue
            with torch.no_grad():
                target.copy_(weight.to(dtype=target.dtype, device=target.device))
            loaded.add(name)
        return loaded
