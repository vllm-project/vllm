# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, device-resident sampling traces using the custom logits processor API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.sample.logits_processor.interface import (
    LogitsContext,
    LogitsProcessor,
)

if TYPE_CHECKING:
    import numpy as np
    import torch

    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.logits_processor.interface import (
        LogitsProcRequestState,
    )


@dataclass(frozen=True)
class SamplingSnapshot:
    """A step before temperature and truncation, with its actual row mapping."""

    logits: torch.Tensor
    request_slots: torch.Tensor
    row_offsets: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor
    batch_slots: np.ndarray
    sequence_lengths: np.ndarray
    labels: dict[int, str | None]


class SamplingTraceProcessor(LogitsProcessor):
    """Observe sampling without changing the logits returned to the sampler.

    Configure tensor storage bounds in additional_config["sampling_trace"].
    Snapshots stay on the original device. Drain them through take_snapshots()
    outside the inference loop before transferring them to the host.
    """

    def __init__(self, vllm_config: VllmConfig, req_states: LogitsProcRequestState):
        options = vllm_config.additional_config.get("sampling_trace", {})
        if not isinstance(options, dict):
            raise ValueError("sampling_trace must be an object")
        self.max_steps = options.get("max_steps", 4)
        self.max_bytes = options.get("max_bytes", 64 * 1024 * 1024)
        for name, value in [
            ("max_steps", self.max_steps),
            ("max_bytes", self.max_bytes),
        ]:
            if type(value) is not int or value < 0:
                raise ValueError(f"sampling_trace.{name} must be a nonnegative integer")
        self._snapshots: list[SamplingSnapshot] = []
        self._labels: dict[int, str | None] = {}
        self._bytes = 0
        self.dropped_steps = 0

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        label = (sampling_params.extra_args or {}).get("sampling_trace_label")
        if label is not None and (not isinstance(label, str) or len(label) > 128):
            raise ValueError(
                "sampling_trace_label must be a string of at most 128 characters"
            )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        self.validate_params(sampling_params)
        self._labels[req_idx] = (sampling_params.extra_args or {}).get(
            "sampling_trace_label"
        )
        # Keep this processor in the chain even though its output is unchanged.
        return self.max_steps > 0 and self.max_bytes > 0

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        if (
            logits.shape[0] == 0
            or len(self._snapshots) >= self.max_steps
            or self.max_bytes == 0
        ):
            return logits
        tensors = (
            logits,
            ctx.expanded_idx_mapping,
            ctx.expanded_local_pos,
            ctx.input_ids,
            ctx.pos,
        )
        required = sum(value.numel() * value.element_size() for value in tensors)
        required += ctx.idx_mapping_np.nbytes + ctx.seq_lens_upper_bound_np.nbytes
        if required > self.max_bytes - self._bytes:
            self.dropped_steps += 1
            return logits
        values = [value.detach().clone() for value in tensors]
        slots = ctx.idx_mapping_np.copy()
        snapshot = SamplingSnapshot(
            *values,
            batch_slots=slots,
            sequence_lengths=ctx.seq_lens_upper_bound_np.copy(),
            labels={int(slot): self._labels.get(int(slot)) for slot in slots},
        )
        self._snapshots.append(snapshot)
        self._bytes += required
        return logits

    def take_snapshots(self) -> tuple[SamplingSnapshot, ...]:
        """Transfer ownership of captured steps and reset the storage budget."""
        snapshots = tuple(self._snapshots)
        self._snapshots.clear()
        self._bytes = 0
        return snapshots


class SamplingTraceWorkerExtension:
    """Drain device snapshots through an explicit worker RPC after generation."""

    def take_sampling_traces(self) -> list[tuple[SamplingSnapshot, ...]]:
        """Return device snapshots; caller must consume them outside inference.

        Use an in-process executor for device tensors. Multiprocess RPC needs
        host serialization; transfer tensors explicitly in an application
        extension after the engine is paused, before returning from the RPC.
        """
        sampler = self.model_runner.sampler
        if sampler is None:
            return []
        return [
            processor.take_snapshots()
            for processor in sampler.logits_processors
            if isinstance(processor, SamplingTraceProcessor)
        ]
