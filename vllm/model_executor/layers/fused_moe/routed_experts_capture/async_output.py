# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asynchronous routed-experts output handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.routed_experts_capture.shared_region import (
    RoutedExpertsWorkerWriter,
)

if TYPE_CHECKING:
    from vllm.v1.outputs import ModelRunnerOutput


class RoutedExpertsTensors(NamedTuple):
    """Store one step of routed-experts tensors pending async D2H."""

    # (num_scheduled_tokens, num_layers, moe_top_k)
    routing_data: torch.Tensor
    # (num_scheduled_tokens,)
    slot_mapping: torch.Tensor

    def to_cpu_nonblocking(self) -> RoutedExpertsTensors:
        """Copy the tensors to CPU without blocking the current stream."""
        if self.routing_data.device.type == "cpu":
            return self
        return RoutedExpertsTensors(
            self.routing_data.to("cpu", non_blocking=True),
            self.slot_mapping.to("cpu", non_blocking=True),
        )

    def tolists(self) -> RoutedExpertsLists:
        """Convert the tensors to the numpy-backed worker representation."""
        return RoutedExpertsLists(
            self.routing_data.cpu().numpy(),
            self.slot_mapping.cpu().numpy(),
        )


class RoutedExpertsLists(NamedTuple):
    """Store one step of CPU routed-experts data and slot indices."""

    # (num_scheduled_tokens, num_layers, moe_top_k)
    routing_data: np.ndarray
    # (num_scheduled_tokens,)
    slot_mapping: np.ndarray


@dataclass
class RoutedExpertsWriteTask:
    """Copy and publish one step of routed-experts output."""

    routed_experts_tensors: RoutedExpertsTensors
    writer: RoutedExpertsWorkerWriter
    _routed_experts_tensors_cpu: RoutedExpertsTensors | None = field(
        init=False, default=None
    )

    def start_copy(self) -> None:
        """Start copying the routed-experts tensors on the current stream."""
        self._routed_experts_tensors_cpu = (
            self.routed_experts_tensors.to_cpu_nonblocking()
        )

    def finalize(self, output: ModelRunnerOutput) -> None:
        """Publish the copied routing data and update the model output."""
        assert self._routed_experts_tensors_cpu is not None, (
            "routed-experts CPU tensors are unavailable; call start_copy first"
        )
        routed_experts = self._routed_experts_tensors_cpu.tolists()
        self.writer.store_batch(
            routed_experts.routing_data,
            routed_experts.slot_mapping,
        )
        output.routed_experts_slots = routed_experts.slot_mapping
