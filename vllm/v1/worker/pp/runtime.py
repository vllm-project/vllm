# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional activation buffering and per-batch PP boundaries for MRv2."""

import json
from collections.abc import Mapping
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.pp.boundary import (
    HEADER_KEY,
    PPBoundary,
    PPBoundaryController,
    PPStrategyReporter,
    validate_fresh_prefill,
)
from vllm.v1.worker.pp.buffer import PPSendBuffer
from vllm.v1.worker.pp.config import get_pp_config
from vllm.v1.worker.pp.feedback import PPDeviceFeedback
from vllm.v1.worker.pp.interface import SupportsDynamicPP

logger = init_logger(__name__)


class PPRuntime:
    def __init__(
        self,
        config: VllmConfig,
        model: torch.nn.Module,
        device: torch.device,
        device_module: Any,
        pp_group: GroupCoordinator,
        tp_group: GroupCoordinator,
    ):
        size, dynamic = get_pp_config(config)
        self.model = model
        self.pp = pp_group
        self.tp = tp_group
        self.batch_id = 0
        self.header: PPBoundary | None = None
        self.controller: PPBoundaryController | None = None
        self.feedback: PPDeviceFeedback | None = None
        self.reporter: PPStrategyReporter | None = None
        self.buffer: PPSendBuffer | None = None
        if dynamic.enabled:
            if not isinstance(model, SupportsDynamicPP):
                raise ValueError("This model has no dynamic PP layer-range adapter")
            spec = model.get_dynamic_pp_layer_spec()
            self.controller = PPBoundaryController(
                dynamic, spec.base_boundary, spec.num_layers, spec.minimum_boundary
            )
            if tp_group.rank_in_group == 0:
                self.feedback = PPDeviceFeedback(
                    device_module,
                    pp_group.cpu_group.get_group_store(),
                    pp_group.rank_in_group,
                    dynamic.feedback_timeout_seconds,
                )
                self.model.register_forward_pre_hook(self.feedback.forward_start)
                self.reporter = PPStrategyReporter(
                    self.controller, pp_group.rank_in_group
                )
        if size and pp_group.world_size > 1 and not pp_group.is_last_rank:
            self.buffer = PPSendBuffer(size, device_module, metadata_handles=1)
            template = model.make_empty_intermediate_tensors(
                config.scheduler_config.max_num_batched_tokens,
                dtype=config.model_config.dtype,
                device=device,
            )
            self.buffer.reserve(template.tensors)
            logger.info(
                "Reserved %d PP send slots (%d bytes)", size, self.buffer.reserved_bytes
            )

    def prepare(self, output: SchedulerOutput) -> None:
        self.header = None
        if not output.total_num_scheduled_tokens:
            return
        if self.controller is not None:
            validate_fresh_prefill(output)
            if self.pp.is_first_rank:
                wire = None
                if self.feedback is not None:
                    self.feedback.poll()
                    self.header = self.controller.choose(
                        self.batch_id,
                        self.feedback.queue_depth,
                        self.feedback.take_samples(),
                        feedback_fresh=self.feedback.fresh,
                    )
                    wire = self.header.to_wire()
                wire = self.tp.broadcast_object(wire)
                if self.header is None:
                    self.header = self.controller.accept(wire, self.batch_id)
        if self.buffer is not None:
            self.buffer.acquire()

    def receive_header(self, tensors: dict[str, Any]) -> None:
        if self.controller is not None:
            self.header = self.controller.accept(
                tensors.pop(HEADER_KEY, None), self.batch_id
            )

    def before_forward(self, output: SchedulerOutput) -> None:
        if self.header is None:
            return
        assert self.controller is not None
        assert isinstance(self.model, SupportsDynamicPP)
        self.model.set_dynamic_pp_layer_range(
            *self.controller.execution_range(self.pp.rank_in_group, self.header)
        )
        if self.feedback is not None:
            self.feedback.begin(
                self.header,
                output.total_num_scheduled_tokens,
                len(output.num_scheduled_tokens),
            )
        if self.reporter is not None:
            pending = self.buffer.pending_count if self.buffer is not None else None
            report = self.reporter.observe(self.header, pending)
            if report is not None:
                assert self.feedback is not None
                report.update(self.feedback.stats())
                logger.info(
                    "Dynamic PP strategy %s", json.dumps(report, sort_keys=True)
                )
        self.batch_id += 1

    def send(self, tensors: Mapping[str, Any]) -> None:
        assert self.buffer is not None
        payload = dict(tensors)
        if self.header is not None:
            payload[HEADER_KEY] = self.header.to_wire()
        self.buffer.send(
            payload,
            lambda value: self.pp.isend_tensor_dict(value, all_gather_group=self.tp),
            on_snapshot=self.feedback.finish if self.feedback is not None else None,
        )

    def sampled(self) -> None:
        if self.feedback is not None and self.pp.is_last_rank:
            self.feedback.finish()

    def cancel(self) -> None:
        if self.buffer is not None:
            self.buffer.cancel()

    def drain(self) -> None:
        if self.buffer is not None:
            self.buffer.drain()

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.controller is not None,
            "batches": self.batch_id,
            "boundary": self.controller.boundary if self.controller else None,
            "boundary_version": self.controller.version if self.controller else None,
            "buffer_sends": self.buffer.num_sends if self.buffer else 0,
            "backpressure_waits": self.buffer.num_backpressure_waits
            if self.buffer
            else 0,
            "device_feedback": self.feedback.stats() if self.feedback else None,
        }
