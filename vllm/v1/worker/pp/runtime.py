# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional activation buffering and per-batch PP boundaries for MRv2."""

import json
from collections.abc import Mapping
from typing import Any, cast

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
    validate_dynamic_decode,
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
        self.device_module = device_module
        self.pp = pp_group
        self.tp = tp_group
        self.batch_id = 0
        self.header: PPBoundary | None = None
        self.controller: PPBoundaryController | None = None
        self.feedback: PPDeviceFeedback | None = None
        self.reporter: PPStrategyReporter | None = None
        self.buffer: PPSendBuffer | None = None
        self.allow_mtp_decode = dynamic.enabled and dynamic.allow_mtp_decode
        self.prefix_caching = (
            dynamic.enabled and config.cache_config.enable_prefix_caching
        )
        self.stateful = self.allow_mtp_decode or self.prefix_caching
        self.kv_transfer_backend = dynamic.kv_transfer_backend
        self.mooncake_protocol = dynamic.mooncake_protocol
        self.mooncake_device_name = dynamic.mooncake_device_name
        self.state_transfer: Any = None
        self.control: Any = None
        self._state_boundary: int | None = None
        if dynamic.enabled:
            if not isinstance(model, SupportsDynamicPP):
                raise ValueError("This model has no dynamic PP layer-range adapter")
            spec = model.get_dynamic_pp_layer_spec()
            self.controller = PPBoundaryController(
                dynamic, spec.base_boundary, spec.num_layers, spec.minimum_boundary
            )
            self._state_boundary = self.controller.boundary
            if self.stateful and not callable(
                getattr(model, "get_dynamic_pp_cache_state", None)
            ):
                raise ValueError(
                    "Dynamic PP history needs an isolated cache-state adapter"
                )
            if self.prefix_caching and not callable(
                getattr(model, "get_dynamic_pp_cache_layout", None)
            ):
                raise ValueError(
                    "Dynamic PP prefix caching needs a cache-layout adapter"
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
                if self.stateful:
                    from vllm.v1.worker.pp.state import PPControlChannel

                    self.control = PPControlChannel(self.feedback.store)
        if size and pp_group.world_size > 1 and not pp_group.is_last_rank:
            self.buffer = PPSendBuffer(size, device_module, metadata_handles=1)
            template = cast(Any, model).make_empty_intermediate_tensors(
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
            if self.allow_mtp_decode:
                validate_dynamic_decode(output, allow_cached_prefix=self.prefix_caching)
            else:
                validate_fresh_prefill(output, allow_cached_prefix=self.prefix_caching)
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
            if self.stateful:
                # Bound unpublished/unconsumed headers by the output ring. A
                # completed send implies the consumer already read its header.
                if self.buffer is not None:
                    self.buffer.acquire()
                if self.pp.is_first_rank:
                    if self.control is not None:
                        assert self.header is not None
                        self.control.send(self.header)
                else:
                    wire = None
                    if self.control is not None:
                        wire = self.control.receive(self.batch_id).to_wire()
                    wire = self.tp.broadcast_object(wire)
                    self.header = self.controller.accept(wire, self.batch_id)
                if self.state_transfer is None:
                    from vllm.v1.worker.pp.state import PPStateTransfer

                    try:
                        adapter = cast(Any, self.model)
                        cache_state = adapter.get_dynamic_pp_cache_state()
                        cache_layout = (
                            adapter.get_dynamic_pp_cache_layout()
                            if self.prefix_caching
                            else None
                        )
                        if self.prefix_caching and not cache_layout:
                            raise ValueError("Prefix cache layout must not be empty")
                    except (TypeError, ValueError, RuntimeError):
                        logger.exception("Dynamic PP cache-state adapter failed")
                        # Both peers still enter the collective validator. It
                        # rejects an empty mapping on every stage, avoiding a
                        # peer waiting alone after a local adapter error.
                        cache_state = {}
                        cache_layout = None
                    self.state_transfer = PPStateTransfer(
                        self.pp,
                        self.device_module,
                        cache_state,
                        cache_layout=cache_layout,
                        backend=self.kv_transfer_backend,
                        mooncake_protocol=self.mooncake_protocol,
                        mooncake_device_name=self.mooncake_device_name,
                    )
                assert self._state_boundary is not None
                assert self.header is not None
                self.state_transfer.move(
                    self._state_boundary, self.header.boundary, self.batch_id
                )
                self._state_boundary = self.header.boundary
        if self.buffer is not None and not self.stateful:
            self.buffer.acquire()

    def receive_header(self, tensors: dict[str, Any]) -> None:
        if self.controller is not None:
            wire = tensors.pop(HEADER_KEY, None)
            if self.stateful:
                if self.header is None or PPBoundary.from_wire(wire) != self.header:
                    raise ValueError("PP activation and cache ownership epochs differ")
            else:
                self.header = self.controller.accept(wire, self.batch_id)

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

    def close(self) -> None:
        self.drain()
        if self.state_transfer is not None:
            self.state_transfer.close()

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.controller is not None,
            "prefix_caching": self.prefix_caching,
            "batches": self.batch_id,
            "boundary": self.controller.boundary if self.controller else None,
            "boundary_version": self.controller.version if self.controller else None,
            "buffer_sends": self.buffer.num_sends if self.buffer else 0,
            "backpressure_waits": self.buffer.num_backpressure_waits
            if self.buffer
            else 0,
            "device_feedback": self.feedback.stats() if self.feedback else None,
            "state_transfer": self.state_transfer.stats()
            if self.state_transfer is not None
            else None,
        }
