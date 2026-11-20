# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation-send storage for fixed-partition pipeline parallelism."""

from collections.abc import Mapping
from typing import Any

from vllm.v1.worker.pp.buffer import PPSendBuffer
from vllm.v1.worker.pp.config import get_pp_buffer_size


class PPSendRuntime:
    def __init__(self, config, model, device, device_module, pp_group, tp_group):
        self.pp = pp_group
        self.tp = tp_group
        self.buffer = None
        size = get_pp_buffer_size(config)
        if size and pp_group.world_size > 1 and not pp_group.is_last_rank:
            self.buffer = PPSendBuffer(size, device_module, metadata_handles=1)
            template = model.make_empty_intermediate_tensors(
                config.scheduler_config.max_num_batched_tokens,
                dtype=config.model_config.dtype,
                device=device,
            )
            self.buffer.reserve(template.tensors)

    def prepare(self, output) -> None:
        if output.total_num_scheduled_tokens and self.buffer is not None:
            self.buffer.acquire()

    def send(self, tensors: Mapping[str, Any]) -> None:
        assert self.buffer is not None
        self.buffer.send(
            tensors,
            lambda value: self.pp.isend_tensor_dict(value, all_gather_group=self.tp),
        )

    def cancel(self) -> None:
        if self.buffer is not None:
            self.buffer.cancel()

    def drain(self) -> None:
        if self.buffer is not None:
            self.buffer.drain()
