# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

logger = init_logger(__name__)


class RdnaCustomAllreduce:
    """Graph-only HIP all-reduce for single-node, homogeneous RDNA TP groups.

    All ranks must construct and close this communicator together. Graphs that
    reference it must finish before close and must not execute concurrently.
    """

    _SUPPORTED_ARCHES = ("gfx1100", "gfx1201")
    _BUFFER_BYTES = 128 * 8192 * 2
    _TUNING_ENV = (
        "VLLM_RDNA3_TP2_BLOCKS",
        "VLLM_RDNA3_TP4_BLOCKS",
        "VLLM_RDNA3_TP4_PAIR_MASK",
        "VLLM_RDNA3_TP4_CROSS_MASK",
        "VLLM_RDNA4_TP2_BLOCKS",
        "VLLM_RDNA4_TP4_BLOCKS",
        "VLLM_RDNA4_TP4_PAIR_MASK",
        "VLLM_RDNA4_TP4_CROSS_MASK",
        "VLLM_RDNA4_TP4_ALGO",
        "VLLM_RDNA4_ONESHOT_MAX_NUMEL",
    )

    def __init__(
        self, group: ProcessGroup, device: torch.device, enabled: bool
    ) -> None:
        self.disabled = True
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.arch = ""
        self._context = 0
        self._shared = 0
        self._opened: list[int] = []
        self._closed = True
        self._ops: Any = None
        self._rank_data: torch.Tensor | None = None
        self._payload = 0

        # Even ranks with the option disabled participate in the decision.
        settings = self._gather((enabled, envs.VLLM_ROCM_USE_RDNA_ALL_REDUCE))
        self.requested = any(enabled or requested for enabled, requested in settings)
        flags = [enabled for enabled, _ in settings]
        if not all(flags):
            if any(flags):
                logger.warning("RDNA all-reduce disabled: inconsistent rank settings")
            return
        if self.world_size not in (2, 4):
            return
        if not all(in_the_same_node_as(group, source_rank=0)):
            return

        error = None
        identity = None
        visible: dict[str, int] = {}
        peers: list[int] = []
        try:
            import vllm._rocm_C  # noqa: F401

            self._ops = torch.ops._rdna_custom_ar
            self._ops.meta_size()
            with torch.accelerator.device_index(device.index):
                props = torch.cuda.get_device_properties(device)
                self.arch = props.gcnArchName.split(":")[0]
                if self.arch not in self._SUPPORTED_ARCHES:
                    raise ValueError(f"unsupported architecture {self.arch}")
                if any(name in os.environ for name in self._TUNING_ENV):
                    raise ValueError(
                        "standalone kernel tuning overrides are unsupported"
                    )
                identity = (self.arch, str(props.uuid))
                visible = {
                    str(torch.cuda.get_device_properties(i).uuid): i
                    for i in range(torch.accelerator.device_count())
                }
                peers = self._ops.get_required_peer_ranks(self.rank, self.world_size)
        except Exception as exc:
            error = str(exc)
        reports = self._gather((identity, error))
        identities = [item[0] for item in reports]
        errors = [item[1] for item in reports if item[1] is not None]
        if errors or len({item[0] for item in identities if item}) != 1:
            logger.warning("RDNA all-reduce disabled: incompatible ranks %s", reports)
            return
        if len({item[1] for item in identities}) != self.world_size:
            logger.warning("RDNA all-reduce disabled: ranks must use distinct GPUs")
            return

        error = None
        try:
            with torch.accelerator.device_index(device.index):
                for peer in peers:
                    ordinal = visible.get(identities[peer][1])
                    if ordinal is None or not torch.cuda.can_device_access_peer(
                        device.index, ordinal
                    ):
                        raise ValueError(f"no visible P2P access to rank {peer}")
        except Exception as exc:
            error = str(exc)
        if not self._agree(error):
            return

        handle = None
        error = None
        self._closed = False
        try:
            with torch.accelerator.device_index(device.index):
                self._shared, handle = self._ops.allocate_shared_buffer_and_handle(
                    self._BUFFER_BYTES, self.world_size
                )
                self._rank_data = torch.empty(
                    self._ops.rank_data_size(), dtype=torch.uint8, device=device
                )
        except Exception as exc:
            error = str(exc)
        if not self._agree(error):
            self.close()
            return
        handles = self._gather(handle)

        error = None
        try:
            with torch.accelerator.device_index(device.index):
                pointers = [0] * self.world_size
                pointers[self.rank] = self._shared
                for peer in peers:
                    pointer = self._ops.open_mem_handle(handles[peer])
                    self._opened.append(pointer)
                    pointers[peer] = pointer
                payloads = [p + self._ops.meta_size() if p else 0 for p in pointers]
                self._context = self._ops.init_custom_ar(
                    pointers, self._rank_data, self.rank, self._BUFFER_BYTES
                )
                self._ops.register_buffer(self._context, payloads)
                self._payload = payloads[self.rank]
        except Exception as exc:
            error = str(exc)
        if not self._agree(error):
            self.close()
            return
        self.disabled = False
        logger.info(
            "Enabled graph-only RDNA HIP all-reduce: %s TP%d",
            self.arch,
            self.world_size,
        )

    def _gather(self, value: Any) -> list[Any]:
        values: list[Any] = [None] * self.world_size
        dist.all_gather_object(values, value, group=self.group)
        return values

    def _agree(self, error: str | None) -> bool:
        errors = self._gather(error)
        if any(item is not None for item in errors):
            logger.warning(
                "RDNA all-reduce initialization failed; using fallback: %s", errors
            )
            return False
        return True

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled or inp.device != self.device:
            return False
        if inp.dtype not in (torch.float16, torch.bfloat16) or inp.dim() != 2:
            return False
        if not is_forward_context_available():
            return False
        batch_descriptor = get_forward_context().batch_descriptor
        if (
            batch_descriptor is None
            or not batch_descriptor.uniform
            or batch_descriptor.num_reqs != batch_descriptor.num_tokens
        ):
            return False
        batch_size = batch_descriptor.num_tokens
        if batch_size <= 0 or inp.shape[0] != batch_size:
            return False
        hidden_size = inp.shape[1]
        if batch_size > 128 or hidden_size <= 0 or hidden_size > 8192:
            return False
        if inp.numel() % 8:
            return False
        # Stay below RDNA3's variable-performance tagged/bulk ring region.
        if self.arch == "gfx1100" and inp.numel() >= 128 * 1024:
            return False
        return torch.cuda.is_current_stream_capturing()

    def custom_all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        if not self.should_custom_ar(inp):
            return None
        # Alignment/strides can differ across ranks. Stage locally instead of
        # letting a rank-local pointer property choose a different collective.
        if not inp.is_contiguous() or inp.data_ptr() % 16:
            inp = inp.clone(memory_format=torch.contiguous_format)
        out = torch.empty_like(inp)
        self._ops.all_reduce(self._context, inp, out, self._payload, self._BUFFER_BYTES)
        return out

    def close(self) -> None:
        """Collectively release IPC mappings after all captured work finishes."""
        if self._closed:
            return
        self.disabled = True
        with torch.accelerator.device_index(self.device.index):
            torch.accelerator.synchronize()
            dist.barrier(group=self.group)
            if self._context:
                self._ops.dispose(self._context)
                self._context = 0
            for pointer in self._opened:
                self._ops.close_mem_handle(pointer)
            self._opened.clear()
            # Exporters may free their allocation only after all peers unmap it.
            dist.barrier(group=self.group)
            if self._shared:
                self._ops.free_shared_buffer(self._shared)
                self._shared = 0
            self._rank_data = None
        self._closed = True
