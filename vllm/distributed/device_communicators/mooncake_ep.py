# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from vllm.config import get_current_vllm_config
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
)
from vllm.distributed.utils import StatelessProcessGroup


@dataclass
class _MooncakeEPBufferState:
    buffer: Any
    max_num_ep_ranks: int
    connected_ep_size: int
    active_ep_size: int


class MooncakeEPAll2AllManager(All2AllManagerBase):
    """All2All communication based on Mooncake EP kernels.
    This backend supports elastic EP with dynamic rank connection/disconnection.
    """

    _buffer: _MooncakeEPBufferState | None = None
    _lock = threading.RLock()
    _mask: torch.Tensor | None = None
    _last_mask: torch.Tensor | None = None

    def __init__(self, cpu_group, tcp_store_group=None):
        if tcp_store_group is None:
            tcp_store_group = StatelessProcessGroup(
                rank=cpu_group.rank(),
                world_size=cpu_group.size(),
                store=dist.PrefixStore("mooncake_ep", cpu_group.get_group_store()),
            )
        super().__init__(cpu_group, tcp_store_group)
        self.support_fault_tolerance = True

    @property
    def max_num_ep_ranks(self) -> int:
        assert MooncakeEPAll2AllManager._buffer is not None
        return MooncakeEPAll2AllManager._buffer.max_num_ep_ranks

    def _init_buffer(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_experts_per_rank: int,
    ) -> None:
        from mooncake.mooncake_ep_buffer import Buffer

        parallel_config = get_current_vllm_config().parallel_config
        max_num_ep_ranks = (
            self.world_size
            * parallel_config.elastic_ep_max_dp_size
            // parallel_config.data_parallel_size
            if parallel_config.enable_elastic_ep
            else self.world_size
        )
        max_num_global_experts = max_num_ep_ranks * num_experts_per_rank
        num_rdma_bytes = Buffer.get_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=max_num_ep_ranks,
            num_experts=max_num_global_experts,
        )
        assert MooncakeEPAll2AllManager._buffer is None, (
            "Mooncake EP buffer already initialized"
        )
        buffer = Buffer.create_elastic(
            rank=self.rank,
            tcp_store_group=self.tcp_store_group.store,
        )
        buffer.update_memory_buffers(
            num_ranks=max_num_ep_ranks,
            num_experts_per_rank=num_experts_per_rank,
            num_rdma_bytes=num_rdma_bytes,
        )
        ranks_to_connect = list(range(self.world_size))
        buffer.connect_ranks(ranks_to_connect)
        MooncakeEPAll2AllManager._buffer = _MooncakeEPBufferState(
            buffer=buffer,
            max_num_ep_ranks=max_num_ep_ranks,
            connected_ep_size=self.world_size,
            active_ep_size=self.world_size,
        )

    def _connect_to_ep_size(self, ep_size: int, *, make_active: bool) -> None:
        assert MooncakeEPAll2AllManager._buffer is not None
        state = MooncakeEPAll2AllManager._buffer
        if ep_size <= state.connected_ep_size:
            return

        state.buffer.set_tcp_store_group(self.tcp_store_group.store)
        ranks_to_connect = list(range(state.connected_ep_size, ep_size))
        state.buffer.connect_ranks(ranks_to_connect, activate=make_active)
        state.connected_ep_size = ep_size
        if make_active:
            state.active_ep_size = ep_size

    def _disconnect_to_ep_size(self, ep_size: int) -> None:
        assert MooncakeEPAll2AllManager._buffer is not None
        state = MooncakeEPAll2AllManager._buffer
        if ep_size >= state.connected_ep_size:
            return

        state.buffer.set_tcp_store_group(self.tcp_store_group.store)
        ranks_to_disconnect = list(range(ep_size, state.connected_ep_size))
        state.buffer.disconnect_ranks(ranks_to_disconnect)
        state.connected_ep_size = ep_size
        state.active_ep_size = min(state.active_ep_size, ep_size)

    def _unmask_connected_ranks(self, target_ep_size: int) -> None:
        assert MooncakeEPAll2AllManager._buffer is not None
        state = MooncakeEPAll2AllManager._buffer
        state.buffer.set_tcp_store_group(self.tcp_store_group.store)
        if target_ep_size <= state.active_ep_size:
            return
        assert state.connected_ep_size >= target_ep_size

        for rank in range(state.active_ep_size, target_ep_size):
            state.buffer.update_mask_buffer(rank, mask=False)
        torch.accelerator.synchronize()
        state.active_ep_size = target_ep_size

    def _stage_ep_size(self) -> None:
        assert MooncakeEPAll2AllManager._buffer is not None
        state = MooncakeEPAll2AllManager._buffer
        target_ep_size = self.world_size

        # Scale-up can safely connect standby ranks while leaving them masked.
        # Scale-down must not disconnect active ranks until commit.
        if target_ep_size > state.connected_ep_size:
            self._connect_to_ep_size(target_ep_size, make_active=False)

    def stage_ep_size(self) -> None:
        with MooncakeEPAll2AllManager._lock:
            self._stage_ep_size()

    def commit_ep_size(self) -> None:
        """Commit the staged EP size to the active communication set."""
        with MooncakeEPAll2AllManager._lock:
            assert MooncakeEPAll2AllManager._buffer is not None
            state = MooncakeEPAll2AllManager._buffer
            target_ep_size = self.world_size

            if target_ep_size < state.connected_ep_size:
                self._disconnect_to_ep_size(target_ep_size)
            elif target_ep_size > state.connected_ep_size:
                self._connect_to_ep_size(target_ep_size, make_active=True)

            self._unmask_connected_ranks(target_ep_size)

    @contextmanager
    def mask_remote_ranks(self) -> Iterator[None]:
        peers = [rank for rank in range(self.world_size) if rank != self.rank]
        assert MooncakeEPAll2AllManager._buffer is not None
        buffer = MooncakeEPAll2AllManager._buffer.buffer
        for rank in peers:
            buffer.update_mask_buffer(rank, mask=True)
        torch.accelerator.synchronize()
        try:
            yield
        finally:
            for rank in peers:
                buffer.update_mask_buffer(rank, mask=False)
            torch.accelerator.synchronize()

    def get_handle(self, kwargs):
        with MooncakeEPAll2AllManager._lock:
            state = MooncakeEPAll2AllManager._buffer
            if state is None:
                max_num_tokens_per_dp_rank = kwargs["max_num_tokens_per_dp_rank"]
                self._init_buffer(
                    max_num_tokens_per_dp_rank=max_num_tokens_per_dp_rank,
                    token_hidden_size=kwargs["token_hidden_size"],
                    num_experts_per_rank=kwargs["num_local_experts"],
                )

            assert MooncakeEPAll2AllManager._buffer is not None
            handle = MooncakeEPAll2AllManager._buffer.buffer
            return handle

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        raise NotImplementedError

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError

    def destroy(self):
        # The manager instance is recreated during
        # scale-up/down, so we cannot destroy the persistent buffer here.
        state = MooncakeEPAll2AllManager._buffer
        if state is not None:
            state.buffer.set_tcp_store_group(None)

    def query_active_mask(self) -> torch.Tensor:
        state = MooncakeEPAll2AllManager._buffer
        assert state is not None
        if MooncakeEPAll2AllManager._mask is None:
            MooncakeEPAll2AllManager._mask = torch.zeros(
                state.max_num_ep_ranks, device="cuda", dtype=torch.int32
            )
        state.buffer.query_mask_buffer(MooncakeEPAll2AllManager._mask)
        return MooncakeEPAll2AllManager._mask[: state.active_ep_size]

    def query_fault(self) -> torch.Tensor:
        current = self.query_active_mask()
        last = MooncakeEPAll2AllManager._last_mask
        if last is None or last.shape != current.shape:
            MooncakeEPAll2AllManager._last_mask = torch.zeros_like(current)
            last = MooncakeEPAll2AllManager._last_mask
        has_fault = (current != last).any()
        return has_fault

    def clean_buffers(self) -> None:
        if MooncakeEPAll2AllManager._buffer is None:
            return
        state = MooncakeEPAll2AllManager._buffer
        state.buffer.get_local_buffer_tensor(
            dtype=torch.int8, use_rdma_buffer=True
        ).zero_()
        torch.accelerator.synchronize()
        state.buffer.clean_mask_buffer()
        torch.accelerator.synchronize()
        MooncakeEPAll2AllManager._last_mask = None

    def update_mask(self, rank: int, masked: bool = True) -> None:
        state = MooncakeEPAll2AllManager._buffer
        assert state is not None
        state.buffer.update_mask_buffer(rank, mask=masked)
        torch.accelerator.synchronize()
        last = MooncakeEPAll2AllManager._last_mask
        if last is None or last.shape[0] != state.active_ep_size:
            MooncakeEPAll2AllManager._last_mask = torch.zeros(
                state.active_ep_size, device="cuda", dtype=torch.int32
            )
            last = MooncakeEPAll2AllManager._last_mask
        last[rank] = int(masked)
