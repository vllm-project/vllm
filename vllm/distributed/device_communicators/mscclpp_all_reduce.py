# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import bisect
import math
import os
from contextlib import contextmanager
from enum import IntEnum

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)


class MscclContextSelection(IntEnum):
    MSCCL1SHOT1NODELL = 1
    MSCCL1SHOT2NODELL = 2


def _is_weak_contiguous(inp: torch.Tensor) -> bool:
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


def _bench_time(func, test_niter: int = 10, warmup_niter: int = 2) -> float:
    for _ in range(warmup_niter):
        func()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    dist.barrier()
    start_event.record()
    for _ in range(test_niter):
        func()
    end_event.record()
    end_event.synchronize()
    return start_event.elapsed_time(end_event) / test_niter * 1000


def _load_mscclpp_ops():
    try:
        import vllm_mscclpp_ops  # noqa: F401

        return torch.ops.vllm_mscclpp
    except ImportError:
        return None


class MscclppAllReduce:
    _SUPPORTED_WORLD_SIZES = [8, 16]
    _SUPPORTED_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
    _DEFAULT_MAX_BYTES = int(os.getenv("VLLM_MSCCLPP_MAX_BYTES", "1048576"))

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_bytes: int | None = None,
    ) -> None:
        self.disabled = True
        self._context = None

        self._ops = _load_mscclpp_ops()
        if self._ops is None:
            logger.warning(
                "MSCCL++ ops not available (vllm_mscclpp_ops not found). "
                "MSCCL++ allreduce disabled."
            )
            return

        self.group = group
        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)

        if world_size == 1:
            return

        if world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "MSCCL++ allreduce disabled: unsupported world size %d. Supported: %s",
                world_size,
                self._SUPPORTED_WORLD_SIZES,
            )
            return

        ranks = dist.get_process_group_ranks(group)
        if abs(ranks[-1] - ranks[0]) != world_size - 1:
            logger.warning(
                "MSCCL++ allreduce disabled: non-consecutive ranks %s",
                ranks,
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        self.max_bytes = max_bytes or self._DEFAULT_MAX_BYTES
        self.rank = rank
        self.world_size = world_size
        self.nranks_per_node = torch.cuda.device_count()

        unique_id = [self._ops.mscclpp_generate_unique_id()] if rank == 0 else [None]
        dist.broadcast_object_list(unique_id, src=ranks[0], group=self.group)
        self.unique_id = unique_id[0]

        self.rank_to_node = [r // 8 for r in range(world_size)]
        self.rank_to_ib = [rank % 8 for _ in range(world_size)]

        if world_size == 8:
            self.context_selection = MscclContextSelection.MSCCL1SHOT1NODELL
        elif world_size == 16:
            self.context_selection = MscclContextSelection.MSCCL1SHOT2NODELL

        self.scratch = torch.empty(
            self.max_bytes * 8, dtype=torch.uint8, device=self.device
        )
        self.put_buffer = torch.empty(
            self.max_bytes * 8 // self.nranks_per_node,
            dtype=torch.uint8,
            device=self.device,
        )

        self._context = self._ops.mscclpp_init_context(
            self.unique_id,
            self.rank,
            self.world_size,
            self.scratch,
            self.put_buffer,
            self.nranks_per_node,
            self.rank_to_node,
            self.rank_to_ib,
            int(self.context_selection),
        )

        self.msg_size_for_finetune = [
            2**i for i in range(10, math.floor(math.log2(self.max_bytes)) + 1)
        ]
        self.msg_size2best_config: dict[int, tuple[int, int]] = {}
        self._pre_tune_config()

        config_list = [self.msg_size2best_config] if rank == 0 else [None]
        dist.broadcast_object_list(config_list, src=ranks[0], group=self.group)
        self.msg_size2best_config = config_list[0]

        self.disabled = True
        logger.info(
            "MSCCL++ allreduce initialized (rank=%d, world_size=%d, "
            "max_bytes=%d, context=%s)",
            rank,
            world_size,
            self.max_bytes,
            self.context_selection.name,
        )

    def _pre_tune_config(self, dtype: torch.dtype = torch.bfloat16) -> None:
        nthreads_to_try = [256, 512, 1024]
        nblocks_to_try = [21, 42, 84]
        inp_randn = torch.ones(
            self.msg_size_for_finetune[-1] // dtype.itemsize,
            dtype=dtype,
            device=self.device,
        )
        oup_randn = torch.empty_like(inp_randn)
        for msg_size in self.msg_size_for_finetune:
            mock_inp = inp_randn[: msg_size // dtype.itemsize]
            mock_outp = oup_randn[: msg_size // dtype.itemsize]
            best_config = None
            best_time = None
            for nthreads in nthreads_to_try:
                for nblocks in nblocks_to_try:
                    cur_cost = _bench_time(
                        lambda nt=nthreads, nb=nblocks, mi=mock_inp, mo=mock_outp: (
                            self._ops.mscclpp_allreduce(self._context, mi, mo, nt, nb)
                        )
                    )
                    if best_time is None or cur_cost < best_time:
                        best_config = (nthreads, nblocks)
                        best_time = cur_cost
            self.msg_size2best_config[msg_size] = best_config
            if self.rank == 0:
                logger.debug(
                    "MSCCL++ tune: msg_size=%d best_config=%s time=%.1fus",
                    msg_size,
                    best_config,
                    best_time,
                )

    def should_mscclpp_allreduce(self, inp: torch.Tensor) -> bool:
        if self.disabled or self._context is None:
            return False
        if inp.dtype not in self._SUPPORTED_DTYPES:
            return False
        if not _is_weak_contiguous(inp):
            return False
        return inp.numel() * inp.element_size() <= self.max_bytes

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        msg_size = inp.numel() * inp.element_size()
        index = bisect.bisect_left(self.msg_size_for_finetune, msg_size)
        if index >= len(self.msg_size_for_finetune):
            index = len(self.msg_size_for_finetune) - 1
        msg_size_finetune = self.msg_size_for_finetune[index]
        nthreads, nblocks = self.msg_size2best_config[msg_size_finetune]
        result = torch.empty_like(inp)
        self._ops.mscclpp_allreduce(self._context, inp, result, nthreads, nblocks)
        return result

    @contextmanager
    def capture(self):
        old_disabled = self.disabled
        self.disabled = False
        try:
            yield
        finally:
            self.disabled = old_disabled

    def destroy(self):
        self._context = None
        self.scratch = None
        self.put_buffer = None
