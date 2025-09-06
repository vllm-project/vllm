# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional, TypeAlias, Union

import numpy as np
from scipy.stats import expon

NaiveBlockAllocator: TypeAlias = deque


@dataclass
class ExtraInfo:
    workload: str
    reach_time: float
    prefilled_time: float
    cur_time: float


def get_hashs(token_ids: list[int], block_size: int = 128):
    is_first = True
    prev = None
    for i in range((len(token_ids)) // block_size):
        cur = token_ids[i * block_size : (i + 1) * block_size]
        h = hash((is_first, prev, *cur))
        is_first = False
        prev = h
        yield h


# system prompt judge threshold
sp_reuse_limit: float = 0.3


class SldingWindow:
    def __init__(self, constructor, window_size: float = 600.0, ty="time"):
        self.window_size = window_size
        self.ty = ty
        if window_size == -1:
            if ty == "time":
                self.window_size = float("inf")
            else:
                self.window_size = int(1e18)
        self.constructor = constructor
        self.data: defaultdict = defaultdict(constructor)
        # type-> queue[(ts, data)]
        self.watcher: defaultdict[str, deque] = defaultdict(deque)

    def add(self, key: str, value, ts, clean_to=None):
        if clean_to:
            self._clean(key, clean_to + self.window_size)
        else:
            if self.ty == "time":
                self._clean(key, ts)
            else:
                self._clean_num(key, int(self.window_size))

        self.watcher[key].append((ts, value))
        if self.constructor is int:
            self.data[key] += value
        elif self.constructor is deque:
            self.data[key].append(value)
        else:
            raise RuntimeError("unimplemented")

    def _clean(self, key, ts):
        # self.watcher[key] = deque(sorted(self.watcher[key], key=lambda x: x[0]))
        while self.watcher[key] and ts - self.watcher[key][0][0] > self.window_size:
            if self.constructor is int:
                self.data[key] -= self.watcher[key][0][1]
            elif self.constructor is deque:
                self.data[key].popleft()
            else:
                raise RuntimeError("unimplemented")
            self.watcher[key].popleft()

    def _clean_num(self, key, num):
        while len(self.watcher[key]) > num:
            if self.constructor is int:
                self.data[key] -= self.watcher[key][0][1]
            elif self.constructor is deque:
                self.data[key].popleft()
            else:
                raise RuntimeError("unimplemented")
            self.watcher[key].popleft()


class Monitor:
    def __init__(self, window_size: int):
        self.sliding_window: SldingWindow = SldingWindow(deque, window_size)
        self.workload_cnt: SldingWindow = SldingWindow(int, window_size)
        self.workload_block_hit_cnt: SldingWindow = SldingWindow(int, window_size)

        self.workload_block_cnt: SldingWindow = SldingWindow(int, window_size)

    def record(
        self,
        extras: ExtraInfo,
        num_block: Optional[int] = None,
        reuse_from: Optional[str] = None,
        reuse_time: Optional[float] = None,
        last_add_time: Optional[float] = None,
    ):
        # record a reuse at reuse_form
        workload = extras.workload
        if reuse_from and reuse_time and reuse_time > sp_reuse_limit:
            self.sliding_window.add(reuse_from, reuse_time, extras.reach_time)
            if self.workload_block_cnt.watcher[reuse_from]:
                assert last_add_time is not None
                self.workload_block_hit_cnt.add(
                    reuse_from,
                    1,
                    last_add_time,
                    clean_to=self.workload_cnt.watcher[reuse_from][0][0],
                )

        self.workload_cnt.add(workload, 1, extras.reach_time)
        if num_block:
            self.workload_block_cnt.add(workload, 1, extras.reach_time)

    def get_hit_coefficient(self) -> dict[str, float]:
        def get_exp_fitting(data: list[float]):
            params = expon.fit(data)
            if params[1] == 0:
                mean_reuse_time = np.mean(data)
                return 1.0 / mean_reuse_time if mean_reuse_time > 0 else 1e9
            lambda_hat = 1 / params[1]
            return lambda_hat

        coefficients = {}
        for workload, reuse_times in self.sliding_window.data.items():
            if len(reuse_times) > 0:
                coefficients[workload] = get_exp_fitting(reuse_times)
        for workload in self.workload_cnt.data:
            if workload not in coefficients:
                coefficients[workload] = (
                    0  # TODO: When a workload never reused, what's the coefficient?
                )
        return coefficients

    def get_next_turn_p(self) -> dict[str, float]:
        res = {}
        for workload, cnt in self.workload_block_hit_cnt.data.items():
            if (
                workload in self.workload_cnt.data
                and self.workload_cnt.data[workload] != 0
            ):
                res[workload] = cnt / self.workload_cnt.data[workload]
            else:
                res[workload] = 0
        for workload in self.workload_cnt.data:
            # TODO: if no monitored data in res, what probability to set?
            if workload not in res:
                res[workload] = 1
        return res

    def get_percentiles(self) -> dict[str, list[float]]:
        res = {}
        for workload, reuse_times in self.sliding_window.data.items():
            if len(reuse_times) > 0:
                percentiles = list(range(1, 100))
                ps = np.percentile(reuse_times, percentiles)
                res[workload] = ps
        for workload in self.workload_cnt.data:
            if workload not in res:
                res[workload] = []
        return res


class BlockPool:
    def __init__(self, device: str, num_block: int):
        self.device = device
        self.num_block: int = num_block

        self.free_blocks: deque[int] = deque([i for i in range(num_block)])  # block idx
        self.ref_counts: defaultdict[int, int] = defaultdict(int)  # block_idx -> count
        self.hit_times: defaultdict[int, list[float]] = defaultdict(
            list
        )  # block_idx -> hit time
        self.reuse_times: defaultdict[int, list[float]] = defaultdict(
            list
        )  # block_idx -> reuse time
        self.last_add_time: dict[int, float] = {}  # block_idx -> last add(update) time
        self.mean_reuse_time: dict[int, float] = {}  # block_idx -> mean reuse time
        self.is_sp: dict[int, bool] = {}  # block_idx -> is system prompt or not
        self.block_workload: defaultdict[int, str] = {}  # block_idx -> workload type
        self.lookup_table: dict[int, int] = dict()  # hash -> block_idx

    def __contains__(self, h: int) -> bool:
        return h in self.lookup_table

    def _is_system_prompt(self, v: Union[list[float], float]):
        return np.mean(v or float("inf")) < sp_reuse_limit

    def alloc(
        self,
        h: int,
        num_prefix_block: int,
        hit_times: list[float],
        reuse_times: list[float],
        extras: ExtraInfo,
        monitor: Monitor,
    ) -> int:
        """Alloc a block for hash, return the allocated block index and
        return a old_hash if this block has."""
        if self.device == "GPU":
            if self.free_blocks:
                block_idx = self.free_blocks.popleft()
            else:
                raise MemoryError(
                    "Avaliable GPU block pool is empty, please"
                    "increase the gpu block number for profiler!"
                )
            self.ref_counts[block_idx] += 1
            self.hit_times[block_idx] = hit_times
            self.reuse_times[block_idx] = reuse_times
            self.last_add_time[block_idx] = extras.reach_time
            self.mean_reuse_time[block_idx] = np.mean(reuse_times or float("inf"))
            self.is_sp[block_idx] = self._is_system_prompt(
                self.mean_reuse_time[block_idx]
            )
            self.block_workload[block_idx] = extras.workload
            self.lookup_table[h] = block_idx
            return block_idx

    def free(self, h: int, num_prefix_block: int, extras: ExtraInfo):
        assert self.device == "GPU"
        assert h in self.lookup_table, f"hash [{h}] is not in lookup table"
        block_idx = self.lookup_table[h]
        assert self.ref_counts[block_idx] > 0, f"hash [{h}] is not in pool now"
        self.ref_counts[block_idx] -= 1
        if self.ref_counts[block_idx] == 0:
            self.lookup_table.pop(h)

    def update(
        self,
        h: int,
        cur_time: float,
        extras: ExtraInfo,
        monitor: Monitor,
    ) -> int:
        assert self.device == "GPU", "update only in GPU"
        if h in self.lookup_table:
            block_idx = self.lookup_table[h]
            self.ref_counts[block_idx] += 1
            reuse_time = cur_time - self.hit_times[block_idx][-1]
            monitor.record(
                extras,
                1,
                self.block_workload[block_idx],
                reuse_time,
                self.last_add_time[block_idx],
            )
            self.block_workload[block_idx] = extras.workload
            self.hit_times[block_idx].append(cur_time)
            new_mean_reuse_time = (
                (
                    0
                    if len(self.reuse_times[block_idx]) == 0
                    else self.mean_reuse_time[block_idx]
                    * len(self.reuse_times[block_idx])
                )
                + reuse_time
            ) / (len(self.reuse_times[block_idx]) + 1)
            self.reuse_times[block_idx].append(reuse_time)
            self.last_add_time[block_idx] = extras.reach_time
            self.mean_reuse_time[block_idx] = new_mean_reuse_time
            self.is_sp[block_idx] = self._is_system_prompt(new_mean_reuse_time)
            return block_idx
        else:
            raise ValueError(f"hash [{h}] is not in this pool")


class BlockManager:
    def __init__(
        self,
        block_size: int,
        num_block: int,
    ):
        self.block_size: int = block_size
        self.num_block: int = num_block

        self.gpu_pool: BlockPool = BlockPool(device="GPU", num_block=num_block)
        # set window_size = -1 means consider all history
        self.monitor: Monitor = Monitor(window_size=-1)
        self.gpu_hit: int = 0

        self.total_alloc: int = 0

    def get_coefficients(self) -> dict:
        return self.monitor.get_hit_coefficient()

    def get_next_turn_p(self) -> dict:
        return self.monitor.get_next_turn_p()

    def alloc(self, token_ids: list[int], extras: ExtraInfo) -> int:
        hit_cnt = 0
        cur_time = extras.cur_time

        for i, h in enumerate(get_hashs(token_ids)):
            self.total_alloc += 1
            if h in self.gpu_pool:  # hit
                self.gpu_pool.update(h, cur_time, extras, self.monitor)
                self.gpu_hit += 1
                hit_cnt += 1
            else:  # new allocate
                self.gpu_pool.alloc(h, i, [cur_time], [], extras, self.monitor)
                self.monitor.record(extras, 1)
        return hit_cnt

    def free(self, token_ids: list[int], extras: ExtraInfo):
        for i, h in enumerate(get_hashs(token_ids)):
            assert h in self.gpu_pool, f"hash [{h}] is not in gpu_pool now"
            self.gpu_pool.free(h, i, extras)
