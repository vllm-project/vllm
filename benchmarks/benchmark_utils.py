# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from types import TracebackType


# Collect time and generate time metrics
#
# Example Usage:
#   collector = TimeCollector(TimeCollector.US)
#   for _ in range(total_iteration):
#      with collector:
#          ...
#   collector.dump_avg_max()
class TimeCollector:
    NS: int = 1
    US: int = NS * 1000
    MS: int = US * 1000
    S: int = MS * 1000

    def __init__(self, scale: int) -> None:
        self.cnt: int = 0
        self._sum: int = 0
        self._max: int | None = None
        self.scale = scale
        self.start_time: int = time.monotonic_ns()

    def collect(self, v: int) -> None:
        self.cnt += 1
        self._sum += v
        if self._max is None:
            self._max = v
        else:
            self._max = max(self._max, v)

    def avg(self) -> float | str:
        return self._sum * 1.0 / self.cnt / self.scale if self.cnt > 0 else "N/A"

    def max(self) -> float | str:
        return self._max / self.scale if self._max else "N/A"

    def dump_avg_max(self) -> list[float | str]:
        return [self.avg(), self.max()]

    def __enter__(self) -> None:
        self.start_time = time.monotonic_ns()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        self.collect(time.monotonic_ns() - self.start_time)
