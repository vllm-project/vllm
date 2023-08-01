from abc import ABC, abstractmethod
from time import perf_counter_ns
from statistics import mean


class KernelBenchmark(ABC):

    def __init__(self, loop_time) -> None:
        super().__init__()
        loop_time = loop_time if loop_time > 2 else 3

        self.loop_time = loop_time
        self.time = []

    def execute(self):
        for i in range(self.loop_time):
            start = perf_counter_ns()
            self._run()
            end = perf_counter_ns()
            self.time.append(end - start)

        self.time.sort()
        avg = mean(self.time[1:-1])
        print("Execution time: {} ns".format(avg))

    @abstractmethod
    def _run(self):
        pass
