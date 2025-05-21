# SPDX-License-Identifier: Apache-2.0

import dataclasses
from collections.abc import Iterable
from typing import Any, Callable, Optional

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement


@dataclasses.dataclass
class CudaGraphBenchParams:
    num_ops_in_cuda_graph: int


@dataclasses.dataclass
class ArgPool:
    """
    When some argument of the benchmarking function is annotated with this type,
    the benchmarking class (BenchMM) will collapse the argument to a pick a
    single value from the given list of values, during function invocation.
    For every invocation during a benchmarking run, it will choose a
    different value from the list.
    """

    values: Iterable[Any]

    def __getitem__(self, index):
        return self.values[index]


class Bench:
    class ArgsIterator:
        def __init__(self, args_list, kwargs_list):
            assert len(args_list) == len(kwargs_list)
            self.args_list = args_list
            self.kwargs_list = kwargs_list
            self.n = len(self.args_list)
            self.idx = 0

        def __next__(self):
            while True:
                yield (self.args_list[self.idx], self.kwargs_list[self.idx])
                self.idx += 1
                self.idx = self.idx % self.n

        def reset(self):
            self.idx = 0

        @property
        def n_args(self):
            return self.n

    def __init__(
        self,
        cuda_graph_params: Optional[CudaGraphBenchParams],
        label: str,
        sub_label: str,
        description: str,
        fn: Callable,
        *args,
        **kwargs,
    ):
        self.cuda_graph_params = cuda_graph_params
        self.use_cuda_graph = self.cuda_graph_params is not None
        self.label = label
        self.sub_label = sub_label
        self.description = description
        self.fn = fn

        # Process args
        self._args = args
        self._kwargs = kwargs
        self.args_list, self.kwargs_list = self.collapse_argpool(*args, **kwargs)
        self.args_iterator = self.ArgsIterator(self.args_list, self.kwargs_list)

        # Cudagraph runner
        self.g = None
        if self.use_cuda_graph:
            self.g = self.get_cuda_graph_runner()

        # benchmark run params
        self.min_run_time = 1

    def collapse_argpool(self, *args, **kwargs):
        argpool_args = [arg for arg in args if isinstance(arg, ArgPool)] + [
            arg for arg in kwargs.values() if isinstance(arg, ArgPool)
        ]
        if len(argpool_args) == 0:
            return [args], [kwargs]

        # Make sure all argpools are of the same size
        argpool_size = len(argpool_args[0].values)
        assert all([argpool_size == len(arg.values) for arg in argpool_args])

        # create copies of the args
        args_list = []
        kwargs_list = []
        for _ in range(argpool_size):
            args_list.append(args)
            kwargs_list.append(kwargs.copy())

        for i in range(argpool_size):
            # collapse args; Just pick the ith value
            args_list[i] = tuple(
                [arg[i] if isinstance(arg, ArgPool) else arg for arg in args_list[i]]
            )

            # collapse kwargs
            kwargs_i = kwargs_list[i]
            arg_pool_keys = [k for k, v in kwargs_i.items() if isinstance(v, ArgPool)]
            for k in arg_pool_keys:
                # again just pick the ith value
                kwargs_i[k] = kwargs_i[k][i]
            kwargs_list[i] = kwargs_i

        return args_list, kwargs_list

    def get_cuda_graph_runner(self):
        assert self.use_cuda_graph
        assert self.args_iterator is not None

        num_graph_ops = self.cuda_graph_params.num_ops_in_cuda_graph

        # warmup
        args_it = self.args_iterator.__next__()
        for _ in range(2):
            args, kwargs = next(args_it)
            self.fn(*args, **kwargs)

        self.args_iterator.reset()
        args_it = self.args_iterator.__next__()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(num_graph_ops):
                    args, kwargs = next(args_it)
                    self.fn(*args, **kwargs)
        return g

    def run_cudagrah(self) -> TMeasurement:
        assert self.use_cuda_graph
        globals = {"g": self.g}

        return TBenchmark.Timer(
            stmt="g.replay()",
            globals=globals,
            label=(
                f"{self.label}"
                f" | cugraph {self.cuda_graph_params.num_ops_in_cuda_graph} ops"
            ),
            sub_label=self.sub_label,
            description=self.description,
        ).blocked_autorange(min_run_time=self.min_run_time)

    def run_eager(self) -> TMeasurement:
        setup = None
        stmt = None
        globals = None

        has_arg_pool = self.args_iterator.n_args > 1
        if has_arg_pool:
            setup = """
                    args_iterator.reset()
                    args_it = args_iterator.__next__()
                    """
            stmt = """
                    args, kwargs = next(args_it)
                    fn(*args, **kwargs)
                    """
            globals = {"fn": self.fn, "args_iterator": self.args_iterator}
        else:
            # no arg pool. Just use the args and kwargs directly
            self.args_iterator.reset()
            args_it = self.args_iterator.__next__()
            args, kwargs = next(args_it)

            setup = ""
            stmt = """
                    fn(*args, **kwargs)
                   """
            globals = {"fn": self.fn, "args": args, "kwargs": kwargs}

        return TBenchmark.Timer(
            stmt=stmt,
            setup=setup,
            globals=globals,
            label=self.label,
            sub_label=self.sub_label,
            description=self.description,
        ).blocked_autorange(min_run_time=self.min_run_time)

    def run(self) -> TMeasurement:
        timer = None
        if self.use_cuda_graph:  # noqa SIM108
            timer = self.run_cudagrah()
        else:
            timer = self.run_eager()
        if not timer.meets_confidence() or timer.has_warnings:
            print("Doesn't meet confidence - re-running bench ...")
            return self.run()
        return timer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            print(f"exc type {exc_type}")
            print(f"exc value {exc_value}")
            print(f"exc traceback {traceback}")
