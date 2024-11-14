import dataclasses
import random
from typing import Any, Callable, Iterable, Optional

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from utils import make_n_rand_tensors

import vllm._custom_ops as ops


@dataclasses.dataclass
class CudaGraphBenchParams:
    num_ops_in_cuda_graph: int


@dataclasses.dataclass
class ArgPool:
    '''
    When some argument of the benchmarking function is annotated with this type,
    the benchmarking class (BenchMM) will collapse the argument to a pick a
    single value from the given list of values, during function invocation.

    For every invocation during a benchmarking run, it will choose a
    different value from the list.
    '''
    values: Iterable[Any]


class BenchMM:

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

    def __init__(self, cuda_graph_params: Optional[CudaGraphBenchParams],
                 label: str, sub_label: str, description: str, fn: Callable,
                 *args, **kwargs):

        self.cuda_graph_params = cuda_graph_params
        self.use_cuda_graph = self.cuda_graph_params is not None
        self.label = label
        self.sub_label = sub_label
        self.description = description
        self.fn = fn

        # Process args
        self._args = args
        self._kwargs = kwargs
        self.args_list, self.kwargs_list = self.collapse_argpool(
            *args, **kwargs)
        self.args_iterator = self.ArgsIterator(self.args_list,
                                               self.kwargs_list)

        # Cudagraph runner
        self.g = None
        if self.use_cuda_graph:
            self.g = self.get_cuda_graph_runner()

        # benchmark run params
        self.min_run_time = 1

    def collapse_argpool(self, *args, **kwargs):
        kwargs = kwargs if kwargs is not None else {}
        assert kwargs is None or all([
            not isinstance(v, ArgPool) for k, v in kwargs.items()
        ]), 'ArgPools in kwargs are not supported yet'

        arg_pool_indices = [
            i for i, x in enumerate(args) if isinstance(x, ArgPool)
        ]
        if len(arg_pool_indices) == 0:
            return [args], [kwargs]

        # make sure all the Arg pools have the same number of choices
        arg_pool_size = len(args[arg_pool_indices[0]].values)
        assert all(
            [len(args[i].values) == arg_pool_size for i in arg_pool_indices])

        # create copies of the args
        args_list = []
        kwargs_list = []
        for _ in range(arg_pool_size):
            args_list.append(args)
            kwargs_list.append(kwargs.copy())

        # collapse the arg pools by simply choosing the ith value
        for i in range(arg_pool_size):
            assert isinstance(args_list[i], tuple)
            # get as list
            args_i = list(args_list[i])
            # collapse - make replacements
            for arg_pool_idx in arg_pool_indices:
                val_from_pool = args_i[arg_pool_idx].values[i]
                args_i[arg_pool_idx] = val_from_pool
            # store back as tuple
            args_list[i] = tuple(args_i)

        return args_list, kwargs_list

    def get_cuda_graph_runner(self):
        assert self.use_cuda_graph
        assert self.args_iterator is not None

        num_graph_ops = self.cuda_graph_params.num_ops_in_cuda_graph

        # warmup
        args_it = self.args_iterator.__next__()
        for _ in range(5):
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
        globals = {'g': self.g}

        return TBenchmark.Timer(
            stmt="g.replay()",
            globals=globals,
            label=self.label,
            sub_label=self.sub_label,
            description=self.description,
        ).blocked_autorange(min_run_time=self.min_run_time)

    def run_eager(self) -> TMeasurement:
        setup = None
        stmt = None
        globals = None

        has_arg_pool = self.args_iterator.n_args > 1
        if has_arg_pool:
            setup = '''
                    args_iterator.reset()
                    args_it = args_iterator.__next__()
                    '''
            stmt = '''
                    args, kwargs = next(args_it)
                    fn(*args, **kwargs)
                    '''
            globals = {'fn': self.fn, 'args_iterator': self.args_iterator}
        else:
            # no arg pool. Just use the args and kwargs directly
            self.args_iterator.reset()
            args_it = self.args_iterator.__next__()
            args, kwargs = next(args_it)

            setup = ""
            stmt = '''
                    fn(*args, **kwargs)
                   '''
            globals = {'fn': self.fn, 'args': args, 'kwargs': kwargs}

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
        #assert timer.meets_confidence()
        #assert not timer.has_warnings, f"Warnings {timer._warnings}"
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


def bench_fp8(dtype: torch.dtype, with_cuda_graph: Optional[int],
              with_arg_pool: Optional[int], m: int, k: int, n: int, label: str,
              sub_label: str) -> Iterable[TMeasurement]:

    arg_pool_size = with_arg_pool if with_arg_pool else 1
    cuda_graph_params: Optional[CudaGraphBenchParams] = None
    if with_cuda_graph:
        num_ops_in_cuda_graph = with_cuda_graph if with_cuda_graph else None
        cuda_graph_params = CudaGraphBenchParams(num_ops_in_cuda_graph)

    assert dtype == torch.float8_e4m3fn

    # Make input As and Bs
    As, Bs = make_n_rand_tensors(arg_pool_size, torch.float8_e4m3fn, m, n, k)
    bf16_As = [x.to(dtype=torch.bfloat16) for x in As]
    bf16_Bs = [x.to(dtype=torch.bfloat16) for x in Bs]
    # shuffle As and Bs to prevent any suspicion of pattern exploitation
    random.shuffle(As)
    random.shuffle(Bs)
    random.shuffle(bf16_As)
    random.shuffle(bf16_Bs)

    # Make scales and biases
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    timers = []

    # pytorch impl w. bf16
    with BenchMM(cuda_graph_params, label, sub_label,
                 "pytorch_bf16_bf16_bf16_matmul-no-scales", torch.mm,
                 ArgPool(bf16_As), ArgPool(bf16_Bs)) as bench:
        timers.append(bench.run())

    ## pytorch impl: bf16 output, without fp8 fast accum
    with BenchMM(cuda_graph_params,
                 label,
                 sub_label,
                 "pytorch_fp8_fp8_bf16_scaled_mm",
                 torch._scaled_mm,
                 ArgPool(As),
                 ArgPool(Bs),
                 scale_a=scale_a,
                 scale_b=scale_b,
                 out_dtype=torch.bfloat16) as bench:
        timers.append(bench.run())

    ## pytorch impl: bf16 output, with fp8 fast accum
    with BenchMM(cuda_graph_params,
                 label,
                 sub_label,
                 "pytorch_fp8_fp8_bf16_scaled_mm_fast_accum",
                 torch._scaled_mm,
                 ArgPool(As),
                 ArgPool(Bs),
                 scale_a=scale_a,
                 scale_b=scale_b,
                 out_dtype=torch.bfloat16,
                 use_fast_accum=True) as bench:
        timers.append(bench.run())

    ## cutlass impl: bf16 output
    with BenchMM(cuda_graph_params, label, sub_label,
                 "cutlass_fp8_fp8_bf16_scaled_mm", ops.cutlass_scaled_mm,
                 ArgPool(As), ArgPool(Bs), scale_a, scale_b,
                 torch.bfloat16) as bench:
        timers.append(bench.run())

    return timers


def bench_v2(dtype: torch.dtype, with_cuda_graph: Optional[int],
             with_arg_pool: Optional[int], m: int, k: int, n: int, label: str,
             sub_label: str) -> Iterable[TMeasurement]:
    if dtype == torch.float8_e4m3fn:
        return bench_fp8(dtype, with_cuda_graph, with_arg_pool, m, k, n, label,
                         sub_label)
    raise ValueError("unsupported type")
