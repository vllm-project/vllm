import dataclasses
import random
from typing import Any, Callable, Iterable, Optional, Tuple, Dict, List

import multiprocessing as mp
from multiprocessing import Process, Queue
from queue import Empty

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from utils import make_n_rand_sparse_tensors

import vllm._custom_ops as ops
import traceback

import json
import os
import hashlib
from datetime import datetime
from pathlib import Path


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

    def run_cudagraph(self) -> TMeasurement:
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
            timer = self.run_cudagraph()
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


def run_single_benchmark_process(kernel_config: Dict, gpu_id: int, queue: Queue):
    """
    Run a single kernel benchmark in an isolated process.
    Puts (success, result, config) tuple in the queue.
    """
    try:
        torch.cuda.set_device(gpu_id)
        
        # Initialize CUDA tensors
        m, k, n = kernel_config['m'], kernel_config['k'], kernel_config['n']
        dtype = kernel_config['dtype']
        
        # Create tensors
        BComps, Es, As, Bs = make_n_rand_sparse_tensors(
            kernel_config.get('arg_pool_size', 1), 
            dtype, m, n, k
        )
        AsT = [x.t() for x in As]
        BsT = [x.t() for x in Bs]
        bf16_As = [x.to(dtype=torch.bfloat16) for x in As]
        bf16_BsT = [x.to(dtype=torch.bfloat16) for x in BsT]
        scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        # Because the transposed output will be computed
        out = torch.zeros((n, m), dtype=torch.bfloat16, device="cuda")

        # Setup benchmark params
        cuda_graph_params = None
        if cgops := kernel_config.get('cuda_graph_ops'):
            cuda_graph_params = CudaGraphBenchParams(cgops)

        label = kernel_config['label']
        sub_label = kernel_config['sub_label']

        # Initialize benchmark based on kernel type
        bench = None
        kernel_type = kernel_config['kernel_type']

        if kernel_type == 'pytorch_mm':
            bench = BenchMM(cuda_graph_params, label, sub_label,
                            "pytorch_bf16_bf16_bf16_matmul-no-scales", 
                            torch.mm,
                            ArgPool(bf16_As), ArgPool(bf16_BsT))

        elif kernel_type == 'pytorch_scaled_mm':
            bench = BenchMM(cuda_graph_params, label, sub_label,
                            "pytorch_fp8_fp8_bf16_scaled_mm",
                            torch._scaled_mm,
                            ArgPool(As), ArgPool(BsT),
                            scale_a=scale_a, scale_b=scale_b,
                            out_dtype=torch.bfloat16)

        elif kernel_type == 'pytorch_scaled_mm_fast':
            bench = BenchMM(cuda_graph_params, label, sub_label,
                            "pytorch_fp8_fp8_bf16_scaled_mm_fast_accum",
                            torch._scaled_mm,
                            ArgPool(As), ArgPool(BsT),
                            scale_a=scale_a, scale_b=scale_b,
                            out_dtype=torch.bfloat16,
                            use_fast_accum=True)

        elif kernel_type == 'cutlass_scaled_mm':
            bench = BenchMM(cuda_graph_params, label, sub_label,
                            "cutlass_fp8_fp8_bf16_scaled_mm_default", 
                            ops.cutlass_scaled_mm,
                            ArgPool(As), ArgPool(BsT), scale_a, scale_b,
                            torch.bfloat16)

        elif kernel_type == 'cutlass_sparse_mm':
            bench = BenchMM(cuda_graph_params, label, sub_label,
                            "cutlass_fp8_fp8_bf16_scaled_sparse_mm_default", 
                            ops.cutlass_scaled_sparse_mm,
                            ArgPool(BComps), ArgPool(Es), ArgPool(AsT), 
                            scale_b, scale_a, torch.bfloat16)


        # Run the benchmark
        result = bench.run()
        queue.put((True, result, kernel_config))

    except Exception as e:
        print(f"Error in benchmark process: {str(e)}")
        print(traceback.format_exc())
        queue.put((False, None, kernel_config))
    finally:
        # Explicit cleanup
        torch.cuda.empty_cache()

def benchmark_gpu_worker(gpu_id: int, task_queue: Queue, result_queue: Queue):
    """Worker process that spawns individual benchmark processes for each kernel."""
    try:
        while True:
            try:
                kernel_config = task_queue.get_nowait()
                if kernel_config is None:  # Poison pill
                    break

                # Create a new process queue for this specific benchmark
                process_queue = Queue()

                # Create and start a new process for this kernel benchmark
                p = Process(target=run_single_benchmark_process, 
                          args=(kernel_config, gpu_id, process_queue))
                p.start()

                # Wait for result with timeout (5 minutes for benchmarking)
                try:
                    success, result, config = process_queue.get(timeout=300)
                    result_queue.put((success, result, config))
                except Empty:
                    print(f"Kernel {kernel_config.get('kernel_type')} benchmark timed out")
                    result_queue.put((False, None, kernel_config))

                # Cleanup
                p.join(timeout=1)  # Give it 1 second to join
                if p.is_alive():
                    p.terminate()
                    p.join()

            except Empty:
                break
            except Exception as e:
                print(f"Error in GPU {gpu_id} worker: {str(e)}")
                print(traceback.format_exc())
                if 'kernel_config' in locals():
                    result_queue.put((False, None, kernel_config))

    finally:
        print(f"GPU {gpu_id} worker finished")

def run_kernels_on_gpus(configs: List[Dict]) -> List[Tuple[bool, Optional[TMeasurement], Dict]]:
    MULTI_GPU_MULTI_PROCESS = False  # Set to False for single GPU testing
    if MULTI_GPU_MULTI_PROCESS:
        gpus_list = [0]
        task_queue = Queue()
        result_queue = Queue()

        configs = configs[:10]

        # Fill task queue
        for config in configs:
            task_queue.put(config)
        for _ in gpus_list:  # Add poison pills
            task_queue.put(None)

        # Start GPU workers
        workers = []
        for gpu_id in gpus_list:
            p = Process(target=benchmark_gpu_worker, args=(gpu_id, task_queue, result_queue))
            p.start()
            workers.append(p)

        # Collect results
        results = []
        completed = 0
        total_tasks = len(configs)

        while completed < total_tasks:
            success, result, config = result_queue.get()
            results.append((success, result, config))
            completed += 1

            # Print progress
            status = "Success" if success else "Failed"
            print(f"{status}: {config['kernel_type']}")

        # Cleanup workers
        for worker in workers:
            worker.join(timeout=1)
            if worker.is_alive():
                worker.terminate()
                worker.join()

        return results
    else:
        """Run kernel benchmarks in a single process."""
        results = []
        gpu_id = 0  # Using the same GPU as before
        torch.cuda.set_device(gpu_id)
        # configs = configs[:10]  # Keep the original slice
        
        for config in configs:
            try:
                # Initialize CUDA tensors
                m, k, n = config['m'], config['k'], config['n']
                dtype = config['dtype']
                
                # Create tensors
                BComps, Es, As, Bs = make_n_rand_sparse_tensors(
                    config.get('arg_pool_size', 1), 
                    dtype, m, n, k
                )
                AsT = [x.t() for x in As]
                BsT = [x.t() for x in Bs]
                bf16_As = [x.to(dtype=torch.bfloat16) for x in As]
                bf16_BsT = [x.to(dtype=torch.bfloat16) for x in BsT]
                scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
                scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
                out = torch.zeros((n, m), dtype=torch.bfloat16, device="cuda")

                # Setup benchmark params
                cuda_graph_params = None
                if cgops := config.get('cuda_graph_ops'):
                    cuda_graph_params = CudaGraphBenchParams(cgops)

                label = config['label']
                sub_label = config['sub_label']

                # Initialize benchmark based on kernel type
                bench = None
                kernel_type = config['kernel_type']

                if kernel_type == 'pytorch_mm':
                    bench = BenchMM(cuda_graph_params, label, sub_label,
                                    "pytorch_bf16_bf16_bf16_matmul-no-scales", 
                                    torch.mm,
                                    ArgPool(bf16_As), ArgPool(bf16_BsT))

                elif kernel_type == 'pytorch_scaled_mm':
                    bench = BenchMM(cuda_graph_params, label, sub_label,
                                    "pytorch_fp8_fp8_bf16_scaled_mm",
                                    torch._scaled_mm,
                                    ArgPool(As), ArgPool(BsT),
                                    scale_a=scale_a, scale_b=scale_b,
                                    out_dtype=torch.bfloat16)

                elif kernel_type == 'pytorch_scaled_mm_fast':
                    bench = BenchMM(cuda_graph_params, label, sub_label,
                                    "pytorch_fp8_fp8_bf16_scaled_mm_fast_accum",
                                    torch._scaled_mm,
                                    ArgPool(As), ArgPool(BsT),
                                    scale_a=scale_a, scale_b=scale_b,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)

                elif kernel_type == 'cutlass_scaled_mm':
                    bench = BenchMM(cuda_graph_params, label, sub_label,
                                    "cutlass_fp8_fp8_bf16_scaled_mm_default", 
                                    ops.cutlass_scaled_mm,
                                    ArgPool(As), ArgPool(BsT), scale_a, scale_b,
                                    torch.bfloat16)

                elif kernel_type == 'cutlass_sparse_mm':
                    bench = BenchMM(cuda_graph_params, label, sub_label,
                                    "cutlass_fp8_fp8_bf16_scaled_sparse_mm_default", 
                                    ops.cutlass_scaled_sparse_mm,
                                    ArgPool(BComps), ArgPool(Es), ArgPool(AsT), 
                                    scale_b, scale_a, torch.bfloat16)

                # Run the benchmark
                result = bench.run()
                
                # Print progress
                print(f"Success: {kernel_type}")
                    
                results.append((True, result, config))
                
                # Cleanup
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in benchmark: {str(e)}")
                print(traceback.format_exc())
                results.append((False, None, config))
                torch.cuda.empty_cache()
                
        return results


def get_cache_path() -> str:
    """Get the path to the cache file for the given configuration hash."""
    return f'{Path(os.path.dirname(os.path.realpath(__file__)))}/stable_kernels.json'


def bench_fp8(dtype: torch.dtype, with_cuda_graph: Optional[int],
              with_arg_pool: Optional[int], m: int, k: int, n: int, label: str,
              sub_label: str) -> Iterable[TMeasurement]:
    
    # Check if context is not set
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    timers = []
    gpus_list = [5]  # Using the same GPU list as original code

    # Base configuration for all kernels
    base_config = {
        'm': m,
        'k': k,
        'n': n,
        'dtype': dtype,
        'cuda_graph_ops': with_cuda_graph,
        'arg_pool_size': with_arg_pool if with_arg_pool else 1,
        'label': label,
        'sub_label': sub_label
    }
    
    # Prepare configs for all kernels
    standard_kernels = [
        {'kernel_type': 'pytorch_mm'},
        {'kernel_type': 'pytorch_scaled_mm'},
        {'kernel_type': 'pytorch_scaled_mm_fast'},
        {'kernel_type': 'cutlass_scaled_mm'},
        {'kernel_type': 'cutlass_sparse_mm'}
    ]
    
    # Create configs for standard kernels
    all_configs = [{**base_config, **kernel} for kernel in standard_kernels]
    
    # Run all kernels distributed across GPUs
    print(f"Running {len(all_configs)} benchmarks across {len(gpus_list)} GPUs...")
    results = run_kernels_on_gpus(all_configs)
    
    # Process results
    for success, result, _ in results:
        if success and result is not None:
            timers.append(result)
    
    return timers


def bench_v2(dtype: torch.dtype, with_cuda_graph: Optional[int],
             with_arg_pool: Optional[int], m: int, k: int, n: int, label: str,
             sub_label: str) -> Iterable[TMeasurement]:
    if dtype == torch.float8_e4m3fn:
        return bench_fp8(dtype, with_cuda_graph, with_arg_pool, m, k, n, label,
                         sub_label)
    raise ValueError("unsupported type")
