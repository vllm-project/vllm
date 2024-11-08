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


def get_autogen_functions():
    import importlib
    from importlib.util import find_spec

    # import vllm nm_cutlass modules so torch._C can find it
    m_idx = 0
    m_name = f'vllm._nm_cutlass_{m_idx}_C'
    while find_spec(m_name):
        # print(f"attempting import {m_name}")
        importlib.import_module(m_name)
        m_idx += 1
        m_name = f'vllm._nm_cutlass_{m_idx}_C'

    dispatch_names = torch._C._dispatch_get_all_op_names()
    autogen_dispatch_names = [x for x in dispatch_names if 'autogen' in x]
    assert all([x.startswith('_nm_cutlass') for x in autogen_dispatch_names])
    autogen_dispatch_modules_names = [(getattr(torch.ops,
                                               x.split('::')[0]),
                                       x.split('::')[1])
                                      for x in autogen_dispatch_names]
    name_fn = [(name, getattr(m, name))
               for m, name in autogen_dispatch_modules_names]
    # print(f"#autogen functions found {len(name_fn)}")
    return name_fn


def run_benchmark_process(kernel_config, queue):
    try:
        # Initialize CUDA tensors
        arg_pool_size = kernel_config.get('arg_pool_size', 1)
        m, k, n = kernel_config['m'], kernel_config['k'], kernel_config['n']
        dtype = kernel_config['dtype']
        
        # Create tensors
        AComps, Es, As, Bs = make_n_rand_sparse_tensors(arg_pool_size, dtype, m, n, k)
        bf16_As = [x.to(dtype=torch.bfloat16) for x in As]
        bf16_Bs = [x.to(dtype=torch.bfloat16) for x in Bs]
        scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        out = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")

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
                            ArgPool(bf16_As), ArgPool(bf16_Bs))
        
        elif kernel_type == 'pytorch_scaled_mm':
            bench = BenchMM(cuda_graph_params, label, sub_label,
                            "pytorch_fp8_fp8_bf16_scaled_mm",
                            torch._scaled_mm,
                            ArgPool(As), ArgPool(Bs),
                            scale_a=scale_a, scale_b=scale_b,
                            out_dtype=torch.bfloat16)
        
        elif kernel_type == 'pytorch_scaled_mm_fast':
            bench = BenchMM(cuda_graph_params, label, sub_label,
                            "pytorch_fp8_fp8_bf16_scaled_mm_fast_accum",
                            torch._scaled_mm,
                            ArgPool(As), ArgPool(Bs),
                            scale_a=scale_a, scale_b=scale_b,
                            out_dtype=torch.bfloat16,
                            use_fast_accum=True)
        
        elif kernel_type == 'cutlass_scaled_mm':
            bench = BenchMM(cuda_graph_params, label, sub_label,
                            "cutlass_fp8_fp8_bf16_scaled_mm", 
                            ops.cutlass_scaled_mm,
                            ArgPool(As), ArgPool(Bs), scale_a, scale_b,
                            torch.bfloat16)
        
        elif kernel_type == 'cutlass_sparse_mm':
            bench = BenchMM(cuda_graph_params, label, sub_label,
                            "cutlass_fp8_fp8_bf16_scaled_sparse_mm", 
                            ops.cutlass_scaled_sparse_mm,
                            ArgPool(AComps), ArgPool(Es), ArgPool(Bs), 
                            scale_a, scale_b, torch.bfloat16)
        
        elif kernel_type == 'autogen_kernel':
            # Get the autogen kernel
            kernel_num = kernel_config['kernel_num']
            autogen_fn = None
            
            # Get the kernel in autogen functions
            kernel_name, autogen_fn = get_autogen_functions()[kernel_num]
            
            if autogen_fn is None:
                raise ValueError(f"Autogen kernel {kernel_name} not found")
            
            # Create appropriate benchmark based on kernel type
            if "scaled_sparse_mm" in kernel_name:
                bench = BenchMM(cuda_graph_params, label, sub_label,
                                kernel_name, autogen_fn, out, 
                                ArgPool(AComps), ArgPool(Es), ArgPool(Bs),
                                scale_a, scale_b)
            else:
                bench = BenchMM(cuda_graph_params, label, sub_label,
                                kernel_name, autogen_fn, out,
                                ArgPool(As), ArgPool(Bs))

        # Run the benchmark
        result = bench.run()
        queue.put((True, result))
        
    except Exception as e:
        print(f"Error in process: {str(e)}")
        print(traceback.format_exc())
        queue.put((False, None))


def gpu_worker(gpu_id: int, task_queue: Queue, result_queue: Queue):
    try:
        torch.cuda.set_device(gpu_id)
        while True:
            try:
                kernel_config = task_queue.get_nowait()
                if kernel_config is None:  # Poison pill
                    break
                
                process_queue = Queue()
                run_benchmark_process(kernel_config, process_queue)
                success, result = process_queue.get()
                
                result_queue.put((success, result, kernel_config))
                
            except Empty:
                break
            except Exception as e:
                print(f"Error in GPU {gpu_id} worker: {str(e)}")
                print(traceback.format_exc())
                result_queue.put((False, None, kernel_config))
                
    except Exception as e:
        print(f"Fatal error in GPU {gpu_id} worker: {str(e)}")
        print(traceback.format_exc())
    finally:
        print(f"GPU {gpu_id} worker finished")

def run_kernels_on_gpus(configs: List[Dict]) -> List[Tuple[bool, Optional[TMeasurement], Dict]]:
    # num_gpus = torch.cuda.device_count()
    gpus_list = [5]
    task_queue = Queue()
    result_queue = Queue()
    
    # Fill task queue
    for config in configs:
        task_queue.put(config)
    for _ in gpus_list:  # Add poison pills
        task_queue.put(None)
    
    # Start GPU workers
    workers = []
    for gpu_id in gpus_list:
        p = Process(target=gpu_worker, args=(gpu_id, task_queue, result_queue))
        p.start()
        workers.append(p)
    
    # Collect results
    results = []
    completed = 0
    total_tasks = len(configs)
    
    while completed < total_tasks:
        result = result_queue.get()
        results.append(result)
        completed += 1
        
        success, _, config = result
        if config['kernel_type'] == 'autogen_kernel':
            kernel_num = config['kernel_num']
            kernel_name = get_autogen_functions()[kernel_num][0]
            status = "Success" if success else "Failed"
            print(f"{status}: autogen {kernel_num + 1}/{total_tasks} {kernel_name}")
        else:
            status = "Success" if success else "Failed"
            print(f"{status}: {config['kernel_type']}")
    
    for worker in workers:
        worker.join()
    
    return results


def bench_fp8(dtype: torch.dtype, with_cuda_graph: Optional[int],
              with_arg_pool: Optional[int], m: int, k: int, n: int, label: str,
              sub_label: str) -> Iterable[TMeasurement]:
    
    # Check if context is not set
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    timers = []

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
    standard_configs = [{**base_config, **kernel} for kernel in standard_kernels]
    
    # Create configs for autogen kernels
    autogen_name_fn = get_autogen_functions()
    # autogen_name_fn = autogen_name_fn[284:288]
    # i_range = [284, 285, 286, 287]
    i_range = range(len(autogen_name_fn))
    autogen_configs = []
    
    for i in i_range:
        config = {
            **base_config,
            'kernel_type': 'autogen_kernel',
            'kernel_num': i
        }
        autogen_configs.append(config)
    
    # Combine all configs
    all_configs = standard_configs + autogen_configs
    
    # Run all kernels distributed across GPUs
    print(f"Running {len(all_configs)} benchmarks across {torch.cuda.device_count()} GPUs...")
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
