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

        elif kernel_type == 'autogen_kernel':
            # Get the autogen kernel
            kernel_num = kernel_config['kernel_num']
            kernel_name, autogen_fn = get_autogen_functions()[kernel_num]

            # Create appropriate benchmark based on kernel type
            bench = BenchMM(cuda_graph_params, label, sub_label,
                            kernel_name, autogen_fn, out, 
                            ArgPool(BComps), ArgPool(Es), ArgPool(AsT),
                            scale_b, scale_a)

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
        gpus_list = [5]
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
            if config['kernel_type'] == 'autogen_kernel':
                kernel_num = config['kernel_num']
                kernel_name = get_autogen_functions()[kernel_num][0]
                status = "Success" if success else "Failed"
                print(f"{status}: autogen {kernel_num} {kernel_name}")
            else:
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
        gpu_id = 5  # Using the same GPU as before
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

                elif kernel_type == 'autogen_kernel':
                    # Get the autogen kernel
                    kernel_num = config['kernel_num']
                    kernel_name, autogen_fn = get_autogen_functions()[kernel_num]

                    # Create appropriate benchmark based on kernel type
                    bench = BenchMM(cuda_graph_params, label, sub_label,
                                    kernel_name, autogen_fn, out, 
                                    ArgPool(BComps), ArgPool(Es), ArgPool(AsT),
                                    scale_b, scale_a)

                # Run the benchmark
                result = bench.run()
                
                # Print progress
                if kernel_type == 'autogen_kernel':
                    kernel_num = config['kernel_num']
                    kernel_name = get_autogen_functions()[kernel_num][0]
                    print(f"Success: autogen {kernel_num} {kernel_name}")
                else:
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



def test_autogen_kernel_process(kernel_config: Dict, gpu_id: int, queue: Queue):
    """
    Test run a single autogen kernel in an isolated process.
    Puts (kernel_num, success) tuple in the queue.
    """
    try:
        torch.cuda.set_device(gpu_id)
        
        # Initialize test tensors (using smaller dimensions for quick testing)
        test_m, test_k, test_n = 256, 256, 256  # Small test dimensions
        dtype = kernel_config['dtype']
        kernel_num = kernel_config['kernel_num']
        
        # Create minimal test tensors
        BComps, Es, As, Bs = make_n_rand_sparse_tensors(1, dtype, test_m, test_n, test_k)
        AsT = [x.t() for x in As]
        BsT = [x.t() for x in Bs]
        scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        out = torch.zeros((test_m, test_n), dtype=torch.bfloat16, device="cuda")
        
        # Get the autogen kernel
        kernel_name, autogen_fn = get_autogen_functions()[kernel_num]
        
        # Test run based on kernel type
        autogen_fn(out, BComps[0], Es[0], AsT[0], scale_a, scale_b)
            
        # Run a second time to ensure stability
        torch.cuda.synchronize()
        autogen_fn(out, BComps[0], Es[0], AsT[0], scale_a, scale_b)
        torch.cuda.synchronize()
        
        queue.put((kernel_num, True))
        
    except Exception as e:
        print(f"Kernel {kernel_num} ({kernel_name if 'kernel_name' in locals() else 'unknown'}) failed test: {str(e)}")
        queue.put((kernel_num, False))
    finally:
        # Explicit cleanup
        torch.cuda.empty_cache()


def test_gpu_worker(gpu_id: int, task_queue: Queue, result_queue: Queue):
    """Worker process that spawns individual test processes for each kernel."""
    try:
        while True:
            try:
                kernel_config = task_queue.get_nowait()
                if kernel_config is None:  # Poison pill
                    break
                
                # Create a new process queue for this specific test
                process_queue = Queue()
                
                # Create and start a new process for this kernel test
                p = Process(target=test_autogen_kernel_process, 
                          args=(kernel_config, gpu_id, process_queue))
                p.start()
                
                # Wait for result with timeout
                try:
                    kernel_num, success = process_queue.get(timeout=30)  # 30 second timeout
                    result_queue.put((kernel_num, success))
                except Empty:
                    print(f"Kernel {kernel_config['kernel_num']} timed out")
                    result_queue.put((kernel_config['kernel_num'], False))
                
                # Cleanup
                p.join(timeout=1)  # Give it 1 second to join
                if p.is_alive():
                    p.terminate()
                    p.join()
                
            except Empty:
                break
            except Exception as e:
                print(f"Error in GPU {gpu_id} test worker: {str(e)}")
                print(traceback.format_exc())
                if 'kernel_config' in locals():
                    result_queue.put((kernel_config['kernel_num'], False))
                
    finally:
        print(f"GPU {gpu_id} test worker finished")


def filter_stable_autogen_kernels(base_config: Dict, gpus_list: List[int]) -> List[int]:
    """
    Test all autogen kernels and return list of kernel numbers that pass the test.
    Each kernel is tested in a completely isolated process.
    """
    task_queue = Queue()
    result_queue = Queue()
    
    # Get all autogen kernels
    autogen_name_fn = get_autogen_functions()
    total_kernels = len(autogen_name_fn)
    
    # Fill task queue with test configs
    for i in range(total_kernels):
        config = {
            **base_config,
            'kernel_type': 'autogen_kernel',
            'kernel_num': i
        }
        task_queue.put(config)
    
    # Add poison pills
    for _ in gpus_list:
        task_queue.put(None)
    
    # Start GPU workers
    workers = []
    for gpu_id in gpus_list:
        p = Process(target=test_gpu_worker, args=(gpu_id, task_queue, result_queue))
        p.start()
        workers.append(p)
    
    # Collect results
    stable_kernels = []
    completed = 0
    
    print(f"Testing {total_kernels} autogen kernels for stability...")
    while completed < total_kernels:
        kernel_num, success = result_queue.get()
        completed += 1
        
        if success:
            kernel_name = get_autogen_functions()[kernel_num][0]
            stable_kernels.append(kernel_num)
            print(f"Kernel {kernel_num} ({kernel_name}) passed stability test")
        
        if completed % 10 == 0:
            print(f"Tested {completed}/{total_kernels} kernels. {len(stable_kernels)} stable so far.")
    
    # Wait for workers to finish
    for worker in workers:
        worker.join(timeout=1)
        if worker.is_alive():
            worker.terminate()
            worker.join()
    
    print(f"Found {len(stable_kernels)} stable kernels out of {total_kernels}")
    return stable_kernels


def get_config_hash(base_config: Dict) -> str:
    """
    Create a hash of the relevant configuration parameters that would affect kernel stability.
    """
    # Extract only the parameters that affect kernel stability
    relevant_params = {
        'dtype': str(base_config['dtype']),  # Convert dtype to string for hashing
        'm': base_config['m'],
        'k': base_config['k'],
        'n': base_config['n'],
    }
    
    # Add CUDA version and PyTorch version to the hash
    relevant_params['cuda_version'] = torch.version.cuda
    relevant_params['torch_version'] = torch.__version__
    
    # Create a sorted string representation for consistent hashing
    param_str = json.dumps(relevant_params, sort_keys=True)
    
    # Create hash
    return hashlib.sha256(param_str.encode()).hexdigest()[:16]


def get_cache_path() -> str:
    """Get the path to the cache file for the given configuration hash."""
    return f'{Path(os.path.dirname(os.path.realpath(__file__)))}/stable_kernels.json'


def load_cached_kernels(cache_path: str) -> Optional[List[int]]:
    """
    Load cached stable kernel list if it exists and is not too old.
    Returns None if cache doesn't exist or is invalid.
    """
    try:
        if not os.path.exists(cache_path):
            return None
            
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
            
        # # Check if cache is too old (e.g., older than 7 days)
        # cache_date = datetime.fromisoformat(cache_data['date'])
        # cache_age = (datetime.now() - cache_date).days
        # if cache_age > 7:
        #     print("Cache is older than 7 days, will rerun stability tests")
        #     return None
            
        # Verify the cached kernel numbers are valid
        total_kernels = len(get_autogen_functions())
        stable_kernels = cache_data['stable_kernels']
        if any(k >= total_kernels for k in stable_kernels):
            print("Cache is invalid (kernel numbers out of range), will rerun stability tests")
            return None
            
        print(f"Loaded {len(stable_kernels)} stable kernels from cache")
        return stable_kernels
        
    except Exception as e:
        print(f"Error loading cache: {str(e)}")
        return None


def save_cached_kernels(cache_path: str, stable_kernels: List[int]):
    """Save the list of stable kernels to cache."""
    try:
        cache_data = {
            'date': datetime.now().isoformat(),
            'stable_kernels': stable_kernels
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
            
        print(f"Saved {len(stable_kernels)} stable kernels to cache")
        
    except Exception as e:
        print(f"Error saving cache: {str(e)}")


def get_stable_autogen_kernels(base_config: Dict, gpus_list: List[int]) -> List[int]:
    """
    Get the list of stable autogen kernels, either from cache or by running tests.
    """
    # Generate config hash and get cache path
    # config_hash = get_config_hash(base_config)
    cache_path = get_cache_path()
    
    # Try to load from cache
    stable_kernels = load_cached_kernels(cache_path)
    
    if stable_kernels is None:
        # Cache miss or invalid cache - run stability tests
        print("Running stability tests for autogen kernels...")
        stable_kernels = filter_stable_autogen_kernels(base_config, gpus_list)
        
        # Save results to cache
        save_cached_kernels(cache_path, stable_kernels)
    
    return stable_kernels


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
    standard_configs = [{**base_config, **kernel} for kernel in standard_kernels]
    
    # Get stable kernels (from cache or by testing)
    stable_kernel_nums = get_stable_autogen_kernels(base_config, gpus_list)
    
    # Create configs only for stable autogen kernels
    autogen_configs = []
    for kernel_num in stable_kernel_nums:
        config = {
            **base_config,
            'kernel_type': 'autogen_kernel',
            'kernel_num': kernel_num
        }
        autogen_configs.append(config)
    
    # Combine all configs
    all_configs = standard_configs + autogen_configs
    
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
