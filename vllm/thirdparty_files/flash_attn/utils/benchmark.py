# Copyright (c) 2023, Tri Dao.
""" Useful functions for writing test code. """

import torch
import torch.utils.benchmark as benchmark


def benchmark_forward(
    fn, *inputs, repeats=10, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_backward(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the backward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Backward pass")
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError("Grad shape does not match output shape")

    def f(*inputs, y, grad):
        # Set .grad to None to avoid extra operation of gradient accumulation
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x.grad = None
        y.backward(grad, retain_graph=True)

    t = benchmark.Timer(
        stmt="f(*inputs, y=y, grad=grad)",
        globals={"f": f, "inputs": inputs, "y": y, "grad": grad},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_combined(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward+backward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward + Backward pass")
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError("Grad shape does not match output shape")

    def f(grad, *inputs, **kwinputs):
        for x in inputs:
            if isinstance(x, torch.Tensor):
                x.grad = None
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            y = fn(*inputs, **kwinputs)
            if type(y) is tuple:
                y = y[0]
        y.backward(grad, retain_graph=True)

    t = benchmark.Timer(
        stmt="f(grad, *inputs, **kwinputs)",
        globals={"f": f, "fn": fn, "inputs": inputs, "grad": grad, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def benchmark_fwd_bwd(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward+backward pass of an arbitrary function."""
    return (
        benchmark_forward(
            fn,
            *inputs,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
        benchmark_backward(
            fn,
            *inputs,
            grad=grad,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
    )


def benchmark_all(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    """Use Pytorch Benchmark on the forward+backward pass of an arbitrary function."""
    return (
        benchmark_forward(
            fn,
            *inputs,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
        benchmark_backward(
            fn,
            *inputs,
            grad=grad,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
        benchmark_combined(
            fn,
            *inputs,
            grad=grad,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
    )


def pytorch_profiler(
    fn,
    *inputs,
    trace_filename=None,
    backward=False,
    amp=False,
    amp_dtype=torch.float16,
    cpu=False,
    verbose=True,
    **kwinputs,
):
    """Wrap benchmark functions in Pytorch profiler to see CUDA information."""
    if backward:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
            g = torch.randn_like(out)
    for _ in range(30):  # Warm up
        if backward:
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    x.grad = None
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
        # Backward should be done outside autocast
        if backward:
            out.backward(g, retain_graph=True)
    activities = ([torch.profiler.ProfilerActivity.CPU] if cpu else []) + [
        torch.profiler.ProfilerActivity.CUDA
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        # profile_memory=True,
        with_stack=True,
    ) as prof:
        if backward:
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    x.grad = None
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
        if backward:
            out.backward(g, retain_graph=True)
    if verbose:
        # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))
        print(prof.key_averages().table(row_limit=50))
    if trace_filename is not None:
        prof.export_chrome_trace(trace_filename)


def benchmark_memory(fn, *inputs, desc="", verbose=True, **kwinputs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*inputs, **kwinputs)
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / ((2**20) * 1000)
    if verbose:
        print(f"{desc} max memory: {mem}GB")
    torch.cuda.empty_cache()
    return mem
