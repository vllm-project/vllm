# Copied From https://github.com/FlagOpen/FlagGems

import inspect

import triton


class LibEntry(triton.KernelInterface):

    def __init__(
        self,
        fn,
    ):
        self.fn = fn
        self.arg_names = fn.arg_names
        self.divisibility = 16
        self.kernel_cache = dict()
        fn = self.fn
        while not isinstance(fn, triton.runtime.JITFunction):
            fn = fn.fn
        self.jit_function: triton.runtime.JITFunction = fn
        self.specialize_indices = [
            p.num for p in self.jit_function.params
            if not p.is_constexpr and not p.do_not_specialize
        ]
        self.do_not_specialize_indices = [
            p.num for p in self.jit_function.params
            if not p.is_constexpr and p.do_not_specialize
        ]

    def key(self, spec_args, dns_args, const_args):
        spec_key = [(arg.dtype, arg.data_ptr() %
                     self.divisibility == 0) if hasattr(arg, "data_ptr") else
                    (type(arg), arg) for arg in spec_args]
        dns_key = [
            arg.dtype if hasattr(
                arg, "data_ptr") else type(arg) if not isinstance(arg, int)
            else "i32" if -(2**31) <= arg and arg <= 2**31 -
            1 else "u64" if 2**63 <= arg and arg <= 2**64 - 1 else "i64"
            for arg in dns_args
        ]
        # const args passed by position
        return tuple(spec_key + dns_key + const_args)

    def run(self, *args, **kwargs):
        grid = kwargs["grid"]
        # collect all the arguments
        spec_args = []  # specialize arguments
        dns_args = []  # do not specialize arguments
        const_args = []  # constexpr arguments
        k_args = []  # kernel arguments
        for i, arg in enumerate(args):
            if i in self.specialize_indices:
                k_args.append(arg)
                spec_args.append(arg)
            elif i in self.do_not_specialize_indices:
                k_args.append(arg)
                dns_args.append(arg)
            else:
                const_args.append(arg)
        for p in self.jit_function.params[len(args):]:
            if p.name in kwargs:
                val = kwargs[p.name]
            elif p.default is inspect._empty:
                continue
            else:
                val = p.default

            if p.is_constexpr:
                const_args.append(val)
            elif p.do_not_specialize:
                dns_args.append(val)
                k_args.append(val)
            else:
                spec_args.append(val)
                k_args.append(val)

        entry_key = self.key(spec_args, dns_args, const_args)

        if entry_key not in self.kernel_cache:
            # compile the kernel also completes the related computations
            kernel = self.fn.run(*args, **kwargs)
            fn = self.fn
            # collect constexpr arguments for grid computation
            constexprs = {}
            while not isinstance(fn, triton.runtime.JITFunction):
                if isinstance(fn, triton.runtime.Autotuner):
                    config = fn.best_config
                    constexprs["num_warps"] = config.num_warps
                    constexprs["num_stages"] = config.num_stages
                    constexprs["num_ctas"] = config.num_ctas
                    constexprs = {**constexprs, **config.kwargs}
                elif isinstance(fn, triton.runtime.Heuristics):
                    for v, heur in fn.values.items():
                        constexprs[v] = heur({
                            **dict(zip(fn.arg_names, args)),
                            **kwargs,
                            **constexprs,
                        })
                else:
                    raise RuntimeError("Invalid Runtime Function")
                fn = fn.fn
            # In vLLM, certain kernels like fused_moe_kernel get the
            # best_config(as kwargs) from a configuration json file, rather
            # than using Autotuner & Heuristics. Therefore, all their constexprs
            # (tl.constexpr) are assigned values through the following loop.
            for p in self.jit_function.params:
                if p.is_constexpr and p.name not in constexprs:
                    constexprs[p.name] = p.default  #default=inspect._empty
            self.kernel_cache[entry_key] = (kernel, constexprs)
        else:
            # load kernel from cache directly
            kernel, constexprs = self.kernel_cache[entry_key]

            if callable(grid):
                # collect all arguments to the grid fn，ie:
                # 1. args,
                # 2. kwargs,
                # 3. all all other captured arguments in CompiledKernel from
                # Autotunner & Heuristics when kwargs & captured args conflict,
                # captured args have higher priority
                # 4. We must filter out captured args with default value firstly
                constexprs = {
                    k: v
                    for k, v in constexprs.items() if v is not inspect._empty
                }
                meta = {
                    **dict(zip(self.arg_names, args)),
                    **kwargs,
                    **constexprs,
                }
                grid = grid(meta)
            if isinstance(grid, tuple):
                grid = grid + (1, 1)
            elif isinstance(grid, list):
                grid = grid + [1, 1]
            kernel[grid[0:3]](*k_args)
        # maintaining the same return type as the JITFunction.run
        return kernel


def libentry():
    """
    Decorator for triton library entries.
    Motivation:
        The runtime overhead of Triton kernels is the reason for the lower 
        performance of small kernels, particularly evident with smaller models. 
        Using this decorator can reduce Triton runtime overhead.
    How:
        The `run` function of JITFunction needs to accomplish:
            - Parameter binding using inspect
            - KernelArg type wrapping
            - Cache key calculation
        When dealing with small size, these steps can become bottlenecks in 
        Triton runtime. Libentry simplifies these steps to reduce runtime 
        overhead, thereby improving the runtime expenses of small kernels.
    NOTE:
        When Triton is upgraded to version 3.0.0, libentry can be removed,
        see: https://github.com/vllm-project/vllm/pull/5036#issuecomment-2243396245
        

    """

    def decorator(fn):
        return LibEntry(fn)

    return decorator
