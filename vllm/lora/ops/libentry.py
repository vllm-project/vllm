# Copied From https://github.com/FlagOpen/FlagGems

import triton



class LibEntry(triton.KernelInterface):
    def __init__(
        self,
        fn,
    ):
        self.fn = fn
        self.arg_names = fn.arg_names
        self.divisibility = 16
        self.config_cache = dict()
        self.kernel_cache = dict()
        if isinstance(fn, triton.runtime.Autotuner):
            self.rt = "Autotuner"
        elif isinstance(fn, triton.runtime.Heuristics):
            self.rt = "Heuristics"
        else:
            self.rt = "JitFunction"

    def run(self, *args, **kwargs):
        key = []
        for arg in args:
            if hasattr(arg, "data_ptr"):
                key.append(arg.dtype)
                key.append(arg.data_ptr() % self.divisibility == 0)
            elif isinstance(arg, int):
                key.append(arg)
        entry_key = tuple(key)

        config = {}
        # Autotuner
        if self.rt == "Autotuner":
            if entry_key not in self.config_cache:
                # tune
                kernel = self.fn.run(*args, **kwargs)
                config = self.fn.best_config.kwargs
                self.config_cache[entry_key] = config
                self.kernel_cache[entry_key] = kernel
                return
            else:
                # tuned
                config = self.config_cache[entry_key]
                kernel = self.kernel_cache[entry_key]
        # Heuristics
        elif self.rt == "Heuristics":
            if entry_key not in self.kernel_cache:
                # compile
                kernel = self.fn.run(*args, **kwargs)
                self.kernel_cache[entry_key] = kernel
                return
            else:
                # compiled
                for v, heur in self.fn.values.items():
                    config[v] = heur(
                        {**dict(zip(self.arg_names, args)), **kwargs}
                    )
                kernel = self.kernel_cache[entry_key]
        # JitFunction
        else:
            if entry_key not in self.kernel_cache:
                # compile
                kernel = self.fn.run(*args, **kwargs)
                self.kernel_cache[entry_key] = kernel
                return
            else:
                # compiled
                args = tuple([
                    arg
                    for i, arg in enumerate(args)
                    if not self.fn.params[i].is_constexpr
                ])
                kernel = self.kernel_cache[entry_key]
        grid = kwargs["grid"]
        if callable(grid):
            # grid_fn
            current = dict(**kwargs, **config)
            meta = {**dict(zip(self.arg_names, args)), **current}
            grid = grid(meta)
        grid = grid + (1, 1)

        kernel[grid[0:3]](*args)
        return


def libentry():
    """
    Decorator for triton library entries.
    """

    def decorator(fn):
        return LibEntry(fn)

    return decorator
