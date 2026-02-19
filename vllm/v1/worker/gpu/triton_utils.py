# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable


class CachedKernel:
    """Caches compiled triton kernels after first invocation to bypass
    the JIT dispatch overhead on subsequent calls.

    Triton specializes compiled kernels based on integer argument values
    (e.g., divisible by 16 vs. not) and constexpr argument values.
    This class maintains a cache of launchers keyed by these
    specialization-relevant properties. Each launcher calls
    ``CompiledKernel.run`` (the CudaLauncher) directly — the same
    internal path used by the JIT — skipping argument binding,
    specialization, and cache key computation.

    Can be used as a decorator — the call-site syntax is unchanged::

        @CachedKernel
        @triton.jit
        def my_kernel(..., BLOCK_SIZE: tl.constexpr):
            ...

        my_kernel[(grid,)](*args, BLOCK_SIZE=1024)

    Constexpr and launch-option kwargs may vary between calls;
    each unique combination gets its own cached launcher.
    """

    __slots__ = ("_kernel", "_launchers", "_key_fn", "_constexpr_keys", "_grid")

    def __init__(self, kernel):
        self._kernel = kernel
        self._launchers: dict = {}
        self._key_fn: Callable | None = None
        self._constexpr_keys: tuple[str, ...] = ()
        self._grid: tuple = ()

    def __getitem__(self, grid):
        self._grid = grid if isinstance(grid, tuple) else (grid,)
        return self

    def __call__(self, *args, **kwargs):
        grid = self._grid
        key_fn = self._key_fn
        if key_fn is not None:
            key = key_fn(args, kwargs)
            launch = self._launchers.get(key)
            if launch is not None:
                launch(grid, *args)
                return

        # Normal JIT dispatch (first call or new specialization).
        self._kernel[grid](*args, **kwargs)
        if key_fn is None:
            key_fn = self._init_key_fn(*args, **kwargs)
            key = key_fn(args, kwargs)
        self._launchers[key] = self._build_launcher(*args, **kwargs)

    def _init_key_fn(self, *args, **kwargs) -> Callable:
        """One-time setup: identify integer params, constexpr params,
        and build the cache key function."""
        constexpr_names = {p.name for p in self._kernel.params if p.is_constexpr}
        self._constexpr_keys = tuple(k for k in kwargs if k in constexpr_names)
        int_indices = tuple(
            i
            for i, p in enumerate(self._kernel.params)
            if not p.is_constexpr and i < len(args) and isinstance(args[i], int)
        )
        key_fn = _make_key_fn(int_indices, tuple(kwargs.keys()))
        self._key_fn = key_fn
        return key_fn

    def _build_launcher(self, *args, **kwargs):
        """Build a fast launch closure that calls CudaLauncher directly."""
        import triton.knobs as knobs
        from triton.runtime import driver

        compiled = self._kernel.warmup(*args, grid=(1,), **kwargs)
        if hasattr(compiled, "result"):
            compiled = compiled.result()

        c_run = compiled.run
        c_fn = compiled.function
        c_pm = compiled.packed_metadata
        enter_hook = knobs.runtime.launch_enter_hook
        exit_hook = knobs.runtime.launch_exit_hook
        device = driver.active.get_current_device()
        get_stream = driver.active.get_current_stream
        c_vals = tuple(kwargs[k] for k in self._constexpr_keys)

        def launch(grid, *args):
            c_run(
                grid[0],
                grid[1] if len(grid) > 1 else 1,
                1,
                get_stream(device),
                c_fn,
                c_pm,
                None,
                enter_hook,
                exit_hook,
                *args,
                *c_vals,
            )

        return launch


def _make_key_fn(int_indices: tuple[int, ...], kw_keys: tuple[str, ...]) -> Callable:
    """Build a function that computes the cache key from args and kwargs.

    The key captures triton's integer arg specialization categories
    (value==1, divisible by 16, or other) and all kwarg values.
    """
    if not int_indices and not kw_keys:
        return lambda args, kw: 0

    def key_fn(args, kw):
        k = 0
        for i in int_indices:
            v = args[i]
            k = k * 3 + (1 if v == 1 else (2 if v % 16 == 0 else 0))
        return (k, *(kw[key] for key in kw_keys)) if kw_keys else k

    return key_fn
