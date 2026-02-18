# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


class CachedKernel:
    """Caches compiled triton kernels after first invocation to bypass
    the JIT dispatch overhead on subsequent calls.

    Triton specializes compiled kernels based on integer argument values
    (e.g., divisible by 16 vs. not). This class maintains a small cache
    of launchers keyed by the specialization signature of the non-constexpr
    integer arguments. Each launcher calls ``CompiledKernel.run`` (the
    CudaLauncher) directly — the same internal path used by the JIT —
    skipping argument binding, specialization, and cache key computation.

    NOTE: All calls must use the same constexpr/option values (e.g.,
    BLOCK_SIZE, num_warps) as the first call.
    """

    __slots__ = ("_kernel", "_launchers", "_spec_key_fn", "_constexpr_vals")

    def __init__(self, kernel):
        self._kernel = kernel
        self._launchers: dict | None = None
        self._spec_key_fn = None
        self._constexpr_vals: tuple = ()

    def __call__(self, grid, *args, **kwargs):
        launchers = self._launchers
        if launchers is not None:
            assert self._spec_key_fn is not None
            launch = launchers.get(self._spec_key_fn(args))
            if launch is not None:
                launch(grid, *args)
                return
        # Normal JIT dispatch (first call or new specialization).
        self._kernel[grid](*args, **kwargs)
        if launchers is None:
            launchers = self._init_params(*args, **kwargs)
        assert self._spec_key_fn is not None
        launchers[self._spec_key_fn(args)] = self._build_launcher(*args, **kwargs)

    def _init_params(self, *args, **kwargs) -> dict:
        """One-time setup: identify non-constexpr integer params and
        cache constexpr values."""
        self._launchers = {}
        constexpr_names = {p.name for p in self._kernel.params if p.is_constexpr}
        int_indices = tuple(
            i
            for i, p in enumerate(self._kernel.params)
            if not p.is_constexpr and i < len(args) and isinstance(args[i], int)
        )
        self._spec_key_fn = _make_spec_key_fn(int_indices)
        self._constexpr_vals = tuple(
            v for k, v in kwargs.items() if k in constexpr_names
        )
        return self._launchers

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
        c_vals = self._constexpr_vals

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


def _make_spec_key_fn(int_indices: tuple[int, ...]):
    """Build a specialized function that computes the triton specialization
    key for the given integer parameter indices.

    Triton specializes integer args into three categories:
    - value == 1: compiled as constexpr (value baked into binary)
    - value % 16 == 0: divisibility hint for vectorization
    - otherwise: no specialization
    """
    n = len(int_indices)
    if n == 0:
        return lambda args: 0
    if n == 1:
        i0 = int_indices[0]
        return lambda args: 1 if args[i0] == 1 else (2 if args[i0] % 16 == 0 else 0)
    if n == 2:
        i0, i1 = int_indices
        return lambda args: (
            (1 if args[i0] == 1 else (2 if args[i0] % 16 == 0 else 0))
            + (1 if args[i1] == 1 else (2 if args[i1] % 16 == 0 else 0)) * 3
        )

    def _spec_key(args):
        k = 0
        for i in int_indices:
            v = args[i]
            k = k * 3 + (1 if v == 1 else (2 if v % 16 == 0 else 0))
        return k

    return _spec_key
