# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skip Triton autotuning under VLLM_TRITON_FORCE_FIRST_CONFIG."""

from vllm.logger import init_logger
from vllm.triton_utils.importing import HAS_TRITON

logger = init_logger(__name__)

_installed: bool = False


def is_installed() -> bool:
    """Return whether the first-valid-config patch is currently installed."""
    return _installed


def install() -> None:
    """Install the Autotuner.run replacement."""
    global _installed
    if _installed:
        return
    if not HAS_TRITON:
        return

    import importlib

    autotuner_mod = importlib.import_module("triton.runtime.autotuner")
    Autotuner = autotuner_mod.Autotuner
    from triton.compiler.errors import CompileTimeAssertionFailure
    from triton.runtime.errors import OutOfResources, PTXASError

    _invalid_config_errors = (OutOfResources, CompileTimeAssertionFailure, PTXASError)
    _picked_cache: dict[tuple, int] = {}
    seen_kernels: set[str] = set()

    def _run_first_valid_config(self, *args, **kwargs):
        if not self.configs:
            return self.fn(*args, **kwargs)

        key_vals = tuple(kwargs[name] for name in self.keys if name in kwargs)
        cache_key = (id(self), key_vals)
        kernel_name = getattr(self.base_fn, "__name__", repr(self.fn))

        cached_idx = _picked_cache.get(cache_key)
        candidate_indices = (
            [cached_idx] if cached_idx is not None else list(range(len(self.configs)))
        )

        last_exc: Exception | None = None
        for idx in candidate_indices:
            config = self.configs[idx]
            if config.pre_hook is not None:
                full_nargs = {
                    **dict(zip(self.arg_names, args)),
                    **kwargs,
                    **config.all_kwargs(),
                }
                config.pre_hook(full_nargs)
            # Prefer self.fn.run(...) — the kernel-launch entrypoint for both
            # JITFunction and Heuristics. Calling JITFunction(...) directly
            # raises "Cannot call @triton.jit'd outside of the scope of a
            # kernel". Fall back to plain call only if .run is missing.
            launch = getattr(self.fn, "run", self.fn)
            try:
                result = launch(*args, **kwargs, **config.all_kwargs())
            except _invalid_config_errors as e:
                last_exc = e
                continue

            if cached_idx is None:
                _picked_cache[cache_key] = idx
                self.best_config = config
                if kernel_name not in seen_kernels:
                    seen_kernels.add(kernel_name)
                    logger.info(
                        "[triton-autotune-disabled] kernel=%s configs=%d "
                        "picked_index=%d picked=%s",
                        kernel_name,
                        len(self.configs),
                        idx,
                        config,
                    )
            return result

        raise RuntimeError(
            f"No valid config for kernel "
            f"{kernel_name} key={key_vals} (tried {len(self.configs)} configs)"
        ) from last_exc

    Autotuner.run = _run_first_valid_config
    _installed = True
