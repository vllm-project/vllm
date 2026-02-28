# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton autotuning control for vLLM.

Provides ``vllm_autotune``, a drop-in replacement for ``@triton.autotune``
that respects the ``VLLM_TRITON_AUTOTUNE`` environment variable.

When ``VLLM_TRITON_AUTOTUNE=0`` (the default), only the first config in
the list is used — this skips all benchmarking while still compiling the
kernel with the correct compile-time constants.

When ``VLLM_TRITON_AUTOTUNE=1``, the full config list is passed through
to ``triton.autotune`` for runtime benchmarking.
"""

from __future__ import annotations

from vllm.triton_utils.importing import HAS_TRITON

if HAS_TRITON:
    from typing import Any

    import vllm.envs as envs
    from vllm.triton_utils import triton

    def vllm_autotune(
        configs: list[triton.Config],
        key: list[str],
        **kwargs: Any,
    ) -> Any:
        """Drop-in replacement for ``@triton.autotune`` that respects
        ``VLLM_TRITON_AUTOTUNE``.

        When ``VLLM_TRITON_AUTOTUNE=0`` (default): uses only ``configs[0]``,
        skipping all benchmarking. The kernel is compiled with the first
        config's compile-time constants (BLOCK_M, num_warps, etc.).

        When ``VLLM_TRITON_AUTOTUNE=1``: delegates to ``triton.autotune``
        with the full config list for runtime benchmarking.

        Args:
            configs: List of ``triton.Config`` objects to benchmark.
            key: List of argument names whose values determine the cache key
                for autotuning results.
            **kwargs: Additional keyword arguments passed to
                ``triton.autotune`` (e.g., ``prune_configs_by``,
                ``warmup``, ``rep``).
        """
        if envs.VLLM_TRITON_AUTOTUNE:
            return triton.autotune(configs=configs, key=key, **kwargs)
        # Single config: Triton skips benchmarking entirely while still
        # compiling the kernel with the config's kwargs as tl.constexpr
        # values and applying num_warps/num_stages.
        return triton.autotune(configs=configs[:1], key=key, **kwargs)

    _original_triton_autotune: Any | None = None

    def disable_autotune_globally() -> None:
        """Monkey-patch ``triton.autotune`` to use first-config-only mode.

        This affects ALL Triton code in the process, including third-party
        libraries (FlashInfer, etc.). Use only when full determinism is
        required (e.g., ``VLLM_BATCH_INVARIANT=1``).
        """
        global _original_triton_autotune
        if _original_triton_autotune is not None:
            return  # Already patched

        _original_triton_autotune = triton.autotune

        def _fixed_autotune(
            configs: list[triton.Config],
            key: list[str],
            **kwargs: Any,
        ) -> Any:
            return _original_triton_autotune(configs=configs[:1], key=key, **kwargs)

        triton.autotune = _fixed_autotune  # type: ignore[assignment]

    def restore_autotune_globally() -> None:
        """Undo the monkey-patch applied by :func:`disable_autotune_globally`.

        Primarily useful in tests.
        """
        global _original_triton_autotune
        if _original_triton_autotune is None:
            return
        triton.autotune = _original_triton_autotune  # type: ignore[assignment]
        _original_triton_autotune = None

else:
    # Fallback when Triton is not installed
    def vllm_autotune(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):  # type: ignore[no-untyped-def]
            return fn

        if args and callable(args[0]):
            return args[0]
        return decorator

    def disable_autotune_globally() -> None:  # type: ignore[misc]
        pass

    def restore_autotune_globally() -> None:  # type: ignore[misc]
        pass
