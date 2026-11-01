# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse ``cute.compile`` results across processes (VLLM_CUTE_DSL_COMPILE_CACHE).

``cute.compile`` forces ``no_cache``, so every process re-runs the full
MLIR -> PTX -> cubin pipeline for every CuTe DSL kernel it compiles.

With the variable set, compile-only builds go through CuTe DSL's own caches:
the in-memory one, then the file cache under ``CUTE_DSL_CACHE_DIR`` (default
``<VLLM_CACHE_ROOT>/cute_dsl``). The key is CuTe DSL's module hash, which covers
the traced IR, the compile-affecting environment, the compile options and the
DSL version, so a hit is the same binary. Tracing still runs to compute the key;
only the compilation is skipped. CuTe DSL publishes each entry with a unique
temporary directory and a rename and checks a CRC32 on load.
``CUTE_DSL_NO_CACHE``, ``CUTE_DSL_KEEP``, ``CUTE_DSL_KEEP_*`` and
``CUTE_DSL_DISABLE_FILE_CACHING`` keep their meaning.
"""

import functools
import os
import threading
import time

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_installed: bool = False
_stats_lock = threading.Lock()
_stats: dict[str, float] = {}


def _record(name: str, value: float = 1) -> None:
    with _stats_lock:
        _stats[name] = _stats.get(name, 0) + value


def get_compile_cache_stats() -> dict[str, float]:
    """Counters of this process's CuTe DSL compile path, if the cache is on.

    ``traces`` counts compile-only traces; ``backend_compiles`` the traces that
    reached the compiler backend (the rest were in-memory hits), of which
    ``file_hits`` loaded the compiled module from ``CUTE_DSL_CACHE_DIR`` and
    ``misses`` ran the full pipeline. ``backend_s`` is the time spent in the
    backend.
    """
    with _stats_lock:
        return dict(_stats)


def _cache_opted_out(envar: object) -> bool:
    """Whether CuTe DSL's own settings ask for uncached builds."""
    return any(
        getattr(envar, name, False)
        for name in ("no_cache", "keep_ptx", "keep_cubin", "keep_sass")
    )


def _ensure_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        if not os.path.isdir(path):
            logger.warning("Cannot create CuTe DSL cache directory %s: %s", path, e)
            return False
    return True


def _install_counters() -> None:
    try:
        from cutlass.base_dsl.compile_backend import CompilerBackend
    except Exception:
        return
    resolve = getattr(CompilerBackend, "_resolve_compile_cache", None)
    compile_ = getattr(CompilerBackend, "compile", None)
    if resolve is None or compile_ is None:
        return

    @functools.wraps(resolve)
    def _resolve_compile_cache(self, ctx):
        state = resolve(self, ctx)
        if not ctx.cache_enabled:
            _record("uncached")
        elif state.cache_hit:
            _record("file_hits" if state.load_from_file_cache else "memory_hits")
        else:
            _record("misses")
        return state

    @functools.wraps(compile_)
    def _compile(self, ctx):
        start = time.perf_counter()
        try:
            return compile_(self, ctx)
        finally:
            _record("backend_compiles")
            _record("backend_s", time.perf_counter() - start)

    CompilerBackend._resolve_compile_cache = _resolve_compile_cache
    CompilerBackend.compile = _compile


def maybe_enable_cute_dsl_compile_cache() -> bool:
    """Enable the cache in this process if VLLM_CUTE_DSL_COMPILE_CACHE is set.

    Must run before the first ``cute.compile`` of the process. Returns whether
    the cache is enabled.
    """
    global _installed
    if _installed:
        return True
    if not envs.VLLM_CUTE_DSL_COMPILE_CACHE:
        return False

    try:
        from cutlass.base_dsl.dsl import BaseDSL
    except Exception:
        logger.debug("CuTe DSL is not available; not caching cute.compile results.")
        return False

    prepare = getattr(BaseDSL, "_prepare_compilation", None)
    setup_cls = getattr(BaseDSL, "_CompilationSetup", None)
    setup_fields = getattr(setup_cls, "__dataclass_fields__", {})
    if prepare is None or not {"no_cache", "compile_only"} <= setup_fields.keys():
        logger.warning(
            "VLLM_CUTE_DSL_COMPILE_CACHE is set, but this CuTe DSL version has "
            "no BaseDSL._prepare_compilation hook; cute.compile results are "
            "not cached."
        )
        return False

    cache_dir = os.environ.get("CUTE_DSL_CACHE_DIR") or os.path.join(
        envs.VLLM_CACHE_ROOT, "cute_dsl"
    )
    # CuTe DSL creates the directory outside its error handling on every
    # write, so a directory that cannot exist would fail compiles.
    if not _ensure_dir(cache_dir):
        return False
    os.environ["CUTE_DSL_CACHE_DIR"] = cache_dir

    @functools.wraps(prepare)
    def _prepare_compilation(self, funcBody, *args, **kwargs):
        setup = prepare(self, funcBody, *args, **kwargs)
        if setup.compile_only:
            _record("traces")
        # cute.compile is the compile-only entry point and the only one that
        # forces no_cache; keep the cache off when CuTe DSL's own settings
        # ask for it.
        if (
            setup.compile_only
            and setup.no_cache
            and not _cache_opted_out(getattr(self, "envar", None))
        ):
            setup.no_cache = False
        return setup

    BaseDSL._prepare_compilation = _prepare_compilation
    _install_counters()
    _installed = True
    logger.info_once(
        "Caching cute.compile results in %s (VLLM_CUTE_DSL_COMPILE_CACHE).",
        cache_dir,
    )
    return True
