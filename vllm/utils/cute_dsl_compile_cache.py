# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse `cute.compile` results across processes (VLLM_CUTE_DSL_COMPILE_CACHE).

`cute.compile` forces `no_cache`, so every process re-runs the full
MLIR -> PTX -> cubin pipeline for every CuTe DSL kernel it compiles.

With the variable set, compile-only builds go through CuTe DSL's own caches:
the in-memory one, then the file cache under `CUTE_DSL_CACHE_DIR` (default
`<VLLM_CACHE_ROOT>/cute_dsl`). The key is CuTe DSL's module hash, which covers
the traced IR, the compile-affecting environment, the compile options and the
DSL version. It does not cover the TVM-FFI argument spec attached after the
compiler passes: calls with identical IR but different specs could reuse the
first call's wrapper. Tracing still runs to compute the key; only compilation
is skipped. CuTe DSL publishes each entry with a unique
temporary directory and a rename and checks a CRC32 on load.
`CUTE_DSL_NO_CACHE`, `CUTE_DSL_KEEP`, `CUTE_DSL_KEEP_*` and
`CUTE_DSL_DISABLE_FILE_CACHING` keep their meaning.
"""

import functools
import importlib
import inspect
import os
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_installed: bool = False


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


def maybe_enable_cute_dsl_compile_cache() -> bool:
    """Enable the cache in this process if VLLM_CUTE_DSL_COMPILE_CACHE is set.

    Must run before the first `cute.compile` of the process. Returns whether
    the cache is enabled.
    """
    global _installed
    if _installed:
        return True
    if not envs.VLLM_CUTE_DSL_COMPILE_CACHE:
        return False

    try:
        from cutlass.base_dsl.compiler import (
            CompileOptions,
            KeepCUBIN,
            KeepPTX,
            KeepSASS,
        )
        from cutlass.base_dsl.dsl import BaseDSL

        cache_helpers: Any = importlib.import_module("cutlass.base_dsl.cache_helpers")
    except Exception:
        logger.debug("CuTe DSL is not available; not caching cute.compile results.")
        return False

    try:
        supported_options = {
            KeepPTX,
            KeepCUBIN,
            KeepSASS,
        } <= CompileOptions().options.keys()
    except Exception:
        supported_options = False
    if not supported_options:
        logger.warning(
            "VLLM_CUTE_DSL_COMPILE_CACHE is set, but this CuTe DSL version has "
            "no supported Keep compile options; cute.compile results are not cached."
        )
        return False

    load_ir = getattr(cache_helpers, "load_ir", None)
    try:
        params = inspect.signature(load_ir).parameters if callable(load_ir) else {}
        supported_load = (
            tuple(params) == ("file", "asBytecode", "bytecode_reader")
            and all(
                p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                for p in params.values()
            )
            and params["asBytecode"].default is False
            and params["bytecode_reader"].default is None
        )
    except (TypeError, ValueError):
        supported_load = False
    if not supported_load or not callable(load_ir):
        logger.warning(
            "VLLM_CUTE_DSL_COMPILE_CACHE is set, but this CuTe DSL version has "
            "no supported cache_helpers.load_ir hook; cute.compile results "
            "are not cached."
        )
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
        # cute.compile is the compile-only entry point and the only one that
        # forces no_cache; keep the cache off when CuTe DSL's own settings
        # ask for it.
        if (
            setup.compile_only
            and setup.no_cache
            and not _cache_opted_out(getattr(self, "envar", None))
            and not any(
                self.compile_options.options[option].value
                for option in (KeepPTX, KeepCUBIN, KeepSASS)
            )
        ):
            setup.no_cache = False
        return setup

    @functools.wraps(load_ir)
    def _load_ir(file, asBytecode=False, bytecode_reader=None):
        # The original creates its own context, so wrapping it in a context
        # would still leave a thread pool alive for every loaded module.
        assert "mlir" in file
        func_name = file.split(".mlir")[0].split("dsl_")[-1]
        with cache_helpers.ir.Context() as ctx:
            ctx.enable_multithreading(False)
            with open(file, "rb" if asBytecode else "r") as f:
                module = (
                    bytecode_reader(f)
                    if bytecode_reader
                    else cache_helpers.ir.Module.parse(f.read())
                )
        return func_name, module

    cache_helpers.load_ir = _load_ir
    BaseDSL._prepare_compilation = _prepare_compilation
    _installed = True
    logger.info_once(
        "Caching cute.compile results in %s (VLLM_CUTE_DSL_COMPILE_CACHE).",
        cache_dir,
    )
    return True
