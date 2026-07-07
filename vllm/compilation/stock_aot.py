# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""AOT-compile driver for the STOCK_TORCH_COMPILE path.

This is an alternative to the lazy eval-frame ``nn.Module.compile()`` driver used
by the default stock path. Instead of arming Dynamo's frame-evaluation hook and
compiling on the first forward, it drives compilation through torch's
``aot_compile`` API (``torch.compile(fn, backend="inductor").aot_compile(...)``),
producing a standalone ``AOTCompiledFunction`` that runs the model WITHOUT the
Dynamo eval frame or per-call guard checks. Selected in STOCK_TORCH_COMPILE mode
when ``VLLM_USE_AOT_COMPILE`` is set.

It mirrors the AOT plumbing in ``decorators.py`` (which the VllmBackend path uses
under the same env var) but targets the stock Inductor backend and is driven by
the model runner rather than the ``@support_torch_compile`` decorator, so the
external cudagraph wrapper and Inductor graph-partition wiring are identical to
the eval-frame stock path -- only the compile driver differs.
"""

import contextlib
import hashlib
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


def stock_aot_available() -> bool:
    """Whether the current torch exposes the aot_compile API this path needs."""
    return hasattr(torch._dynamo.config, "enable_aot_compile") and hasattr(
        torch.compiler, "load_compiled_function"
    )


@contextlib.contextmanager
def _enable_aot_compile():
    if hasattr(torch._dynamo.config, "enable_aot_compile"):
        with torch._dynamo.config.patch(enable_aot_compile=True):
            yield
    else:
        yield


def _mark_dynamic_batch_dim(*args: Any, **kwargs: Any) -> None:
    """Mark dim 0 (the token/batch dim) dynamic on every tensor input.

    The stock eval-frame path gets this from ``nn.Module.compile``'s automatic
    dynamic handling; aot_compile bakes in a single graph, so the dynamic dims
    must be marked explicitly on the example inputs or the graph specializes to
    the profile-run shape and cannot serve other batch sizes.
    """
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            torch._dynamo.mark_dynamic(value, 0)
        elif isinstance(value, IntermediateTensors):
            for tensor in value.tensors.values():
                torch._dynamo.mark_dynamic(tensor, 0)


class _StockAOTForward:
    """Lazy aot-compile wrapper installed onto a module's ``forward``.

    On the first forward (during ``profile_run``) it aot-compiles the original
    bound forward against the real inputs, then stores and calls the resulting
    ``AOTCompiledFunction`` directly. When a serialized artifact is present it is
    loaded eagerly at install time so warm starts skip the first-forward compile
    entirely.
    """

    def __init__(
        self,
        module: nn.Module,
        vllm_config: VllmConfig,
        options: dict[str, Any] | None,
        aot_path: str | None,
    ) -> None:
        self.module = module
        self.vllm_config = vllm_config
        self.options = options
        self.aot_path = aot_path
        self._original_forward = module.forward
        self._aot_fn: Callable[..., Any] | None = None
        self._loaded_from_disk = False
        self._evaluate_guards = (
            vllm_config.compilation_config.dynamic_shapes_config.evaluate_guards
        )

        # Drop all guards, exactly as TorchCompileWithNoGuardsWrapper does for the
        # VllmBackend path. vLLM guarantees a single full graph with the dynamic dims
        # marked, so the guards are unnecessary; dropping them also avoids serializing
        # guards that reference unpicklable C++ objects (e.g. mxfp4 kernels), which
        # otherwise breaks aot_compile's guard serialization.
        compile_options = dict(options or {})
        if hasattr(torch.compiler, "skip_all_guards_unsafe"):
            compile_options["guard_filter_fn"] = torch.compiler.skip_all_guards_unsafe
        else:
            compile_options["guard_filter_fn"] = lambda guards: [False for _ in guards]

        # enable_aot_compile must be active at construction time for torch.compile
        # to attach the .aot_compile method to the returned callable (matching
        # TorchCompileWithNoGuardsWrapper.__init__).
        with _enable_aot_compile():
            self._compiled_callable = torch.compile(
                self._original_forward,
                fullgraph=True,
                dynamic=False,
                backend="inductor",
                options=compile_options,
            )

    def install(self) -> None:
        if self.aot_path is not None and not envs.VLLM_DISABLE_COMPILE_CACHE:
            self._aot_fn = self._try_load()
            if self._aot_fn is not None:
                self._loaded_from_disk = True
        self.module.forward = self  # type: ignore[method-assign, assignment]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._aot_fn is None:
            self._compile(args, kwargs)
        # AOTCompiledFunction is compiled from the *unbound* forward code, so the
        # module (self.module) must be supplied as the first ("self") argument,
        # matching decorators.py's `self.aot_compiled_fn(self, *args, **kwargs)`.
        return self._aot_fn(self.module, *args, **kwargs)

    def _compile(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        _mark_dynamic_batch_dim(*args, **kwargs)
        with _enable_aot_compile():
            aot_fn = self._compiled_callable.aot_compile((args, kwargs))
        if not self._evaluate_guards:
            # vLLM guarantees a single full graph and pads to captured shapes, so
            # the per-call guard check is pure overhead -- drop it to realize the
            # eval-frame-free benefit of the aot path.
            aot_fn.disable_guard_check()
        self._aot_fn = aot_fn
        compilation_counter.num_aot_compiles += 1
        compilation_counter.stock_torch_compile_count += 1
        self._maybe_save(aot_fn)

    def _try_load(self) -> Callable[..., Any] | None:
        from vllm.compilation.decorators import _verify_source_unchanged

        assert self.aot_path is not None
        if not os.path.exists(self.aot_path):
            return None
        try:
            with open(self.aot_path, "rb") as f:
                loaded_fn = torch.compiler.load_compiled_function(
                    f, f_globals=self._original_forward.__func__.__globals__
                )
            _verify_source_unchanged(loaded_fn.source_info(), self.vllm_config)
            # Some backends expose finalize_loading to eagerly materialize the
            # compiled artifact; the stock inductor compiled_fn does not, and its
            # partitions bind to the persistently-installed partition wrapper lazily
            # on the first call, so only call it when present.
            compiled_fn = loaded_fn._artifacts.compiled_fn
            if hasattr(compiled_fn, "finalize_loading"):
                compiled_fn.finalize_loading(self.vllm_config)
            if not self._evaluate_guards:
                loaded_fn.disable_guard_check()
            compilation_counter.num_aot_artifacts_loaded += 1
            logger.info("Loaded stock AOT compilation from %s", self.aot_path)
            return loaded_fn
        except Exception as e:
            logger.warning(
                "Recompiling: failed to load stock AOT artifact from %s (%s)",
                self.aot_path,
                repr(e),
            )
            if envs.VLLM_FORCE_AOT_LOAD:
                raise
            return None

    def _maybe_save(self, aot_fn: Callable[..., Any]) -> None:
        if self.aot_path is None or envs.VLLM_DISABLE_COMPILE_CACHE:
            return
        if self._loaded_from_disk:
            return
        try:
            os.makedirs(os.path.dirname(self.aot_path), exist_ok=True)
            tmp = f"{self.aot_path}.{os.getpid()}.tmp"
            aot_fn.save_compiled_function(tmp)
            os.replace(tmp, self.aot_path)
            compilation_counter.num_aot_artifacts_saved += 1
            logger.info_once("Saved stock AOT compilation to %s", self.aot_path)
        except Exception as e:
            logger.warning(
                "Unable to save stock AOT artifact to %s: %s", self.aot_path, e
            )


def _aot_cache_path(module: nn.Module, vllm_config: VllmConfig) -> str | None:
    """Per-(config, model, rank) path for the serialized aot artifact, and set the
    process Inductor cache dir underneath it so the warm path reuses codegen."""
    from vllm.compilation.caching import aot_compile_hash_factors

    try:
        factors = aot_compile_hash_factors(vllm_config)
        factors.append(module.__class__.__qualname__)
        factors.append(module.forward.__func__.__qualname__)
    except Exception:
        return None
    hash_key = hashlib.sha256(str(factors).encode()).hexdigest()
    base = os.path.join(
        envs.VLLM_CACHE_ROOT, "torch_compile_cache", "stock_aot_compile", hash_key
    )
    inductor_cache = os.path.join(base, "inductor_cache")
    os.makedirs(inductor_cache, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
    rank = vllm_config.parallel_config.rank
    dp_rank = vllm_config.parallel_config.data_parallel_index
    return os.path.join(base, f"rank_{rank}_{dp_rank}", "model")


def install_stock_aot_forward(
    module: nn.Module,
    vllm_config: VllmConfig,
    options: dict[str, Any] | None,
) -> None:
    """Install the lazy aot-compile forward on ``module`` for the stock path."""
    aot_path = _aot_cache_path(module, vllm_config)
    _StockAOTForward(module, vllm_config, options, aot_path).install()
