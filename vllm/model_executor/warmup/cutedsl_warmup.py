# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orchestrate CuTeDSL warmup providers."""

from __future__ import annotations

from collections.abc import Callable, Generator, Hashable, Iterable, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Literal

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.tracing import instrument

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

CuTeDSLCompileFn = Callable[[], None]
CuTeDSLWarmupCoverage = Literal["targeted", "full"]
CuTeDSLWarmupTokenScope = Literal["rank_local", "data_parallel", "world"]
_VALID_WARMUP_COVERAGE = {"targeted", "full"}
_VALID_TOKEN_SCOPES = {"rank_local", "data_parallel", "world"}
_CUTE_DSL_CACHE_ENABLED_ENV = "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"
_CUTE_DSL_CACHE_DIR_ENV = "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"
_CUTEDSL_WARMUP_TARGET_ATTRS = (
    "impl",
    "prefill_backend",
    "quant_method",
    "moe_kernel",
    "fused_experts",
)


@dataclass(frozen=True)
class CuTeDSLCompileUnit:
    name: str
    key: Hashable
    compile: CuTeDSLCompileFn


def compile_cutedsl_units(
    units: Iterable[CuTeDSLCompileUnit],
    *,
    provider: str,
    seen: set[Hashable] | None = None,
) -> None:
    seen_keys = seen if seen is not None else set()
    for unit in units:
        key = unit.key
        if key in seen_keys:
            logger.debug(
                "Skipping duplicate CuTeDSL compile unit provider=%s "
                "name=%s key=%r.",
                provider,
                unit.name,
                key,
            )
            continue
        seen_keys.add(key)
        unit.compile()


@dataclass(frozen=True)
class CuTeDSLWarmupContext:
    """Parallelism context used to scale provider warmup shapes."""

    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_enabled: bool = False

    @property
    def world_size(self) -> int:
        return (
            self.data_parallel_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size
        )


@dataclass(frozen=True)
class CuTeDSLWarmupPlan:
    """Warmup work requested by one CuTeDSL integration.

    CuTeDSL-backed classes can expose this by defining
    ``get_cutedsl_warmup_plan(runner)`` on the existing model module, attention
    impl, prefill backend, or kernel integration object.
    """

    provider: str
    """Human-readable provider name used in logs and error messages."""

    compile_units: tuple[CuTeDSLCompileUnit, ...] = ()
    """Concrete compile-only work that does not depend on token sizes."""


def _get_cutedsl_warmup_context(runner: "GPUModelRunner") -> CuTeDSLWarmupContext:
    parallel_config = getattr(runner.vllm_config, "parallel_config", None)
    return CuTeDSLWarmupContext(
        data_parallel_size=max(
            1, int(getattr(parallel_config, "data_parallel_size", 1) or 1)
        ),
        tensor_parallel_size=max(
            1, int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
        ),
        pipeline_parallel_size=max(
            1, int(getattr(parallel_config, "pipeline_parallel_size", 1) or 1)
        ),
        expert_parallel_enabled=bool(
            getattr(parallel_config, "enable_expert_parallel", False)
        ),
    )


def get_cutedsl_warmup_token_sizes(
    runner: "GPUModelRunner",
    coverage: CuTeDSLWarmupCoverage = "targeted",
    token_size_scope: CuTeDSLWarmupTokenScope = "rank_local",
) -> list[int]:
    if coverage not in _VALID_WARMUP_COVERAGE:
        raise ValueError(
            "Invalid CuTeDSL warmup coverage "
            f"{coverage!r}. Valid coverage values are "
            f"{sorted(_VALID_WARMUP_COVERAGE)}."
        )
    if token_size_scope not in _VALID_TOKEN_SCOPES:
        raise ValueError(
            "Invalid CuTeDSL warmup token size scope "
            f"{token_size_scope!r}. Valid scopes are "
            f"{sorted(_VALID_TOKEN_SCOPES)}."
        )

    kernel_config = runner.vllm_config.kernel_config
    max_tokens = runner.scheduler_config.max_num_batched_tokens
    if coverage == "full":
        token_sizes = _derive_cutedsl_warmup_token_sizes(
            runner,
            coverage=coverage,
            max_tokens=max_tokens,
        )
    else:
        configured_sizes = kernel_config.cutedsl_warmup_token_sizes
        token_sizes = _normalize_cutedsl_warmup_token_sizes(
            configured_sizes,
            max_tokens=max_tokens,
        )
        if not token_sizes:
            token_sizes = _derive_cutedsl_warmup_token_sizes(
                runner,
                coverage=coverage,
                max_tokens=max_tokens,
            )

    context = _get_cutedsl_warmup_context(runner)
    return _scale_cutedsl_token_sizes(token_sizes, token_size_scope, context)


def _get_cutedsl_warmup_token_sizes(
    runner: "GPUModelRunner",
    coverage: CuTeDSLWarmupCoverage = "targeted",
) -> list[int]:
    return get_cutedsl_warmup_token_sizes(runner, coverage)


def _normalize_cutedsl_warmup_token_sizes(
    token_sizes: Sequence[int],
    *,
    max_tokens: int,
) -> list[int]:
    return sorted(
        {
            min(size, max_tokens)
            for size in token_sizes
            if isinstance(size, int) and size > 0
        }
    )


def _derive_cutedsl_warmup_token_sizes(
    runner: "GPUModelRunner",
    *,
    coverage: CuTeDSLWarmupCoverage,
    max_tokens: int,
) -> list[int]:
    if coverage == "full":
        return list(range(1, max_tokens + 1))

    compilation_config = getattr(runner.vllm_config, "compilation_config", None)
    cudagraph_sizes = _normalize_cutedsl_warmup_token_sizes(
        getattr(compilation_config, "cudagraph_capture_sizes", ()) or (),
        max_tokens=max_tokens,
    )
    if cudagraph_sizes:
        return cudagraph_sizes

    # Match vLLM's eager/profile warmup shape. Eager startup does not sweep all
    # token sizes; it profiles a single dummy batch at max_num_batched_tokens.
    return [max_tokens]


def _iter_cutedsl_warmup_targets(
    model: torch.nn.Module,
) -> Iterable[object]:
    seen: set[int] = set()

    for module in model.modules():
        candidates = [module]
        for candidate in candidates:
            if candidate is None:
                continue
            candidate_id = id(candidate)
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            yield candidate
            for attr in _CUTEDSL_WARMUP_TARGET_ATTRS:
                candidates.append(getattr(candidate, attr, None))


def _coerce_cutedsl_warmup_plans(
    value: object,
) -> list[CuTeDSLWarmupPlan]:
    if value is None:
        return []
    if isinstance(value, CuTeDSLWarmupPlan):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        plans = list(value)
        if all(isinstance(plan, CuTeDSLWarmupPlan) for plan in plans):
            return plans
    raise TypeError(
        "get_cutedsl_warmup_plan must return CuTeDSLWarmupPlan, "
        "an iterable of CuTeDSLWarmupPlan, or None"
    )


def _get_cutedsl_warmup_plans(
    runner: "GPUModelRunner",
    model: torch.nn.Module,
) -> list[CuTeDSLWarmupPlan]:
    plans: list[CuTeDSLWarmupPlan] = []

    for target in _iter_cutedsl_warmup_targets(model):
        get_plan = getattr(target, "get_cutedsl_warmup_plan", None)
        if get_plan is not None:
            plans.extend(_coerce_cutedsl_warmup_plans(get_plan(runner)))
    return plans


def _reset_cutedsl_persistent_cache() -> bool:
    if os.environ.get(_CUTE_DSL_CACHE_ENABLED_ENV) != "1":
        return False

    cache_dir = os.environ.get(_CUTE_DSL_CACHE_DIR_ENV)
    if not cache_dir:
        return False

    cache_path = Path(cache_dir)
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        for child in cache_path.rglob("*"):
            if child.name == ".warmup.lock":
                continue
            if child.is_file():
                child.unlink(missing_ok=True)
    except Exception:
        logger.exception("Failed to reset CuTeDSL persistent cache at %s.", cache_path)
        return False

    logger.warning(
        "Reset CuTeDSL persistent cache at %s after warmup failed.",
        cache_path,
    )
    return True


def _clear_cutedsl_in_memory_caches() -> None:
    cache_attrs = (
        "_flash_attn_fwd",
        "_flash_attn_fwd_combine",
        "_flash_attn_bwd",
        "_bwd_preprocess",
        "_bwd_postprocess_convert",
    )
    cleared = 0
    for module_name, module in list(sys.modules.items()):
        if not module_name.endswith(".cute.interface"):
            continue
        for attr_name in cache_attrs:
            fn = getattr(module, attr_name, None)
            cache = getattr(fn, "compile_cache", None)
            clear = getattr(cache, "clear", None)
            if clear is None:
                continue
            clear()
            cleared += 1
    if cleared:
        logger.warning("Cleared %d CuTeDSL in-memory compile caches.", cleared)


def _scale_cutedsl_token_sizes(
    token_sizes: Sequence[int],
    scope: CuTeDSLWarmupTokenScope,
    context: CuTeDSLWarmupContext,
) -> list[int]:
    if scope == "rank_local":
        factor = 1
    elif scope == "data_parallel":
        factor = context.data_parallel_size
    else:
        factor = context.world_size
    return [size * factor for size in token_sizes]


def _get_world_group_or_none():
    try:
        from vllm.distributed.parallel_state import get_world_group

        return get_world_group()
    except Exception:
        return None


def _barrier_after_cutedsl_warmup() -> None:
    world = _get_world_group_or_none()
    if world is None or getattr(world, "world_size", 1) <= 1:
        return
    world.barrier()


def _should_lock_cutedsl_persistent_cache() -> bool:
    if os.environ.get(_CUTE_DSL_CACHE_ENABLED_ENV) != "1":
        return False

    world = _get_world_group_or_none()
    return world is None or getattr(world, "world_size", 1) <= 1


@contextmanager
def _cutedsl_persistent_cache_lock() -> Generator[None, None, None]:
    if os.environ.get(_CUTE_DSL_CACHE_ENABLED_ENV) != "1":
        yield
        return

    cache_dir = os.environ.get(_CUTE_DSL_CACHE_DIR_ENV)
    if not cache_dir:
        yield
        return

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path / ".warmup.lock"
    with open(lock_path, "a+") as lock_file:
        logger.debug("Acquiring CuTeDSL persistent cache warmup lock: %s", lock_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _run_cutedsl_warmup_once(
    plans: list[CuTeDSLWarmupPlan],
) -> None:
    seen: set[Hashable] = set()
    with torch.inference_mode():
        for plan in plans:
            if plan.compile_units:
                compile_cutedsl_units(
                    plan.compile_units,
                    provider=plan.provider,
                    seen=seen,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def _count_unique_cutedsl_compile_units(
    plans: Iterable[CuTeDSLWarmupPlan],
) -> int:
    return len({unit.key for plan in plans for unit in plan.compile_units})


@instrument(span_name="CuTeDSL warmup")
def cutedsl_warmup(runner: "GPUModelRunner") -> None:
    """Run CuTeDSL warmup providers before serving."""
    if not current_platform.is_cuda():
        logger.info("Skipping CuTeDSL warmup on non-CUDA platform.")
        return

    if runner.is_pooling_model:
        logger.info("Skipping CuTeDSL warmup for pooling model.")
        return

    model = runner.get_model()
    plans = _get_cutedsl_warmup_plans(runner, model)
    if not plans:
        logger.info(
            "Skipping CuTeDSL warmup because no providers requested warmup."
        )
        return

    provider_names = list(dict.fromkeys(plan.provider for plan in plans))
    context = _get_cutedsl_warmup_context(runner)

    logger.info(
        "Warming up CuTeDSL providers=%s with compile_units=%d, "
        "parallel_context=%s.",
        provider_names,
        _count_unique_cutedsl_compile_units(plans),
        context,
    )

    start_time = time.perf_counter()
    lock_context = (
        _cutedsl_persistent_cache_lock()
        if _should_lock_cutedsl_persistent_cache()
        else nullcontext()
    )
    with lock_context:
        try:
            _run_cutedsl_warmup_once(
                plans,
            )
        except Exception:
            if not _reset_cutedsl_persistent_cache():
                logger.exception("CuTeDSL warmup providers=%s failed.", provider_names)
                raise

            _clear_cutedsl_in_memory_caches()
            logger.warning(
                "CuTeDSL warmup providers=%s failed. Retrying once after "
                "persistent cache reset.",
                provider_names,
                exc_info=True,
            )
            try:
                _run_cutedsl_warmup_once(
                    plans,
                )
            except Exception:
                logger.exception(
                    "CuTeDSL warmup providers=%s failed after persistent "
                    "cache reset.",
                    provider_names,
                )
                raise

    _barrier_after_cutedsl_warmup()
    logger.info("CuTeDSL warmup completed in %.2f s.", time.perf_counter() - start_time)
