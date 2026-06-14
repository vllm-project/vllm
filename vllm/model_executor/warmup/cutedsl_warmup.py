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

CuTeDSLWarmupCallback = Callable[["GPUModelRunner", Sequence[int]], None]
CuTeDSLWarmupMode = Literal["prefill", "mixed", "uniform_decode"]
CuTeDSLWarmupCoverage = Literal["targeted", "full"]
CuTeDSLWarmupTokenScope = Literal["rank_local", "data_parallel", "world"]
_VALID_MODEL_RUNNER_MODES = {"prefill", "mixed", "uniform_decode"}
_VALID_WARMUP_COVERAGE = {"targeted", "full"}
_VALID_TOKEN_SCOPES = {"rank_local", "data_parallel", "world"}
_CUTE_DSL_CACHE_ENABLED_ENV = "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"
_CUTE_DSL_CACHE_DIR_ENV = "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"


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

    model_runner_modes: tuple[CuTeDSLWarmupMode, ...] = ()
    """Generic ``GPUModelRunner._dummy_run`` batch modes to execute."""

    cudagraph_capture_modes: bool = False
    """Warm eager dummy runs for the runner's CUDA graph capture descriptors."""

    warmup_callbacks: tuple[CuTeDSLWarmupCallback, ...] = ()
    """Provider-specific warmup callbacks for paths generic dummy runs miss."""

    coverage: CuTeDSLWarmupCoverage = "targeted"
    """Warmup token-size coverage requested by this provider."""

    token_size_scope: CuTeDSLWarmupTokenScope = "rank_local"
    """How callback token sizes should be interpreted in distributed runs."""

    dedupe_key: Hashable | None = None
    """Optional key for skipping equivalent warmup plans across modules."""


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


def _get_cutedsl_warmup_token_sizes(
    runner: "GPUModelRunner",
    coverage: CuTeDSLWarmupCoverage = "targeted",
) -> list[int]:
    kernel_config = runner.vllm_config.kernel_config
    max_tokens = runner.scheduler_config.max_num_batched_tokens
    if coverage == "full":
        return list(range(1, max_tokens + 1))

    configured_sizes = kernel_config.cutedsl_warmup_token_sizes

    token_sizes = {
        min(size, max_tokens)
        for size in configured_sizes
        if isinstance(size, int) and size > 0
    }
    return sorted(token_sizes)


def _iter_cutedsl_warmup_targets(
    model: torch.nn.Module,
) -> Iterable[object]:
    seen: set[int] = set()

    for module in model.modules():
        candidates = [
            module,
            getattr(module, "impl", None),
            getattr(module, "prefill_backend", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            candidate_id = id(candidate)
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            yield candidate


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


def _validate_cutedsl_warmup_plan(plan: CuTeDSLWarmupPlan) -> None:
    invalid_modes = set(plan.model_runner_modes) - _VALID_MODEL_RUNNER_MODES
    if invalid_modes:
        raise ValueError(
            "Invalid CuTeDSL warmup model runner mode(s) "
            f"{sorted(invalid_modes)} for provider {plan.provider}. "
            f"Valid modes are {sorted(_VALID_MODEL_RUNNER_MODES)}."
        )
    if plan.coverage not in _VALID_WARMUP_COVERAGE:
        raise ValueError(
            "Invalid CuTeDSL warmup coverage "
            f"{plan.coverage!r} for provider {plan.provider}. "
            f"Valid coverage values are {sorted(_VALID_WARMUP_COVERAGE)}."
        )
    if plan.token_size_scope not in _VALID_TOKEN_SCOPES:
        raise ValueError(
            "Invalid CuTeDSL warmup token size scope "
            f"{plan.token_size_scope!r} for provider {plan.provider}. "
            f"Valid scopes are {sorted(_VALID_TOKEN_SCOPES)}."
        )


def _dedupe_cutedsl_warmup_plans(
    plans: list[CuTeDSLWarmupPlan],
) -> list[CuTeDSLWarmupPlan]:
    deduped: list[CuTeDSLWarmupPlan] = []
    seen: set[Hashable] = set()

    for plan in plans:
        if plan.dedupe_key is None:
            deduped.append(plan)
            continue
        key = (plan.provider, plan.dedupe_key)
        if key in seen:
            logger.debug(
                "Skipping duplicate CuTeDSL warmup plan provider=%s key=%r.",
                plan.provider,
                plan.dedupe_key,
            )
            continue
        seen.add(key)
        deduped.append(plan)
    return deduped


def _get_cutedsl_warmup_plans(
    runner: "GPUModelRunner",
    model: torch.nn.Module,
) -> list[CuTeDSLWarmupPlan]:
    plans: list[CuTeDSLWarmupPlan] = []

    for target in _iter_cutedsl_warmup_targets(model):
        get_plan = getattr(target, "get_cutedsl_warmup_plan", None)
        if get_plan is not None:
            plans.extend(_coerce_cutedsl_warmup_plans(get_plan(runner)))
    for plan in plans:
        _validate_cutedsl_warmup_plan(plan)
    return _dedupe_cutedsl_warmup_plans(plans)


def _run_mixed_dummy_warmup(
    runner: "GPUModelRunner",
    token_sizes: Sequence[int],
) -> None:
    for num_tokens in token_sizes:
        runner._dummy_run(
            num_tokens=num_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )


def _run_prefill_dummy_warmup(
    runner: "GPUModelRunner",
    token_sizes: Sequence[int],
) -> None:
    for num_tokens in token_sizes:
        runner._dummy_run(
            num_tokens=num_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
        )


def _run_uniform_decode_dummy_warmup(runner: "GPUModelRunner") -> None:
    decode_tokens = min(
        runner.scheduler_config.max_num_seqs,
        runner.scheduler_config.max_num_batched_tokens,
    )
    if decode_tokens <= 0:
        return

    runner._dummy_run(
        num_tokens=decode_tokens,
        skip_eplb=True,
        is_profile=True,
        force_attention=True,
        uniform_decode=True,
    )


def _run_cudagraph_capture_mode_warmup(runner: "GPUModelRunner") -> None:
    """Warm eager forwards for CUDA graph capture descriptors.

    CUDA graph capture descriptors are vLLM's concrete, configuration-specific
    batch shapes. Running them eagerly warms JIT'ed kernels without capturing
    graphs or adding a separate shape grid.
    """
    cudagraph_dispatcher = getattr(runner, "cudagraph_dispatcher", None)
    if cudagraph_dispatcher is None:
        logger.info(
            "Skipping CuTeDSL CUDA graph descriptor warmup because the runner "
            "does not have a cudagraph dispatcher."
        )
        return

    capture_descs = cudagraph_dispatcher.get_capture_descs()
    if not capture_descs:
        logger.info(
            "Skipping CuTeDSL CUDA graph descriptor warmup because no CUDA "
            "graph capture descriptors are available."
        )
        return

    descriptors: list[tuple[int, bool, int]] = []
    seen_descs: set[tuple[int, bool, int]] = set()
    for _runtime_mode, batch_descs in capture_descs:
        for desc in batch_descs:
            key = (desc.num_tokens, desc.uniform, desc.num_active_loras)
            if key in seen_descs:
                continue
            seen_descs.add(key)
            descriptors.append(key)

    logger.info(
        "Warming CuTeDSL eager CUDA graph descriptor shapes: %s.",
        descriptors,
    )

    for num_tokens, uniform, num_active_loras in descriptors:
        runner._dummy_run(
            num_tokens=num_tokens,
            cudagraph_runtime_mode=None,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            uniform_decode=uniform,
            allow_microbatching=False,
            remove_lora=False,
            num_active_loras=num_active_loras,
        )


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
    runner: "GPUModelRunner",
    plans: list[CuTeDSLWarmupPlan],
    token_sizes: list[int],
    model_runner_modes: set[CuTeDSLWarmupMode],
    use_cudagraph_capture_modes: bool,
    context: CuTeDSLWarmupContext,
) -> None:
    with torch.inference_mode():
        if token_sizes:
            for plan in plans:
                callback_token_sizes = _scale_cutedsl_token_sizes(
                    token_sizes,
                    plan.token_size_scope,
                    context,
                )
                for callback in plan.warmup_callbacks:
                    callback(runner, callback_token_sizes)
            if "prefill" in model_runner_modes:
                _run_prefill_dummy_warmup(runner, token_sizes)
            if "mixed" in model_runner_modes:
                _run_mixed_dummy_warmup(runner, token_sizes)
            if "uniform_decode" in model_runner_modes:
                _run_uniform_decode_dummy_warmup(runner)
        if use_cudagraph_capture_modes:
            _run_cudagraph_capture_mode_warmup(runner)


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

    provider_names = [plan.provider for plan in plans]
    model_runner_modes = {
        mode for plan in plans for mode in plan.model_runner_modes
    }
    has_cudagraph_capture_modes = any(
        plan.cudagraph_capture_modes for plan in plans
    )
    use_cudagraph_capture_modes = (
        has_cudagraph_capture_modes
        and runner.vllm_config.kernel_config.cutedsl_warmup_use_cudagraph_descriptors
    )
    coverage: CuTeDSLWarmupCoverage = (
        "full" if any(plan.coverage == "full" for plan in plans) else "targeted"
    )
    token_sizes = _get_cutedsl_warmup_token_sizes(runner, coverage)
    context = _get_cutedsl_warmup_context(runner)
    uses_token_sizes = bool(model_runner_modes) or any(
        plan.warmup_callbacks for plan in plans
    )
    if not token_sizes and uses_token_sizes:
        logger.info(
            "Skipping CuTeDSL token-size warmup because no token sizes were "
            "selected."
        )
        if not use_cudagraph_capture_modes:
            return

    logger.info(
        "Warming up CuTeDSL providers=%s with token_sizes=%s "
        "coverage=%s, model_runner_modes=%s, cudagraph_capture_modes=%s, "
        "use_cudagraph_descriptors=%s, parallel_context=%s.",
        provider_names,
        token_sizes,
        coverage,
        sorted(model_runner_modes),
        has_cudagraph_capture_modes,
        use_cudagraph_capture_modes,
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
                runner,
                plans,
                token_sizes,
                model_runner_modes,
                use_cudagraph_capture_modes,
                context,
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
                    runner,
                    plans,
                    token_sizes,
                    model_runner_modes,
                    use_cudagraph_capture_modes,
                    context,
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
