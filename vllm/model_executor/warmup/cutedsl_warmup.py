# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Orchestrate CuTeDSL warmup providers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
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
_VALID_MODEL_RUNNER_MODES = {"prefill", "mixed", "uniform_decode"}
_CUTE_DSL_CACHE_ENABLED_ENV = "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"
_CUTE_DSL_CACHE_DIR_ENV = "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"


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


def _get_cutedsl_warmup_token_sizes(runner: "GPUModelRunner") -> list[int]:
    kernel_config = runner.vllm_config.kernel_config
    max_tokens = runner.scheduler_config.max_num_batched_tokens
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
    return plans


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


def _run_cutedsl_warmup_once(
    runner: "GPUModelRunner",
    plans: list[CuTeDSLWarmupPlan],
    token_sizes: list[int],
    model_runner_modes: set[CuTeDSLWarmupMode],
    use_cudagraph_capture_modes: bool,
) -> None:
    with torch.inference_mode():
        if token_sizes:
            for plan in plans:
                for callback in plan.warmup_callbacks:
                    callback(runner, token_sizes)
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
    token_sizes = _get_cutedsl_warmup_token_sizes(runner)
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
        "model_runner_modes=%s, cudagraph_capture_modes=%s, "
        "use_cudagraph_descriptors=%s.",
        provider_names,
        token_sizes,
        sorted(model_runner_modes),
        has_cudagraph_capture_modes,
        use_cudagraph_capture_modes,
    )

    start_time = time.perf_counter()
    try:
        _run_cutedsl_warmup_once(
            runner,
            plans,
            token_sizes,
            model_runner_modes,
            use_cudagraph_capture_modes,
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
            )
        except Exception:
            logger.exception(
                "CuTeDSL warmup providers=%s failed after persistent cache reset.",
                provider_names,
            )
            raise

    logger.info("CuTeDSL warmup completed in %.2f s.", time.perf_counter() - start_time)
