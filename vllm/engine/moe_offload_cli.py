# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CLI/config wiring for MoE CPU offload expert paging.

This module intentionally does not change runtime behavior. It only exposes
user-facing flags and attaches a MoEOffloadConfig object to VllmConfig so the
Case 1 passive path and Case 2 prefetch path can consume it.
"""

from __future__ import annotations

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from math import ceil
from typing import Any

from vllm.config import MoEOffloadConfig
from vllm.logger import init_logger

_PATCHED_ATTR = "_moe_offload_cli_patched"
_CASE_1_TRANSFER_METHOD = "passive"
_CASE_2_MODE = "prefetch"
_LOW_MEMORY_PREFETCH_GPU_UTILIZATION = 0.5
_LOW_MEMORY_PREFETCH_MAX_MODEL_LEN = 196_608

logger = init_logger(__name__)


def _parser_has_option(parser: ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ArgumentTypeError(
            "--moe-gpu-prefetch must be a positive integer"
        ) from exc
    if parsed <= 0:
        raise ArgumentTypeError("--moe-gpu-prefetch must be a positive integer")
    return parsed


def _add_moe_offload_args(parser: ArgumentParser) -> ArgumentParser:
    if _parser_has_option(parser, "--moe-cpu-offload"):
        return parser

    group = parser.add_argument_group(
        title="MoEOffloadConfig",
        description="Sparse-MoE CPU offload expert paging options.",
    )
    group.add_argument(
        "--moe-cpu-offload",
        action="store_true",
        default=MoEOffloadConfig.enabled,
        help=(
            "Enable sparse-MoE CPU offload expert paging mode. "
            "Default: disabled."
        ),
    )
    group.add_argument(
        "--moe-gpu-prefetch",
        type=_positive_int,
        default=MoEOffloadConfig.gpu_prefetch,
        metavar="N",
        help=(
            "Enable sparse-MoE Case 2 GPU active expert prefetch mode and "
            "keep up to N active experts resident on GPU. Case 2 takes "
            "precedence over --moe-cpu-offload. Default: disabled."
        ),
    )
    return parser


def _make_config_from_args(args_obj: Any) -> MoEOffloadConfig:
    gpu_prefetch = getattr(args_obj, "moe_gpu_prefetch", None)
    if gpu_prefetch is not None:
        return MoEOffloadConfig(
            enabled=True,
            mode=_CASE_2_MODE,
            gpu_prefetch=int(gpu_prefetch),
            effective_gpu_prefetch=int(gpu_prefetch),
        )
    return MoEOffloadConfig(
        enabled=bool(getattr(args_obj, "moe_cpu_offload", False)),
        mode="passive" if getattr(args_obj, "moe_cpu_offload", False) else "disabled",
    )


def _effective_gpu_prefetch(
    requested: int,
    active_experts: int | None,
) -> int:
    if active_experts is None or requested >= active_experts:
        return requested
    return ceil(active_experts * 1.5)


def _get_num_experts(model_config: Any) -> int | None:
    if model_config is None:
        return None
    get_num_experts = getattr(model_config, "get_num_experts", None)
    if get_num_experts is None:
        return None
    try:
        return int(get_num_experts())
    except Exception:
        return None


def _get_active_expert_count(model_config: Any) -> int | None:
    if model_config is None:
        return None

    candidates = (
        "top_k_experts",
        "num_experts_per_tok",
        "moe_top_k",
        "num_experts_per_token",
    )
    for config in (
        getattr(model_config, "hf_text_config", None),
        getattr(model_config, "hf_config", None),
    ):
        if config is None:
            continue
        for name in candidates:
            value = getattr(config, name, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
    return None


def _maybe_cap_low_memory_prefetch_max_model_len(
    vllm_config: Any,
    explicit_max_model_len: int | None,
) -> tuple[int, int] | None:
    if explicit_max_model_len is not None:
        return None

    cache_config = getattr(vllm_config, "cache_config", None)
    gpu_memory_utilization = getattr(cache_config, "gpu_memory_utilization", 1.0)
    if float(gpu_memory_utilization) > _LOW_MEMORY_PREFETCH_GPU_UTILIZATION:
        return None

    model_config = getattr(vllm_config, "model_config", None)
    max_model_len = getattr(model_config, "max_model_len", None)
    if max_model_len is None or max_model_len <= _LOW_MEMORY_PREFETCH_MAX_MODEL_LEN:
        return None

    model_config.max_model_len = _LOW_MEMORY_PREFETCH_MAX_MODEL_LEN
    return int(max_model_len), _LOW_MEMORY_PREFETCH_MAX_MODEL_LEN


def patch_engine_args() -> None:
    """Patch EngineArgs with MoE offload CLI/config plumbing."""
    from vllm.engine.arg_utils import EngineArgs

    if getattr(EngineArgs, _PATCHED_ATTR, False):
        return

    original_add_cli_args = EngineArgs.add_cli_args
    original_from_cli_args = EngineArgs.from_cli_args.__func__
    original_create_engine_config = EngineArgs.create_engine_config

    def add_cli_args(parser):  # type: ignore[no-untyped-def]
        parser = original_add_cli_args(parser)
        return _add_moe_offload_args(parser)

    @classmethod
    def from_cli_args(cls, args: Namespace):  # type: ignore[no-untyped-def]
        engine_args = original_from_cli_args(cls, args)
        engine_args.moe_cpu_offload = bool(
            getattr(args, "moe_cpu_offload", MoEOffloadConfig.enabled)
        )
        engine_args.moe_gpu_prefetch = getattr(
            args,
            "moe_gpu_prefetch",
            MoEOffloadConfig.gpu_prefetch,
        )
        engine_args.moe_offload_config = _make_config_from_args(engine_args)
        return engine_args

    def create_engine_config(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        vllm_config = original_create_engine_config(self, *args, **kwargs)
        moe_offload_config = _make_config_from_args(self)
        model_config = getattr(vllm_config, "model_config", None)
        is_moe_model = bool(getattr(model_config, "is_moe", False))
        if moe_offload_config.enabled and not is_moe_model:
            requested_case = moe_offload_config.mode
            moe_offload_config = MoEOffloadConfig(enabled=False)
            logger.info(
                "MoE offload ignored: %s was requested, but the model is not "
                "a MoE model.",
                requested_case,
            )
        elif moe_offload_config.enabled and moe_offload_config.mode == "prefetch":
            self.enforce_eager = True
            if model_config is not None:
                model_config.enforce_eager = True
            num_experts = _get_num_experts(model_config)
            active_experts = _get_active_expert_count(model_config)
            requested_prefetch = int(moe_offload_config.gpu_prefetch or 1)
            effective_prefetch = _effective_gpu_prefetch(
                requested_prefetch,
                active_experts,
            )
            moe_offload_config = MoEOffloadConfig(
                enabled=True,
                mode="prefetch",
                gpu_prefetch=requested_prefetch,
                effective_gpu_prefetch=effective_prefetch,
            )
            max_model_len_cap = _maybe_cap_low_memory_prefetch_max_model_len(
                vllm_config,
                getattr(self, "max_model_len", None),
            )
            logger.info(
                "MoE GPU prefetch enabled: total experts=%s, active experts=%s, "
                "requested prefetch=%s, effective prefetch=%s.",
                num_experts if num_experts is not None else "unknown",
                active_experts if active_experts is not None else "unknown",
                requested_prefetch,
                effective_prefetch,
            )
            if max_model_len_cap is not None:
                original_max_model_len, capped_max_model_len = max_model_len_cap
                logger.info(
                    "MoE GPU prefetch capped max model length from %s to %s "
                    "because gpu_memory_utilization is <= %.2f.",
                    original_max_model_len,
                    capped_max_model_len,
                    _LOW_MEMORY_PREFETCH_GPU_UTILIZATION,
                )
            if getattr(self, "moe_cpu_offload", False):
                logger.info(
                    "MoE CPU offload ignored: --moe-gpu-prefetch takes "
                    "precedence over --moe-cpu-offload."
                )
        elif moe_offload_config.enabled:
            self.enforce_eager = True
            if model_config is not None:
                model_config.enforce_eager = True
            num_experts = _get_num_experts(model_config)
            active_experts = _get_active_expert_count(model_config)
            moe_offload_config = MoEOffloadConfig(
                enabled=True,
                mode="passive",
            )
            logger.info(
                "MoE CPU offload enabled: total experts=%s, active experts=%s, "
                "active expert transfer=%s.",
                num_experts if num_experts is not None else "unknown",
                active_experts if active_experts is not None else "unknown",
                _CASE_1_TRANSFER_METHOD,
            )
        vllm_config.moe_offload_config = moe_offload_config
        return vllm_config

    EngineArgs.add_cli_args = staticmethod(add_cli_args)
    EngineArgs.from_cli_args = from_cli_args
    EngineArgs.create_engine_config = create_engine_config
    setattr(EngineArgs, _PATCHED_ATTR, True)


patch_engine_args()
