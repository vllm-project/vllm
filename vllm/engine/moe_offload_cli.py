# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage-1 CLI/config wiring for MoE CPU offload expert paging.

This module intentionally does not change runtime behavior. It only exposes
user-facing flags and attaches a MoEOffloadConfig object to VllmConfig so later
stages can consume it behind the --moe-cpu-offload master flag.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any

from vllm.config import MoEOffloadConfig
from vllm.logger import init_logger

_PATCHED_ATTR = "_moe_offload_cli_patched"
_ACTIVE_EXPERT_FETCH_METHOD = "passive"

logger = init_logger(__name__)


def _parser_has_option(parser: ArgumentParser, option: str) -> bool:
    return any(option in action.option_strings for action in parser._actions)


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
    return parser


def _make_config_from_args(args_obj: Any) -> MoEOffloadConfig:
    return MoEOffloadConfig(
        enabled=bool(getattr(args_obj, "moe_cpu_offload", False)),
    )


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


def patch_engine_args() -> None:
    """Patch EngineArgs with Stage-1 MoE offload CLI/config plumbing."""
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
        engine_args.moe_offload_config = _make_config_from_args(engine_args)
        return engine_args

    def create_engine_config(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        vllm_config = original_create_engine_config(self, *args, **kwargs)
        moe_offload_config = _make_config_from_args(self)
        model_config = getattr(vllm_config, "model_config", None)
        is_moe_model = bool(getattr(model_config, "is_moe", False))
        if moe_offload_config.enabled and not is_moe_model:
            moe_offload_config = MoEOffloadConfig(
                enabled=False,
            )
            logger.info(
                "MoE CPU offload ignored: --moe-cpu-offload was set, "
                "but the model is not a MoE model."
            )
        elif moe_offload_config.enabled:
            self.enforce_eager = True
            if model_config is not None:
                model_config.enforce_eager = True
            num_experts = _get_num_experts(model_config)
            active_experts = _get_active_expert_count(model_config)
            moe_offload_config = MoEOffloadConfig(
                enabled=True,
            )
            logger.info(
                "MoE CPU offload enabled: total experts=%s, active experts=%s, "
                "active expert transfer=%s.",
                num_experts if num_experts is not None else "unknown",
                active_experts if active_experts is not None else "unknown",
                _ACTIVE_EXPERT_FETCH_METHOD,
            )
        vllm_config.moe_offload_config = moe_offload_config
        return vllm_config

    EngineArgs.add_cli_args = staticmethod(add_cli_args)
    EngineArgs.from_cli_args = from_cli_args
    EngineArgs.create_engine_config = create_engine_config
    setattr(EngineArgs, _PATCHED_ATTR, True)


patch_engine_args()
