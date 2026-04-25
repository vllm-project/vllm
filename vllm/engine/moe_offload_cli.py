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

_PATCHED_ATTR = "_moe_offload_cli_patched"


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
    group.add_argument(
        "--moe-gpu-limit",
        type=float,
        default=MoEOffloadConfig.gpu_limit,
        help=(
            "Maximum fraction of GPU memory for the offload-managed "
            "working set. Default: 0.4."
        ),
    )
    group.add_argument(
        "--moe-active-expert-budget",
        type=int,
        default=MoEOffloadConfig.active_expert_budget,
        help=(
            "Maximum number of active expert models resident on GPU. "
            "Default: 2."
        ),
    )
    group.add_argument(
        "--moe-max-pipeline-depth",
        type=int,
        default=MoEOffloadConfig.max_pipeline_depth,
        help=(
            "Maximum routed expert bucket pipeline depth for expert reuse. "
            "Default: 4."
        ),
    )
    return parser


def _make_config_from_args(args_obj: Any) -> MoEOffloadConfig:
    return MoEOffloadConfig(
        enabled=bool(getattr(args_obj, "moe_cpu_offload", False)),
        gpu_limit=float(getattr(args_obj, "moe_gpu_limit", MoEOffloadConfig.gpu_limit)),
        active_expert_budget=int(
            getattr(
                args_obj,
                "moe_active_expert_budget",
                MoEOffloadConfig.active_expert_budget,
            )
        ),
        max_pipeline_depth=int(
            getattr(
                args_obj,
                "moe_max_pipeline_depth",
                MoEOffloadConfig.max_pipeline_depth,
            )
        ),
    )


def patch_engine_args() -> None:
    """Patch EngineArgs with Stage-1 MoE offload CLI/config plumbing."""
    from vllm.engine.arg_utils import EngineArgs

    if getattr(EngineArgs, _PATCHED_ATTR, False):
        return

    original_add_cli_args = EngineArgs.add_cli_args
    original_from_cli_args = EngineArgs.from_cli_args
    original_create_engine_config = EngineArgs.create_engine_config

    def add_cli_args(parser):  # type: ignore[no-untyped-def]
        parser = original_add_cli_args(parser)
        return _add_moe_offload_args(parser)

    @classmethod
    def from_cli_args(cls, args: Namespace):  # type: ignore[no-untyped-def]
        engine_args = original_from_cli_args(args)
        engine_args.moe_cpu_offload = bool(args.moe_cpu_offload)
        engine_args.moe_gpu_limit = args.moe_gpu_limit
        engine_args.moe_active_expert_budget = args.moe_active_expert_budget
        engine_args.moe_max_pipeline_depth = args.moe_max_pipeline_depth
        engine_args.moe_offload_config = _make_config_from_args(engine_args)
        return engine_args

    def create_engine_config(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        vllm_config = original_create_engine_config(self, *args, **kwargs)
        vllm_config.moe_offload_config = _make_config_from_args(self)
        return vllm_config

    EngineArgs.add_cli_args = staticmethod(add_cli_args)
    EngineArgs.from_cli_args = from_cli_args
    EngineArgs.create_engine_config = create_engine_config
    setattr(EngineArgs, _PATCHED_ATTR, True)


patch_engine_args()
