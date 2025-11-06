# ABOUTME: EPS package initialization utilities.
# ABOUTME: Exposes runtime helpers for EigenPage Summaries.

from .block_filter import apply_union_mask
from .config import EpsRuntimeConfig, to_runtime_config
from .context import (
    EpsForwardContext,
    EpsGroupRuntime,
    EpsLayerInfo,
    EpsRequestRuntime,
    eps_context,
    get_eps_context,
)
from .reporter import EpsReporter, EpsAggregator
from .selector import build_union_visit_set
from .state import EpsJLState
from .summarizer import jl_update_block, jl_update_once
from .telemetry import EpsStepCounters
from .writer import apply_eps_prefill_updates
from .runtime import build_eps_forward_context
from .device_union import apply_device_union, union_select_for_request
from .union_pass import run_union_prepass

__all__ = [
    "apply_union_mask",
    "apply_eps_prefill_updates",
    "eps_context",
    "build_eps_forward_context",
    "EpsForwardContext",
    "EpsGroupRuntime",
    "EpsLayerInfo",
    "EpsRequestRuntime",
    "get_eps_context",
    "EpsReporter",
    "EpsAggregator",
    "build_union_visit_set",
    "EpsRuntimeConfig",
    "EpsJLState",
    "EpsStepCounters",
    "to_runtime_config",
    "jl_update_block",
    "jl_update_once",
    "run_union_prepass",
    "apply_device_union",
    "union_select_for_request",
]
