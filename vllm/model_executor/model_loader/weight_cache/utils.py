# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers shared by the weight cache daemon, the IPC loader and the engine.

These live outside ``protocol`` so callers that only need to know whether a
draft is cached, or how a daemon group is named, do not have to import the
wire format.
"""

from typing import Any, TypeGuard

from vllm.config import SpeculativeConfig
from vllm.model_executor.models.interfaces import SupportsEagleBase

# Speculative methods whose draft model the daemon caches in its own group.
# Other drafts keep loading from disk in the engine.
WEIGHT_CACHE_DRAFT_METHODS = frozenset({"mtp", "eagle", "eagle3"})

# Python-side flags that weight loading sets on EAGLE-style drafts; the engine
# never runs load_weights for cached models, so the daemon ships them.
EXPORTED_MODEL_ATTRS = tuple(SupportsEagleBase.__annotations__)


def export_model_attrs(model: Any) -> dict[str, bool]:
    return {
        name: bool(getattr(model, name))
        for name in EXPORTED_MODEL_ATTRS
        if hasattr(model, name)
    }


def caches_draft_model(
    speculative_config: SpeculativeConfig | None,
) -> TypeGuard[SpeculativeConfig]:
    """Whether the daemon serves the speculative draft as a separate role."""
    return (
        speculative_config is not None
        and speculative_config.method in WEIGHT_CACHE_DRAFT_METHODS
        and speculative_config.draft_model_config is not None
    )


def format_daemon_role(draft_model_idx: int | None) -> str:
    """Name of a daemon group: the target model, or a draft by index."""
    return "target" if draft_model_idx is None else f"draft{draft_model_idx}"


def format_socket_role_suffix(draft_model_idx: int | None) -> str:
    """Socket-name suffix keeping each draft group distinct from the target."""
    if draft_model_idx is None:
        return ""
    return f"_{format_daemon_role(draft_model_idx)}"
