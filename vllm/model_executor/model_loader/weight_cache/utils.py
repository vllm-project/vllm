# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers shared by the weight cache daemon, the IPC loader and the engine.

These live outside ``protocol`` so callers that only need to know whether a
draft is cached, or how a daemon group is named, do not have to import the
wire format.
"""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import SpeculativeConfig, VllmConfig, replace, set_current_vllm_config
from vllm.model_executor.models.interfaces import SupportsEagleBase
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@dataclass
class ExternalWeightCharges:
    """Track a worker's external weight charges and those freed by disk fallback."""

    fallback_bytes: int = field(default=0, init=False)
    _charges: dict[tuple[str, bool], int] = field(default_factory=dict, init=False)

    @property
    def current_bytes(self) -> int:
        """External bytes held right now: charges minus fallback removals."""
        return sum(self._charges.values())

    @contextmanager
    def track(self) -> Generator[None, None, None]:
        token = _active_external_weight_charges.set(self)
        try:
            yield
        finally:
            _active_external_weight_charges.reset(token)


_active_external_weight_charges: ContextVar[ExternalWeightCharges | None] = ContextVar(
    "active_external_weight_charges", default=None
)


def record_external_weight_charge(
    socket_path: str, is_draft: bool, memory_bytes: int
) -> None:
    if accounting := _active_external_weight_charges.get():
        accounting._charges[socket_path, is_draft] = memory_bytes


def record_external_weight_fallback(socket_path: str, is_draft: bool) -> None:
    if accounting := _active_external_weight_charges.get():
        accounting.fallback_bytes += accounting._charges.pop((socket_path, is_draft), 0)


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


def is_draft_model_cacheable(speculative_config: SpeculativeConfig | None) -> bool:
    """Whether the daemon serves the speculative draft as a separate role."""
    return (
        speculative_config is not None
        and speculative_config.method in WEIGHT_CACHE_DRAFT_METHODS
        and speculative_config.draft_model_config is not None
    )


def format_daemon_role(is_draft: bool) -> str:
    """Name of a daemon group: the target model or the draft."""
    return "draft" if is_draft else "target"


def format_socket_role_suffix(is_draft: bool) -> str:
    """Socket-name suffix keeping the draft group distinct from the target."""
    return "_draft" if is_draft else ""


def build_warmup_runner(
    vllm_config: VllmConfig,
    local_rank: int,
    *,
    is_draft: bool,
    model_config: "ModelConfig",
) -> "GPUModelRunner":
    """Create a model runner for the daemon's warmup dummy runs."""
    if is_draft:
        vllm_config = replace(
            vllm_config, model_config=model_config, speculative_config=None
        )
    device = torch.device(current_platform.device_type, local_rank)
    with set_current_vllm_config(vllm_config):
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        return GPUModelRunner(vllm_config, device)
