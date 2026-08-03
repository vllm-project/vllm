# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Commit-time validation for graph-visible storage across weight reload."""

import contextlib
import os
from collections.abc import Iterator

import torch

from vllm.logger import init_logger
from vllm.model_executor.reload_arena import (
    ArenaIdentity,
    snapshot_model_arenas,
    verify_model_arenas,
)
from vllm.model_executor.reload_manifest import check_global_storage

from .layerwise import get_layer_arena_findings

logger = init_logger(__name__)


def _validate_reload_storage(
    model: torch.nn.Module,
    arena_snapshots: dict[str, dict[str, ArenaIdentity]],
    *,
    enforce_gate: bool,
) -> None:
    arena_problems = verify_model_arenas(model, arena_snapshots)
    per_layer = get_layer_arena_findings()
    model_finding_set = frozenset(arena_problems)
    layer_finding_set = frozenset(per_layer)
    if model_finding_set != layer_finding_set:
        missing_per_layer = sorted(model_finding_set - layer_finding_set)
        extra_per_layer = sorted(layer_finding_set - model_finding_set)
        logger.warning(
            "Reload arena verification mismatch: per-layer missed %s "
            "and additionally found %s. Model-level: %s | Per-layer: %s",
            missing_per_layer[:10],
            extra_per_layer[:10],
            arena_problems[:10],
            per_layer[:10],
        )

    if arena_problems:
        msg = (
            "Reload violated graph-visible storage identity on "
            f"{len(arena_problems)} slot(s):\n  "
            + "\n  ".join(arena_problems[:20])
        )
        gate = os.environ.get("VLLM_RELOAD_GATE", "strict")
        if not enforce_gate:
            logger.error(msg)
        elif gate == "warn":
            logger.warning(msg)
        elif gate != "off":
            raise RuntimeError(
                msg
                + "\nCaptured CUDA graphs may reference freed or stale "
                "storage; serving would risk corruption or an illegal memory "
                "access. Set VLLM_RELOAD_GATE=warn to downgrade (unsafe)."
            )

    manifest_gate = os.environ.get("VLLM_RELOAD_GLOBAL_MANIFEST", "warn")
    if enforce_gate and manifest_gate == "off":
        return

    report = check_global_storage()
    if report is not None and not report.is_clean:
        msg = (
            "Reload rebound module-level storage that was live when graphs "
            "were captured:\n" + report.format()
        )
        if not enforce_gate:
            logger.error(msg)
        elif manifest_gate == "strict":
            raise RuntimeError(
                msg
                + "\nSet VLLM_RELOAD_GLOBAL_MANIFEST=warn to downgrade."
            )
        logger.warning(msg)
    elif report is not None:
        logger.debug("Global storage manifest clean (%d checked)", report.checked)


@contextlib.contextmanager
def reload_storage_guard(model: torch.nn.Module) -> Iterator[None]:
    """Validate graph-visible storage after an in-place weight reload."""
    arena_snapshots = snapshot_model_arenas(model)
    try:
        yield
    except BaseException as reload_error:
        add_note = getattr(reload_error, "add_note", None)
        try:
            _validate_reload_storage(
                model,
                arena_snapshots,
                enforce_gate=False,
            )
        except Exception as validation_error:
            logger.exception(
                "Storage validation also failed after weight reload failed"
            )
            if add_note is not None:
                add_note(
                    "Post-failure storage validation raised "
                    f"{type(validation_error).__name__}: {validation_error}"
                )

        if add_note is not None:
            add_note(
                "The in-place reload may have partially mutated the model. "
                "This worker must not continue serving and should be restarted."
            )
        logger.error(
            "In-place weight reload failed after mutation may have started; "
            "the worker must be restarted before serving more requests."
        )
        raise
    else:
        _validate_reload_storage(
            model,
            arena_snapshots,
            enforce_gate=True,
        )
