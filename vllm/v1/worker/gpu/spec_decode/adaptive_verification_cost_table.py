# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dump and pin the cost curves of adaptive verification.

Adaptive verification prices its draft budget from step costs profiled once at
startup, and the profile of rank 0 is broadcast to every rank. A boot on a busy
node (another process on the GPU, a clock that has not ramped up) therefore skews
the cost tables for the whole life of the server, and two boots of the same
configuration can disagree. These helpers let one boot write the profiled curves
to a file and later boots read them back instead of trusting their own profile.

The file holds the profiled *curves*, not the derived tables, together with a
fingerprint of everything that shapes them. A file that is missing, corrupt or was
profiled for another configuration is never used silently: rank 0 logs a warning
and keeps its own profile.
"""

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_FORMAT = 1

Curve = list[tuple[int, float]]


def check_cost_table_envs() -> None:
    """Fail at startup when both the dump and the load path are set."""
    if envs.VLLM_ADAPTIVE_VERIFICATION_COST_TABLE_DUMP and (
        envs.VLLM_ADAPTIVE_VERIFICATION_COST_TABLE_LOAD
    ):
        raise RuntimeError(
            "VLLM_ADAPTIVE_VERIFICATION_COST_TABLE_DUMP and "
            "VLLM_ADAPTIVE_VERIFICATION_COST_TABLE_LOAD are both set. Dumping the "
            "profiled cost curves and pinning stored ones at the same time is "
            "ambiguous; set only one of them."
        )


def _digest(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _try(fn):
    """A fingerprint field that cannot be read is recorded as None."""
    try:
        return fn()
    except Exception:
        return None


def cost_table_fingerprint(manager: Any) -> dict[str, Any]:
    """Everything that shapes the profiled curves.

    Deliberately over-broad: a false mismatch only costs a fall back to the
    freshly profiled curves, a false match would pin the wrong costs.
    """
    import torch

    from vllm.config import get_current_vllm_config

    fingerprint: dict[str, Any] = {
        "num_speculative_steps": int(manager.num_speculative_steps),
        "max_num_reqs": int(manager.req_states.max_num_reqs),
        "max_num_batched_tokens": int(manager.req_states.max_num_batched_tokens),
        "cudagraph_limit": int(manager._cudagraph_limit),
        "profile_context_len": int(envs.VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN),
        "gpu_name": _try(lambda: str(torch.cuda.get_device_name())),
        "gpu_sm_count": _try(
            lambda: int(torch.cuda.get_device_properties(0).multi_processor_count)
        ),
        "gpu_capability": _try(lambda: list(torch.cuda.get_device_capability())),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
    }
    config = _try(get_current_vllm_config)
    if config is not None:
        fingerprint["vllm_config_hash"] = _try(lambda: str(config.compute_hash()))
        fingerprint["model"] = _try(lambda: str(config.model_config.model))
        fingerprint["tp_size"] = _try(lambda: int(config.parallel_config.world_size))
        fingerprint["cudagraph_capture_sizes"] = _try(
            lambda: sorted(
                int(s) for s in config.compilation_config.cudagraph_capture_sizes
            )
        )
        fingerprint["speculative_method"] = _try(
            lambda: str(config.speculative_config.method)
        )
    # Round-trip through JSON so the fingerprint compares equal to a loaded one.
    return json.loads(json.dumps(fingerprint, default=str))


def dump_cost_curves(
    path: str,
    fingerprint: Mapping[str, Any],
    curves: Mapping[str, Sequence[tuple[int, float]] | None],
) -> None:
    """Write the curves atomically: a reader never sees a half-written file."""
    document = {
        "format": _FORMAT,
        "fingerprint_hash": _digest(fingerprint),
        "fingerprint": fingerprint,
        "curves": {
            name: None if curve is None else [[int(x), float(y)] for x, y in curve]
            for name, curve in curves.items()
        },
    }
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(document, f, indent=1)
    os.replace(tmp_path, path)


def load_cost_curves(
    path: str,
    fingerprint: Mapping[str, Any],
    names: Sequence[str],
) -> dict[str, Curve | None] | None:
    """The curves stored in `path`, or None (with a warning) if they cannot be used.

    `names` are the curves the running code profiles; a file with a different set
    of curves, or with a fingerprint that differs in any field, is rejected.
    """
    try:
        with open(path) as f:
            document = json.load(f)
        if document["format"] != _FORMAT:
            raise ValueError(f"unknown format {document['format']!r}")
        saved = document["fingerprint"]
        raw = document["curves"]
        if set(raw) != set(names):
            raise ValueError(
                f"stored curves {sorted(raw)} differ from the profiled ones "
                f"{sorted(names)}"
            )
        curves: dict[str, Curve | None] = {}
        for name in names:
            stored = raw[name]
            curves[name] = (
                None if stored is None else [(int(x), float(y)) for x, y in stored]
            )
            for _, cost in curves[name] or []:
                if not math.isfinite(cost):
                    raise ValueError(f"non-finite cost in curve {name!r}")
        # The two curves every cost table is built from must exist.
        if not curves.get("draft") or not curves.get("verify"):
            raise ValueError("the draft and verify curves must not be empty")
    except Exception as e:
        logger.warning(
            "VLLM_ADAPTIVE_VERIFICATION_COST_TABLE_LOAD=%s is IGNORED, the file "
            "could not be used (%s: %s); keeping the profiled cost curves.",
            path,
            type(e).__name__,
            e,
        )
        return None
    differing = sorted(
        key
        for key in set(saved) | set(fingerprint)
        if saved.get(key) != fingerprint.get(key)
    )
    if differing:
        logger.warning(
            "VLLM_ADAPTIVE_VERIFICATION_COST_TABLE_LOAD=%s is IGNORED, it was "
            "profiled for another configuration (fingerprint %s, running %s, "
            "differs in %s); keeping the profiled cost curves.",
            path,
            document.get("fingerprint_hash"),
            _digest(fingerprint),
            differing,
        )
        return None
    return curves


def dump_or_load_cost_curves(
    manager: Any, curves: Mapping[str, Curve | None]
) -> dict[str, Curve | None]:
    """Rank 0 only, before the broadcast: dump the profiled curves, or replace
    them by the stored ones. Returns the curves to broadcast."""
    dump_path = envs.VLLM_ADAPTIVE_VERIFICATION_COST_TABLE_DUMP
    load_path = envs.VLLM_ADAPTIVE_VERIFICATION_COST_TABLE_LOAD
    fingerprint = cost_table_fingerprint(manager)
    if dump_path:
        try:
            dump_cost_curves(dump_path, fingerprint, curves)
            logger.info(
                "Adaptive verification cost curves written to %s (fingerprint %s).",
                dump_path,
                _digest(fingerprint),
            )
        except OSError as e:
            logger.warning(
                "VLLM_ADAPTIVE_VERIFICATION_COST_TABLE_DUMP=%s could not be "
                "written (%s: %s).",
                dump_path,
                type(e).__name__,
                e,
            )
    if load_path:
        loaded = load_cost_curves(load_path, fingerprint, list(curves))
        if loaded is not None:
            logger.info(
                "Adaptive verification cost curves pinned from %s (fingerprint "
                "%s); the profiled curves are not used.",
                load_path,
                _digest(fingerprint),
            )
            return loaded
    return dict(curves)
