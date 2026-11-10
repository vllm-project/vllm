# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rust (`vllm._rust_core`) implementations of the store's per-block hot paths.

Used when `VLLM_USE_RUST_CORE` is set. Each returns `None` when the
extension is missing or does not support the layout, so callers keep the
bitwise-identical Python path.
"""

from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def _construct(name: str, source: Any) -> Any | None:
    if not envs.VLLM_USE_RUST_CORE:
        return None
    try:
        import vllm._rust_core as rust_core
    except ImportError:
        logger.warning_once(
            "VLLM_USE_RUST_CORE is set but vllm._rust_core is not built; "
            "using the Python Mooncake store paths."
        )
        return None
    try:
        return getattr(rust_core, name)(source)
    except Exception as e:
        logger.info_once("Using the Python %s: %s", name, e)
        return None


def make_rust_coordinator(coord: Any) -> Any | None:
    """Rust mirror of a `MooncakeStoreCoordinator`."""
    return _construct("MooncakeStoreCoordinator", coord)


def make_rust_lookup(worker: Any) -> Any | None:
    """Rust `MooncakeStoreWorker.lookup`, built from the worker's lookup
    state (`coord`, `token_dbs`, `_lookup_key_prefixes`)."""
    return _construct("MooncakeStoreLookup", worker)
