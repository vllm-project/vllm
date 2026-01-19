# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Loom integration for vLLM.

Keep this package import lightweight.

Some environments import parts of vLLM without having the compiled CUDA
extension available/compatible (e.g. when only using CPU utilities). Eagerly
importing connector code can trigger platform resolution and attempt to load
`vllm._C`, which can fail even when users only need pure-Python pieces.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "LoomConnector",
    "LoomConnectorMetadata",
    "LoomConnectorScheduler",
    "LoomConnectorWorker",
    "LoomOffloadingSpec",
]


def __getattr__(name: str):
    if name == "LoomConnector":
        return import_module(".connector.connector", __name__).LoomConnector
    if name == "LoomConnectorMetadata":
        return import_module(".connector.metadata", __name__).LoomConnectorMetadata
    if name == "LoomConnectorScheduler":
        return import_module(".connector.scheduler", __name__).LoomConnectorScheduler
    if name == "LoomConnectorWorker":
        return import_module(".connector.worker", __name__).LoomConnectorWorker
    if name == "LoomOffloadingSpec":
        return import_module(".kv_offload.spec", __name__).LoomOffloadingSpec
    raise AttributeError(name)


if TYPE_CHECKING:
    from .connector.connector import LoomConnector as LoomConnector
    from .connector.metadata import LoomConnectorMetadata as LoomConnectorMetadata
    from .connector.scheduler import LoomConnectorScheduler as LoomConnectorScheduler
    from .connector.worker import LoomConnectorWorker as LoomConnectorWorker
    from .kv_offload.spec import LoomOffloadingSpec as LoomOffloadingSpec
