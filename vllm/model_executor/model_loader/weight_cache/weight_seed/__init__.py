# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Daemon-to-daemon weight shard seeding: movers, and both ends.

- :mod:`.core`: the mover interface and the factory.
- :mod:`.utils`: the storage manifest wire format and fingerprint helpers.
- :mod:`.peer_ipc` / :mod:`.rdma`: the two movers (CUDA IPC / Mooncake).
- :mod:`.seed`: :class:`WeightShardSeed`, the source side served by a daemon.
- :mod:`.loader`: :class:`PeerModelLoader`, the ``peer_cache`` load format
  that fills a mirror daemon's model from a source replica.
"""

from vllm.model_executor.model_loader.weight_cache.weight_seed.core import (
    PEER_IPC_BACKEND,
    RDMA_BACKEND,
    WeightMover,
    get_mover,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.seed import (
    WeightShardSeed,
)
from vllm.model_executor.model_loader.weight_cache.weight_seed.utils import (
    build_manifest,
    manifest_dtype,
    manifest_nbytes,
    rebuild_tensors,
)

__all__ = [
    "PEER_IPC_BACKEND",
    "RDMA_BACKEND",
    "WeightMover",
    "WeightShardSeed",
    "build_manifest",
    "get_mover",
    "manifest_dtype",
    "manifest_nbytes",
    "rebuild_tensors",
]
