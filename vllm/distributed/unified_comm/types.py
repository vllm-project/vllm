# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common type definitions shared by the unified_comm subpackage.

This module deliberately avoids importing any other ``unified_comm``
submodule so that it can be referenced from anywhere without creating
circular imports.
"""

from __future__ import annotations

from enum import Enum, auto


class TransferProtocol(Enum):
    """Underlying transport protocol."""

    COLLECTIVE = auto()  # backed by collectives (AllGather / AllReduce)
    P2P = auto()  # point-to-point (Send / Recv)
    RDMA = auto()  # raw RDMA
    SHM = auto()  # shared memory
    STORE = auto()  # via a Store (e.g. torch.distributed.Store)


class TransferType(Enum):
    """Logical transfer plane type."""

    KV_CACHE = auto()  # KV cache transfer (prefill -> decode disagg)
    EC = auto()  # Expert-Centric transfer (MoE multi-plane)
    WEIGHT = auto()  # weight transfer (load / sync / hot-update)
    ACTIVATION = auto()  # intermediate activation (pipeline parallel)
    CUSTOM = auto()  # user-defined transfer
