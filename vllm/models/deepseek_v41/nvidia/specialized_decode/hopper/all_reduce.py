# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Hopper executor's fused all-reduce + mHC: ``BoundaryAllReduce`` with
BF16 layer input, built after a TP barrier."""

from transformers import PreTrainedConfig

from vllm.distributed import get_tp_group

from ..boundary.all_reduce import BoundaryAllReduce


class HopperBoundaryAllReduce(BoundaryAllReduce):
    """``BoundaryAllReduce`` with BF16 layer input."""

    def __init__(self, config: PreTrainedConfig, max_num_tokens: int) -> None:
        # Align the ranks first: the mailbox's symmetric-memory rendezvous
        # waits at most 300 s for its peers, and weight loading (the Engram
        # tables' pinning) can skew the ranks by about that much.
        get_tp_group().barrier()
        super().__init__(
            config, quantize_layer_input=False, max_num_tokens=max_num_tokens
        )
