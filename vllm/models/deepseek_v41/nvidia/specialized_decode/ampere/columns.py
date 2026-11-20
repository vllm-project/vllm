# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM80 and SM90 TP all-gathers along the last dimension: logits and
Engram.

Some of a decode step's tensors are split by columns across the TP ranks and
all-gathered side by side, every rank's columns of a row next to each other:

* the vocabulary-parallel logits of the target and the DSpark draft model:
  each rank's LM head computes every row over its slice of the vocabulary
  (16,160 BF16 values at TP8) and sampling needs all of it;
* the two Engram layers' embedded hash heads (split by heads) and their
  `wkv` output (split by output columns).

vLLM gathers them with NCCL's ring into rank-major blocks and then copies the
blocks side by side: at 96 rows on 8x A100 about 170 + 60 us for a step's
logits and 35 + 70 us (with their copies) for an Engram layer. Here they go
through the Lamport all-gather of `all_gather.py` laid out side by side,
which writes the gathered rows in place. The kernel uses SM80 instructions
only, so H100 (SM90) takes it too.

`init_column_gathers` builds one `AmpereAllGather` per row size in use
(a mailbox each), collectively, where `supports_column_gathers` holds.
`all_gather_columns` picks the one for a tensor's row size and falls back to
the TP group's all-gather for shapes it does not take (more than
`MAX_ROWS` rows, another row size, or before `init_column_gathers`).
Every rank makes the same calls, so all take the same path.
"""

from typing import TYPE_CHECKING, cast

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.platforms import current_platform

from .all_gather import AmpereAllGather
from .all_reduce import FRAGMENT_BYTES, MAX_ROWS

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )

# The side-by-side all-gathers by the bytes of a rank's row.
_gathers: dict[int, AmpereAllGather] = {}


def supports_column_gathers(vllm_config: VllmConfig) -> bool:
    """Whether the model gathers its logits and Engram columns with the
    Lamport kernel: on SM80 and SM90 at TP 2, 4 or 8 where vLLM's custom
    all-reduce runs, so the GPUs are fully connected with peer access.
    Collective decisions follow from it, so it holds on every rank or none."""
    if vllm_config.parallel_config.tensor_parallel_size not in (2, 4, 8) or not (
        current_platform.is_device_capability(80)
        or current_platform.is_device_capability(90)
    ):
        return False
    comm = cast("CudaCommunicator", get_tp_group().device_communicator).ca_comm
    return comm is not None and not comm.disabled and comm.fully_connected


def column_gathers_ready() -> bool:
    """Whether `init_column_gathers` has built any gather: the target model
    builds them before its DSpark draft model, whose head then shares the
    logits gather (a row size without a gather keeps the TP group's)."""
    return bool(_gathers)


def init_column_gathers(row_bytes: list[int]) -> None:
    """Build the side-by-side all-gathers for rows of these sizes, up to
    `MAX_ROWS` rows a call; sizes that are not whole 16-byte fragments
    keep the TP group's all-gather. Collective over the TP group: every rank
    calls it with the same sizes."""
    group = get_tp_group()
    for size in row_bytes:
        if size % FRAGMENT_BYTES == 0 and size not in _gathers:
            _gathers[size] = AmpereAllGather(
                group.device_group, size, MAX_ROWS, group.device, side_by_side=True
            )


def all_gather_columns(x: torch.Tensor) -> torch.Tensor:
    """The TP all-gather of a 2-D `x` along its last dimension, `[rows,
    world * columns]`.

    Args:
        x: This rank's columns of every row.

    """
    gather = _gathers.get(x.shape[-1] * x.element_size()) if x.dim() == 2 else None
    if gather is not None:
        out = gather.gather_columns(x)
        if out is not None:
            return out
    return tensor_model_parallel_all_gather(x)


class AmpereLogitsProcessor(LogitsProcessor):
    """`LogitsProcessor` gathering the vocabulary shards with
    `all_gather_columns`."""

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """The logits over the whole vocabulary, on every rank."""
        if self.use_all_gather and logits.dim() == 2:
            return all_gather_columns(logits)
        return super()._gather_logits(logits)
