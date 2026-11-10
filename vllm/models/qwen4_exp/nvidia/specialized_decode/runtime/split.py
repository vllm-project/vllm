# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The row-split hyper-connection reads of the executor's large TP4 steps.

Problem. Every decoder block of a tensor-parallel step ends with an all-reduce
of the ranks' partial outputs, which the boundary kernel (`boundary.py`)
runs together with the next hyper-connection read's combine-norm. The read's
projections then follow on every rank: the down projection (`hc_down`) and
the up/mix (`hc_up_mix`) of all the step's rows, the same work on all four
GPUs of TP4. At 32 rows and more that replicated work costs more than
exchanging its result.

Split. From the MLP read of the last layer with per-layer embeddings (PLE; of
layer 0 without one) through the final mixer, rank `q` of the TP group owns
the step's rows `[q R, q R + R)`, `R = rows / 4`, and keeps only those rows
of the residual streams and of the injection logits. Each read then runs:

1. the boundary kernel's split call (`Boundaries.read` with `split`): each
   rank sends every row of its partial block output to the rank that owns the
   row, which sums the four contributions (a reduce-scatter), combines its own
   rows of the streams and normalizes them;
2. the down projection of the owned rows;
3. the up/mix of the owned rows, all-gathered over the TP group in the same
   kernel (`HcUpMixGather`, `RowSplit.project`): every rank gets the block
   input of all rows and runs the block (attention or MoE) on them as before.

The step's last read, the final mixer's, all-gathers the sampled hidden states
the same way; `RowsAllGather` then gathers its owned streams into the MTP
drafter's input buffer (`RowSplit.streams`). The drafter's passes do not
split.

Equality. Each read's outputs are those of the replicated reads bit for bit:
the owner of a row sums the ranks' contributions in rank order in FP32 from
zero and rounds once, as the one-shot all-reduce does, and the down and up/mix
projections compute each row by itself, in a fixed order. The gathers carry
negative zeros as positive zeros (the empty-slot sentinel of their mailboxes),
as the logits gather does; the two are equal in every later product and sum.

Cost. One site (a boundary and its read) of a CUPTI microbenchmark of 16
chained sites at TP4 on GB300, attention and folded MoE boundaries
alternating: 19.8 to 14.3 us at 64 rows, 16.8 to 13.9 at 48, 13.9 to 13.2 at
32 (`SPLIT_MIN_ROWS`). Above 64 rows the replicated reads are the stock
reads (the read kernels take at most 64 rows), while a split rank still reads
at most 64: steps of up to `SPLIT_TP * OWNED_MAX_ROWS` (256) rows split, 64
requests with three draft tokens.

Which steps. `RowSplit.build` returns None, and every step reads replicated,
unless the TP group has 4 ranks (the measured shape), the largest step that
can split (`split_max_rows`) has `SPLIT_MIN_ROWS` rows or more, and the
boundary kernel serves the
attention blocks and every MoE block from the split's first read on (a stock
all-reduce would hand the next read all rows). A step takes the split
(`RowSplit.takes`) when its rows are a multiple of 4 between
`SPLIT_MIN_ROWS` and `max_rows` and it has no deepstack embeddings (which
end the delayed combine). These decisions read the layers, the TP size, the
MoE sites the ranks compared when the boundaries were built, and host state of
the step that every rank shares, so every rank makes the same calls in the
same order, as the exchanges require.

Liveness and generations. The three exchanges (the boundary kernel's split
calls, the up/mix gather, the rows gather) each keep their own mailbox of two
generations in symmetric memory. Every rank owns `R >= 8` rows of every
call and polls every other rank's data for them, so no rank completes call
`n + 1` before every rank has started it, which a rank does after its call
`n` completed; a rank's call `n + 2` therefore writes a peer's slots of
call `n` only after that peer has read them. Each kernel's grid fits on the
GPU at once and launches its dependents only once every CTA has published
(module docstrings of `..boundary.all_reduce_hc`, `..hc.up_mix` and
`..logits.gather`).
"""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.distributed import get_tp_group
from vllm.logger import init_logger

from .. import HC_HIDDEN, HC_STREAMS

if TYPE_CHECKING:
    from ...hyperconnection import GatedResidual
    from ...model import Qwen4ExpDecoderLayer
    from .boundary import Boundaries

logger = init_logger(__name__)

# The tensor-parallel size the split serves.
SPLIT_TP = 4
# The smallest step (token rows) that takes the split (module docstring, Cost).
SPLIT_MIN_ROWS = 32
# The most rows a rank owns in a split step: the most the down projection and
# the up/mix gather read in one call (``..hc.down.MAX_ROWS``,
# ``..hc.up_mix.GATHER_MAX_ROWS``; ``RowSplit`` checks both).
OWNED_MAX_ROWS = 64


def split_max_rows(max_rows: int) -> int:
    """The rows of the largest step that can split when the executor's
    largest step has `max_rows` rows: at most `OWNED_MAX_ROWS` rows a rank
    of `SPLIT_TP`."""
    return min(max_rows, SPLIT_TP * OWNED_MAX_ROWS)


def wants_split(max_rows: int) -> bool:
    """Whether the executor's boundaries should have the split calls: TP4 and
    a largest step that can split (`max_rows`, `split_max_rows`) of
    `SPLIT_MIN_ROWS` rows or more."""
    return get_tp_group().world_size == SPLIT_TP and max_rows >= SPLIT_MIN_ROWS


def split_start(layers: "list[Qwen4ExpDecoderLayer]") -> int:
    """The position of the layer whose MLP read is the first split read: the
    last PLE layer's, 0 without one."""
    start = 0
    for position, layer in enumerate(layers):
        if layer.ple is not None:
            start = position
    return start


def split_unsupported(boundaries: "Boundaries", start: int) -> str:
    """What the boundaries lack for a split from the MLP read of the layer at
    `start` on, or an empty string."""
    if not boundaries.kernel.split:
        return "the boundary kernel's split calls"
    if not boundaries.attention:
        return "the boundary kernel to serve the attention blocks"
    unserved = [
        position
        for position in range(start, len(boundaries.moe_layers))
        if boundaries.moe_layers[position] is None
    ]
    if unserved:
        return (
            "the boundary kernel to serve every MoE block from the split's first "
            f"read on (positions {unserved} keep the stock all-reduce)"
        )
    return ""


class RowSplit:
    """The row split of the executor's steps (module docstring).

    Construction is collective over `group`: the up/mix and rows gathers map
    every rank's mailbox.

    Args:
        group: The TP group's device group (`SPLIT_TP` ranks).
        max_rows: Rows of the largest step that splits (a multiple of the
            group's size, at most `OWNED_MAX_ROWS` a rank).
        start: The position of the layer whose MLP read is the first split
            read (`split_start`).
        device: This rank's GPU.

    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        max_rows: int,
        start: int,
        device: torch.device,
    ) -> None:
        # Imported here: CuTe DSL kernels (CUDA builds only).
        from ..hc.read import MAX_ROWS as DOWN_MAX_ROWS
        from ..hc.read import hc_project
        from ..hc.up_mix import GATHER_MAX_ROWS, HcUpMixGather
        from ..logits.gather import RowsAllGather

        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        assert self.world >= 2 and max_rows % self.world == 0
        self.max_rows = max_rows
        self.start = start
        owned = max_rows // self.world
        assert owned <= OWNED_MAX_ROWS <= min(DOWN_MAX_ROWS, GATHER_MAX_ROWS)
        self._project = hc_project
        self.gather = HcUpMixGather(group, owned, device)
        self.streams = RowsAllGather(group, HC_STREAMS * HC_HIDDEN, owned, device)

    @classmethod
    def build(
        cls,
        layers: "list[Qwen4ExpDecoderLayer]",
        boundaries: "Boundaries",
        max_rows: int,
    ) -> "RowSplit | None":
        """The split of the steps of `layers` up to `max_rows` rows
        (`split_max_rows`), or None when the module docstring's conditions
        (Which steps) do not hold. Collective over the TP group.

        Args:
            layers: The executor's decoder layers, in the order of their calls.
            boundaries: Their boundaries (`Boundaries.build` with `split`
                when `wants_split`).
            max_rows: Rows of the largest step that can split
                (`split_max_rows` of the executor's largest step).

        """
        if not wants_split(max_rows):
            return None
        start = split_start(layers)
        reason = split_unsupported(boundaries, start)
        if reason:
            logger.info_once(
                "Qwen4Exp specialized decode: the hyper-connection reads do not "
                "split their rows among the TP ranks (the split needs %s).",
                reason,
            )
            return None
        tp_group = get_tp_group()
        rows = max_rows - max_rows % tp_group.world_size
        device = layers[0].mlp_hyper_connection.hc_norm.weight.device
        split = cls(tp_group.device_group, rows, start, device)
        logger.info_once(
            "Qwen4Exp specialized decode: steps of %d-%d rows (multiples of %d) "
            "split the hyper-connection reads' rows among the TP ranks from "
            "layer %d's MLP read on.",
            SPLIT_MIN_ROWS,
            rows,
            tp_group.world_size,
            layers[start].layer_idx,
        )
        return split

    def takes(self, rows: int, deepstack: bool) -> bool:
        """Whether a step of `rows` rows (`deepstack`: with deepstack
        embeddings) splits; the same on every rank."""
        return (
            not deepstack
            and SPLIT_MIN_ROWS <= rows <= self.max_rows
            and rows % self.world == 0
        )

    def own(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's rows of a step's `x` (`[rows, ...]`), a view."""
        owned = x.shape[0] // self.world
        return x[self.rank * owned : (self.rank + 1) * owned]

    def project(
        self, hc: "GatedResidual", xn: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """A split read's projections (`Boundaries.read`'s `project`):
        `hc_project` of this rank's normalized streams `xn`, with the
        block input of every rank's rows."""
        return self._project(hc, xn, self.gather)

    def warm_up(self) -> None:
        """Compile and run the up/mix and rows gathers once (collective)."""
        self.gather.warm_up()
        self.streams.warm_up()
