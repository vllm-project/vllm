# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The target logits gather of the TP specialized decode on a Lamport mailbox.

`LogitsProcessor` gathers the vocab-parallel lm_head's [M, V / TP] shards with
`tensor_model_parallel_all_gather`: NCCL's ring all-gather into [TP, M, V / TP],
then a copy to [M, V]. The target's `compute_logits` runs eagerly once per decode
step with a few rows (1 + draft tokens per request), so that gather is latency-bound.

For the HEAD_GATHER rows it runs as the DSpark drafter's logits gather does
(`drafter.comm.Gather`, the same kernel and mailbox class): one kernel multicasts
the rank's shard into row m of every rank's [cap, V] mailbox at columns
rank * V / TP (the layout of NCCL's all-gather), polls the peers' fragments and
copies the row to the [M, V] output, re-arming the mailbox. The bytes are moved
unchanged, but for the BF16 pair (+0, -0) (the sentinel word), which arrives as
(+0, +0): the same values to the sampler (comparisons, sums, exp).
"""

from functools import partial

import torch

from vllm.logger import init_logger

from ..cutedsl.common.policy import batch_table

logger = init_logger(__name__)

# Rows (tokens of one ``compute_logits`` call) whose logits gather runs on the
# Lamport mailbox, per (TP size, DCP size); other pairs and rows: NCCL's all-gather.
# TP16 x DCP16 serves DSpark verify steps of one request up to 7 draft tokens
# (chosen by an end-to-end A/B there); other pairs are not measured yet.
HEAD_GATHER = {(16, 16): "1-8:1"}


def install_logits_gather(model, group, tp, rank, device, dcp=1):
    """Route `model`'s logits gather (`logits_processor._gather_logits`, an
    instance attribute: other LogitsProcessors keep theirs) through a HeadGather for
    the HEAD_GATHER rows of this (TP, DCP) pair. Collective over `group` (the TP
    device group) when a table entry applies: every TP rank runs it at load, before
    any capture, in the same order. Returns the rows served (empty: the stock
    gather); a processor that already has one keeps it."""
    lp = getattr(model, "logits_processor", None)
    installed = getattr(lp, "head_gather", None)
    if installed is not None:  # installed by an earlier call: keep that one
        return tuple(sorted(installed.rows))
    head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    if (
        lp is None
        or weight is None
        or weight.dim() != 2
        or weight.dtype != torch.bfloat16
        or lp.logits_as_input
        or getattr(lp, "head_dtype", None) not in (None, torch.bfloat16)
        or getattr(head, "tp_size", 1) != tp
    ):
        return ()
    table = batch_table(HEAD_GATHER.get((tp, dcp), ""))
    rows = tuple(sorted(r for r, on in table.items() if on))
    if not rows or tp == 1 or not lp.use_all_gather:
        return ()
    served = _install_gather(lp, group, device, tp, rank, weight.shape[0], rows)
    if served:
        logger.info_once("Specialized decode logits gather: Lamport rows %s", served)
    return served


class HeadGather:
    """The vocab-parallel lm_head's gather [M, V / TP] -> [M, V] (BF16) for `rows`
    on two parities of a Lamport mailbox ([2, max rows, V] BF16, `drafter.comm`):
    rank r's shard lands at columns r * V / TP, the layout of NCCL's all-gather. Bytes
    move unchanged but for the sentinel pair (+0, -0), sent as (+0, +0).

    The parity alternates per call (every rank calls in the same order): two uses of
    one parity are separated by a use of the other, and a rank pushes into that one
    only after its previous gather kernel finished reading and re-arming. This
    assumes every call on a rank runs on one stream, in call order (the target's
    `compute_logits` on the model runner's stream): calls issued from another
    stream could overlap a gather of the other parity. Calls under a graph capture,
    and other shapes, take `stock`.

    Construction allocates the mailbox (a collective over `group`) and arms it;
    the caller orders every rank's construction before the first call
    (`_install_gather`'s all-reduce)."""

    def __init__(self, group, device, tp, rank, shard, rows):
        from ..drafter.comm import PAIR_SENT, LamportRows, compile_gather, gather_ctas

        self.tp, self.rank, self.shard = tp, rank, shard
        self.rows, self.cap = frozenset(rows), max(rows)
        words = shard // 2
        self.box = LamportRows(group, device, (2, self.cap, words * tp), PAIR_SENT)
        self.kernel = compile_gather(tp, words, self.cap, PAIR_SENT, True, True, device)
        self.ctas = {m: gather_ctas(m, words * tp // 4) for m in rows}
        self.parity = 0
        torch.accelerator.synchronize(device)

    def __call__(self, logits, stock):
        rows = logits.shape[0] if logits.dim() == 2 else 0
        if (
            rows not in self.rows
            or logits.dtype != torch.bfloat16
            or logits.shape[1] != self.shard
            or torch.cuda.is_current_stream_capturing()
        ):
            return stock(logits)
        box, mc = self.box.view(self.parity)
        self.parity ^= 1
        out = logits.new_empty((rows, self.shard * self.tp))
        shard = logits.contiguous().view(torch.int32)
        self.kernel(shard, out.view(torch.int32), box, mc, self.rank, self.ctas[rows])
        return out


def _install_gather(lp, group, device, tp, rank, shard, rows):
    """`lp`'s `_gather_logits` on a HeadGather for `rows` (NCCL's otherwise);
    the rows it serves."""
    if shard % 8:
        return ()
    gather: HeadGather | None = None
    error: object = "it failed on another TP rank"
    try:
        gather = HeadGather(group, device, tp, rank, shard, rows)
    except Exception as exc:  # capability fallback: NCCL's all-gather
        error = exc
    # One decision for the TP group, so that every rank takes the same gather: the
    # minimum of the ranks' outcomes. It also orders every rank's armed mailbox
    # before any rank's first push.
    built = torch.tensor([int(gather is not None)], dtype=torch.int32, device=device)
    torch.distributed.all_reduce(built, op=torch.distributed.ReduceOp.MIN, group=group)
    if not built.item():
        logger.warning_once("Specialized decode logits gather stays stock: %s", error)
        return ()
    assert gather is not None
    stock = lp._gather_logits
    lp.head_gather = gather
    lp._gather_logits = partial(gather, stock=stock)
    return tuple(sorted(gather.rows))
