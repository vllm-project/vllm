# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LM head outputs in peer-mapped memory: the vocabulary-parallel logits
without an NCCL all-gather.

The stock LM heads (target and draft) gather their `[rows, V / TP]` shards
with an NCCL ring all-gather and a copy that moves the rank dimension into
the vocabulary. Here each rank writes its shard with the head's own GEMM into
a peer-mapped buffer (the TP group's peer transport), pushes it into every rank's
`[rows, V]` logits buffer over NVLink, and one flag barrier makes every
shard visible (`PeerPushedLogits`): for the target sampler
(`runtime/model_state.py`) and the DSpark Markov sampling (`drafter.py`).

Reuse is safe without a second barrier: between two uses of a buffer every
rank runs collectives that the other ranks can only join after finishing
their reads (the draft step's, the target forward's).

The buffers, their peer handshake and the kernels' compiles happen in
`warm_up`, which the model state runs on every rank at start-up, before
memory profiling, so serving allocates and compiles nothing here.
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit(do_not_specialize=["rank"])
def _peer_barrier_kernel(epoch_ptr, flag_table_ptr, rank, WORLD: tl.constexpr):
    """Every rank's previous kernels are done: bump the local epoch, store it
    into this rank's slot of every rank's flags, wait for every slot of ours.

    No rank can be a barrier ahead (see the module docstring), so a slot holds
    the previous epoch or the current one. Epochs are int64: they never wrap.
    """
    epoch = tl.load(epoch_ptr) + 1
    tl.store(epoch_ptr, epoch)
    for peer in tl.static_range(WORLD):
        flags = tl.load(flag_table_ptr + peer).to(tl.pointer_type(tl.int64))
        tl.atomic_xchg(flags + rank, epoch, sem="release", scope="sys")
    mine = tl.load(flag_table_ptr + rank).to(tl.pointer_type(tl.int64))
    for peer in tl.static_range(WORLD):
        seen = tl.atomic_add(mine + peer, 0, sem="acquire", scope="sys")
        while seen < epoch:
            seen = tl.atomic_add(mine + peer, 0, sem="acquire", scope="sys")


@triton.jit(do_not_specialize=["rank", "shard_size", "vocab_size"])
def _push_shard_kernel(
    src_ptr,  # this rank's [rows, shard] GEMM output
    table_ptr,  # int64 [WORLD]: every rank's [max_rows, vocab] full buffer
    fence_ptr,  # int32 scratch for the release that ends each program
    rank,
    shard_size,
    vocab_size,
    DTYPE: tl.constexpr,
    WORLD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy one block of this rank's shard row into every rank's full logits
    (remote writes), then release them at system scope."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < shard_size
    v = tl.load(src_ptr + row * shard_size + col, mask=mask)
    dst = row * vocab_size + rank * shard_size + col
    for p in tl.static_range(WORLD):
        base = tl.multiple_of(tl.load(table_ptr + p).to(tl.pointer_type(DTYPE)), 16)
        tl.store(base + dst, v, mask=mask)
    tl.debug_barrier()
    tl.atomic_add(fence_ptr, 0, sem="release", scope="sys")


class PeerShards:
    """`[max_rows, shard]` of this rank's LM head output in a peer-mapped
    buffer, every rank's address (`table`), and the flag barrier."""

    def __init__(
        self, weight: torch.Tensor, max_rows: int, name: str, extra_bytes: int = 0
    ) -> None:
        from vllm.distributed import get_tp_group

        from .kernels._comm import make_transport

        group = get_tp_group()
        self.world, self.rank = group.world_size, group.rank_in_group
        self.shard = weight.shape[0]
        self.dtype = weight.dtype
        self.max_rows = max_rows
        nbytes = max_rows * self.shard * weight.element_size()
        self._transport = make_transport(
            "auto",
            group=group.device_group,
            cpu_group=group.cpu_group,
            rank=self.rank,
            world=self.world,
            capacity=nbytes + extra_bytes + 8192,
        )
        logits = self._transport.alloc(name, nbytes)
        flags = self._transport.alloc(name + "_flags", 8 * self.world)
        self.local = logits.local.view(self.dtype).view(max_rows, self.shard)
        device = weight.device
        self.table = torch.tensor(logits.peer_ptrs, dtype=torch.int64, device=device)
        self.flag_table = torch.tensor(
            flags.peer_ptrs, dtype=torch.int64, device=device
        )
        self.epoch = torch.zeros(1, dtype=torch.int64, device=device)

    def barrier(self) -> None:
        _peer_barrier_kernel[(1,)](
            self.epoch, self.flag_table, self.rank, WORLD=self.world, num_warps=1
        )


_HEAD_DTYPES = (torch.bfloat16, torch.float16)


def _tl_dtype(dtype: torch.dtype):
    return {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[dtype]


def peer_logits_unsupported(lm_head: nn.Module, logits_processor: nn.Module) -> str:
    """Why an LM head's logits cannot be read in place ("" when they can)."""
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.dim() != 2 or weight.dtype not in _HEAD_DTYPES:
        return "a quantized LM head"
    if getattr(lm_head, "bias", None) is not None:
        return "an LM head bias"
    if logits_processor.head_dtype not in (None, weight.dtype):
        return "a separate head dtype"
    if logits_processor.logits_as_input:
        return "logits as input"
    if logits_processor.scale != 1.0 or logits_processor.soft_cap is not None:
        return "scaled or soft-capped logits"
    if getattr(lm_head, "tp_size", 1) == 1:
        return "a replicated LM head"
    if (
        lm_head.num_embeddings_padded != logits_processor.org_vocab_size
        or lm_head.tp_size * weight.shape[0] != logits_processor.org_vocab_size
    ):
        return "a padded vocabulary"
    return ""


class PeerPushedLogits:
    """`compute_logits` of a vocabulary-parallel LM head with each rank
    pushing its shard into every rank's full logits buffer (remote writes)
    instead of every rank reading the others' shards after the barrier:
    the head's GEMM into the local shard buffer, one push, the barrier; the
    logits are this rank's full buffer (rows reused by the next call)."""

    BLOCK = 4096

    def __init__(self, lm_head, logits_processor, max_rows: int, name: str):
        self._lm_head = lm_head
        self._name = name
        self.vocab_size = logits_processor.org_vocab_size
        self.max_rows = max_rows
        self._shards: PeerShards | None = None
        self._full: torch.Tensor | None = None
        self._table: torch.Tensor | None = None
        self._fence: torch.Tensor | None = None

    def warm_up(self) -> None:
        """Collective (every rank, same order, outside graph capture):
        allocate the buffers, exchange their peer addresses and compile both
        kernels with one push of a zero row and one barrier."""
        weight = self._lm_head.weight
        nbytes = self.max_rows * self.vocab_size * weight.element_size()
        self._shards = PeerShards(weight, self.max_rows, self._name, nbytes + 256)
        tr = self._shards._transport
        full = tr.alloc(self._name + "_full", nbytes)
        self._full = full.local.view(weight.dtype).view(self.max_rows, self.vocab_size)
        self._table = torch.tensor(
            full.peer_ptrs, dtype=torch.int64, device=weight.device
        )
        self._fence = torch.zeros(1, dtype=torch.int32, device=weight.device)
        self._shards.local[:1].zero_()
        self._push(1)
        torch.accelerator.synchronize()

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        rows = hidden_states.shape[0]
        if hidden_states.dim() != 2 or rows == 0 or rows > self.max_rows:
            return None
        if self._shards is None:
            raise RuntimeError(f"{self._name}: peer logits used before warm_up")
        torch.mm(hidden_states, self._lm_head.weight.t(), out=self._shards.local[:rows])
        self._push(rows)
        return self._full[:rows]  # type: ignore[index]

    def _push(self, rows: int) -> None:
        shards: PeerShards = self._shards  # type: ignore[assignment]
        weight = self._lm_head.weight
        local = shards.local[:rows]
        _push_shard_kernel[(rows, triton.cdiv(shards.shard, self.BLOCK))](
            local,
            self._table,
            self._fence,
            shards.rank,
            shards.shard,
            self.vocab_size,
            DTYPE=_tl_dtype(weight.dtype),
            WORLD=shards.world,
            BLOCK=self.BLOCK,
            num_warps=8,
        )
        shards.barrier()


def peer_logits(lm_head, logits_processor, max_rows: int, name: str = "target_logits"):
    """The peer-memory logits of an LM head: each rank pushes its shard
    (faster than every rank reading the shards after the barrier)."""
    return PeerPushedLogits(lm_head, logits_processor, max_rows, name)
