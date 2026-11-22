# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`collectives/logits_exchange_common` (communicator `logits_exchange_comm`, op its
`gemm_exchange` method): the lm_head shard GEMM with a push exchange of its output, for

* the target's `compute_logits` (all-gather mode: every rank receives all rows, the
  drop-in for GEMM + `tensor_model_parallel_all_gather`),
* the DSpark drafter's `compute_draft_logits` (all-gather mode, inside the draft's
  CUDA graph; its own communicator),
* the batch-sharded sampler (`compute_logits_owned`: owner mode, the rows sorted by
  owner; each rank receives only the rows it samples), switch `SHARDED`.

Two communicators (target, draft). The target's has two receive buffers that its calls
alternate (a fast sender never overwrites rows a slow rank still reads); the returned
logits are a view of the receive buffer, valid until the second next exchange. The
draft's, captured in a graph, always uses buffer 0 (its logits are consumed by the
Markov chain before the next target step, whose collectives order every rank's read
before any rank's next draft exchange) and has that one buffer only. Above the
communicator's rows (prefill steps, prompt logprobs), for a call the entry rejects
(`logits_gemm_exchange_unsupported_reason`) or for a scaled / soft-capped / non-bf16
head the stock path runs. The split-K GEMM bands share one workspace per device and
spin-wait for their partner workgroup: they need the stream to themselves (the model's
aux streams are idle there) and all 256 workgroups co-resident.
"""

from typing import Any

import torch

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd import akr_config, akr_nonlayer

logger = init_logger(__name__)

MAX_ROWS = 256  # rows the receive buffers are sized for (2 x rows x vocab bf16 a rank)
_state: dict[str, Any] = {}


def _lib():
    from amd_kernels.collectives import logits_exchange_common

    return logits_exchange_common


class _Exchange:
    """One communicator + its alternating buffer + the persistent helper tensors."""

    def __init__(self, name: str, comm, max_rows: int, device: torch.device):
        self.name, self.max_rows, self.comm = name, max_rows, comm
        self.rank = akr_nonlayer.tp_group().rank_in_group
        self.arange = torch.arange(MAX_ROWS, dtype=torch.int64, device=device)
        self.buf = 0

    def next_buf(self) -> int:
        b = self.buf
        self.buf ^= 1
        return b

    def warm_up(self, w: torch.Tensor, owner: bool) -> None:
        """Collective: compile every variant of the all-gather mode and, with `owner`
        (the batch-sharded sampler can call it), of the owner mode (rows <= / > the push
        crossovers), and take the first-use allocations of the library."""
        x = torch.zeros(
            self.max_rows, w.shape[1], dtype=torch.bfloat16, device=w.device
        )
        self.comm.warmup()
        for rows in sorted(
            {min(8, self.max_rows), min(32, self.max_rows), self.max_rows}
        ):
            self.comm.gemm_exchange(x[:rows], w, self.arange[:rows], None, 0)
        for rows in sorted({min(8, self.max_rows), self.max_rows} if owner else ()):
            world = akr_nonlayer.tp_group().world_size
            counts = torch.tensor(
                [rows // world + int(r < rows % world) for r in range(world)],
                dtype=torch.int32,
                device=w.device,
            )
            self.comm.gemm_exchange(x[:rows], w, self.arange[:rows], counts, 0)
        torch.accelerator.synchronize(w.device)

    def all_gather(
        self, x: torch.Tensor, w: torch.Tensor, buf: int | None = None
    ) -> torch.Tensor:
        rows = x.shape[0]
        b = self.next_buf() if buf is None else buf
        self.comm.gemm_exchange(x, w, self.arange[:rows], None, b)
        return self.comm.recv(rows, b)

    def owned(self, x, w, row_indices, counts_dev, num_local: int) -> torch.Tensor:
        b = self.next_buf()
        self.comm.gemm_exchange(x, w, row_indices, counts_dev, b)
        return self.comm.recv(num_local, b)


def _head_reason(lm_head, logits_processor) -> str | None:
    """vLLM-side gates (the shard layout the exchange assumes); the kernel's shape gates
    are the entry's."""
    lib = _lib()
    if not lib.available():
        return "needs gfx950 with FlyDSL"
    g = akr_nonlayer.tp_group()
    w = lm_head.weight
    key = (*w.shape, g.world_size)
    if w.dim() != 2 or key not in lib.logits_gemm_exchange_shapes():
        return f"no logits exchange for (lm_head shard, world) {key}"
    if w.dtype != torch.bfloat16 or not w.is_contiguous():
        return f"lm_head shard {w.dtype} not contiguous bf16"
    if getattr(lm_head, "bias", None) is not None:
        return "lm_head has a bias"
    if (
        lm_head.org_vocab_size != lm_head.num_embeddings
        or lm_head.org_vocab_size != w.shape[0] * g.world_size
    ):
        return "padded or added vocabulary"
    if lm_head.shard_indices.org_vocab_start_index != g.rank_in_group * w.shape[0]:
        return "vocabulary shards are not in TP rank order"
    lp = logits_processor
    if lp.scale != 1.0 or lp.soft_cap is not None or lp.logits_as_input:
        return "scaled / soft-capped logits"
    if lp.head_dtype is not None and lp.head_dtype != torch.bfloat16:
        return "head dtype is not bf16"
    return None


def head_supported(lm_head, logits_processor) -> bool:
    key = id(lm_head)
    ok = _state.setdefault("head_ok", {})
    if key not in ok:
        ok[key] = not akr_nonlayer.blocked(
            "LMHEAD", _head_reason(lm_head, logits_processor)
        )
    return ok[key]


def _rows_for_target(vllm_config) -> int:
    spec = vllm_config.speculative_config
    k = 0 if spec is None else spec.num_speculative_tokens
    return min(MAX_ROWS, vllm_config.scheduler_config.max_num_seqs * (k + 1))


def _rows_for_draft(vllm_config) -> int:
    spec = vllm_config.speculative_config
    k = 0 if spec is None else spec.num_speculative_tokens
    return min(MAX_ROWS, vllm_config.scheduler_config.max_num_seqs * max(k, 1))


def _buffers_for(kind: str) -> int:
    """Receive buffers of the `kind` communicator: the draft's exchanges all take buffer
    0 (`logits`), so it has one."""
    return 1 if kind == "draft" else 2


def setup(kind: str, lm_head, logits_processor, vllm_config) -> bool:
    """Collective (every TP rank, same order, after the weights load and before
    capture): build the `kind` ('target' or 'draft') communicator and compile its GEMM
    variants."""
    if kind in _state:
        return True
    if not akr_nonlayer.switch("LMHEAD") or not head_supported(
        lm_head, logits_processor
    ):
        return False
    rows = (
        _rows_for_target(vllm_config)
        if kind == "target"
        else _rows_for_draft(vllm_config)
    )
    rows = max(rows, 8)
    g = akr_nonlayer.tp_group()
    w = lm_head.weight
    lib = _lib()
    kw = dict(
        n_shard=w.shape[0], k=w.shape[1], max_rows=rows, buffers=_buffers_for(kind)
    )
    if akr_nonlayer.blocked(
        "LMHEAD",
        lib.logits_exchange_comm_unsupported_reason(
            g.rank_in_group, g.cpu_group, w.device, **kw
        ),
    ):
        return False
    comm = lib.logits_exchange_comm(g.rank_in_group, g.cpu_group, w.device, **kw)
    ex = _Exchange(kind, comm, rows, w.device)
    ex.warm_up(w, owner=kind == "target" and akr_nonlayer.switch("SHARDED"))
    _state[kind] = ex
    gb = kw["buffers"] * ex.max_rows * lm_head.org_vocab_size * 2 / 2**30
    logger.info(
        "amd_kernels lm_head exchange (%s): max %d rows, %.3f GB receive buffers per "
        "rank",
        kind,
        ex.max_rows,
        gb,
    )
    return True


def logits(kind: str, lm_head, normed: torch.Tensor) -> torch.Tensor | None:
    """All-gathered logits [rows, vocab] bf16 of the normed hidden rows, or None
    (stock path) when this call is not the kernel's case."""
    ex = _state.get(kind)
    if ex is None or normed.dim() != 2:
        return None
    rows = normed.shape[0]
    buf = 0 if kind == "draft" else ex.buf
    reason = _lib().logits_gemm_exchange_unsupported_reason(
        ex.comm, normed, lm_head.weight, ex.arange[:rows], None, buf
    )
    if akr_config.call_unsupported("gemm", reason):
        return None
    return ex.all_gather(normed, lm_head.weight, 0 if kind == "draft" else None)


def owned_logits(
    lm_head, norm, hidden_states, sorted_logits_indices, md
) -> torch.Tensor | None:
    """The sharded sampler's logits: rows sorted by owner in, [num_local, 163840] out.
    None (stock `compute_logits_local` + all-to-all) when not the kernel's case."""
    ex = _state.get("target")
    total = sum(md.num_logits_per_rank)
    if (
        ex is None
        or not 1 <= total <= ex.max_rows
        or sorted_logits_indices.numel() != total
        or hidden_states.dim() != 2
    ):
        return None
    if hidden_states.shape[0] <= ex.max_rows:
        # decode: every row is a logits row; the row gather is fused into the GEMM
        x = norm(hidden_states.contiguous(), None)
        idx = sorted_logits_indices
    else:
        x = norm(hidden_states[sorted_logits_indices], None)
        idx = ex.arange[:total]
    # pinned staging through the caching host allocator (safe to reuse once the copy is
    # done)
    counts = (
        torch.tensor(md.num_logits_per_rank, dtype=torch.int32, device="cpu")
        .pin_memory()
        .to(x.device, non_blocking=True)
    )
    reason = _lib().logits_gemm_exchange_unsupported_reason(
        ex.comm, x, lm_head.weight, idx, counts, ex.buf
    )
    if akr_config.call_unsupported("gemm", reason):
        return None
    return ex.owned(x, lm_head.weight, idx, counts, md.num_local_logits)
