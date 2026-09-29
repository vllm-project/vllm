# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduced vocabulary for the MTP draft head only.

The draft projects every step against the full ``lm_head``; that GEMV is on its
memory roofline. With a list of frequent token ids it projects only against
those rows and writes ``-inf`` elsewhere. The draft still produces logits over
the full vocabulary, and rejection sampling uses that same distribution, so the
output is exact: only the acceptance rate moves. The target model's
verification is untouched.

The listed rows are very unevenly spread across the TP shards of the
``lm_head``, so they are gathered once at load time with an all-reduce and
re-split evenly (``n / tp`` rows per rank). This runs in ``load_draft_model``,
before CUDA graph capture, and checks itself against the original
``compute_logits`` on random input; if the check fails the full vocabulary is
kept.

Enabled with ``VLLM_MTP_DRAFT_VOCAB=<path to a sorted int64 .npy of ids>``.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def maybe_install_draft_vocab(model: nn.Module, vocab_size: int, max_rows: int) -> None:
    """Replace ``model.compute_logits`` with the reduced-vocabulary projection.

    Args:
        model: The loaded MTP draft model.
        vocab_size: Full vocabulary size the logits must cover.
        max_rows: Largest number of hidden-state rows per call; larger calls
            fall back to the original projection.
    """
    path = os.environ.get("VLLM_MTP_DRAFT_VOCAB")
    if not path:
        return
    try:
        _install(model, path, vocab_size, max_rows)
    except Exception as e:  # noqa: BLE001  keep the full vocabulary
        logger.warning("MTP draft vocabulary not installed: %s", e)


def _install(model: nn.Module, path: str, vocab_size: int, max_rows: int) -> None:
    lm = model.lm_head
    weight = lm.weight
    dev, hidden = weight.device, weight.shape[1]
    tp = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()
    ids_np = np.load(path).astype(np.int64)
    n = len(ids_np)
    if n % tp or len(np.unique(ids_np)) != n or ids_np.max() >= vocab_size:
        raise ValueError(f"invalid id list: {n} ids, tp={tp}, max {ids_np.max()}")
    ids = torch.from_numpy(np.sort(ids_np)).to(dev)

    # Rows of the list held by this shard, re-split evenly across ranks.
    shard = lm.shard_indices
    lo, hi = shard.org_vocab_start_index, shard.org_vocab_end_index
    mine = (ids >= lo) & (ids < hi)
    full = torch.zeros(n, hidden, dtype=weight.dtype, device=dev)
    full[mine] = weight[ids[mine] - lo]
    count = tensor_model_parallel_all_reduce(mine.to(torch.float32).sum().reshape(1))
    if int(count.item()) != n:
        raise ValueError(f"id list not covered exactly by the shards: {count}/{n}")
    full = tensor_model_parallel_all_reduce(full)
    k = n // tp
    rows = full[rank * k : (rank + 1) * k].contiguous()
    del full

    original = model.compute_logits
    gen = torch.Generator(device="cpu").manual_seed(1234)
    probe = torch.randn(4, hidden, generator=gen).to(dev, weight.dtype)
    ref = original(probe)
    if ref.shape[-1] != vocab_size:
        raise ValueError(f"compute_logits gives {ref.shape[-1]} columns")
    out_buf = torch.full(
        (max_rows, vocab_size), float("-inf"), dtype=ref.dtype, device=dev
    )

    def compute_logits(hidden_states: torch.Tensor) -> torch.Tensor:
        num = hidden_states.shape[0]
        if num > max_rows:
            return original(hidden_states)
        local = F.linear(hidden_states.to(rows.dtype), rows)
        gathered = tensor_model_parallel_all_gather(local, dim=-1)
        out = out_buf[:num]
        out.index_copy_(1, ids, gathered.to(out.dtype))
        return out

    # Control: equal to the original on the listed columns, -inf elsewhere,
    # and the list shifted by one id must NOT match.
    new = compute_logits(probe).clone()
    outside = torch.ones(vocab_size, dtype=torch.bool, device=dev)
    outside[ids] = False
    scale = ref[:, ids].float().abs().max()
    err = float((new[:, ids].float() - ref[:, ids].float()).abs().max() / scale)
    err_shift = float(
        (new[:, ids[:-1]].float() - ref[:, ids[1:]].float()).abs().max() / scale
    )
    neg_inf = bool(torch.isneginf(new[:, outside]).all())
    if not (err < 2e-2 and err_shift > 10 * max(err, 1e-3) and neg_inf):
        raise ValueError(
            f"self-check failed: err {err:.2e}, shifted {err_shift:.2e}, "
            f"-inf outside {neg_inf}"
        )
    out_buf.fill_(float("-inf"))
    model.compute_logits = compute_logits
    logger.info(
        "MTP draft vocabulary: %d ids (%d rows/rank of %d), self-check err %.2e",
        n,
        k,
        weight.shape[0],
        err,
    )
