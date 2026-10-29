# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replacement of the Engram table lookup (``zero`` / ``per_token`` modes).

Only the lookup output is replaced. The rows handed to ``wkv`` have the same
shape and dtype as the real lookup ([num_tokens, part_n_hash_cols, head_dim],
bf16, this rank's head columns); ``wkv``, the gate, the norms and the residual
add run unchanged. The logic only needs torch and numpy and runs on CPU.

``per_token`` reads one row per token from a ``[compressed_vocab, n_hash_cols,
head_dim]`` table stored in ``<replacement_tables>/U_L<layer>_bf16.npy`` (one
file per Engram layer, bf16 bits stored as ``uint16``) and indexed by the
compressed id of the current token. ``zero`` returns zeros and loads nothing.
"""

import os

import numpy as np
import torch
from torch import nn

from vllm.config.engram import per_token_table_path

MODE_ZERO = "zero"
MODE_PER_TOKEN = "per_token"


class ReplacementShard(nn.Module):
    """Stand-in for ``ParallelEngramEmbedding`` in the replacement modes.

    Carries the shard geometry the Engram module reads; it has no parameters
    and holds no table. Only the ROCm ``Engram.prepare_embeddings``/``embed``
    path reads this geometry; the config check rejects the replacement modes on
    other platforms.
    """

    def __init__(self, n_hash_cols: int, head_dim: int, tp_size: int, tp_rank: int):
        super().__init__()
        self.n_hash_cols = n_hash_cols
        self.dim = head_dim
        self.tp_size = tp_size
        self.dp_size = 1
        self.cpu_offload = False
        self.part_n_hash_cols = -(-n_hash_cols // tp_size)
        self.head_start = tp_rank * self.part_n_hash_cols


def load_per_token_table(
    path: str,
    comp_vocab: int,
    n_hash_cols: int,
    head_dim: int,
    head_start: int = 0,
    part_n_hash_cols: int | None = None,
) -> torch.Tensor:
    """Load the hash columns ``[head_start, head_start + part_n_hash_cols)`` of
    a per-token table as a contiguous bf16 CPU tensor.

    The last shard is zero-padded when it overruns ``n_hash_cols``; the
    padding is cropped after the TP all-gather, like the real lookup's.
    """
    part = n_hash_cols if part_n_hash_cols is None else part_n_hash_cols
    if not os.path.isfile(path):
        raise FileNotFoundError(f"engram per-token table not found: {path}")
    try:
        # Memory-mapped: only this rank's columns are read.
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    except (ValueError, OSError) as e:
        raise ValueError(f"cannot read engram per-token table {path}: {e}") from e
    if arr.dtype != np.uint16:
        raise ValueError(
            f"engram per-token table {path} must store bf16 as uint16, "
            f"got dtype {arr.dtype}"
        )
    expected = (comp_vocab, n_hash_cols, head_dim)
    if tuple(arr.shape) != expected:
        raise ValueError(
            f"engram per-token table {path} has shape {tuple(arr.shape)}, "
            f"expected {expected} (compressed_vocab, n_hash_cols, head_dim)"
        )
    out = np.zeros((comp_vocab, part, head_dim), dtype=np.uint16)
    stop = min(head_start + part, n_hash_cols)
    if stop > head_start:
        out[:, : stop - head_start] = arr[:, head_start:stop]
    del arr
    return torch.from_numpy(out.view(np.int16)).view(torch.bfloat16)


class EngramReplacement(nn.Module):
    """Stand-in for one Engram layer's lookup: ``fill(rows, comp_ids)`` writes
    ``rows`` (= ``staged_rows[:num_tokens]``) in place."""

    def __init__(
        self,
        mode: str,
        layer_id: int,
        shard: ReplacementShard,
        comp_vocab: int,
        table_dir: str | None,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        if mode not in (MODE_ZERO, MODE_PER_TOKEN):
            raise ValueError(f"not an engram replacement mode: {mode!r}")
        self.mode = mode
        self.layer_id = layer_id
        # A plain attribute: not a parameter or buffer, so it is not part of
        # the state dict, the weight loader or ``.to()``. The table is read once
        # at construction; weight reload and sleep level 2 do not restore it.
        self.table: torch.Tensor | None = None
        if mode == MODE_PER_TOKEN:
            if not table_dir:
                raise ValueError("engram mode 'per_token' requires replacement_tables")
            self.table = load_per_token_table(
                per_token_table_path(table_dir, layer_id),
                comp_vocab,
                shard.n_hash_cols,
                shard.dim,
                shard.head_start,
                shard.part_n_hash_cols,
            ).to(device)

    def fill(self, rows: torch.Tensor, comp_ids: torch.Tensor | None) -> None:
        if self.table is None:
            rows.zero_()
            return
        if comp_ids is None or comp_ids.shape[0] != rows.shape[0]:
            raise ValueError(
                "engram mode 'per_token' needs one compressed token id per row"
            )
        assert rows.shape[1:] == self.table.shape[1:] and rows.dtype == self.table.dtype
        torch.index_select(self.table, 0, comp_ids, out=rows)


def is_ngram_table_weight(name: str) -> bool:
    """Whether a checkpoint tensor belongs to an Engram n-gram table (skipped
    when the lookup is replaced)."""
    return ".engram.embed_tokens." in name


def per_token_comp_ids(
    token_map: torch.Tensor, input_ids: torch.Tensor, dead_mask: torch.Tensor
) -> torch.Tensor:
    """Compressed id of every position's current token.

    Dead positions (image spans, profile runs) map to id 0; their rows are
    masked out of the gate, so any valid id will do. Ids beyond the token map
    are clamped (no device-side assert), like the hash kernels' masked loads.
    """
    ids = torch.where(dead_mask, 0, input_ids).long()
    return token_map[ids.clamp(0, token_map.shape[0] - 1)].long()
