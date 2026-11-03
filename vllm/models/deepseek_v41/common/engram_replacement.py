# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replacement of the Engram table lookup (``per_token`` mode).

Only the lookup output is replaced. The rows handed to ``wkv`` have the same
shape and dtype as the real lookup ([num_tokens, part_n_hash_cols, head_dim],
bf16, this rank's head columns); ``wkv``, the gate, the norms and the residual
add run unchanged. The logic only needs torch and runs on CPU.

``per_token`` reads one row per token from a ``[compressed_vocab, n_hash_cols,
head_dim]`` bf16 table, indexed by the compressed id of the current token. The
table is the checkpoint tensor ``layers.<i>.engram.per_token_table.weight`` and
is loaded like any other weight; each TP rank keeps its own hash columns.
"""

import torch
from torch import nn

from vllm.model_executor.utils import set_weight_attrs

MODE_PER_TOKEN = "per_token"


class ReplacementShard(nn.Module):
    """Stand-in for ``ParallelEngramEmbedding`` in the replacement mode.

    Carries the shard geometry the Engram module reads; it has no parameters
    and holds no table. Only the ROCm ``Engram.prepare_embeddings``/``embed``
    path reads this geometry; the config check rejects the replacement mode on
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


class PerTokenTable(nn.Module):
    """One Engram layer's per-token table: ``weight`` holds this rank's hash
    columns ``[head_start, head_start + part_n_hash_cols)`` of the checkpoint's
    ``[comp_vocab, n_hash_cols, head_dim]`` bf16 tensor.

    The last shard is zero-padded when it overruns ``n_hash_cols``; the padding
    is cropped after the TP all-gather, like the real lookup's.
    """

    def __init__(self, shard: ReplacementShard, comp_vocab: int):
        super().__init__()
        self.comp_vocab = comp_vocab
        self.n_hash_cols = shard.n_hash_cols
        self.head_dim = shard.dim
        self.head_start = shard.head_start
        self.part_n_hash_cols = shard.part_n_hash_cols
        self.weight = nn.Parameter(
            torch.empty(
                comp_vocab, self.part_n_hash_cols, self.head_dim, dtype=torch.bfloat16
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        expected = (self.comp_vocab, self.n_hash_cols, self.head_dim)
        if tuple(loaded_weight.shape) != expected:
            raise ValueError(
                f"engram per_token_table.weight has shape "
                f"{tuple(loaded_weight.shape)}, expected {expected} "
                "(compressed_vocab, n_hash_cols, head_dim)"
            )
        if loaded_weight.dtype != torch.bfloat16:
            raise ValueError(
                "engram per_token_table.weight must be bfloat16, "
                f"got {loaded_weight.dtype}"
            )
        used = max(min(self.n_hash_cols - self.head_start, self.part_n_hash_cols), 0)
        param.data[:, :used].copy_(
            loaded_weight[:, self.head_start : self.head_start + used]
        )
        param.data[:, used:].zero_()

    def fill(self, rows: torch.Tensor, comp_ids: torch.Tensor | None) -> None:
        """Write the per-token rows into ``rows`` (= ``staged_rows[:num_tokens]``)."""
        if comp_ids is None or comp_ids.shape[0] != rows.shape[0]:
            raise ValueError(
                "engram mode 'per_token' needs one compressed token id per row"
            )
        assert (
            rows.shape[1:] == self.weight.shape[1:] and rows.dtype == self.weight.dtype
        )
        torch.index_select(self.weight, 0, comp_ids, out=rows)


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
