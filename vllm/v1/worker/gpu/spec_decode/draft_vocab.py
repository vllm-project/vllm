# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduced draft vocabulary (FR-Spec) for drafters that share the target lm_head.

Only drafting changes: the target still verifies with its full lm_head.
"""

import json
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbeddingShardIndices,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)


def load_draft_token_ids(path: str, extra_ids: list[int]) -> torch.Tensor:
    """Load a draft token map and return its sorted, unique target ids.

    Args:
        path: SGLang `--speculative-token-map` file (`.pt`) or a JSON list.
        extra_ids: Ids that are always included, e.g. EOS.

    Returns:
        int64 CPU tensor of target token ids.

    """
    if path.endswith(".json"):
        with open(path) as f:
            ids = json.load(f)
    else:
        ids = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(ids, torch.Tensor):
            ids = ids.flatten().tolist()
    if not isinstance(ids, (list, tuple)):
        raise ValueError(f"draft_token_map {path!r} must hold a flat list of ids.")

    token_ids = torch.tensor([int(i) for i in ids] + extra_ids, dtype=torch.int64)
    if token_ids.numel() == 0:
        raise ValueError("draft_token_map is empty.")
    return torch.unique(token_ids, sorted=True)


class DraftVocabHead(nn.Module):
    """One TP rank's rows of a draft vocabulary, padded to a common count.

    Exposes the `VocabParallelEmbedding` attributes `LogitsProcessor` reads, so
    the gather and the local-argmax reduction work unchanged.
    """

    def __init__(self, weight: torch.Tensor, tp_size: int, tp_rank: int, pad: int):
        super().__init__()
        self.register_buffer("weight", weight, persistent=False)
        self.tp_size = tp_size
        rows = weight.shape[0]
        start, end = tp_rank * rows, (tp_rank + 1) * rows
        self.shard_indices = VocabParallelEmbeddingShardIndices(
            start, end, end, end, start, end - pad, end, end
        )
        self.quant_method = UnquantizedEmbeddingMethod()
        self.quant_method.process_weights_after_loading(self)


class DraftVocab:
    """A draft vocabulary built on a shared, vocab-parallel lm_head.

    Attributes:
        head: Reduced head that replaces the shared lm_head in the drafter.
        target_ids: Sorted target id of each draft token, shape `(K,)`.
        col_to_target: Target id of each gathered logit column, including
            per-rank padding columns.
        valid_cols: Non-padding gathered columns, or `None` without padding.

    """

    def __init__(self, token_ids: torch.Tensor, lm_head: nn.Module):
        lm_head = getattr(lm_head, "base_layer", lm_head)
        if not hasattr(lm_head, "shard_indices"):
            raise ValueError(
                "draft_token_map needs a vocab-parallel lm_head, got "
                f"{type(lm_head).__name__}."
            )
        unquantized = (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        if not isinstance(lm_head.quant_method, unquantized):
            raise ValueError("draft_token_map needs an unquantized lm_head.")
        if getattr(lm_head, "bias", None) is not None:
            raise ValueError("draft_token_map does not support an lm_head bias.")
        vocab_size = lm_head.org_vocab_size
        bad = token_ids[(token_ids < 0) | (token_ids >= vocab_size)]
        if bad.numel() > 0:
            raise ValueError(
                f"draft_token_map has {bad.numel()} ids outside [0, {vocab_size}), "
                f"e.g. {bad[:5].tolist()}."
            )
        tp_size, tp_rank = lm_head.tp_size, lm_head.tp_rank
        shards = [
            type(lm_head)._get_indices(
                lm_head.num_embeddings_padded,
                lm_head.org_vocab_size_padded,
                lm_head.num_embeddings,
                lm_head.org_vocab_size,
                rank,
                tp_size,
            )
            for rank in range(tp_size)
        ]
        rank_ids = [
            token_ids[
                (token_ids >= s.org_vocab_start_index)
                & (token_ids < s.org_vocab_end_index)
            ]
            for s in shards
        ]
        rows_per_rank = max(ids.numel() for ids in rank_ids)
        col_to_target = torch.zeros(tp_size, rows_per_rank, dtype=torch.int64)
        valid = torch.zeros(tp_size, rows_per_rank, dtype=torch.bool)
        for rank, ids in enumerate(rank_ids):
            col_to_target[rank, : ids.numel()] = ids
            valid[rank, : ids.numel()] = True

        device = lm_head.weight.device
        rows = (rank_ids[tp_rank] - shards[tp_rank].org_vocab_start_index).to(device)
        weight = lm_head.weight.index_select(0, rows)
        pad = rows_per_rank - rows.numel()
        weight = torch.cat([weight, weight.new_zeros(pad, weight.shape[1])])

        self.head = DraftVocabHead(weight, tp_size, tp_rank, pad)
        self.target_ids = token_ids.to(device)
        self.col_to_target = col_to_target.flatten().to(device)
        valid = valid.flatten()
        self.valid_cols = None if valid.all() else valid.nonzero()[:, 0].to(device)
        logger.info(
            "Draft vocabulary: %d of %d tokens, %d lm_head rows per TP rank.",
            token_ids.numel(),
            vocab_size,
            rows_per_rank,
        )

    def install(self, draft_model: nn.Module, target_lm_head: nn.Module) -> None:
        """Replace every drafter `lm_head`/`head` that is the target's head."""
        replaced = 0
        for module in list(draft_model.modules()):
            for name in ("lm_head", "head"):
                if module._modules.get(name) is target_lm_head:
                    setattr(module, name, self.head)
                    replaced += 1
        if replaced == 0:
            raise ValueError(
                "draft_token_map requires a drafter that shares the target "
                f"model's lm_head; {type(draft_model).__name__} has its own."
            )

    def restrict(self, logits: torch.Tensor) -> torch.Tensor:
        """Drop the padding columns of gathered `(N, tp_size * rows)` logits."""
        if self.valid_cols is None:
            return logits
        return logits.index_select(-1, self.valid_cols)

    def scatter(self, logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Expand `(N, K)` logits to `(N, vocab_size)`, `-inf` off the list."""
        out = logits.new_full((logits.shape[0], vocab_size), float("-inf"))
        return out.index_copy_(1, self.target_ids, logits)


def make_draft_vocab(
    token_map: str,
    model_config: "ModelConfig",
    draft_model: nn.Module,
    target_lm_head: nn.Module,
) -> DraftVocab:
    """Build a draft vocabulary, always including EOS, and install it."""
    eos_ids: list[int] = []
    for eos in (
        getattr(model_config.hf_text_config, "eos_token_id", None),
        model_config.try_get_generation_config().get("eos_token_id"),
    ):
        if eos is not None:
            eos_ids.extend(eos if isinstance(eos, (list, tuple)) else [eos])
    token_ids = load_draft_token_ids(token_map, eos_ids)
    draft_vocab = DraftVocab(token_ids, target_lm_head)
    draft_vocab.install(draft_model, target_lm_head)
    return draft_vocab
