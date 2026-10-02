# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduced draft vocabulary for drafters that share the target lm_head.

The drafter projects onto a subset of the target vocabulary (FR-Spec). Only
drafting changes: the target still verifies with its full lm_head.
"""

import json
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING

import regex as re
import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_DEQUANT_CHUNK = 512


def load_draft_token_map(path: str) -> list[int]:
    """Read draft token ids from SGLang's ``--speculative-token-map`` format
    (``.pt``/``.pth``), a JSON list, or whitespace/comma separated text. As in
    SGLang, a path that does not exist locally is ``<hf_repo_id>/<filename>``.
    """
    if not os.path.exists(path):
        repo_id, filename = os.path.split(path)
        if not repo_id:
            raise FileNotFoundError(f"draft_token_map file {path!r} not found.")
        from vllm.transformers_utils.repo_utils import hf_api

        path = hf_api().hf_hub_download(repo_id=repo_id, filename=filename)

    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".pt", ".pth"):
        ids = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(ids, torch.Tensor):
            ids = ids.flatten().tolist()
    elif suffix == ".json":
        with open(path) as f:
            ids = json.load(f)
    else:
        with open(path) as f:
            ids = [tok for tok in re.split(r"[\s,]+", f.read()) if tok]
    if not isinstance(ids, (list, tuple)):
        raise ValueError(f"draft_token_map {path!r} must hold a flat list of ids.")
    return [int(i) for i in ids]


def build_draft_token_ids(
    token_ids: list[int], vocab_size: int, always_include: list[int]
) -> torch.Tensor:
    """Validate, merge, dedupe and sort the draft vocabulary (int64, CPU)."""
    ids = torch.tensor(list(token_ids) + list(always_include), dtype=torch.int64)
    if ids.numel() == 0:
        raise ValueError("draft_token_map is empty.")
    bad = ids[(ids < 0) | (ids >= vocab_size)]
    if bad.numel() > 0:
        raise ValueError(
            f"draft_token_map has {bad.numel()} ids outside [0, {vocab_size}), "
            f"e.g. {bad[:5].tolist()}."
        )
    return torch.unique(ids, sorted=True)


def _dequantize_rows(
    lm_head: nn.Module, rows: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Materialize rows of a quantized lm_head by probing its quant method with
    identity rows. Activation quantization of a one-hot row may scale every
    column by one constant, which a least-squares fit against the full head on
    random inputs removes."""
    hidden, device = lm_head.embedding_dim, rows.device
    out = torch.empty(rows.numel(), hidden, dtype=dtype, device=device)
    for start in range(0, hidden, _DEQUANT_CHUNK):
        n = min(_DEQUANT_CHUNK, hidden - start)
        probe = torch.zeros(n, hidden, dtype=dtype, device=device)
        probe[:, start : start + n].fill_diagonal_(1.0)
        cols = lm_head.quant_method.apply(lm_head, probe, bias=None)
        out[:, start : start + n] = cols.index_select(1, rows).t().to(dtype)

    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(16, hidden, generator=gen, device=device).to(dtype)
    ref = lm_head.quant_method.apply(lm_head, x, bias=None).index_select(1, rows)
    approx = (x @ out.t()).float()
    scale = float((ref.float() * approx).sum() / (approx * approx).sum())
    if not scale > 0.0:
        raise ValueError("Could not materialize the quantized lm_head rows.")
    logger.info(
        "Materialized %d quantized lm_head rows in %s for the draft vocabulary "
        "(rescale %.4f).",
        rows.numel(),
        dtype,
        scale,
    )
    return out.mul_(scale)


class DraftVocabHead(nn.Module):
    """lm_head rows of a draft vocabulary, vocab-parallel over TP.

    Each rank holds the listed rows of its own shard, zero-padded to the same
    count on every rank so the TP gather stays uniform. Exposes the attributes
    ``LogitsProcessor`` reads from a ``VocabParallelEmbedding``.
    """

    def __init__(self, weight: torch.Tensor, tp_size: int, tp_rank: int, pad: int):
        super().__init__()
        self.register_buffer("weight", weight, persistent=False)
        self.quant_method = UnquantizedEmbeddingMethod()
        self.tp_size = tp_size
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=tp_rank * weight.shape[0],
            num_org_vocab_padding=pad,
        )
        self.quant_method.process_weights_after_loading(self)


class DraftVocab:
    """A draft vocabulary built on a shared, vocab-parallel lm_head.

    Attributes:
        head: Reduced head that replaces the shared lm_head in the drafter.
        target_ids: Sorted target id of each draft token, shape ``(K,)``.
        col_to_target: Target id of each gathered logit column, including
            per-rank padding columns.
        valid_cols: Non-padding gathered columns, or ``None`` without padding.

    """

    def __init__(self, token_ids: torch.Tensor, lm_head: nn.Module, dtype):
        lm_head = getattr(lm_head, "base_layer", lm_head)
        if not hasattr(lm_head, "shard_indices"):
            raise ValueError(
                "draft_token_map needs a vocab-parallel lm_head, got "
                f"{type(lm_head).__name__}."
            )
        if getattr(lm_head, "bias", None) is not None:
            raise ValueError("draft_token_map does not support an lm_head bias.")
        tp_size, tp_rank = lm_head.tp_size, getattr(lm_head, "tp_rank", 0)
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
        unquantized = (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        if isinstance(lm_head.quant_method, unquantized):
            weight = lm_head.weight.index_select(0, rows)
        else:
            weight = _dequantize_rows(lm_head, rows, dtype)
        pad = rows_per_rank - rows.numel()
        weight = torch.cat([weight, weight.new_zeros(pad, weight.shape[1])])

        self.head = DraftVocabHead(weight, tp_size, tp_rank, pad)
        self.target_ids = token_ids.to(device)
        self.col_to_target = col_to_target.flatten().to(device)
        valid = valid.flatten()
        self.valid_cols = None if valid.all() else valid.nonzero()[:, 0].to(device)
        logger.info(
            "Draft vocabulary: %d of %d tokens; the drafter reads %d lm_head "
            "rows per TP rank instead of %d.",
            token_ids.numel(),
            lm_head.org_vocab_size,
            rows_per_rank,
            lm_head.shard_indices.num_org_elements,
        )

    def install(self, draft_model: nn.Module, target_lm_head: nn.Module) -> None:
        """Replace each drafter ``lm_head``/``head`` that is the target head.
        The target model keeps its full head."""
        if getattr(draft_model, "draft_id_to_target_id", None) is not None:
            raise ValueError(
                "draft_token_map cannot be combined with a drafter that has its "
                "own draft vocabulary (draft_id_to_target_id)."
            )
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
        """Drop padding columns of the gathered logits: ``(N, K)``."""
        if self.valid_cols is None:
            return logits
        return logits.index_select(-1, self.valid_cols)

    def scatter(self, logits: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Write ``(N, K)`` logits into the target columns of a ``-inf``-filled
        ``out``, so softmax has zero mass outside the draft vocabulary."""
        out = out[: logits.shape[0]]
        out.index_copy_(1, self.target_ids, logits.to(out.dtype))
        return out


def make_draft_vocab(
    token_map: str,
    model_config: "ModelConfig",
    draft_model: nn.Module,
    target_lm_head: nn.Module,
) -> DraftVocab:
    """Load a token map, always adding EOS ids, and install it on a drafter
    that shares ``target_lm_head``."""
    eos_ids: list[int] = []
    for eos in (
        getattr(model_config.hf_text_config, "eos_token_id", None),
        model_config.try_get_generation_config().get("eos_token_id"),
    ):
        if eos is not None:
            eos_ids.extend(eos if isinstance(eos, (list, tuple)) else [eos])
    base_head = getattr(target_lm_head, "base_layer", target_lm_head)
    vocab_size = getattr(base_head, "org_vocab_size", model_config.get_vocab_size())
    token_ids = build_draft_token_ids(
        load_draft_token_map(token_map), vocab_size, eos_ids
    )
    draft_vocab = DraftVocab(token_ids, target_lm_head, model_config.dtype)
    draft_vocab.install(draft_model, target_lm_head)
    return draft_vocab
