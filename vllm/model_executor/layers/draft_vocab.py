# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft lm_heads that cover a subset of the target vocabulary.

`DraftVocab` serves two kinds of drafters:

- Checkpoints with a pruned draft lm_head and `draft_id_to_target_id` offsets
  (EAGLE3, DFlash, DSpark). This used to live in each model file.
- MTP drafters that share the target's full lm_head, restricted at load time
  by `speculative_config.draft_token_map`. On bandwidth-bound GPUs the full
  head dominates the draft step, so each draft token computes exact logits
  only for:
  1. the listed rows, a fixed list of frequent tokens, and
  2. optionally `draft_token_map_dynamic_rows` more rows per draft token,
     picked by a rank-256 projection of the lm_head (built from its weights
     at load time), which recovers what the list misses, e.g. other languages.

Either way `compute_logits` returns target-vocabulary logits that are `-inf`
outside the subset, so speculators need no special handling. The target still
verifies with its full head, so only the acceptance length can change.
"""

import json
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_UNQUANTIZED = (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
# Rank of the lm_head projection that scores the dynamic rows. On Qwen3.8
# MTP, rank 256 alone puts 97.5% of the greedy draft tokens in its top 16k.
_DYNAMIC_RANK = 256


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


# Rows per step when copying lm_head rows, which bounds the
# temporaries.
_CHUNK = 16384


class _Rows(ReplicatedLinear):
    """lm_head rows as a linear layer."""

    def __init__(self, weight: torch.Tensor):
        with torch.device(weight.device):
            super().__init__(
                weight.shape[1],
                weight.shape[0],
                bias=False,
                params_dtype=weight.dtype,
                return_bias=False,
                disable_tp=True,
            )
        self.weight_loader(self.weight, weight)
        self.quant_method.process_weights_after_loading(self)


# The kernel below computes `x[n] @ weight[ids[n, j]]` for the dynamic rows,
# reading only the picked rows instead of gathering them into a copy. One
# program handles BLOCK_ROWS picked rows of one token.
@triton.jit
def _gather_rows_gemv_kernel(
    x_ptr,
    weight_ptr,
    ids_ptr,
    out_ptr,
    num_rows,
    hidden_size,
    weight_stride,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    token = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < num_rows
    ids = tl.load(ids_ptr + token * num_rows + rows, mask=row_mask, other=0)
    acc = tl.zeros((BLOCK_ROWS,), dtype=tl.float32)
    for start in range(0, hidden_size, BLOCK_HIDDEN):
        cols = start + tl.arange(0, BLOCK_HIDDEN)
        col_mask = cols < hidden_size
        x = tl.load(x_ptr + token * hidden_size + cols, mask=col_mask, other=0.0)
        w = tl.load(
            weight_ptr + ids[:, None].to(tl.int64) * weight_stride + cols[None, :],
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(w.to(tl.float32) * x[None, :].to(tl.float32), axis=1)
    tl.store(
        out_ptr + token * num_rows + rows,
        acc.to(out_ptr.dtype.element_ty),
        mask=row_mask,
    )


def gather_rows_gemv(
    x: torch.Tensor, weight: torch.Tensor, ids: torch.Tensor
) -> torch.Tensor:
    """`out[n, j] = x[n] @ weight[ids[n, j]]` without materializing the rows.

    Args:
        x: Activations, shape `(N, H)`.
        weight: lm_head weight, shape `(V, H)`.
        ids: Row ids per token, shape `(N, J)`.

    """
    if not x.is_cuda:
        return torch.einsum("nd,njd->nj", x, weight[ids])
    num_tokens, num_rows = ids.shape
    out = x.new_empty(num_tokens, num_rows)
    grid = (num_tokens, triton.cdiv(num_rows, 64))
    _gather_rows_gemv_kernel[grid](
        x.contiguous(),
        weight,
        ids.contiguous(),
        out,
        num_rows,
        x.shape[-1],
        weight.stride(0),
        BLOCK_ROWS=64,
        BLOCK_HIDDEN=128,
    )
    return out


def scatter_draft_logits(
    vocab_size: int,
    listed: torch.Tensor,
    token_ids: torch.Tensor,
    picked: torch.Tensor,
    picks: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    """Target-vocabulary logits, `-inf` outside the listed and picked rows.

    `out[n, token_ids] = listed[n]` and `out[n, candidate_ids[picks[n]]] =
    picked[n]`.
    """
    out = listed.new_full((listed.shape[0], vocab_size), float("-inf"))
    out[:, token_ids] = listed
    out.scatter_(1, candidate_ids[picks], picked)
    return out


class DraftVocab:
    """Draft logits over a subset of the target vocabulary.

    Args:
        logits_processor: Logits processor of the draft lm_head.
        vocab_size: Target vocabulary size. A smaller draft vocabulary gets a
            `draft_id_to_target_id` parameter for the model to register and
            load.

    """

    def __init__(self, logits_processor: LogitsProcessor, vocab_size: int):
        self.logits_processor = logits_processor
        self.vocab_size = vocab_size
        self.draft_id_to_target_id: nn.Parameter | None = None
        if logits_processor.vocab_size < vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(logits_processor.vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        # Set by load_token_map.
        self.token_ids: torch.Tensor | None = None
        self.rows: _Rows | None = None
        self.num_dynamic_rows = 0

    def compute_logits(
        self, lm_head: VocabParallelEmbedding, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Target-vocabulary logits, `-inf` outside the draft vocabulary."""
        if self.token_ids is not None:
            listed, picked = self._token_map_logits(hidden_states)
            if picked is not None:
                return scatter_draft_logits(
                    self.vocab_size, listed, self.token_ids, *picked, self.candidate_ids
                )
            out = listed.new_full((listed.shape[0], self.vocab_size), float("-inf"))
            out[:, self.token_ids] = listed
            return out

        logits = self.logits_processor(lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits
        base = torch.arange(logits.shape[-1], device=logits.device)
        targets = base + self.draft_id_to_target_id
        out = logits.new_full((logits.shape[0], self.vocab_size), float("-inf"))
        out[:, targets] = logits
        return out

    def get_top_tokens(
        self, lm_head: VocabParallelEmbedding, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Greedy target token ids without materializing full logits."""
        if self.token_ids is not None:
            listed, picked = self._token_map_logits(hidden_states)
            listed_max, listed_col = listed.max(dim=-1)
            top = self.token_ids[listed_col]
            if picked is None:
                return top
            logits, picks = picked
            picked_max, picked_col = logits.max(dim=-1)
            pick = picks.gather(1, picked_col.unsqueeze(1)).squeeze(1)
            # Ties go to the listed row, as an argmax over [listed, picked] would.
            return torch.where(listed_max >= picked_max, top, self.candidate_ids[pick])

        top = self.logits_processor.get_top_tokens(lm_head, hidden_states)
        return self.map_draft_to_target(top)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Target ids of draft-vocabulary ids from the pruned lm_head."""
        if self.draft_id_to_target_id is None:
            return draft_ids
        return draft_ids + self.draft_id_to_target_id[draft_ids]

    def load_token_map(
        self,
        lm_head: VocabParallelEmbedding,
        token_ids: torch.Tensor,
        dynamic_rows: int = 0,
    ) -> None:
        """Restrict drafting to the `token_ids` rows of `lm_head`.

        The rows are copied out of `lm_head`, so the drafter keeps sharing it
        with the target, which verifies with the full vocabulary.

        Args:
            lm_head: The drafter's full-vocabulary lm_head.
            token_ids: Sorted, unique target ids of the listed rows.
            dynamic_rows: Rows outside the list picked per draft token.

        """
        head = getattr(lm_head, "base_layer", lm_head)
        lp = self.logits_processor
        if (
            self.draft_id_to_target_id is not None
            or not isinstance(head, VocabParallelEmbedding)
            or not isinstance(head.quant_method, _UNQUANTIZED)
            or getattr(head, "bias", None) is not None
            or lp.head_dtype not in (None, head.weight.dtype)
            or lp.scale != 1.0
            or lp.soft_cap is not None
        ):
            raise ValueError(
                "draft_token_map needs a full-vocabulary, unquantized lm_head "
                "without bias, logit scaling or soft capping, run in its weight "
                "dtype."
            )
        if head.tp_size > 1:
            raise ValueError("draft_token_map does not support tensor parallelism.")

        vocab_size = head.org_vocab_size
        bad = token_ids[(token_ids < 0) | (token_ids >= vocab_size)]
        if bad.numel() > 0:
            raise ValueError(
                f"draft_token_map has {bad.numel()} ids outside [0, {vocab_size}), "
                f"e.g. {bad[:5].tolist()}."
            )

        weight = head.weight[:vocab_size]
        self.token_ids = token_ids.to(weight.device)
        self.rows = _Rows(weight[self.token_ids])
        self.num_dynamic_rows = dynamic_rows
        if dynamic_rows > 0:
            self._init_dynamic_rows(weight)
        logger.info(
            "Draft vocabulary: %d of %d tokens plus %d dynamic rows per draft token.",
            token_ids.numel(),
            vocab_size,
            dynamic_rows,
        )

    def _init_dynamic_rows(self, weight: torch.Tensor):
        """Score the rows outside the list with a rank-256 lm_head factor.

        The basis is the top right singular vectors of the lm_head (the top
        eigenvectors of `W^T W`), so
        `(x @ basis) @ (weight[i] @ basis)` approximates the logit of row `i`.
        The best `num_dynamic_rows` per token then get exact logits.
        """
        vocab_size, hidden_size = weight.shape
        assert self.token_ids is not None
        is_listed = torch.zeros(vocab_size, dtype=torch.bool, device=weight.device)
        is_listed[self.token_ids] = True
        self.candidate_ids = (~is_listed).nonzero()[:, 0]
        if self.num_dynamic_rows > self.candidate_ids.numel():
            raise ValueError(
                "draft_token_map_dynamic_rows must be at most "
                f"{self.candidate_ids.numel()}, got {self.num_dynamic_rows}."
            )

        gram = weight.new_zeros(hidden_size, hidden_size, dtype=torch.float32)
        for start in range(0, vocab_size, _CHUNK):
            block = weight[start : start + _CHUNK].float()
            gram.addmm_(block.t(), block)
        basis = torch.linalg.eigh(gram).eigenvectors[:, -_DYNAMIC_RANK:].flip(-1)
        del gram
        self.basis = basis.to(weight.dtype).contiguous()
        scores = torch.cat(
            [
                (weight[ids].float() @ basis).to(weight.dtype)
                for ids in self.candidate_ids.split(_CHUNK)
            ]
        )
        self.scorer = _Rows(scores)
        self.weight = weight

    def _token_map_logits(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Logits of the listed rows, and of the picked rows with the picks as
        positions in `candidate_ids`."""
        assert self.rows is not None
        logits = self.rows(hidden_states)
        if self.num_dynamic_rows == 0:
            return logits, None
        scores = self.scorer(hidden_states @ self.basis)
        picks = scores.topk(self.num_dynamic_rows, dim=-1, sorted=False).indices
        picked = gather_rows_gemv(hidden_states, self.weight, self.candidate_ids[picks])
        return logits, (picked, picks)


def load_draft_token_map(model: nn.Module, vllm_config: "VllmConfig") -> None:
    """Restrict a drafter's `DraftVocab` to `speculative_config.draft_token_map`.

    EOS ids of the target are always included.
    """
    spec = vllm_config.speculative_config
    assert spec is not None and spec.draft_token_map is not None
    draft_vocab = getattr(model, "draft_vocab", None)
    if not isinstance(draft_vocab, DraftVocab):
        raise ValueError(f"draft_token_map is not supported by {type(model).__name__}.")
    model_config = vllm_config.model_config
    eos_ids: list[int] = []
    for eos in (
        getattr(model_config.hf_text_config, "eos_token_id", None),
        model_config.try_get_generation_config().get("eos_token_id"),
    ):
        if eos is not None:
            eos_ids.extend(eos if isinstance(eos, (list, tuple)) else [eos])
    draft_vocab.load_token_map(
        model.lm_head,
        load_draft_token_ids(spec.draft_token_map, eos_ids),
        spec.draft_token_map_dynamic_rows,
    )
