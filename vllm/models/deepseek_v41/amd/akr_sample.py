# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route `markov`: DSpark's sequential Markov draft steps (Markov bias GEMV,
Gumbel sample, draft-logits cache write) for every request in one amd_kernels
launch, in place of about nine launches per step.

Its workgroups wait for each other, so it needs every CU: it is the last launch
of the draft step and nothing runs beside it.
"""

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v41.amd import akr_config, akr_hooks, akr_kernels

logger = init_logger(__name__)


def unsupported_reason(spec, model, base_logits: torch.Tensor) -> str | None:
    """Why the kernel cannot serve this speculator and batch (host checks)."""
    if not getattr(spec, "_fuse_markov_bias", False):
        return "the Markov bias is not fusable for this speculator"
    if spec.use_confidence_head:
        return "the confidence head needs the per-step Markov embeddings"
    if getattr(spec, "_draft_topk", None) is not None:
        return "top-k draft vocabulary"
    if getattr(spec, "_d2t_scatter_index", None) is not None:
        return "reduced draft vocabulary"
    if getattr(spec, "use_fp64_gumbel", False):
        return "fp64 Gumbel"
    if not spec.sample_from_anchor:
        return "drafts not sampled from the anchor"
    cache = spec.draft_logits
    if cache is None or cache.row_mapping is not None:
        return "no replicated draft-logits cache"
    if cache.values.dtype != torch.bfloat16 or not cache.values.is_contiguous():
        return "the draft-logits cache is not contiguous bf16"
    if base_logits.dtype != torch.bfloat16 or base_logits.stride(-1) != 1:
        return "the base logits are not bf16 rows"
    head = model.model.markov_head
    for w in (head.markov_w1.weight, head.markov_w2.weight):
        if w.dtype != torch.bfloat16 or not w.is_contiguous():
            return "the Markov weights are not contiguous bf16"
    return None


def markov_chain(
    spec,
    model,
    base_logits: torch.Tensor,
    prev: torch.Tensor,
    idx_map: torch.Tensor,
    sample_pos: torch.Tensor,
    out_tokens: torch.Tensor,
    prev_indices: torch.Tensor | None,
) -> torch.Tensor | None:
    """All Markov steps for `base_logits` `[n, steps, V]`; the drafts land in
    `out_tokens`. Returns an empty tensor (no confidence head) when done, or
    None, having done nothing, for the speculator's loop."""
    if prev_indices is None or not akr_config.serves("markov", base_logits.shape[0]):
        return None
    reason = unsupported_reason(spec, model, base_logits)
    if reason is not None:
        logger.info_once("amd_kernels markov: stock path (%s)", reason)
        return None
    head = model.model.markov_head
    if not akr_kernels.markov_chain_ok(head.markov_w1.weight, idx_map):
        return None
    n, steps, vocab = base_logits.shape
    akr_kernels.markov_chain(
        base_logits.reshape(n * steps, vocab),
        head.markov_w1.weight.data,
        head.markov_w2.weight.data,
        prev,
        prev_indices,
        idx_map,
        sample_pos,
        spec.temperature,
        spec.seeds,
        spec.draft_logits.values,
        out_tokens,
    )
    # a row without a finite logit gets an out-of-vocabulary id from some
    # kernel versions; keep the embedding lookups downstream in range
    out_tokens[:n].clamp_(max=vocab - 1)
    return base_logits.new_empty(0)


def post_load(model) -> None:
    from vllm.config import get_current_vllm_config

    if not akr_hooks.is_draft(model):
        return
    spec = get_current_vllm_config().speculative_config
    w1 = model.model.markov_head.markov_w1.weight
    if spec is not None and akr_kernels.markov_chain_ok(
        w1, w1.new_empty(1, spec.num_speculative_tokens, dtype=torch.int64)
    ):
        akr_kernels.markov_chain_warmup(w1, spec.num_speculative_tokens)
