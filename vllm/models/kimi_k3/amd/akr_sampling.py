# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The amd_kernels `sampling` route outside the lm_head exchange (entries
`sampling/top_p_mask_common` and `sampling/markov_gumbel_common`; each call the entry's
`*_unsupported_reason` rejects stays on the stock path):

* `top_p_mask` through `topk_topp_triton.split_top_p_override` (switch `TOPP`): the
  seven-launch Triton split top-p chain in one launch, bit for bit the chain's masking,
  for the rows / vocabularies the entry serves with mask value -inf; else the chain
  runs.
* `markov_gumbel_fused` for `DSparkSpeculator._sample_sequential` (switch `MARKOV`):
  all k steps of the Markov bias GEMV + add + Gumbel argmax in one persistent launch of
  co-resident workgroups (nothing else may hold CUs of the device while it runs: the
  draft graph is single-stream), up to `markov_max_rows` requests, one zeroed scratch
  buffer per device (allocated in `setup_speculator`, never inside a capture). It
  writes `draft_tokens` and the draft logits cache (honouring
  `DraftLogitsCache.row_mapping`) exactly where vLLM's loop does, so both the
  `synthetic` and the `block` rejection samplers read the same cache.
"""

import torch

from vllm.logger import init_logger
from vllm.models.kimi_k3.amd import akr_config, akr_nonlayer

logger = init_logger(__name__)


# ---------------------------------------------------------------- top-p
def install_top_p(vocab: int) -> bool:
    """`vocab`: the sampled logits' width (the lm_head's unpadded vocabulary)."""
    if not akr_nonlayer.switch("TOPP"):
        return False
    from vllm.v1.sample.ops import topk_topp_triton

    try:
        from amd_kernels.sampling.top_p_mask_common import (
            available,
            top_p_mask,
            top_p_mask_shapes,
            top_p_mask_unsupported_reason,
        )
    except ImportError:
        return False
    if akr_nonlayer.blocked(
        "TOPP", None if available() else "needs gfx950 with FlyDSL"
    ):
        return False
    if akr_nonlayer.blocked(
        "TOPP",
        None if (vocab,) in top_p_mask_shapes() else f"no top_p_mask for vocab {vocab}",
    ):
        return False

    def override(logits: torch.Tensor, p: torch.Tensor, mask_value: float) -> bool:
        if mask_value != float("-inf") or top_p_mask_unsupported_reason(logits, p):
            return False
        top_p_mask(logits, p)
        return True

    topk_topp_triton.split_top_p_override = override
    logger.info(
        "amd_kernels sampling: top_p_mask replaces the Triton split top-p chain"
    )
    return True


# ---------------------------------------------------------------- Markov chain
def _markov_reason(spec) -> str | None:
    """vLLM-side gates: the drafting modes the fused chain reproduces."""
    model = spec.model
    head = getattr(getattr(model, "model", None), "markov_head", None)
    if head is None:
        return "draft has no Markov head"
    if spec._draft_topk is not None:
        return "top-k Markov drafting"
    if spec.use_confidence_head or spec.acceptance_estimator is not None:
        return "confidence head / acceptance estimator needs the per-step logits"
    if spec.draft_watermarker is not None or spec.use_fp64_gumbel:
        return "watermarking / fp64 Gumbel noise"
    if spec._d2t_scatter_index is not None or hasattr(
        model, "apply_output_logit_transform"
    ):
        return "reduced draft vocabulary or output logit transform"
    if spec.draft_logits is None:
        return "no draft logits cache (greedy drafting)"
    return None


def _markov_args(spec, base: torch.Tensor, n: int, scratch: torch.Tensor) -> tuple:
    """`markov_gumbel_fused`'s arguments of an `n`-request step over `base`
    [n, k, V]."""
    head = spec.model.model.markov_head
    num_sample = n * spec.num_speculative_steps
    return (
        base,
        head.markov_w1.weight,
        head.markov_w2.weight,
        spec.input_buffers.input_ids,
        spec._anchor_idx,
        spec.draft_tokens,
        spec.sample_idx_mapping[:num_sample].view(n, -1),
        spec.sample_pos[:num_sample].view(n, -1),
        spec.temperature,
        spec.seeds,
        spec.draft_logits.values,
        scratch,
        spec.draft_logits.row_mapping,
    )


def setup_speculator(spec) -> None:
    """From `DSparkSpeculator.load_model`, before any capture: allocate the scratch
    and compile the chain's variants (warm-up entry) with private dummy buffers."""
    spec._akr_markov = None
    if not akr_nonlayer.switch("MARKOV"):
        return
    if akr_nonlayer.blocked("MARKOV", _markov_reason(spec)):
        return
    from amd_kernels.sampling.markov_gumbel_common import (
        available,
        markov_gumbel_fused,
        markov_gumbel_fused_unsupported_reason,
        markov_max_rows,
        markov_scratch_dwords,
    )

    if akr_nonlayer.blocked(
        "MARKOV", None if available() else "needs gfx950 with FlyDSL"
    ):
        return
    dev = spec.device
    k = spec.num_speculative_steps
    head = spec.model.model.markov_head
    w1 = head.markov_w1.weight
    vocab = w1.shape[0]
    if akr_nonlayer.blocked(
        "MARKOV", None if spec.vocab_size == vocab else f"vocab {spec.vocab_size}"
    ):
        return
    reason = None
    try:
        max_rows = markov_max_rows(w1)
    except ValueError as e:
        reason = str(e)
    if akr_nonlayer.blocked("MARKOV", reason):
        return
    scratch = torch.zeros(
        markov_scratch_dwords(w1, max_rows), dtype=torch.int32, device=dev
    )
    # the entry's reason on this speculator's buffers at its largest step
    rows = min(max_rows, spec.max_num_reqs)
    base = torch.empty(rows, k, vocab, dtype=torch.bfloat16, device=dev)
    reason = markov_gumbel_fused_unsupported_reason(
        *_markov_args(spec, base, rows, scratch)
    )
    del base
    if akr_nonlayer.blocked("MARKOV", reason):
        return
    # warm-up: the first call of the mode compiles every row-tile variant; a private
    # zero batch with invalid rows (idx_mapping < 0) and its own scratch / cache
    n = min(16, max_rows)
    zi = lambda *s: torch.zeros(*s, dtype=torch.int64, device=dev)  # noqa: E731
    markov_gumbel_fused(
        torch.zeros(n, k, vocab, dtype=torch.bfloat16, device=dev),
        w1,
        head.markov_w2.weight,
        spec.input_buffers.input_ids,
        zi(n),
        zi(n, k),
        torch.full((n, k), -1, dtype=torch.int64, device=dev),
        zi(n, k),
        spec.temperature,
        spec.seeds,
        torch.zeros(1, k, vocab, dtype=torch.bfloat16, device=dev),
        torch.zeros(markov_scratch_dwords(w1, n), dtype=torch.int32, device=dev),
    )
    torch.accelerator.synchronize(dev)
    spec._akr_markov = (
        markov_gumbel_fused,
        markov_gumbel_fused_unsupported_reason,
        scratch,
        max_rows,
    )
    logger.info("amd_kernels sampling: fused Markov chain (k=%d, n<=%d)", k, max_rows)


def markov_sequential(spec, num_reqs: int, head_hidden: torch.Tensor) -> bool:
    """`DSparkSpeculator._sample_sequential`'s loop on the fused chain. False, having
    done nothing, when this step is not the kernel's case."""
    state = getattr(spec, "_akr_markov", None)
    if state is None:
        return False
    fused, reason_of, scratch, max_rows = state
    if not 1 <= num_reqs <= max_rows:
        return False
    k = spec.num_speculative_steps
    num_sample = num_reqs * k
    sample_hidden = head_hidden[spec.sample_indices[:num_sample]]
    base = spec.model.compute_draft_logits(sample_hidden)
    args = None
    if base.dim() == 2 and base.shape[0] == num_sample and base.is_contiguous():
        args = _markov_args(spec, base.view(num_reqs, k, -1), num_reqs, scratch)
    if args is None or akr_config.call_unsupported("sampling", reason_of(*args)):
        return _stock_tail(spec, num_reqs, sample_hidden, base)
    fused(*args)
    return True


def _stock_tail(
    spec, num_reqs: int, sample_hidden: torch.Tensor, base: torch.Tensor
) -> bool:
    """The base logits were already computed (not in the kernel's layout): run vLLM's
    loop on them (the part of `_sample_sequential` after `compute_draft_logits`)."""
    k = spec.num_speculative_steps
    base = base.view(num_reqs, k, -1)
    num_sample = num_reqs * k
    idx_map = spec.sample_idx_mapping[:num_sample].view(num_reqs, k)
    sample_pos = spec.sample_pos[:num_sample].view(num_reqs, k)
    prev = spec.input_buffers.input_ids[spec._anchor_idx[:num_reqs]]
    for i in range(k):
        bias = spec.model.markov_bias(spec.model.markov_embed(prev))
        sampled = spec._sample_logits(
            base[:, i] + bias, idx_map[:, i], sample_pos[:, i], i
        )
        spec.draft_tokens[:num_reqs, i] = sampled
        prev = sampled
    return True
