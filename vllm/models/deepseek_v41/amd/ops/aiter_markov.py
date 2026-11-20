# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark's Markov draft sampling as one AITER FlyDSL launch on gfx950.

`DSparkSpeculator._sample_markov_chain` runs, for each of the 5 draft steps,
the Markov embedding of the previous token, the bias GEMV over the vocabulary,
the Gumbel sample with the bias fused in and the token copy. AITER's
`flydsl_markov_chain_sample` runs all 5 steps of up to 160 requests in one
persistent launch. Its Gumbel draws are vLLM's (`gumbel_sample` with
`is_drafting=True`, fp32), and it writes the same `bf16(base + bias)` rows
into the draft logits cache. The bias is summed in another order than
vLLM's GEMV, so a value can differ in its last bf16 bit, as in
fused_markov.py. That changes only which tokens the draft proposes, because
the rejection sampler verifies every draft token against the target.

Enabled by VLLM_ROCM_USE_AITER=1 and VLLM_ROCM_USE_AITER_FLYDSL_MARKOV_SAMPLER=1.
`resolve` decides once per speculator from its load-time state and compiles
the kernels, before any CUDA graph capture. `sample_markov_chain` then checks
only host-side tensor metadata, so every captured row count records the
path it takes.

The kernel's workgroups wait for each other, so it needs the GPU's CUs to
itself: it is the last launch of the draft step, on the current stream.
"""

from types import ModuleType

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _why_not(spec, model, flydsl: ModuleType) -> str | None:
    """Why this speculator keeps vLLM's Markov loop, or None."""
    if spec.draft_logits is None:
        return "greedy drafts"
    if spec.draft_logits.row_mapping is not None:
        return "a batch-sharded draft logits cache"
    if spec._draft_topk is not None:
        return "top-k Markov sampling"
    if not spec.sample_from_anchor:
        return "fill-in draft blocks (sample_from_anchor=False)"
    if hasattr(model, "apply_output_logit_transform"):
        return "an output logit transform"
    lp = model.logits_processor
    if lp.scale != 1.0 or lp.soft_cap is not None:
        return "a scaled or soft-capped Markov bias"
    if lp.head_dtype not in (None, torch.bfloat16):
        return f"a {lp.head_dtype} Markov bias"
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )

    head = model.model.markov_head
    if not isinstance(head.markov_w2.quant_method, UnquantizedEmbeddingMethod):
        return "a quantized Markov head"
    d2t = spec._d2t_scatter_index
    return flydsl.markov_chain_sample_unsupported_reason(
        head.markov_w1.weight,
        head.markov_w2.weight,
        spec.draft_logits.values,
        num_speculative_steps=spec.num_speculative_steps,
        use_fp64_gumbel=spec.use_fp64_gumbel,
        draft_vocab_size=spec.vocab_size if d2t is None else d2t.numel(),
        has_confidence_head=spec.use_confidence_head,
        has_draft_watermarker=spec.draft_watermarker is not None,
        has_acceptance_estimator=spec.acceptance_estimator is not None,
    )


def resolve(spec, model) -> ModuleType | None:
    """The AITER op module if it runs this speculator's Markov sampling, else
    None. Compiles the kernels and allocates their workspace, so it must run
    outside CUDA graph capture."""
    if not current_platform.is_rocm():
        return None
    from vllm._aiter_ops import rocm_aiter_ops

    flydsl = rocm_aiter_ops.flydsl_markov_sampler()
    if flydsl is None:
        return None
    why = _why_not(spec, model, flydsl)
    if why is not None:
        logger.info_once(
            "AITER FlyDSL Markov sampler off: %s. Using vLLM's Markov loop.", why
        )
        return None
    flydsl.warmup_markov_chain_sample(
        spec.device,
        idx_dtype=spec.sample_idx_mapping.dtype,
        ids_dtype=spec.input_buffers.input_ids.dtype,
    )
    logger.info_once("Using the AITER FlyDSL DSpark Markov sampler.")
    return flydsl


def sample_markov_chain(
    flydsl: ModuleType,
    spec,
    model,
    base_logits: torch.Tensor,
    prev: torch.Tensor,
    idx_map: torch.Tensor,
    sample_pos: torch.Tensor,
    out_tokens: torch.Tensor,
    prev_indices: torch.Tensor | None,
) -> bool:
    """`DSparkSpeculator._sample_markov_chain` for `base_logits`
    `[rows, steps, vocab]` in one launch: writes `out_tokens` and the draft
    logits cache. Returns False, having done nothing, for a call the kernel
    does not take."""
    rows, steps, vocab = base_logits.shape
    if (
        prev_indices is None
        or base_logits.dtype != torch.bfloat16
        or base_logits.stride(2) != 1
        or base_logits.stride(1) % 2
        or base_logits.stride(0) != steps * base_logits.stride(1)
    ):
        return False
    head = model.model.markov_head
    flydsl.flydsl_markov_chain_sample(
        base_logits.view(rows * steps, vocab),
        head.markov_w1.weight,
        head.markov_w2.weight,
        prev,
        prev_indices,
        idx_map,
        sample_pos,
        spec.temperature,
        spec.seeds,
        spec.draft_logits.values,
        out_tokens,
    )
    return True
