# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark's Markov draft sampling as one AITER FlyDSL launch on gfx950.

AITER's `flydsl_markov_chain_sample` runs all 5 draft steps of up to 160
requests in one launch, with vLLM's Gumbel draws (`gumbel_sample` with
`is_drafting=True`), and writes the same `bf16(base + bias)` draft logits
cache rows up to the last bf16 bit of the bias, which it sums in another order
than vLLM's GEMV. The kernel's workgroups wait for each other, so it is the
last launch of the draft step, on the current stream.
"""

from types import ModuleType

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _why_not(spec, model, flydsl: ModuleType) -> str | None:
    """Returns why `spec` keeps vLLM's Markov loop, or None."""
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
    """Decides once per speculator whether the AITER op samples its drafts.

    Compiles the kernels and allocates their workspace, so it must run outside
    CUDA graph capture.

    Args:
        spec: The `DSparkSpeculator`, after `load_model`.
        model: Its DeepSeek-V4.1 draft model.

    Returns:
        The AITER op module, or None to keep vLLM's loop.

    """
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
    """Runs `DSparkSpeculator._sample_markov_chain` in one launch.

    Checks only host-side metadata of the per-call tensors, so each captured
    row count records its path. The speculator's persistent buffers (input
    ids, maps, positions, temperature, seeds, draft tokens) are not
    re-checked: the op raises `ValueError` if their layout ever changes.

    Args:
        flydsl: The module `resolve` returned.
        spec: The `DSparkSpeculator`.
        model: Its draft model.
        base_logits: `[rows, steps, vocab]` draft logits before the bias.
        prev: Input ids; step 0's previous token is `prev[prev_indices]`.
        idx_map: `[rows, steps]` request-state slots.
        sample_pos: `[rows, steps]` positions of the predicted tokens.
        out_tokens: `[rows, steps]` draft tokens to write.
        prev_indices: Anchor indices into `prev`.

    Returns:
        False, having done nothing, if the op does not take this call.

    """
    rows, steps, vocab = base_logits.shape
    if (
        prev_indices is None
        or prev_indices.dtype != torch.int64
        or base_logits.dtype != torch.bfloat16
        or base_logits.stride(2) != 1
        or base_logits.stride(1) % 2
        or base_logits.stride(0) != steps * base_logits.stride(1)
        or base_logits.data_ptr() % 8
        or rows * base_logits.stride(0) * 2 >= 1 << 31
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
