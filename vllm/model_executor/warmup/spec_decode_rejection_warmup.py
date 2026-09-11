# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up spec-decode rejection-sampler Triton kernels.

The rejection sampler kernels (``_compute_local_logits_stats_kernel``,
``_rejection_kernel``, ``_resample_kernel``) are JIT-compiled by Triton on
first use. Without warmup, the first spec-decode request pays a multi-second
compilation cost. This pre-compiles them with dummy data matching the
server's vocab size, speculative config and watermark config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


@torch.inference_mode()
def spec_decode_rejection_warmup(worker: Worker) -> None:
    spec_config = worker.vllm_config.speculative_config
    if spec_config is None:
        return

    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

    model_config = worker.vllm_config.model_config
    vocab_size = model_config.get_vocab_size()
    num_spec = spec_config.num_speculative_tokens
    if num_spec <= 0 or vocab_size <= 0:
        return

    # Mirror the constexpr-relevant flags the runtime uses.
    rejection_method = getattr(spec_config, "rejection_sample_method", None)
    use_block_verification = rejection_method == "block"
    use_synthetic = rejection_method == "synthetic"
    # ``HAS_DRAFT_LOGITS`` is a constexpr of all three kernels. ``Speculator``
    # allocates ``draft_logits`` only for ``draft_sample_method`` ==
    # "probabilistic"; with the default "greedy" the runtime passes
    # ``draft_logits=None``, so warming a real tensor compiles a specialization
    # no such engine launches.
    use_draft_logits = (
        getattr(spec_config, "draft_sample_method", "greedy") == "probabilistic"
    )

    device = worker.device
    num_reqs = 1
    tokens_per_req = num_spec + 1
    num_logits = num_reqs * tokens_per_req

    # ``WATERMARK`` and ``CONTEXT_WIDTH`` are constexprs of ``_resample_kernel``,
    # so a watermarked engine launches a different specialization than the stock
    # one. Warm the variant the runtime will actually use: RejectionSampler
    # passes the watermark triple when a watermark config is present, so
    # the unwatermarked variant is never launched in that case.
    watermark_config = worker.vllm_config.watermark_config
    watermark_kwargs: dict[str, Any] = {}
    if watermark_config is not None:
        from vllm.v1.watermarking.spec_decode import speculative_target_watermark_key

        try:
            watermark_key = speculative_target_watermark_key(watermark_config)
        except Exception:
            logger.warning(
                "Skipping spec-decode rejection sampler warmup: could not derive "
                "the watermark recovery key.",
                exc_info=True,
            )
            return
        # Contexts are int32 [num_logits, context_width] and contiguous, matching
        # GPUWatermarkSampler._get_contexts, which reads the int32 request-state
        # token ids (the pointer dtype and the row stride are part of the
        # specialization key). The key halves are do_not_specialize'd, which drops
        # the divisibility hints but not the i32/i64 type Triton infers from an
        # int argument's magnitude, so warm the key the engine actually uses.
        watermark_kwargs = {
            "contexts": torch.zeros(
                (num_logits, watermark_config.context_width),
                dtype=torch.int32,
                device=device,
            ),
            "watermarking": torch.zeros(num_reqs, dtype=torch.bool, device=device),
            "watermark_key": watermark_key,
        }

    # Triton JIT-specializes on tensor dtypes. The target logits may be fp32
    # (apply_sampling_params copies to fp32 when processing is needed) or the
    # model dtype (pass-through otherwise), while draft logits are always the
    # model dtype. Warm every (target, draft) combination the runtime can hit;
    # ``None`` means the runtime launches the HAS_DRAFT_LOGITS=False variant,
    # for which the draft dtype does not exist.
    model_dtype = model_config.dtype
    warmup_dtype_pairs: set[tuple[torch.dtype, torch.dtype | None]]
    if use_draft_logits:
        warmup_dtype_pairs = {
            (model_dtype, model_dtype),
            (torch.float32, torch.float32),
            (torch.float32, model_dtype),
            (model_dtype, torch.float32),
        }
    else:
        warmup_dtype_pairs = {(model_dtype, None), (torch.float32, None)}

    logger.info(
        "Warming up spec-decode rejection sampler kernels "
        "(vocab=%d, num_spec=%d, dtype_pairs=%s, block_verify=%s, watermark=%s).",
        vocab_size,
        num_spec,
        [(str(t), str(d)) for t, d in warmup_dtype_pairs],
        use_block_verification,
        watermark_config is not None,
    )
    for tgt_dtype, draft_dtype in warmup_dtype_pairs:
        target_logits = torch.zeros(
            (num_logits, vocab_size), dtype=tgt_dtype, device=device
        )
        draft_logits = (
            torch.zeros(
                (num_reqs, num_spec, vocab_size), dtype=draft_dtype, device=device
            )
            if draft_dtype is not None
            else None
        )
        synthetic_rates = (
            torch.full((num_spec,), 0.5, dtype=torch.float32, device=device)
            if use_synthetic
            else None
        )
        try:
            rejection_sample(
                target_logits=target_logits,
                draft_logits=draft_logits,
                # ``draft_sampled`` is a slice of ``InputBatch.input_ids``, which
                # is int32; ``pos`` slices ``InputBatch.positions``, which is int64.
                draft_sampled=torch.zeros(num_logits, dtype=torch.int32, device=device),
                cu_num_logits=torch.tensor(
                    [0, num_logits], dtype=torch.int32, device=device
                ),
                pos=torch.zeros(num_logits, dtype=torch.int64, device=device),
                # ``idx_mapping`` is built from an ``np.intp`` array (and from
                # ``torch.arange(..., dtype=torch.int64)`` for dummy batches), so
                # it is int64; ``expanded_idx_mapping`` is either that same tensor
                # or ``idx_mapping.new_empty(...)``, hence int64 too.
                # ``expanded_local_pos`` is separately allocated as int32.
                idx_mapping=torch.zeros(num_reqs, dtype=torch.int64, device=device),
                expanded_idx_mapping=torch.zeros(
                    num_logits, dtype=torch.int64, device=device
                ),
                expanded_local_pos=torch.arange(
                    num_logits, dtype=torch.int32, device=device
                ),
                temperature=torch.zeros(num_reqs, dtype=torch.float32, device=device),
                seed=torch.full((num_reqs,), 42, dtype=torch.int64, device=device),
                num_speculative_steps=num_spec,
                synthetic_conditional_rates=synthetic_rates,
                use_fp64=False,
                use_block_verification=use_block_verification,
                **watermark_kwargs,
            )
        except Exception:
            logger.warning(
                "Skipping spec-decode rejection sampler warmup.", exc_info=True
            )
            return
