# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up speculative-decoding rejection kernels."""

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

    rejection_method = getattr(spec_config, "rejection_sample_method", None)
    use_block_verification = rejection_method == "block"
    use_synthetic = rejection_method == "synthetic"
    # HAS_DRAFT_LOGITS is a constexpr, so the warmup must match the runtime.
    use_draft_logits = (
        getattr(spec_config, "draft_sample_method", "greedy") == "probabilistic"
    )

    device = worker.device
    num_reqs = 1
    tokens_per_req = num_spec + 1
    num_logits = num_reqs * tokens_per_req

    watermark_config = getattr(worker.vllm_config, "watermark_config", None)
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
        # Tensor dtypes and integer widths affect Triton specialization.
        watermark_kwargs = {
            "contexts": torch.zeros(
                (num_logits, watermark_config.context_width),
                dtype=torch.int32,
                device=device,
            ),
            "watermarking": torch.zeros(num_reqs, dtype=torch.bool, device=device),
            "watermark_key": watermark_key,
        }

    # Sampling-parameter processing may promote either distribution to fp32.
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
                draft_sampled=torch.zeros(num_logits, dtype=torch.int32, device=device),
                cu_num_logits=torch.tensor(
                    [0, num_logits], dtype=torch.int32, device=device
                ),
                pos=torch.zeros(num_logits, dtype=torch.int64, device=device),
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
