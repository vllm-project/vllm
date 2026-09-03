# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up spec-decode rejection-sampler Triton kernels.

The rejection sampler kernels (``_compute_local_logits_stats_kernel``,
``_rejection_kernel``, ``_resample_kernel``) are JIT-compiled by Triton on
first use. Without warmup, the first spec-decode request pays a multi-second
compilation cost. This pre-compiles them with dummy data matching the
server's vocab size and speculative config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    device = torch.device("cuda")
    num_reqs = 1
    tokens_per_req = num_spec + 1
    num_logits = num_reqs * tokens_per_req

    # Triton JIT-specializes on tensor dtypes. The target logits may be fp32
    # (apply_sampling_params copies to fp32 when processing is needed) or the
    # model dtype (pass-through otherwise), while draft logits are always the
    # model dtype. Warm every (target, draft) combination the runtime can hit.
    model_dtype = model_config.dtype
    warmup_dtype_pairs = {
        (model_dtype, model_dtype),
        (torch.float32, torch.float32),
        (torch.float32, model_dtype),
        (model_dtype, torch.float32),
    }

    logger.info(
        "Warming up spec-decode rejection sampler kernels "
        "(vocab=%d, num_spec=%d, dtype_pairs=%s, block_verify=%s).",
        vocab_size,
        num_spec,
        [(str(t), str(d)) for t, d in warmup_dtype_pairs],
        use_block_verification,
    )
    for tgt_dtype, draft_dtype in warmup_dtype_pairs:
        target_logits = torch.zeros(
            (num_logits, vocab_size), dtype=tgt_dtype, device=device
        )
        draft_logits = torch.zeros(
            (num_reqs, num_spec, vocab_size), dtype=draft_dtype, device=device
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
                draft_sampled=torch.zeros(num_logits, dtype=torch.int64, device=device),
                cu_num_logits=torch.tensor(
                    [0, num_logits], dtype=torch.int32, device=device
                ),
                pos=torch.zeros(num_logits, dtype=torch.int64, device=device),
                idx_mapping=torch.zeros(num_reqs, dtype=torch.int32, device=device),
                expanded_idx_mapping=torch.zeros(
                    num_logits, dtype=torch.int32, device=device
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
            )
        except Exception:
            logger.warning(
                "Skipping spec-decode rejection sampler warmup.", exc_info=True
            )
            return
