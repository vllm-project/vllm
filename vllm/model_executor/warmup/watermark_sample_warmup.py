# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the watermarked sampler's Triton kernels.

Watermark kernels are JIT-compiled once per specialization, which includes the
logits dtype and whether a skip mask is passed. The generic sampler warmup
samples with ``SamplingParams.for_sampler_warmup()``, whose feature-heavy logits
processing forces the fp32 copy in ``apply_sampling_params``, so the first
ordinary temperature-1.0 request is the first launch with model-dtype logits and
pays the compilation inside inference. This samples through the sampler's
watermarker once per logits dtype and skip-mask variant the sampler path can
pass, which launches whatever kernels the configured algorithm uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_NUM_WARMUP_TOKENS = 1


@torch.inference_mode()
def watermark_sample_warmup(worker: Worker) -> None:
    watermark_config = getattr(worker.vllm_config, "watermark_config", None)
    if watermark_config is None:
        return
    # Watermarkers launch their kernels only on CUDA; elsewhere they take the
    # torch path and there is nothing to compile.
    if not current_platform.is_cuda_alike():
        return

    from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
    from vllm.v1.watermarking.watermarker import RandomSampler

    # Sample through the sampler the runner built instead of rebuilding it from
    # config: its watermarker (the target role under a speculative config), its
    # context deduplication setting and its fp64 flag are what the launches have
    # to match, and cannot drift from the runner. A rank that does not sample --
    # a non-final pipeline stage, a pooling model -- holds no such sampler.
    sampler = getattr(worker.model_runner, "sampler", None)
    if not isinstance(sampler, GPUWatermarkSampler):
        return

    model_config = worker.vllm_config.model_config
    device = worker.device
    try:
        # The sampler passes model-dtype logits through when no logits
        # processing is needed and fp32 otherwise. ``ModelConfig.dtype`` is
        # resolved to a torch dtype during validation.
        dtypes = {cast("torch.dtype", model_config.dtype), torch.float32}
        # ``GPUWatermarkSampler._sample_random`` builds a skip mask whenever
        # context deduplication is on; with it off only a batch that mixes in an
        # opted-out or greedy row does, and the mask-free specialization stays
        # reachable.
        with_skip_mask = (
            (True,) if sampler.deduplicate_contexts != "none" else (True, False)
        )

        vocab_size = model_config.get_vocab_size()
        # Dtypes mirror the runtime buffers the sampler passes and are part of
        # the specialization key: int32 request-state token ids, int64 index and
        # seed buffers, fp32 sampling temperatures.
        contexts = torch.zeros(
            (_NUM_WARMUP_TOKENS, sampler.watermarker.context_width),
            dtype=torch.int32,
            device=device,
        )
        skip_mask = torch.zeros(_NUM_WARMUP_TOKENS, dtype=torch.bool, device=device)
        random_sampler = RandomSampler(
            expanded_idx_mapping=torch.zeros(
                _NUM_WARMUP_TOKENS, dtype=torch.int64, device=device
            ),
            temperatures=torch.ones(
                _NUM_WARMUP_TOKENS, dtype=torch.float32, device=device
            ),
            seeds=torch.zeros(_NUM_WARMUP_TOKENS, dtype=torch.int64, device=device),
            positions=torch.zeros(_NUM_WARMUP_TOKENS, dtype=torch.int64, device=device),
            use_fp64=sampler.use_fp64_gumbel,
        )

        logger.info(
            "Warming up watermark sampler kernels (algorithm=%s, vocab=%d, "
            "dtypes=%s, skip_mask=%s).",
            watermark_config.algorithm,
            vocab_size,
            [str(dtype) for dtype in dtypes],
            list(with_skip_mask),
        )
        for dtype in dtypes:
            logits = torch.zeros(
                (_NUM_WARMUP_TOKENS, vocab_size), dtype=dtype, device=device
            )
            for use_skip_mask in with_skip_mask:
                sampler.watermarker.sample(
                    logits,
                    contexts,
                    random_sampler,
                    skip_mask=skip_mask if use_skip_mask else None,
                )
    except Exception:
        logger.warning("Skipping watermark sampler warmup.", exc_info=True)
