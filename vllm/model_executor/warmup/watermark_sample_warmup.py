# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up the watermarked sampler's Triton kernel.

``_philox_gumbel_kernel`` is JIT-compiled once per (key, logits dtype,
skip-mask, ``USE_FP64``) specialization. The generic sampler warmup samples with
``SamplingParams.for_sampler_warmup()``, whose feature-heavy logits processing
forces the fp32 copy in ``apply_sampling_params``, so the first ordinary
temperature-1.0 request is the first launch with model-dtype logits and pays
the compilation inside inference. This pre-compiles every specialization the
sampler path can launch for the configured watermark.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.watermarking.watermarker import Watermarker
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_NUM_WARMUP_TOKENS = 1


def _philox_key(watermarker: Watermarker) -> int | None:
    """None when the watermarker takes the torch path and compiles no kernel."""
    from vllm.v1.watermarking.gumbel import GumbelWatermarker
    from vllm.v1.watermarking.prfs import PhiloxPRF

    if not isinstance(watermarker, GumbelWatermarker):
        return None
    prf = watermarker.prf
    return prf.key if type(prf) is PhiloxPRF else None


def _philox_sampler_keys(watermarker: Watermarker) -> list[int]:
    """Philox keys the sampler launches the kernel with."""
    from vllm.v1.watermarking.gumbel import DualKeyGumbelWatermarker

    watermarkers: list[Watermarker] = [watermarker]
    if isinstance(watermarker, DualKeyGumbelWatermarker):
        # ``DualKeyGumbelWatermarker.sample`` launches once per key, and
        # collapses to a single key at the alpha bounds, where the other key's
        # specialization is never launched.
        if watermarker.alpha == 0:
            watermarkers = [watermarker.draft_watermarker]
        elif watermarker.alpha == 1:
            watermarkers = [watermarker.target_watermarker]
        else:
            watermarkers = [
                watermarker.draft_watermarker,
                watermarker.target_watermarker,
            ]
    return [key for key in map(_philox_key, watermarkers) if key is not None]


@torch.inference_mode()
def watermark_sample_warmup(worker: Worker) -> None:
    if getattr(worker.vllm_config, "watermark_config", None) is None:
        return
    # ``GumbelWatermarker`` launches the kernel only on CUDA
    # (vllm/v1/watermarking/gumbel.py:63,:82); elsewhere the sampler takes the
    # torch path and warming here would compile a kernel nothing launches.
    if not current_platform.is_cuda_alike():
        return

    from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
    from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_sample

    # Read the sampler the runner built instead of rebuilding it from config:
    # its watermarker (the target role under a speculative config), its context
    # deduplication setting and its fp64 flag are what the launches have to
    # match, and cannot drift from the runner. A rank that does not sample -- a
    # non-final pipeline stage, a pooling model -- holds no such sampler.
    sampler = getattr(worker.model_runner, "sampler", None)
    if not isinstance(sampler, GPUWatermarkSampler):
        return

    model_config = worker.vllm_config.model_config
    device = worker.device
    try:
        keys = _philox_sampler_keys(sampler.watermarker)
        if not keys:
            return

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
        contexts = torch.zeros(
            (_NUM_WARMUP_TOKENS, sampler.watermarker.context_width),
            dtype=torch.int32,
            device=device,
        )
        # Dtypes mirror the runtime buffers the sampler passes and are part of
        # the specialization key: int32 request-state token ids, int64 index and
        # seed buffers, fp32 sampling temperatures.
        sampling_state = {
            "skip_mask": torch.zeros(
                _NUM_WARMUP_TOKENS, dtype=torch.bool, device=device
            ),
            "expanded_idx_mapping": torch.zeros(
                _NUM_WARMUP_TOKENS, dtype=torch.int64, device=device
            ),
            "temperatures": torch.ones(
                _NUM_WARMUP_TOKENS, dtype=torch.float32, device=device
            ),
            "seeds": torch.zeros(_NUM_WARMUP_TOKENS, dtype=torch.int64, device=device),
            "positions": torch.zeros(
                _NUM_WARMUP_TOKENS, dtype=torch.int64, device=device
            ),
        }

        logger.info(
            "Warming up watermark sampler kernel (vocab=%d, keys=%d, dtypes=%s, "
            "skip_mask=%s).",
            vocab_size,
            len(keys),
            [str(dtype) for dtype in dtypes],
            list(with_skip_mask),
        )
        for dtype in dtypes:
            logits = torch.zeros(
                (_NUM_WARMUP_TOKENS, vocab_size), dtype=dtype, device=device
            )
            for key in keys:
                for use_skip_mask in with_skip_mask:
                    philox_gumbel_sample(
                        logits,
                        contexts,
                        key,
                        # Only the mixed path forwards the sampler's setting;
                        # ``GumbelWatermarker._sample_watermarked`` launches
                        # the mask-free variant with the fp32 default.
                        use_fp64=use_skip_mask and sampler.use_fp64_gumbel,
                        **(sampling_state if use_skip_mask else {}),
                    )
    except Exception:
        logger.warning("Skipping watermark sampler warmup.", exc_info=True)
