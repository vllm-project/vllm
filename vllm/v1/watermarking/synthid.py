# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SynthID-Text watermark generation primitives."""

import math

import torch

from vllm.config.watermarking import WatermarkPRFName
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, triton
from vllm.v1.watermarking.detector import WatermarkDetector
from vllm.v1.watermarking.prfs import PhiloxPRF, create_prf
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    Watermarker,
    WatermarkSample,
    _validate_context_width,
)

logger = init_logger(__name__)

# High stream word ("SYNT") keeps SynthID-Text off Gumbel-max's Philox streams.
_STREAM_DOMAIN = 0x53594E54 << 32


def _fused_kernels_error() -> str:
    triton_version = triton.__version__ if HAS_TRITON else "not installed"
    return (
        "SynthID-Text watermarking on GPUs runs only through its fused Gluon "
        "kernels, which need an NVIDIA CUDA GPU with compute capability 8.0 or "
        "newer and Triton 3.8 or newer with Gluon (found platform "
        f"{current_platform.device_name!r}, Triton {triton_version}). Install "
        "Triton 3.8 or newer (the version torch 2.14 requires), or run without "
        "SynthID-Text watermarking."
    )


def check_synthid_text_platform() -> None:
    """Raise if this platform's GPUs cannot run the fused SynthID-Text kernels.

    The torch tournament serves only CPUs and CUDA GPUs below sm80. It is
    several times slower than the kernels, so it is not a fallback for GPUs
    that should run them.
    """
    if not current_platform.is_cuda_alike():
        return
    if current_platform.is_cuda() and not current_platform.has_device_capability(80):
        return
    # The kernels import vllm.v1.watermarking, so import them lazily.
    from vllm.v1.worker.gpu.sample.synthid import is_supported

    if not is_supported():
        raise ValueError(_fused_kernels_error())


def _validate_depth(depth: int) -> None:
    if isinstance(depth, bool):
        raise ValueError("SynthID-Text depth must be an integer, not a bool")
    if not 1 <= depth <= 32:
        raise ValueError("SynthID-Text depth must be between 1 and 32")


class SynthIDWatermarker(Watermarker):
    def __init__(
        self,
        key: int,
        context_width: int = 4,
        depth: int = 22,
        prf: WatermarkPRFName = "philox",
    ) -> None:
        _validate_context_width(context_width)
        _validate_depth(depth)
        selected_prf = create_prf(prf, key)
        if not isinstance(selected_prf, PhiloxPRF):
            raise ValueError("SynthID-Text requires the Philox PRF")
        self.prf = selected_prf
        self._context_width = context_width
        self.depth = depth
        # The first fused call runs in the engine's profile run, before CUDA
        # graph capture. A kernel that cannot compile on this GPU fails there,
        # and startup stops with a clear error.
        self._fused_verified = False

    @property
    def context_width(self) -> int:
        return self._context_width

    @staticmethod
    def _update_probs(
        probs: torch.Tensor,
        words: torch.Tensor,
        num_bits: int,
    ) -> torch.Tensor:
        """Apply up to 32 SynthID-Text tournament layers from one Philox word."""
        for bit in range(num_bits):
            # [B, V] -> [B, V]
            g = ((words >> bit) & 1).to(probs.dtype)

            # [B, V] -> [B, 1]
            g_mass = (g * probs).sum(dim=1, keepdim=True)

            probs = probs * (1 + g - g_mass)  # [B, V]

        return probs

    def watermark_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        vocabulary = torch.arange(logits.shape[-1], device=logits.device)
        probs = torch.softmax(logits, dim=1, dtype=torch.float32)  # [B, V]

        # [B, C] and [V] -> [B, V]
        words = self.prf.uint32(
            contexts,
            vocabulary,
            stream=_STREAM_DOMAIN,
        )

        probs = self._update_probs(
            probs,
            words,
            self.depth,
        )

        log_probs = torch.log(probs)
        return torch.where(
            torch.isfinite(log_probs) | torch.isneginf(logits),
            log_probs,
            torch.finfo(log_probs.dtype).min,
        )

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if random_sampler is None:
            raise ValueError("SynthID-Text requires a random sampler")

        if self._use_fused(logits):
            if self._fused_verified:
                return self._sample_fused(logits, contexts, random_sampler, skip_mask)
            try:
                sample = self._sample_fused(logits, contexts, random_sampler, skip_mask)
            except Exception as e:
                raise RuntimeError(
                    "The fused SynthID-Text kernels failed to compile or launch. "
                    + _fused_kernels_error()
                ) from e
            self._fused_verified = True
            return sample

        logger.warning_once(
            "SynthID-Text is sampling with the torch tournament, which is much "
            "slower than the fused kernels and meant for correctness, not "
            "serving. The fused kernels need an NVIDIA GPU with compute "
            "capability 8.0 or newer and Triton 3.8 or newer."
        )
        watermarked_logits = self.watermark_logits(logits, contexts)
        if skip_mask is not None:
            watermarked_logits = torch.where(
                skip_mask.unsqueeze(-1), logits, watermarked_logits
            )
        return WatermarkSample(
            token_ids=random_sampler(watermarked_logits),
            logits=logits,
        )

    def _use_fused(self, logits: torch.Tensor) -> bool:
        from vllm.v1.worker.gpu.sample.synthid import is_supported

        # Tensors on the CPU, as in tests, keep the torch tournament, and so do
        # fp64 logits, which the kernels do not take.
        return logits.is_cuda and logits.dtype != torch.float64 and is_supported()

    def _sample_fused(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler,
        skip_mask: torch.Tensor | None,
    ) -> WatermarkSample:
        from vllm.v1.worker.gpu.sample.synthid import synthid_sample

        token_ids = synthid_sample(
            logits,
            contexts,
            self.prf.key,
            self.depth,
            random_sampler.expanded_idx_mapping,
            random_sampler.temperatures,
            random_sampler.seeds,
            random_sampler.positions,
            stream=_STREAM_DOMAIN,
            skip_mask=skip_mask,
            use_fp64=random_sampler.use_fp64,
        )
        return WatermarkSample(token_ids=token_ids, logits=logits)

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise ValueError("SynthID-Text requires a random sampler")


def _binomial_survival(hits: int, trials: int) -> float:
    """Return P[Binomial(trials, 0.5) >= hits]."""
    if hits <= 0:
        return 1.0
    if hits > trials:
        return 0.0
    if hits <= trials // 2:
        return 1.0 - _binomial_survival(trials - hits + 1, trials)

    log_term = (
        math.lgamma(trials + 1)
        - math.lgamma(hits + 1)
        - math.lgamma(trials - hits + 1)
        - trials * math.log(2)
    )
    term = math.exp(log_term)
    total = term
    for count in range(hits, trials):
        term *= (trials - count) / (count + 1)
        total += term
    return min(1.0, total)


class SynthIDWatermarkDetector(WatermarkDetector):
    """Detect a surplus of Philox-derived SynthID-Text bits in generated tokens.

    The score is the mean g value. The p-value uses an ideal-PRF
    Binomial(num_scored_tokens * depth, 0.5) null distribution.
    """

    def __init__(
        self,
        key: int,
        context_width: int = 4,
        depth: int = 22,
        p_value_threshold: float = 0.01,
        prf: WatermarkPRFName = "philox",
        deduplicate_contexts: bool = True,
    ) -> None:
        _validate_context_width(context_width)
        _validate_depth(depth)
        selected_prf = create_prf(prf, key)
        if not isinstance(selected_prf, PhiloxPRF):
            raise ValueError("SynthID-Text requires the Philox PRF")
        super().__init__(context_width, p_value_threshold, deduplicate_contexts)
        self.prf = selected_prf
        self.depth = depth

    def _score_tokens(
        self, contexts: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        # [N, C] and [N, 1] -> [N, 1] -> [N]
        words = self.prf.uint32(
            contexts,
            targets.unsqueeze(-1),
            stream=_STREAM_DOMAIN,
        ).squeeze(-1)

        # [depth]
        bit_positions = torch.arange(
            self.depth,
            device=words.device,
        )

        # [N, 1] and [depth] -> [N, depth]
        return (words.unsqueeze(-1) >> bit_positions) & 1

    def _aggregate_scores(self, token_scores: torch.Tensor) -> float:
        return token_scores.to(torch.float64).mean().item()

    def _get_p_value(self, score: float, num_scored_tokens: int) -> float:
        trials = num_scored_tokens * self.depth
        return _binomial_survival(round(score * trials), trials)
