# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, cast

import torch

from vllm.config import SpeculativeConfig
from vllm.config.watermarking import WatermarkConfig
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.watermarking.gumbel import GumbelWatermarker
from vllm.v1.watermarking.prfs import PhiloxPRF
from vllm.v1.watermarking.watermarker import (
    SupportsSpeculativeDecoding,
    Watermarker,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample


class DraftWatermarker:
    def __init__(
        self,
        watermarker: Watermarker,
        max_num_reqs: int,
        device: torch.device,
    ) -> None:
        self.watermarker = watermarker
        self.contexts = torch.zeros(
            max_num_reqs,
            watermarker.context_width,
            dtype=torch.int64,
            device=device,
        )
        self.enabled = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)

    def prepare(self, contexts: torch.Tensor, enabled: torch.Tensor) -> None:
        num_reqs = contexts.shape[0]
        self.contexts[:num_reqs].copy_(contexts)
        self.enabled[:num_reqs].copy_(enabled)

    def sample(
        self,
        logits: torch.Tensor,
        ordinary_sampled: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        request_temperatures = temperature[idx_mapping]
        processed_logits = logits / torch.where(
            request_temperatures == 0, 1, request_temperatures
        ).unsqueeze(-1)
        watermarked = self.watermarker.sample(
            processed_logits,
            self.contexts[: logits.shape[0]],
            lambda _: ordinary_sampled,
        ).token_ids
        enabled = self.enabled[: logits.shape[0]] & (request_temperatures != 0)
        sampled = torch.where(enabled, watermarked, ordinary_sampled)
        contexts = self.contexts[: logits.shape[0]]
        contexts.copy_(torch.cat((contexts[:, 1:], sampled.unsqueeze(-1)), dim=-1))
        return sampled


def create_speculative_target_watermarker(watermarker: Watermarker) -> Watermarker:
    if isinstance(watermarker, SupportsSpeculativeDecoding):
        return watermarker.create_target_watermarker()
    return watermarker


def create_speculative_draft_watermarker(
    watermarker: Watermarker,
    max_num_reqs: int,
    device: torch.device,
    allow_target_only: bool,
) -> DraftWatermarker | None:
    if isinstance(watermarker, SupportsSpeculativeDecoding):
        return DraftWatermarker(
            watermarker.create_draft_watermarker(), max_num_reqs, device
        )
    if allow_target_only:
        return None
    raise ValueError(
        f"The {type(watermarker).__name__} watermarking algorithm does not support "
        "speculative decoding. Set allow_target_only_watermarking=true to "
        "leave draft tokens unwatermarked."
    )


def _philox_key(watermarker: Watermarker) -> int:
    if not (
        isinstance(watermarker, GumbelWatermarker)
        and type(watermarker.prf) is PhiloxPRF
    ):
        raise NotImplementedError(
            "In-kernel watermarked recovery supports only the Philox PRF "
            "GumbelWatermarker"
        )
    return watermarker.prf.key


def _resolve_watermark_key(watermarker: Watermarker) -> int:
    """Philox key of the watermarker driving the in-kernel recovery draw.

    Raises:
        ValueError: if a role-splitting watermarker is passed instead of its
            target role, which would key the recovery draw with the draft key.
    """
    key = _philox_key(watermarker)
    if (
        isinstance(watermarker, SupportsSpeculativeDecoding)
        and _philox_key(watermarker.create_target_watermarker()) != key
    ):
        raise ValueError(
            f"{type(watermarker).__name__} keys the target role separately from "
            "the draft. Pass create_speculative_target_watermarker(watermarker) "
            "so the recovery draw carries the target's key."
        )
    return key


def speculative_target_watermark_key(watermark_config: WatermarkConfig) -> int:
    """Philox key the resample kernel is launched with for ``watermark_config``.

    Mirrors how the model runner builds the sampler's watermarker, so callers
    that never construct a sampler (the JIT warmup) can reproduce the exact
    kernel argument the engine will use.
    """
    from vllm.v1.watermarking.factory import create_watermarker

    return _resolve_watermark_key(
        create_speculative_target_watermarker(create_watermarker(watermark_config))
    )


def watermarked_rejection_sample(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor | None,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    pos: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    num_speculative_steps: int,
    contexts: torch.Tensor,
    watermarking: torch.Tensor,
    watermarker: Watermarker,
    use_fp64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rejection sampling whose recovered token carries the target's key.

    The token the target supplies itself (the residual draw after the first
    rejection, or the bonus token) is drawn inside the resample kernel with
    keyed Philox noise over the row's watermark context. Rows that are opted
    out, greedy, or padded keep the stock seeded draw.
    """
    assert contexts.shape == (target_logits.shape[0], watermarker.context_width)
    return rejection_sample(
        target_logits,
        draft_logits,
        draft_sampled,
        cu_num_logits,
        pos,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_speculative_steps,
        use_fp64=use_fp64,
        contexts=contexts,
        watermarking=watermarking,
        watermark_key=_resolve_watermark_key(watermarker),
    )


class WatermarkedRejectionSampler(RejectionSampler):
    def __init__(
        self,
        sampler: GPUWatermarkSampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
        watermarker: Watermarker,
    ) -> None:
        super().__init__(sampler, spec_config, device)
        self.watermarker = watermarker
        self._watermark_key = _resolve_watermark_key(watermarker)

    def _extra_rejection_sample_kwargs(
        self,
        draft_sampled: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> dict[str, Any]:
        sampler = cast("GPUWatermarkSampler", self.sampler)
        return {
            "contexts": sampler._get_contexts(
                expanded_idx_mapping,
                expanded_local_pos,
                draft_sampled,
            ),
            "watermarking": sampler.watermarking.gpu,
            "watermark_key": self._watermark_key,
        }
