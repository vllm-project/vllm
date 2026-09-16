# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config.watermarking import WatermarkConfig
from vllm.v1.watermarking.factory import create_watermarker
from vllm.v1.watermarking.gumbel import GumbelWatermarker
from vllm.v1.watermarking.prfs import PhiloxPRF
from vllm.v1.watermarking.watermarker import (
    SupportsSpeculativeDecoding,
    Watermarker,
)
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
        ).token_ids
        enabled = self.enabled[: logits.shape[0]] & (request_temperatures != 0)
        sampled = torch.where(enabled, watermarked, ordinary_sampled)
        contexts = self.contexts[: logits.shape[0]]
        contexts.copy_(torch.cat((contexts[:, 1:], sampled.unsqueeze(-1)), dim=-1))
        return sampled


def create_speculative_target_watermarker(watermarker: Watermarker) -> Watermarker:
    if isinstance(watermarker, SupportsSpeculativeDecoding):
        return watermarker.target_watermarker
    return watermarker


def create_speculative_draft_watermarker(
    watermarker: Watermarker,
    max_num_reqs: int,
    device: torch.device,
    allow_target_only: bool,
) -> DraftWatermarker | None:
    if isinstance(watermarker, SupportsSpeculativeDecoding):
        return DraftWatermarker(watermarker.draft_watermarker, max_num_reqs, device)
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
    """Return the Philox recovery key."""
    if isinstance(watermarker, SupportsSpeculativeDecoding):
        raise ValueError(
            f"{type(watermarker).__name__} keys the target role separately from "
            "the draft. Pass create_speculative_target_watermarker(watermarker) "
            "so the recovery draw carries the target's key."
        )
    return _philox_key(watermarker)


def speculative_target_watermark_key(
    watermark_config: WatermarkConfig | None,
) -> int | None:
    """Resolve the resample kernel's Philox key from configuration."""
    if watermark_config is None:
        return None

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
    """Use the target key for rejection recovery and bonus tokens."""
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
