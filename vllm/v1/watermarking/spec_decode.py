# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.config import SpeculativeConfig
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.watermarking.watermarker import (
    AcceptanceRandomness,
    SpeculativeVerification,
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
        "speculative decoding. Set allow_target_only_speculative_decoding=true to "
        "leave draft tokens unwatermarked."
    )


def watermarked_rejection_sample(
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
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
    sampled, num_sampled = rejection_sample(
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
    )

    _apply_watermarked_recovery(
        sampled,
        num_sampled,
        target_logits,
        draft_logits,
        draft_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        contexts,
        watermarking,
        watermarker,
    )
    return sampled, num_sampled


def _apply_watermarked_recovery(
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    contexts: torch.Tensor,
    watermarking: torch.Tensor,
    watermarker: Watermarker,
) -> None:
    resample_steps = num_sampled.to(torch.int64) - 1
    resample_rows = cu_num_logits[:-1].to(torch.int64) + resample_steps
    req_state_indices = expanded_idx_mapping[resample_rows].to(torch.int64)
    request_temperatures = temperature[req_state_indices]
    enabled = watermarking[req_state_indices] & (request_temperatures != 0)
    target_rows = target_logits[resample_rows]
    is_bonus = resample_rows == cu_num_logits[1:] - 1
    next_rows = (resample_rows + 1).clamp_max(draft_sampled.shape[0] - 1)
    has_rejected_draft = draft_sampled[next_rows] >= 0
    needs_residual = ~is_bonus & has_rejected_draft

    draft_steps = resample_steps.clamp_max(draft_logits.shape[1] - 1)
    draft_rows = draft_logits[req_state_indices, draft_steps].float()
    vocab_size = min(target_rows.shape[-1], draft_rows.shape[-1])
    target_rows = target_rows[:, :vocab_size]
    draft_rows = draft_rows[:, :vocab_size]
    target_rows = torch.where(target_rows.isnan(), float("-inf"), target_rows).float()
    draft_rows = torch.where(draft_rows.isnan(), float("-inf"), draft_rows)
    safe_temperatures = torch.where(request_temperatures == 0, 1, request_temperatures)
    draft_rows = draft_rows / safe_temperatures.unsqueeze(-1)
    target_log_probs = torch.log_softmax(target_rows, dim=-1)
    draft_log_probs = torch.log_softmax(draft_rows, dim=-1)
    ratio = torch.exp(draft_log_probs - target_log_probs)
    residual_logits = torch.where(
        ratio < 1,
        target_log_probs + torch.log1p(-ratio.clamp_max(1)),
        float("-inf"),
    )
    recovery_logits = torch.where(
        needs_residual.unsqueeze(-1), residual_logits, target_rows
    )

    output_positions = num_sampled.to(torch.int64) - 1
    request_indices = torch.arange(sampled.shape[0], device=sampled.device)
    ordinary_recovery = sampled[request_indices, output_positions]
    watermarked_recovery = watermarker.sample(
        recovery_logits,
        contexts[resample_rows],
        lambda _: ordinary_recovery,
    ).token_ids
    sampled[request_indices, output_positions] = torch.where(
        enabled, watermarked_recovery, ordinary_recovery
    )


class WatermarkedRejectionSampler(RejectionSampler):
    def __init__(
        self,
        sampler: GPUWatermarkSampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
        watermarker: Watermarker,
    ) -> None:
        if isinstance(watermarker, SupportsSpeculativeDecoding) and (
            watermarker.speculative_verification is not SpeculativeVerification.STANDARD
            or watermarker.acceptance_randomness is not AcceptanceRandomness.RANDOM
        ):
            raise NotImplementedError("Unsupported speculative watermarking protocol")
        super().__init__(sampler, spec_config, device)
        self.watermarker = watermarker

    def _verify(
        self,
        logits: torch.Tensor,
        draft_logits: torch.Tensor | None,
        draft_sampled: torch.Tensor,
        pos: torch.Tensor,
        cu_num_logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        expanded_idx_mapping: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        processed_logits, sampled, num_sampled = super()._verify(
            logits,
            draft_logits,
            draft_sampled,
            pos,
            cu_num_logits,
            idx_mapping,
            idx_mapping_np,
            expanded_idx_mapping,
            expanded_local_pos,
        )
        sampler = self.sampler
        assert isinstance(sampler, GPUWatermarkSampler)
        assert draft_logits is not None
        contexts = sampler._get_contexts(
            expanded_idx_mapping,
            expanded_local_pos,
            draft_sampled,
        )
        _apply_watermarked_recovery(
            sampled,
            num_sampled,
            processed_logits,
            draft_logits,
            draft_sampled,
            cu_num_logits,
            expanded_idx_mapping,
            sampler.sampling_states.temperature.gpu,
            contexts,
            sampler.watermarking.gpu,
            self.watermarker,
        )
        return processed_logits, sampled, num_sampled
