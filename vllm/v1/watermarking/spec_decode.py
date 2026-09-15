# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config.watermarking import WatermarkConfig, WatermarkContextScope
from vllm.v1.watermarking.factory import create_watermarker
from vllm.v1.watermarking.gumbel import GumbelWatermarker
from vllm.v1.watermarking.prfs import PhiloxPRF
from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    SupportsSpeculativeDecoding,
    Watermarker,
)
from vllm.v1.worker.gpu.sample.watermark import draft_watermarking_mask
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample


class DraftWatermarker:
    def __init__(
        self,
        watermarker: Watermarker,
        max_num_reqs: int,
        device: torch.device,
        num_speculative_steps: int,
        deduplicate_contexts: WatermarkContextScope,
        deduplicate_contexts_max_history: int | None,
    ) -> None:
        self.watermarker = watermarker
        self.num_speculative_steps = num_speculative_steps
        self.deduplicate_contexts = deduplicate_contexts
        self.deduplicate_contexts_max_history = deduplicate_contexts_max_history
        self.contexts = torch.zeros(
            max_num_reqs,
            watermarker.context_width,
            dtype=torch.int64,
            device=device,
        )
        self.enabled = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        self.prior_contexts = torch.zeros(
            max_num_reqs,
            num_speculative_steps,
            watermarker.context_width,
            dtype=torch.int64,
            device=device,
        )
        self.all_token_ids: torch.Tensor | None = None
        self.prompt_lens: torch.Tensor | None = None
        self.total_lens: torch.Tensor | None = None

    def prepare(
        self,
        contexts: torch.Tensor,
        enabled: torch.Tensor,
        all_token_ids: torch.Tensor,
        prompt_lens: torch.Tensor,
        total_lens: torch.Tensor,
    ) -> None:
        num_reqs = contexts.shape[0]
        self.contexts[:num_reqs].copy_(contexts)
        self.enabled[:num_reqs].copy_(enabled)
        self.all_token_ids = all_token_ids
        self.prompt_lens = prompt_lens
        self.total_lens = total_lens

    def _sampling_state(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        request_temperatures: torch.Tensor,
        draft_step: int | torch.Tensor,
        contexts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_rows = logits.shape[0]
        steps = torch.as_tensor(draft_step, device=logits.device, dtype=torch.int64)
        if steps.ndim == 0:
            steps = steps.expand(num_rows)
        enabled = self.enabled[:num_rows] & (request_temperatures != 0)
        if self.deduplicate_contexts == "none":
            return steps, enabled
        assert self.all_token_ids is not None
        assert self.prompt_lens is not None
        assert self.total_lens is not None
        enabled = draft_watermarking_mask(
            self.all_token_ids,
            idx_mapping,
            self.prompt_lens,
            self.total_lens,
            self.prior_contexts,
            contexts,
            steps,
            enabled,
            self.deduplicate_contexts_max_history,
            include_prompt=self.deduplicate_contexts == "all",
        )
        return steps, enabled

    def sample(
        self,
        logits: torch.Tensor,
        *,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        positions: torch.Tensor,
        draft_step: int | torch.Tensor,
        draft_logits: torch.Tensor,
        use_fp64: bool,
    ) -> torch.Tensor:
        num_rows = logits.shape[0]
        request_temperatures = temperature[idx_mapping]
        contexts = self.contexts[:num_rows]
        steps, enabled = self._sampling_state(
            logits, idx_mapping, request_temperatures, draft_step, contexts
        )

        processed_logits = logits / torch.where(
            request_temperatures == 0, 1, request_temperatures
        ).unsqueeze(-1)
        random_sampler = RandomSampler(
            expanded_idx_mapping=idx_mapping,
            temperatures=temperature,
            seeds=seeds,
            positions=positions,
            use_fp64=use_fp64,
            is_drafting=True,
            logits_cache=draft_logits,
            logits_cache_col=steps,
            logits_cache_source=logits,
        )
        sampled = self.watermarker.sample(
            processed_logits,
            contexts,
            random_sampler=random_sampler,
            skip_mask=~enabled,
        ).token_ids

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
    num_speculative_steps: int = 1,
    deduplicate_contexts: WatermarkContextScope = "none",
    deduplicate_contexts_max_history: int | None = 8192,
) -> DraftWatermarker | None:
    if isinstance(watermarker, SupportsSpeculativeDecoding):
        return DraftWatermarker(
            watermarker.draft_watermarker,
            max_num_reqs,
            device,
            num_speculative_steps,
            deduplicate_contexts,
            deduplicate_contexts_max_history,
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
    watermarking_skip_mask: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the target key except at contexts selected for ordinary sampling."""
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
        watermarking_skip_mask=watermarking_skip_mask,
        watermark_key=_resolve_watermark_key(watermarker),
    )
