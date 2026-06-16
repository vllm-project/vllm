# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental MinerU-Diffusion entry point.

This module provides the config/model registry surface and MinerU's remasking
primitives.  The full native executor still needs the heavier ModelState work:
prompt KV write, SDAR block-local bidirectional denoising, and commit through
vLLM's speculative-token scheduling path.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import WeightsMapper

from .interfaces import SupportsMultiModal


def get_num_transfer_tokens(block_length: int, denoising_steps: int) -> list[int]:
    """Return MinerU's uniform per-step transfer budget."""
    if block_length <= 0:
        raise ValueError(f"block_length must be positive, got {block_length}")
    if denoising_steps <= 0:
        raise ValueError(f"denoising_steps must be positive, got {denoising_steps}")

    base = block_length // denoising_steps
    remainder = block_length % denoising_steps
    return [base + (1 if step < remainder else 0) for step in range(denoising_steps)]


def sample_with_temperature_topk_topp(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample token ids and return sampled-token probability as confidence."""
    if temperature <= 0:
        token_ids = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits.float(), dim=-1)
        confidence = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        return token_ids, confidence

    filtered = logits.float() / temperature
    if top_k > 0 and top_k < filtered.shape[-1]:
        kth = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < kth, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(sorted_remove).scatter(
            -1, sorted_indices, sorted_remove
        )
        filtered = filtered.masked_fill(remove, float("-inf"))

    probs = torch.softmax(filtered, dim=-1)
    flat_probs = probs.reshape(-1, probs.shape[-1])
    token_ids = torch.multinomial(flat_probs, num_samples=1).reshape(probs.shape[:-1])
    confidence = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return token_ids, confidence


def select_transfer_indices(
    confidence: torch.Tensor,
    *,
    threshold: float,
    transfer_count: int,
) -> torch.Tensor:
    """Select positions transferred from sampled block output this step."""
    if transfer_count < 0:
        raise ValueError(f"transfer_count must be non-negative, got {transfer_count}")

    selected = torch.zeros_like(confidence, dtype=torch.bool)
    if transfer_count == 0:
        return selected

    batch = confidence.reshape(-1, confidence.shape[-1])
    high = (confidence > threshold).reshape_as(batch)
    out = selected.reshape_as(batch)
    k = min(transfer_count, batch.shape[-1])

    for row_idx in range(batch.shape[0]):
        row_high = high[row_idx]
        if int(row_high.sum().item()) >= transfer_count:
            out[row_idx] = row_high
        else:
            top_idx = torch.topk(batch[row_idx], k=k).indices
            out[row_idx, top_idx] = True
    return selected


class MinerUDiffusionForConditionalGeneration(nn.Module, SupportsMultiModal):
    """Registration anchor for native MinerU-Diffusion support."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.",
            "language_model.": "language_model.",
            "model.visual.": "visual.",
            "visual.": "visual.",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.text_config = vllm_config.model_config.hf_text_config
        self.prefix = prefix

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        raise NotImplementedError

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        raise NotImplementedError(
            "MinerU-Diffusion native execution is experimental. "
            "This branch registers the config/model surface and MinerU "
            "remasking primitives; SDAR ModelState, weight loading, and "
            "multimodal forward execution still need to be implemented."
        )
