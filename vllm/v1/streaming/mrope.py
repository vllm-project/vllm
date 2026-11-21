# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Preserve cached RoPE positions across bounded-session updates."""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.model_executor.models.interfaces import SupportsMRoPE
    from vllm.multimodal.inputs import MultiModalFeatureSpec


def compute_chunk_mrope_positions(
    model: "SupportsMRoPE",
    chunk_tokens: list[int],
    chunk_mm_features: list["MultiModalFeatureSpec"],
    base_position: int,
    num_dims: int = 3,
) -> tuple[torch.Tensor, int]:
    """Compute chunk-local positions translated above the cached watermark."""
    if not chunk_tokens:
        return torch.empty((num_dims, 0), dtype=torch.long), base_position - 1
    positions, _ = model.get_mrope_input_positions(chunk_tokens, chunk_mm_features)
    positions = positions + base_position
    return positions, int(positions.max().item())


def make_chunk_relative_mm_features(
    cumulative_mm_features: list["MultiModalFeatureSpec"], prev_num_tokens: int
) -> list["MultiModalFeatureSpec"]:
    """Copy the new chunk's multimodal features with chunk-local offsets."""
    return [
        replace(
            feature,
            mm_position=replace(
                feature.mm_position, offset=feature.mm_position.offset - prev_num_tokens
            ),
        )
        for feature in cumulative_mm_features
        if feature.mm_position.offset >= prev_num_tokens
    ]


@dataclass
class StreamingMropeState:
    """CPU positions for the previous prompt, including surviving cache gaps."""

    positions: torch.Tensor
    max_position: int

    @classmethod
    def from_prompt(
        cls,
        model: "SupportsMRoPE",
        token_ids: list[int],
        mm_features: list["MultiModalFeatureSpec"],
    ) -> "StreamingMropeState":
        positions, _ = model.get_mrope_input_positions(token_ids, mm_features)
        return cls(positions.cpu(), int(positions.max().item()))

    def update(
        self,
        model: "SupportsMRoPE",
        token_ids: list[int],
        mm_features: list["MultiModalFeatureSpec"],
        evicted_ranges: list[tuple[int, int]],
    ) -> None:
        """Replay ordered evictions, then extend through kept output and input."""
        for start, end in evicted_ranges:
            self.positions = torch.cat(
                (self.positions[:, :start], self.positions[:, end:]), dim=1
            )
        previous_len = self.positions.shape[1]
        if previous_len > len(token_ids):
            raise ValueError("Streaming prompt shrank without matching eviction ranges")
        if previous_len == len(token_ids):
            return
        chunk_positions, self.max_position = compute_chunk_mrope_positions(
            model,
            token_ids[previous_len:],
            make_chunk_relative_mm_features(mm_features, previous_len),
            self.max_position + 1,
            num_dims=self.positions.shape[0],
        )
        self.positions = torch.cat((self.positions, chunk_positions.cpu()), dim=1)

    @property
    def delta(self) -> int:
        return self.max_position + 1 - self.positions.shape[1]

    def record_computed_tokens(self, computed_end: int) -> None:
        """Extend through computed decode tokens before any later eviction."""
        length = self.positions.shape[1]
        if computed_end <= length:
            return
        decoded = torch.arange(
            length + self.delta, computed_end + self.delta, dtype=self.positions.dtype
        ).expand(self.positions.shape[0], -1)
        self.positions = torch.cat((self.positions, decoded), dim=1)
        self.max_position = int(decoded[:, -1].max().item())

    def computed_watermark(self, computed_end: int) -> int:
        """Highest position computed through the supplied exclusive token index."""
        prompt_end = min(computed_end, self.positions.shape[1])
        maximum = (
            int(self.positions[:, :prompt_end].max().item()) if prompt_end > 0 else -1
        )
        if computed_end > self.positions.shape[1]:
            maximum = max(maximum, computed_end - 1 + self.delta)
        return maximum
