# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

from pydantic.dataclasses import dataclass

from vllm.config.utils import config


@config
@dataclass
class ReasoningConfig:
    """Configuration for reasoning models."""

    think_start_str: Optional[str] = None
    """String that indicates the start of reasoning."""
    think_end_str: Optional[str] = None
    """String that indicates the end of reasoning."""
    think_start_token_ids: Optional[list[int]] = None
    """Token ID that indicates the start of reasoning."""
    think_end_token_ids: Optional[list[int]] = None
    """Token ID that indicates the end of reasoning."""

    def is_thinking_enabled(self) -> bool:
        """Check if both start and end thinking token IDs
        are set to enable thinking token budget logic."""
        return (self.think_start_token_ids is not None
                and self.think_end_token_ids is not None
                and len(self.think_start_token_ids) > 0
                and len(self.think_end_token_ids) > 0)
