# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight configuration and plugin registration for Uno."""

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    SpeculatorConfig,
    register_speculative_method,
)

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class UnoOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uno_lora_path: str = Field(min_length=1)
    uno_mask_token_id: int | None = Field(default=None, gt=1)
    uno_noise_seed: int = 0


class UnoSpeculatorConfig(SpeculatorConfig):
    speculator_class = "vllm.v1.worker.gpu.spec_decode.uno.UnoSpeculator"
    supported_options = frozenset({"speculator_options"})
    supports_request_lora = False

    @classmethod
    def default_model(cls, spec: "SpeculativeConfig") -> str:
        return "uno"

    @classmethod
    def configure(cls, spec: "SpeculativeConfig") -> None:
        if spec.target_model_config is None or spec.target_parallel_config is None:
            raise ValueError("Uno requires target model and parallel configurations")
        if spec.model not in (None, "uno"):
            raise ValueError(
                "Uno shares the target model; do not specify a draft model"
            )
        if spec.num_speculative_tokens is None:
            raise ValueError("Uno requires num_speculative_tokens")
        super().configure(spec)
        spec.draft_sample_method = "probabilistic"

    @classmethod
    def apply_options(cls, spec: "SpeculativeConfig") -> None:
        options = UnoOptions.model_validate(spec.speculator_options or {})
        if not options.uno_lora_path.strip():
            raise ValueError("method='uno' requires uno_lora_path")
        vocab_size = spec.target_model_config.get_vocab_size()
        if options.uno_mask_token_id is None:
            options.uno_mask_token_id = vocab_size
        if not 1 < options.uno_mask_token_id <= vocab_size:
            raise ValueError("uno_mask_token_id must be in (1, target vocabulary size]")
        spec.speculator_options = options.model_dump()

    @property
    def options(self) -> UnoOptions:
        return UnoOptions.model_validate(self.spec_config.speculator_options)

    @property
    def num_lookahead_tokens(self) -> int:
        return self.num_speculative_tokens

    @property
    def max_num_new_slots_for_drafting(self) -> int:
        return self.num_speculative_tokens - 1


def register() -> None:
    register_speculative_method(
        "uno", "vllm.v1.worker.gpu.spec_decode.uno_config.UnoSpeculatorConfig"
    )
