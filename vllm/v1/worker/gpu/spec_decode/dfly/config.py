# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.spec_decode.speculator_config import DraftModelSpeculatorConfig

if TYPE_CHECKING:
    from transformers import PreTrainedConfig

    from vllm.config import SpeculativeConfig


class DFlySpeculatorConfig(DraftModelSpeculatorConfig):
    """DFly behavior shared by model-specific configuration subclasses."""

    base_method = "dfly"
    uses_target_hidden_states = True
    shares_target_positions = False
    parallel_drafting = True

    @classmethod
    def default_model(cls, spec: "SpeculativeConfig") -> str:
        if spec.target_model_config is None:
            raise ValueError("DFly requires target_model_config")
        return spec.target_model_config.model

    @classmethod
    def draft_hf_overrides(
        cls, spec: "SpeculativeConfig"
    ) -> Callable[["PreTrainedConfig"], "PreTrainedConfig"]:
        target = spec.target_model_config
        if target is None:
            raise ValueError("DFly requires target_model_config")
        raise ValueError(
            "No DFly configuration is registered for target model type "
            f"{target.hf_text_config.model_type!r}"
        )

    @classmethod
    def resolve_num_speculative_tokens(cls, spec: "SpeculativeConfig") -> None:
        n = spec.num_speculative_tokens
        if n is None:
            hf_config = spec.target_model_config.hf_config
            checkpoint_spec = getattr(hf_config, "speculative_config", None)
            if checkpoint_spec is not None:
                if not isinstance(checkpoint_spec, Mapping):
                    raise ValueError(
                        "DFly checkpoint speculative_config must be a mapping"
                    )
                n = checkpoint_spec.get("num_speculative_tokens")
            else:
                block_size = getattr(hf_config, "dfly_block_size", None)
                if type(block_size) is not int or not 2 <= block_size <= 17:
                    raise ValueError("DFly requires dfly_block_size in [2, 17]")
                # The checkpoint block includes the separate anchor query.
                n = block_size - 1
        if type(n) is not int or not 1 <= n <= 16:
            raise ValueError("DFly requires num_speculative_tokens in [1, 16]")
        spec.num_speculative_tokens = n

    @classmethod
    def apply_options(cls, spec: "SpeculativeConfig") -> None:
        if spec.enable_adaptive_verification:
            raise ValueError("DFly does not support adaptive verification")
        if spec.rejection_sample_method not in ("standard", "synthetic"):
            raise ValueError("DFly requires standard or synthetic target verification")

    @property
    def num_lookahead_tokens(self) -> int:
        return self.num_speculative_tokens + 1

    def max_num_draft_seqs(self, max_num_batched_tokens: int) -> int:
        return max_num_batched_tokens // self.num_lookahead_tokens
