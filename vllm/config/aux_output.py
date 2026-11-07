# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for execution auxiliary outputs."""

from pydantic import Field

from vllm.config.utils import CompileFactors, config, get_compile_factors


@config
class AuxOutputConfig:
    """Configuration for auxiliary-output delivery."""

    enable_return_routed_experts: bool = False
    """Capture and return routed-experts auxiliary outputs."""

    enable_hidden_states: bool = False
    """Allow request-local windowed hidden-state capture (MRV2 CUDA only)."""

    max_hidden_capture_bytes: int = Field(default=256 * 1024**2, gt=0)
    """Upper bound on retained hidden/logprob payload bytes across max_num_seqs.
    Admission reserves the requested window size for each possible active request.
    Temporary model logits and allocator caches are not included."""

    max_bytes: int | None = Field(default=None, gt=0)
    """Initial capacity, or ``None`` to derive it from the KV cache capacity.
    Grows when every object is referenced."""

    @property
    def enabled(self) -> bool:
        """Whether any execution auxiliary output is enabled."""
        return self.enable_return_routed_experts or self.enable_hidden_states

    def compile_factors(self) -> CompileFactors:
        """WARNING: Whenever a new field is added to this config, add it to
        `ignored_factors` if it does not affect the computation graph.
        """
        ignored_factors = {
            "max_bytes",
        }
        return get_compile_factors(self, ignored_factors)
