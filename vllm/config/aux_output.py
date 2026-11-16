# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for execution auxiliary outputs."""

from typing import Self

from pydantic import Field, model_validator

from vllm.config.utils import CompileFactors, config, get_compile_factors


@config
class AuxOutputConfig:
    """Configuration for auxiliary-output delivery."""

    enable_return_routed_experts: bool = False
    """Capture and return routed-experts auxiliary outputs."""

    enable_return_indexer_topk: bool = False
    """Capture and return sparse-attention index selections."""

    enable_return_hidden_states: bool = False
    """Store the hidden states the model's forward pass returns for every
    computed position of session requests (`SamplingParams.aux_session`) in
    their objects. Requires `object_key_prefix`."""

    return_topk_logprobs: int = Field(default=0, ge=0)
    """Store the top-k token ids and log-probabilities of the next token at
    every computed position of session requests (`SamplingParams.aux_session`)
    in their objects; 0 disables. Prefix-cache hits keep the rows of the
    request that computed them. Requires `object_key_prefix`; not supported
    with LoRA."""

    max_bytes: int | None = Field(default=None, gt=0)
    """Initial capacity, or `None` to derive it from the KV cache capacity.
    Grows when every object is referenced.
    Under pipeline parallelism, this is the total across all stages."""

    object_key_prefix: str | None = None
    """First segment of the keys of the shared objects that deliver the
    auxiliary outputs of requests with `SamplingParams.aux_session`. Must not
    be reused: choose a new one for every engine start."""

    @model_validator(mode="after")
    def _check_object_key_prefix(self) -> Self:
        prefix = self.object_key_prefix
        if prefix is None:
            for name in ("enable_return_hidden_states", "return_topk_logprobs"):
                if getattr(self, name):
                    raise ValueError(f"{name} requires object_key_prefix.")
        if prefix is not None and not (self.enabled and prefix and "/" not in prefix):
            raise ValueError(
                "object_key_prefix requires enable_return_routed_experts or "
                "enable_return_indexer_topk and must be non-empty and without '/'."
            )
        return self

    @property
    def enabled(self) -> bool:
        """Whether any execution auxiliary output is enabled."""
        return self.enable_return_routed_experts or self.enable_return_indexer_topk

    def compile_factors(self) -> CompileFactors:
        """WARNING: Whenever a new field is added to this config, add it to
        `ignored_factors` if it does not affect the computation graph.
        """
        ignored_factors = {
            "enable_return_hidden_states",
            "max_bytes",
            "object_key_prefix",
            "return_topk_logprobs",
        }
        return get_compile_factors(self, ignored_factors)
