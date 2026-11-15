# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for execution auxiliary outputs."""

from typing import Any, Self

from pydantic import Field, model_validator

from vllm.config.utils import CompileFactors, config, get_compile_factors


@config
class AuxOutputConfig:
    """Configuration for auxiliary-output delivery."""

    enable_return_routed_experts: bool = False
    """Capture and return routed-experts auxiliary outputs."""

    max_bytes: int | None = Field(default=None, gt=0)
    """Initial capacity, or `None` to derive it from the KV cache capacity.
    Grows when every object is referenced."""

    object_store: dict[str, Any] | None = None
    """Deliver the routed experts of requests that set
    `SamplingParams.aux_session` through a shared object store: the response
    lists the keys of the objects that cover its rows, one object per KV cache
    block, followed by the rows past the last full block. Keys: `backend`
    (`"mooncake"`, configured by `MOONCAKE_CONFIG_PATH`) and `key_prefix`
    (default `"vllm-aux"`), which should be unique per job."""

    session_idle_timeout_s: float = Field(default=3600.0, gt=0)
    """Seconds after which the engine forgets a session with no running
    request: its next request hits only objects it publishes again."""

    @property
    def object_key_prefix(self) -> str:
        """Prefix of every shared object key."""
        return (self.object_store or {}).get("key_prefix", "vllm-aux")

    @model_validator(mode="after")
    def _check_object_store(self) -> Self:
        if self.object_store is None:
            return self
        if not self.enable_return_routed_experts:
            raise ValueError("object_store requires enable_return_routed_experts.")
        if self.object_store.get("backend") != "mooncake":
            raise ValueError(
                "object_store backend must be 'mooncake', got "
                f"{self.object_store.get('backend')!r}."
            )
        return self

    @property
    def enabled(self) -> bool:
        """Whether any execution auxiliary output is enabled."""
        return self.enable_return_routed_experts

    def compile_factors(self) -> CompileFactors:
        """WARNING: Whenever a new field is added to this config, add it to
        `ignored_factors` if it does not affect the computation graph.
        """
        ignored_factors = {
            "max_bytes",
            "object_store",
            "session_idle_timeout_s",
        }
        return get_compile_factors(self, ignored_factors)
