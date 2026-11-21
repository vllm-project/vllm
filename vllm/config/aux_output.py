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

    max_bytes: int | None = Field(default=None, gt=0)
    """Initial capacity, or `None` to derive it from the KV cache capacity.
    Grows when every object is referenced."""

    object_key_prefix: str | None = None
    """First segment of the keys of the shared objects that deliver the
    routed experts of requests with `SamplingParams.aux_namespace`. Choose a
    new one for every engine start. Requires `enable_return_routed_experts`."""

    mooncake_config_path: str | None = None
    """Mooncake configuration file of the store that holds the objects of
    `object_key_prefix`, in the format `MOONCAKE_CONFIG_PATH` names; `None`
    reads `MOONCAKE_CONFIG_PATH`. Use a master and pool separate from any
    Mooncake KV store: a KV store evicts and can be reset. Requires
    `object_key_prefix`."""

    object_hard_pin: bool = False
    """Write objects hard-pinned, so the store never evicts them: for a
    memory-only store. By default objects are not pinned, for a store whose
    master offloads to SSD on eviction; without that offload, an object that
    is not pinned is lost when memory evicts it."""

    object_put_stall_timeout_s: float = Field(default=600.0, gt=0)
    """Seconds object writes may be pending without a successful put before
    the engine drops them. A write the store refuses is retried every 0.2 s,
    since it may fit once the store has offloaded or released objects.
    Dropped objects are missing from the store; the engine keeps running. The
    engine uses at most half of `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, so that
    a step waiting for the store does not time out."""

    @model_validator(mode="after")
    def _check_object_key_prefix(self) -> Self:
        prefix = self.object_key_prefix
        if prefix is None and self.mooncake_config_path is not None:
            raise ValueError("mooncake_config_path requires object_key_prefix.")
        if prefix is not None and not (prefix and "/" not in prefix):
            raise ValueError("object_key_prefix must be non-empty and without '/'.")
        if prefix is not None and not self.enable_return_routed_experts:
            raise ValueError("object_key_prefix requires enable_return_routed_experts.")
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
            "mooncake_config_path",
            "object_hard_pin",
            "object_key_prefix",
            "object_put_stall_timeout_s",
        }
        return get_compile_factors(self, ignored_factors)
