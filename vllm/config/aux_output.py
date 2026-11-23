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

    object_key_prefix: str | None = None
    """First segment of the keys of the shared objects that deliver the
    routed experts of requests with `SamplingParams.aux_namespace`. Choose a
    new one for every engine start. Requires `enable_return_routed_experts`."""

    object_store: str = "mooncake"
    """Object store of those objects: `"mooncake"`, or the qualified name of a
    class that implements
    `vllm.distributed.aux_output_connector.object_store.AuxObjectStore`,
    `create` included."""

    object_store_extra_config: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments of the object store's `create`, read only by it. For
    Mooncake: `"config_path"`, its configuration file (default:
    `MOONCAKE_CONFIG_PATH`), with a master and pool separate from any Mooncake
    KV store; `"hard_pin"`, to write objects hard-pinned for a memory-only
    store (default: not pinned, for a store that offloads to SSD on
    eviction)."""

    @model_validator(mode="after")
    def _check_object_key_prefix(self) -> Self:
        prefix = self.object_key_prefix
        if prefix is not None and not (prefix and "/" not in prefix):
            raise ValueError("object_key_prefix must be non-empty and without '/'.")
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
            "object_key_prefix",
            "object_store",
            "object_store_extra_config",
        }
        return get_compile_factors(self, ignored_factors)
