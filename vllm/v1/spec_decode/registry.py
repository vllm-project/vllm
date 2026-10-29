# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Registry for out-of-tree speculative decoding methods.

A package registers a method name once, usually from a ``vllm.general_plugins``
entry point. ``SpeculativeConfig`` then accepts that name for ``method`` and
Model Runner V2 builds the registered speculator instead of one of the built-in
ones. Nothing is registered by default, so built-in methods are unaffected.

Example::

    def register():
        register_speculative_method(
            "my_method",
            SpeculativeMethodSpec(speculator_cls="my_pkg.spec:MySpeculator"),
        )

The registered ``speculator_cls`` is a ``BaseSpeculator`` subclass (see
``vllm/v1/worker/gpu/spec_decode/speculator.py``) and receives
``(vllm_config, device)``. Method specific options go in
``SpeculativeConfig.method_config``, which vLLM does not interpret.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_args

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.config.speculative import SpeculativeConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class SpeculativeMethodSpec:
    """What vLLM needs to know about an out-of-tree speculative method."""

    speculator_cls: str
    """Lazy ``"package.module:Class"`` (or ``"package.module.Class"``) of the
    ``BaseSpeculator`` subclass. Imported only when the method is used."""

    configure: Callable[["SpeculativeConfig"], None] | None = None
    """Called at the end of ``SpeculativeConfig.__post_init__``. It may fill
    ``num_speculative_tokens``, ``num_speculative_tokens_per_batch_size`` and
    ``draft_model_config`` from ``method_config`` and the hardware."""

    uses_target_hidden_states: bool = True
    """The drafter reads the target's hidden states. This makes
    ``SpeculativeConfig.use_eagle()`` true, which turns on the draft KV cache
    group, the prefill lookahead and the connector handling that EAGLE-style
    drafters need."""

    uses_aux_hidden_states: bool = False
    """The target must also return auxiliary (intermediate layer) hidden
    states for the drafter."""

    num_lookahead_tokens: Callable[["SpeculativeConfig"], int] | None = None
    """KV slots to reserve past the scheduled query. ``None`` means
    ``num_speculative_tokens``, the value EAGLE uses."""

    linear_attn_verify_extension: str | None = None
    """Lazy ``"package.module:factory"`` of a callable ``(vllm_config) ->
    LinearAttnVerifyExtension`` (see
    ``vllm/model_executor/layers/mamba/verify_extension.py``). It lets the
    method verify non-chain drafts on gated delta net and Mamba2 layers of
    hybrid models."""

    full_attn_verify_extension: str | None = None
    """Lazy ``"package.module:factory"`` of a callable ``(vllm_config) ->
    FullAttentionVerifyExtension | None`` (see
    ``vllm/v1/attention/verify_extension.py``). A full-attention backend must
    explicitly implement the matching builder and forward dispatch ABI."""

    def resolve_speculator_cls(self) -> type:
        return resolve_obj_by_qualname(self.speculator_cls.replace(":", "."))


_REGISTRY: dict[str, SpeculativeMethodSpec] = {}


def register_speculative_method(name: str, spec: SpeculativeMethodSpec) -> None:
    """Register an out-of-tree speculative method under ``name``.

    Raises ``ValueError`` if ``name`` is a built-in method, or is already
    registered with a different spec (re-registering an equal spec is a no-op,
    so a plugin entry point may run more than once in one process).
    """
    if not name or not isinstance(name, str):
        raise ValueError("A speculative method name must be a non-empty string.")
    from vllm.config.speculative import SpeculativeMethod  # it imports this module

    if name in get_args(SpeculativeMethod):
        raise ValueError(f"'{name}' is a built-in speculative method.")
    existing = _REGISTRY.get(name)
    if existing is not None and existing != spec:
        raise ValueError(f"Speculative method '{name}' is already registered.")
    _REGISTRY[name] = spec
    logger.debug("Registered speculative method '%s' (%s)", name, spec.speculator_cls)


def get_speculative_method(name: Any) -> SpeculativeMethodSpec | None:
    """Return the spec registered under ``name``, or ``None``."""
    if not isinstance(name, str):
        return None
    return _REGISTRY.get(name)


def registered_speculative_methods() -> list[str]:
    return sorted(_REGISTRY)
