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

import logging
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal, get_args

if TYPE_CHECKING:
    from vllm.config.speculative import SpeculativeConfig

logger = logging.getLogger(__name__)


MTPModelTypes = Literal[
    "deepseek_mtp",
    "dots3_note_mtp",
    "mimo_mtp",
    "mimo_v2_mtp",
    "glm4_moe_mtp",
    "glm4_moe_lite_mtp",
    "glm_ocr_mtp",
    "ernie_mtp",
    "nemotron_h_mtp",
    "exaone_moe_mtp",
    "exaone4_5_mtp",
    "qwen3_next_mtp",
    "qwen4_exp_mtp",
    "qwen3_5_mtp",
    "longcat_flash_mtp",
    "bailing_hybrid_v3_mtp",
    "minimax_m3_mtp",
    "bailing_hybrid_mtp",
    "mtp",
    "kimi_k3_mtp",
    "pangu_ultra_moe_mtp",
    "step3p5_mtp",
    "step5_mtp",
    "hy_v3_mtp",
    "hy_v4_mtp",
    "gemma4_mtp",
    "inkling_mtp",
    "glm5_next_mtp",
]
NgramGPUTypes = Literal["ngram_gpu"]
DFlashModelTypes = Literal["dflash"]
DSparkModelTypes = Literal["dspark"]
EagleModelTypes = Literal[
    "eagle", "eagle3", "extract_hidden_states", MTPModelTypes, DFlashModelTypes
]
SpeculativeMethod = Literal[
    "ngram",
    "draft_model",
    "suffix",
    "custom_class",
    EagleModelTypes,
    NgramGPUTypes,
    DSparkModelTypes,
]


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

    supports_pipeline_parallelism: bool = False
    supports_data_parallelism: bool = False
    supports_context_parallelism: bool = False
    supports_dbo: bool = False
    supports_batch_sharded_sampling: bool = False
    """Parallel execution features the plugin has implemented. Tensor parallelism
    uses the engine's TP group; other execution modes require explicit opt-in.
    Batch-sharded sampling also requires synchronizing verifier-owned state
    (such as an accepted tree path) across TP ranks."""

    num_lookahead_tokens: Callable[["SpeculativeConfig"], int] | None = None
    """KV slots to reserve past the scheduled query. ``None`` means
    ``num_speculative_tokens``, the value EAGLE uses."""

    linear_attn_verify_extension: str | None = None
    """Lazy ``"package.module:factory"`` of a callable ``(vllm_config) ->
    LinearAttnVerifyExtension`` (see ``vllm/model_executor/layers/mamba/gdn/
    verify_extension.py``). It lets the method verify non-chain drafts on
    gated delta net layers of hybrid models."""

    def resolve_speculator_cls(self) -> type:
        module, name = self.speculator_cls.replace(":", ".").rsplit(".", 1)
        cls = getattr(import_module(module), name)
        if not isinstance(cls, type):
            raise TypeError(f"Speculator {self.speculator_cls!r} must be a class")
        return cls


_REGISTRY: dict[str, SpeculativeMethodSpec] = {}


def register_speculative_method(
    name: str, spec: SpeculativeMethodSpec, *, replace: bool = False
) -> None:
    """Register an out-of-tree speculative method under ``name``.

    Raises ``ValueError`` if ``name`` is a built-in method, or is already
    registered with a different spec (re-registering an equal spec is a no-op,
    so a plugin entry point may run more than once in one process).
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A speculative method name must be a non-empty string.")
    if not isinstance(spec, SpeculativeMethodSpec):
        raise TypeError("spec must be a SpeculativeMethodSpec")
    if name in get_args(SpeculativeMethod):
        raise ValueError(f"'{name}' is a built-in speculative method.")
    existing = _REGISTRY.get(name)
    if existing is not None and existing != spec and not replace:
        raise ValueError(
            f"Speculative method '{name}' is already registered; pass replace=True "
            "to override it."
        )
    _REGISTRY[name] = spec
    logger.debug("Registered speculative method '%s' (%s)", name, spec.speculator_cls)


def get_speculative_method(name: Any) -> SpeculativeMethodSpec | None:
    """Return the spec registered under ``name``, or ``None``."""
    if not isinstance(name, str):
        return None
    return _REGISTRY.get(name)


def registered_speculative_methods() -> list[str]:
    return sorted(_REGISTRY)


def unregister_speculative_method(name: str) -> None:
    """Remove a registration (for tests and plugin teardown)."""
    _REGISTRY.pop(name, None)
