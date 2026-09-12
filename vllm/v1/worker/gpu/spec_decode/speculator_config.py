# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-speculator configuration for the V2 model runner.

Config, scheduler, KV cache and model code needs to know how the speculator
behaves (KV lookahead slots, whether it consumes target hidden states, which
features it supports, ...) before the speculator is built, and in processes
that never build one. Each speculator implementation therefore has a
`SpeculatorConfig` subclass, in a `config.py` next to it, which describes that
behavior and names the implementation class. `SpeculativeConfig` creates one
(`SpeculativeConfig.speculator_config`), and code queries it instead of checking
speculative method names.

Speculator config modules must stay free of heavy imports, so that importing
one doesn't import any speculator implementation (or the model runner).
"""

from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias

if TYPE_CHECKING:
    from vllm.config import ParallelConfig, SpeculativeConfig

TargetEmbedSharing: TypeAlias = Literal["never", "always", "target_opt_in"]

_PKG = "vllm.v1.worker.gpu.spec_decode"


class SpeculatorConfig:
    """Describes a speculator's behavior that matters outside the speculator.

    The defaults describe a speculator with no special requirements, which
    doesn't write draft KV. This is also the base class for custom speculator
    configs (`SpeculativeConfig.speculator_config_class`). A custom speculator
    that behaves like a built-in one should subclass that speculator's config
    instead, e.g. `DraftModelSpeculatorConfig` for a speculator that runs its
    own draft model, and override `speculator_class`.
    """

    # Fully-qualified name of the V2 speculator class (a `BaseSpeculator`
    # subclass), or None if the V2 model runner has no implementation.
    speculator_class: ClassVar[str | None] = None

    # Drafts from the target model's hidden states (EAGLE-style). Such drafters
    # read the next prompt token during chunked prefill, and their KV for the
    # trailing cache block depends on target state that is not cached, so the
    # block is dropped from the prefix cache.
    uses_target_hidden_states: ClassVar[bool] = False
    # Drafts in a separate query pass of `num_lookahead_tokens` query tokens per
    # request, rather than by adding rows to the target batch.
    uses_separate_draft_pass: ClassVar[bool] = False
    # Runs drafting in its own workspace lane, so draft kernels don't share
    # scratch memory with the target's.
    uses_separate_workspace_lane: ClassVar[bool] = False
    # Needs every request's full token history on the GPU.
    needs_dense_token_history: ClassVar[bool] = False
    # Whether the drafter reuses the target's input embedding: never, always,
    # or only when the target model opts in.
    target_embed_sharing: ClassVar[TargetEmbedSharing] = "never"

    supports_watermarking: ClassVar[bool] = False
    supports_parallel_drafting: ClassVar[bool] = False
    supports_draft_lm_head_quantization: ClassVar[bool] = False
    # Prefill context parallelism.
    supports_pcp: ClassVar[bool] = False

    def __init__(self, spec_config: "SpeculativeConfig"):
        # Read the spec config lazily, since it may still be updated (e.g. by
        # model-specific config hooks) after this is created.
        self.spec_config = spec_config

    @property
    def num_speculative_tokens(self) -> int:
        return self.spec_config.num_speculative_tokens

    @property
    def uses_aux_hidden_states(self) -> bool:
        """Whether the target must also output intermediate hidden states."""
        return False

    @property
    def num_lookahead_tokens(self) -> int:
        """KV slots to reserve past the tokens the target is scheduled for.

        See `VllmConfig.num_lookahead_tokens`.
        """
        return 0

    @property
    def num_prefill_lookahead_tokens(self) -> int:
        """Prefill tokens past the computed range that the drafter reads.

        See `VllmConfig.num_prefill_lookahead_tokens`.
        """
        # Hidden-state drafters read only the immediate next token.
        return 1 if self.uses_target_hidden_states else 0

    @property
    def max_num_new_slots_for_drafting(self) -> int:
        """Query slots per request that drafting adds to the target batch.

        See `SpeculativeConfig.max_num_new_slots_for_drafting`.
        """
        if self.spec_config.parallel_drafting:
            # The existing query is reused; only masked queries need new slots.
            return self.num_speculative_tokens - 1
        return 0

    def pcp_unsupported_reason(self, parallel_config: "ParallelConfig") -> str | None:
        """Why prefill context parallelism can't be used with this speculator,
        or None if it can."""
        if self.supports_pcp:
            return None
        return (
            "MRV2 PCP only supports DSpark or single-module MTP speculative decoding."
        )


class UnsupportedSpeculatorConfig(SpeculatorConfig):
    """A speculative method that the V2 model runner doesn't implement."""


def _builtin(module: str, name: str) -> str:
    return f"{_PKG}.{module}.{name}"


_CONFIG_BY_METHOD: dict[str, str] = {
    "extract_hidden_states": _builtin(
        "extract_hidden_states_config", "ExtractHiddenStatesSpeculatorConfig"
    ),
    "draft_model": _builtin("standalone_ar.config", "DraftModelSpeculatorConfig"),
    "ngram": _builtin("ngram.config", "NgramSpeculatorConfig"),
    "ngram_gpu": _builtin("ngram.config", "NgramGPUSpeculatorConfig"),
    "eagle": _builtin("eagle.config", "EagleSpeculatorConfig"),
    "eagle3": _builtin("eagle.config", "Eagle3SpeculatorConfig"),
    "mtp": _builtin("mtp.config", "MTPSpeculatorConfig"),
    "dflash": _builtin("dflash.config", "DFlashSpeculatorConfig"),
    "dspark": _builtin("dspark.config", "DSparkSpeculatorConfig"),
}


def get_speculator_config_class(
    spec_config: "SpeculativeConfig",
) -> type[SpeculatorConfig]:
    """Select the speculator config class for an initialized spec config.

    This is the only place that maps speculative methods (and draft
    architectures) to speculators.
    """
    from vllm.utils.import_utils import resolve_obj_by_qualname

    if spec_config.speculator_config_class is not None:
        qualname = spec_config.speculator_config_class
        config_cls = resolve_obj_by_qualname(qualname)
        if not (
            isinstance(config_cls, type) and issubclass(config_cls, SpeculatorConfig)
        ):
            raise TypeError(
                f"speculator_config_class '{qualname}' must be a subclass of "
                f"{SpeculatorConfig.__module__}.{SpeculatorConfig.__qualname__}."
            )
        if config_cls.speculator_class is None:
            raise ValueError(
                f"speculator_config_class '{qualname}' must set speculator_class."
            )
        return config_cls

    method = spec_config.method
    if method == "dflash":
        from vllm.config.speculative import XPRESS_ARCHITECTURES

        architectures = spec_config.draft_model_config.architectures
        if XPRESS_ARCHITECTURES.intersection(architectures):
            qualname = _builtin("xpress.config", "XPressSpeculatorConfig")
        elif "LiLiCorrDraftModel" in architectures:
            qualname = _builtin("lilicorr.config", "LiLiCorrSpeculatorConfig")
        elif {"DFlash2DraftModel", "DFlash2BeamDraftModel"}.intersection(architectures):
            qualname = _builtin("dflash2.config", "DFlash2SpeculatorConfig")
        else:
            qualname = _CONFIG_BY_METHOD[method]
    elif method == "mtp" and spec_config.use_gemma4_mtp():
        qualname = _builtin("gemma4.config", "Gemma4MTPSpeculatorConfig")
    elif method == "mtp" and spec_config.use_multi_module_mtp():
        qualname = _builtin("multi_module_mtp.config", "MultiModuleMTPSpeculatorConfig")
    elif method in _CONFIG_BY_METHOD:
        qualname = _CONFIG_BY_METHOD[method]
    else:
        # e.g. suffix, or a V1 custom_class proposer.
        return UnsupportedSpeculatorConfig
    return resolve_obj_by_qualname(qualname)
