# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-speculator configuration.

Each speculator has a `SpeculatorConfig` subclass, in a `config.py` next to its
V2 model runner implementation, which sets up `SpeculativeConfig` for it,
describes its behavior to code outside the speculator (queried through
`SpeculativeConfig.speculator_config` rather than by method name), and names
its implementation class. Config modules must not import the speculators, as
they're used in every process. Custom speculators are registered with
`register_speculative_method`.

Parts only needed by the V1 model runner are marked "V1 model runner only", and
will be removed with it.
"""

from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias, get_args

from vllm.utils.import_utils import resolve_obj_by_qualname

# vllm.config imports this module, so it imports vllm.config lazily.
if TYPE_CHECKING:
    from vllm.config import ParallelConfig, SpeculativeConfig

TargetEmbedSharing: TypeAlias = Literal["never", "always", "target_opt_in"]

_PKG = "vllm.v1.worker.gpu.spec_decode"


class SpeculatorConfig:
    """Configuration of a kind of speculator.

    The defaults describe a speculator without a draft checkpoint or special
    requirements. A custom speculator can subclass this, or the config of the
    built-in speculator it behaves like (overriding `speculator_class`), whose
    setup and behavior it then inherits.
    """

    # For in-tree speculators, the `SpeculativeConfig.method` that enables it.
    # For out-of-tree custom speculators, the method of the built-in one it subclasses.
    base_method: ClassVar[str | None] = None
    # Fully-qualified name of the V2 `BaseSpeculator` subclass, if any.
    speculator_class: ClassVar[str | None] = None
    # Method-specific `SpeculativeConfig` options that this speculator accepts
    # (e.g. "speculator_options" for a custom speculator's own options).
    supported_options: ClassVar[frozenset[str]] = frozenset()

    # Drafts from the target's hidden states (EAGLE-style).
    uses_target_hidden_states: ClassVar[bool] = False
    # Drafts in its own workspace lane.
    uses_separate_workspace_lane: ClassVar[bool] = False
    # For performance, prefers requests' full token history on the GPU instead of
    # UVA memory (e.g. ngram).
    prefer_dense_token_history: ClassVar[bool] = False
    # Reuses the target's input embedding: never, always, or on target opt-in.
    target_embed_sharing: ClassVar[TargetEmbedSharing] = "never"
    # Proposes draft tokens (unlike e.g. hidden state extraction).
    proposes_draft_tokens: ClassVar[bool] = True
    adaptive_verification_uses_draft_logits: ClassVar[bool] = True
    # The draft model has its own vocabulary, which must match the target's
    # unless use_heterogeneous_vocab.
    draft_has_own_vocab: ClassVar[bool] = False
    # The draft model takes the target's positions, so its
    # max_position_embeddings may be raised to the target's max_model_len.
    shares_target_positions: ClassVar[bool] = False

    supports_watermarking: ClassVar[bool] = False
    # Prefill context parallelism.
    supports_pcp: ClassVar[bool] = False

    # --- SpeculativeConfig setup ---

    @classmethod
    def matches(cls, spec: "SpeculativeConfig") -> bool:
        """Whether the draft checkpoint is for this speculator, other than by
        its model's declaration (e.g. by name). Used to detect the method."""
        return False

    @classmethod
    def specialize(cls, spec: "SpeculativeConfig") -> type["SpeculatorConfig"]:
        """This config, or a more specific subclass for the settings."""
        return cls

    @classmethod
    def default_model(cls, spec: "SpeculativeConfig") -> str | None:
        """The `model` to use when it isn't given."""
        return None

    @classmethod
    def configure(cls, spec: "SpeculativeConfig") -> None:
        """Set up the draft model config (and related options)."""
        spec.prompt_lookup_max = 0
        spec.prompt_lookup_min = 0
        # No draft model; the target's config provides the vocab size etc.
        spec.draft_model_config = spec.target_model_config
        spec.draft_parallel_config = spec.target_parallel_config

    @classmethod
    def apply_options(cls, spec: "SpeculativeConfig") -> None:
        """Apply method-specific options to the draft model config."""
        return

    # --- Behavior ---

    def __init__(self, spec_config: "SpeculativeConfig"):
        # Read lazily, since the spec config may still be updated.
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
        """See `VllmConfig.num_lookahead_tokens`."""
        return 0

    @property
    def num_prefill_lookahead_tokens(self) -> int:
        """See `VllmConfig.num_prefill_lookahead_tokens`."""
        # Hidden-state drafters read only the immediate next token.
        return 1 if self.uses_target_hidden_states else 0

    @property
    def max_num_new_slots_for_drafting(self) -> int:
        """See `SpeculativeConfig.max_num_new_slots_for_drafting`."""
        if self.spec_config.parallel_drafting:
            # The existing query is reused; only masked queries need new slots.
            return self.num_speculative_tokens - 1
        return 0

    def max_num_draft_seqs(self, max_num_batched_tokens: int) -> int | None:
        """The most requests the draft forward pass fits in
        max_num_batched_tokens, which limits max_num_active_seqs, or None if
        unlimited (e.g. when it's sized by the target's scheduled tokens)."""
        return None

    def hash_factors(self) -> list[Any]:
        """Extra factors of `SpeculativeConfig.compute_hash`."""
        return []

    def pcp_unsupported_reason(self, parallel_config: "ParallelConfig") -> str | None:
        """Why prefill context parallelism can't be used, or None."""
        if self.supports_pcp:
            return None
        return (
            "MRV2 PCP only supports DSpark or single-module MTP speculative decoding."
        )


class CheckpointSpeculatorConfig(SpeculatorConfig):
    """A speculator whose draft model is loaded from the `model` checkpoint."""

    # Whether an explicitly configured method is checked against the draft
    # checkpoint (otherwise it's trusted).
    validates_checkpoint: ClassVar[bool] = False
    shares_target_positions = True

    @classmethod
    def default_model(cls, spec: "SpeculativeConfig") -> str | None:
        raise ValueError(
            "num_speculative_tokens was provided but without speculative model."
        )

    @classmethod
    def configure(cls, spec: "SpeculativeConfig") -> None:
        from vllm.config import ModelConfig, SpeculativeConfig

        spec.prompt_lookup_max = 0
        spec.prompt_lookup_min = 0
        if spec.model is None:
            return

        target = spec.target_model_config
        # Callable target hf_overrides also apply to the draft; dict ones don't.
        draft_hf_overrides = SpeculativeConfig.compose_draft_hf_overrides(
            target.hf_overrides
        )
        spec.draft_model_config = ModelConfig(
            model=spec.model,
            runner="draft",
            # use_heterogeneous_vocab is V1 model runner only.
            tokenizer=(
                spec.model if spec.use_heterogeneous_vocab else target.tokenizer
            ),
            tokenizer_mode=target.tokenizer_mode,
            trust_remote_code=target.trust_remote_code,
            allowed_local_media_path=target.allowed_local_media_path,
            allowed_media_domains=target.allowed_media_domains,
            dtype=target.dtype,
            seed=target.seed,
            revision=spec.revision,
            code_revision=spec.code_revision,
            tokenizer_revision=target.tokenizer_revision,
            max_model_len=spec.max_model_len,  # type: ignore[arg-type]
            spec_target_max_model_len=target.max_model_len,
            quantization=spec.quantization,
            enforce_eager=target.enforce_eager,
            max_logprobs=target.max_logprobs,
            hf_overrides=draft_hf_overrides,
            config_format=target.config_format,
        )

        config_cls = _resolve_checkpoint_config_class(spec, cls)
        if spec.method is None:
            spec.method = config_cls.base_method  # type: ignore[assignment]

        if config_cls.shares_target_positions:
            SpeculativeConfig._maybe_override_draft_max_position_embeddings(
                spec.draft_model_config.hf_config, target.max_model_len
            )
        config_cls.adapt_draft_model_config(spec)
        config_cls.resolve_num_speculative_tokens(spec)
        if spec.num_speculative_tokens is None:
            raise ValueError(
                "A speculative model was provided, but "
                "`num_speculative_tokens` was not provided"
            )
        config_cls.validate_draft_model_config(spec)

        spec.draft_tensor_parallel_size = SpeculativeConfig._verify_and_get_draft_tp(
            spec.target_parallel_config, spec.draft_tensor_parallel_size
        )
        spec.draft_model_config.max_model_len = (
            SpeculativeConfig._maybe_override_draft_max_model_len(
                spec.max_model_len,
                spec.draft_model_config.max_model_len,
                target.max_model_len,
            )
        )
        # From the final draft config, so EP agrees with its expert count.
        spec.draft_parallel_config = SpeculativeConfig.create_draft_parallel_config(
            spec.target_parallel_config,
            spec.draft_tensor_parallel_size,
            draft_model_config=spec.draft_model_config,
        )

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        return

    @classmethod
    def resolve_num_speculative_tokens(cls, spec: "SpeculativeConfig") -> None:
        """Default or validate `num_speculative_tokens` from the checkpoint."""
        n_predict = getattr(spec.draft_model_config.hf_config, "n_predict", None)
        if n_predict is None:
            return
        if spec.num_speculative_tokens is None:
            # Default to max value defined in draft model config.
            spec.num_speculative_tokens = n_predict
        elif (
            spec.num_speculative_tokens > n_predict
            and spec.num_speculative_tokens % n_predict != 0
        ):
            # Ensure divisibility for MTP module reuse.
            raise ValueError(
                f"num_speculative_tokens:{spec.num_speculative_tokens}"
                f" must be divisible by {n_predict=}"
            )

    @classmethod
    def validate_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        """Validate the draft model config, given `num_speculative_tokens`."""
        return


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


# V1 model runner only (see `v1_speculator_configs`).
_V1_ONLY_METHODS = ("suffix", "custom_class")

# Custom speculative methods' configs (or their qualified names), by name.
_CUSTOM_CONFIGS: dict[str, "type[SpeculatorConfig] | str"] = {}


def register_speculative_method(
    name: str, config_cls: "type[SpeculatorConfig] | str"
) -> None:
    """Register a custom speculative method, used with `method=name`.

    `config_cls` is its `SpeculatorConfig` subclass, or its qualified name to
    import it lazily. Register it from a vLLM general plugin, so that it's
    registered in every process. Custom methods require the V2 model runner.
    This is an experimental interface that may change between releases.
    """
    from vllm.config.speculative import SpeculativeMethod

    if name in get_args(SpeculativeMethod) or name in _CONFIG_BY_METHOD:
        raise ValueError(f"'{name}' is a built-in speculative method.")
    if name in _CUSTOM_CONFIGS and _CUSTOM_CONFIGS[name] != config_cls:
        raise ValueError(f"Speculative method '{name}' is already registered.")
    _CUSTOM_CONFIGS[name] = config_cls


def registered_speculative_methods() -> list[str]:
    """The names of the registered custom speculative methods."""
    return sorted(_CUSTOM_CONFIGS)


def get_custom_speculator_config_class(
    method: object,
) -> type[SpeculatorConfig] | None:
    """The config of a registered custom speculative method, or None."""
    if not isinstance(method, str) or method not in _CUSTOM_CONFIGS:
        return None
    config_cls = _CUSTOM_CONFIGS[method]
    if isinstance(config_cls, str):
        config_cls = resolve_obj_by_qualname(config_cls)
    if not (isinstance(config_cls, type) and issubclass(config_cls, SpeculatorConfig)):
        raise TypeError(
            f"The config of speculative method '{method}' must be a subclass of "
            f"{SpeculatorConfig.__module__}.{SpeculatorConfig.__qualname__}."
        )
    if config_cls.speculator_class is None:
        raise ValueError(
            f"The config of speculative method '{method}' must set speculator_class."
        )
    return config_cls


def get_method_config_class(spec_config: "SpeculativeConfig") -> type[SpeculatorConfig]:
    """The config class for `spec_config.method`, regardless of the draft
    checkpoint (for setting up `spec_config`)."""
    if (
        custom_cls := get_custom_speculator_config_class(spec_config.method)
    ) is not None:
        return custom_cls
    method = spec_config.method
    if method is None:
        # Not configured: a draft model, unless the checkpoint shows otherwise.
        method = "draft_model"
    if method in _V1_ONLY_METHODS:
        from vllm.v1.worker.gpu.spec_decode.v1_speculator_configs import (
            SuffixSpeculatorConfig,
            V1CustomProposerConfig,
        )

        if method == "suffix":
            return SuffixSpeculatorConfig
        return V1CustomProposerConfig
    if method not in _CONFIG_BY_METHOD:
        raise ValueError(f"Unsupported speculative method: '{method}'")
    return resolve_obj_by_qualname(_CONFIG_BY_METHOD[method])


def get_speculator_config_class(
    spec_config: "SpeculativeConfig",
) -> type[SpeculatorConfig]:
    """The config class for a set-up spec config."""
    config_cls = get_method_config_class(spec_config)
    if get_custom_speculator_config_class(spec_config.method) is not None:
        return config_cls
    declared_cls = _get_draft_declared_config_class(spec_config)
    if declared_cls is not None and issubclass(declared_cls, config_cls):
        return declared_cls
    return config_cls.specialize(spec_config)


def _get_draft_declared_config_class(
    spec_config: "SpeculativeConfig",
) -> type[SpeculatorConfig] | None:
    """The config declared by the draft model (see `SpeculativeDraftModel`)."""
    draft = spec_config.draft_model_config
    qualname = getattr(draft, "speculator_config_class", None)
    return resolve_obj_by_qualname(qualname) if qualname is not None else None


# Methods detected by `matches` (in order) if the draft model declares none.
_CHECKPOINT_DETECTION_ORDER = ("eagle", "eagle3", "dflash", "dspark", "mtp")


def _resolve_checkpoint_config_class(
    spec_config: "SpeculativeConfig",
    configured_cls: type[CheckpointSpeculatorConfig],
) -> type[CheckpointSpeculatorConfig]:
    """The built-in config for the draft checkpoint: detected if no method is
    configured, else the configured one (checked if it `validates_checkpoint`).
    """
    if spec_config.method is not None:
        if configured_cls.validates_checkpoint:
            recognized_cls = _recognize_checkpoint_config_class(spec_config)
            if not (
                # The same built-in speculator (or one based on it).
                recognized_cls.base_method == configured_cls.base_method
                if recognized_cls is not None
                else configured_cls.matches(spec_config)
            ):
                raise ValueError(
                    _checkpoint_mismatch_message(spec_config, recognized_cls)
                )
        return configured_cls

    declared_cls = _get_draft_declared_config_class(spec_config)
    if declared_cls is not None:
        assert issubclass(declared_cls, CheckpointSpeculatorConfig)
        return declared_cls
    for method in _CHECKPOINT_DETECTION_ORDER:
        config_cls = resolve_obj_by_qualname(_CONFIG_BY_METHOD[method])
        if config_cls.matches(spec_config):
            return config_cls
    return configured_cls


def _recognize_checkpoint_config_class(
    spec_config: "SpeculativeConfig",
) -> type[SpeculatorConfig] | None:
    """The config the checkpoint is structurally recognized as (by declaration
    or MTP model type, not by name)."""
    declared_cls = _get_draft_declared_config_class(spec_config)
    if declared_cls is not None:
        return declared_cls
    mtp_cls = resolve_obj_by_qualname(_CONFIG_BY_METHOD["mtp"])
    return mtp_cls if mtp_cls.matches(spec_config) else None


def _checkpoint_mismatch_message(
    spec_config: "SpeculativeConfig",
    recognized_cls: type[SpeculatorConfig] | None,
) -> str:
    draft = spec_config.draft_model_config
    if recognized_cls is not None:
        recognized = recognized_cls.base_method
        looks = f"looks like a {recognized!r} checkpoint"
        hint = f" Use method={recognized!r} if that is what you intended."
    else:
        looks = (
            "is not recognized as any drafter type (a plain causal LM can be "
            "used with method='draft_model')"
        )
        hint = ""
    return (
        f"speculative_config requested method={spec_config.method!r}, but the "
        f"draft model {draft.model!r} {looks} "
        f"(model_type={getattr(draft.hf_config, 'model_type', None)!r}, "
        f"architectures={getattr(draft, 'architectures', None)}). Pass a matching "
        f"method or a matching draft checkpoint.{hint}"
    )
