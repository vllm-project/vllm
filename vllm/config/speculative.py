# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import functools
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import Field, SkipValidation, field_validator, model_validator
from typing_extensions import Self

from vllm.config import LoadConfig
from vllm.config.cache import CacheDType
from vllm.config.kernel import MoEBackend
from vllm.config.model import HfOverrides, ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_text_config
from vllm.utils.hashing import safe_hash
from vllm.utils.import_utils import LazyLoader, has_arctic_inference
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    import vllm.model_executor.layers.quantization as me_quant
else:
    PretrainedConfig = Any

    me_quant = LazyLoader(
        "model_executor", globals(), "vllm.model_executor.layers.quantization"
    )

logger = init_logger(__name__)

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
    "qwen3_5_mtp",
    "longcat_flash_mtp",
    "bailing_hybrid_v3_mtp",
    "minimax_m3_mtp",
    "bailing_hybrid_mtp",
    "mtp",
    "kimi_k3_mtp",
    "pangu_ultra_moe_mtp",
    "step3p5_mtp",
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
    "medusa",
    "mlp_speculator",
    "draft_model",
    "suffix",
    "custom_class",
    EagleModelTypes,
    NgramGPUTypes,
    DSparkModelTypes,
]
RejectionSampleMethod = Literal["standard", "synthetic", "block"]
DraftSampleMethod = Literal["greedy", "probabilistic"]

_QWEN3_OMNI_TARGET_ARCHITECTURES = frozenset(
    {
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
    }
)
_QWEN3_OMNI_DSPARK_ARCHITECTURE = "Qwen3OmniDSparkModel"


def _is_qwen3_omni_target(model_config: ModelConfig) -> bool:
    hf_config = model_config.hf_config
    architectures = set(getattr(model_config, "architectures", ()) or ())
    architectures.update(getattr(hf_config, "architectures", ()) or ())
    return getattr(hf_config, "model_type", None) == "qwen3_omni_moe" or bool(
        architectures & _QWEN3_OMNI_TARGET_ARCHITECTURES
    )


def _get_qwen3_omni_text_config(model_config: ModelConfig) -> Any:
    thinker_config = getattr(model_config.hf_config, "thinker_config", None)
    text_config = getattr(thinker_config, "text_config", None)
    if text_config is None:
        text_config = getattr(model_config, "hf_text_config", None)
    return text_config


def _get_attention_head_dim(config: Any) -> int | None:
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        return head_dim
    hidden_size = getattr(config, "hidden_size", None)
    num_attention_heads = getattr(config, "num_attention_heads", None)
    if (
        isinstance(hidden_size, int)
        and isinstance(num_attention_heads, int)
        and num_attention_heads > 0
        and hidden_size % num_attention_heads == 0
    ):
        return hidden_size // num_attention_heads
    return None


def _get_nested_config_value(config: Any, section: str, name: str) -> Any:
    nested_config = getattr(config, section, None)
    if isinstance(nested_config, Mapping):
        return nested_config.get(name)
    return getattr(nested_config, name, None)


def _get_qwen3_dspark_value(config: Any, name: str) -> Any:
    # Match DFlashQwen3Model's precedence: dflash_config overrides
    # eagle_config, and top-level values are the modern fallback.
    value = _get_nested_config_value(config, "dflash_config", name)
    if value is None:
        value = _get_nested_config_value(config, "eagle_config", name)
    if value is None:
        value = getattr(config, name, None)
    return value


def _validate_qwen3_omni_dspark(
    target_model_config: ModelConfig,
    draft_model_config: ModelConfig,
    num_speculative_tokens: int,
) -> None:
    """Validate the standalone Qwen3-Omni DSpark checkpoint contract."""
    if not _is_qwen3_omni_target(target_model_config):
        return

    draft_hf_config = draft_model_config.hf_config
    draft_architectures = set(getattr(draft_model_config, "architectures", ()) or ())
    draft_architectures.update(getattr(draft_hf_config, "architectures", ()) or ())
    if _QWEN3_OMNI_DSPARK_ARCHITECTURE not in draft_architectures:
        raise ValueError(
            "Qwen3-Omni DSpark requires a standalone draft checkpoint with "
            f"architectures=['{_QWEN3_OMNI_DSPARK_ARCHITECTURE}']; generic "
            "Qwen3DSparkModel checkpoints must be converted first."
        )

    block_size = _get_qwen3_dspark_value(draft_hf_config, "block_size")
    if block_size is None:
        block_size = _get_qwen3_dspark_value(draft_hf_config, "dspark_block_size")
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError(
            "Qwen3-Omni DSpark requires a positive integer block_size in the "
            "draft config."
        )
    if num_speculative_tokens != block_size:
        raise ValueError(
            "Qwen3-Omni DSpark requires num_speculative_tokens to match the "
            f"trained block_size ({block_size}); got {num_speculative_tokens}."
        )

    target_text_config = _get_qwen3_omni_text_config(target_model_config)
    if target_text_config is None:
        raise ValueError(
            "Qwen3-Omni DSpark could not resolve the target thinker text_config."
        )
    target_hidden_size = getattr(
        target_text_config,
        "hidden_size",
        target_model_config.get_hidden_size(),
    )
    draft_target_hidden_size = getattr(draft_hf_config, "target_hidden_size", None)
    if draft_target_hidden_size != target_hidden_size:
        raise ValueError(
            "Qwen3-Omni DSpark draft target_hidden_size must match the target "
            f"text hidden size ({target_hidden_size}); got {draft_target_hidden_size}."
        )
    draft_hidden_size = getattr(draft_hf_config, "hidden_size", None)
    if draft_hidden_size != target_hidden_size:
        raise ValueError(
            "Qwen3-Omni DSpark draft hidden_size must match the target thinker "
            f"hidden size ({target_hidden_size}); got {draft_hidden_size}."
        )

    target_attention = {
        "num_attention_heads": getattr(target_text_config, "num_attention_heads", None),
        "num_key_value_heads": getattr(
            target_text_config,
            "num_key_value_heads",
            getattr(target_text_config, "num_attention_heads", None),
        ),
        "head_dim": _get_attention_head_dim(target_text_config),
    }
    draft_attention = {
        "num_attention_heads": getattr(draft_hf_config, "num_attention_heads", None),
        "num_key_value_heads": getattr(
            draft_hf_config,
            "num_key_value_heads",
            getattr(draft_hf_config, "num_attention_heads", None),
        ),
        "head_dim": _get_attention_head_dim(draft_hf_config),
    }
    for name, target_value in target_attention.items():
        if (
            not isinstance(target_value, int)
            or isinstance(target_value, bool)
            or target_value <= 0
        ):
            raise ValueError(
                "Qwen3-Omni DSpark requires a positive integer "
                f"{name} in the target thinker text_config; got {target_value}."
            )
        draft_value = draft_attention[name]
        if draft_value != target_value:
            raise ValueError(
                f"Qwen3-Omni DSpark draft {name} must match the target thinker "
                f"value ({target_value}); got {draft_value}."
            )

    target_layer_ids = _get_nested_config_value(
        draft_hf_config, "dflash_config", "target_layer_ids"
    )
    if target_layer_ids is None:
        target_layer_ids = getattr(draft_hf_config, "dspark_target_layer_ids", None)
    if target_layer_ids is None:
        target_layer_ids = getattr(draft_hf_config, "target_layer_ids", None)
    if not isinstance(target_layer_ids, (list, tuple)) or not target_layer_ids:
        raise ValueError(
            "Qwen3-Omni DSpark requires a non-empty target_layer_ids list in "
            "the draft config."
        )
    if any(
        not isinstance(layer_id, int) or isinstance(layer_id, bool)
        for layer_id in target_layer_ids
    ):
        raise ValueError(
            "Qwen3-Omni DSpark target_layer_ids must contain only integers."
        )
    if list(target_layer_ids) != sorted(set(target_layer_ids)):
        raise ValueError(
            "Qwen3-Omni DSpark target_layer_ids must be unique and strictly increasing."
        )
    target_num_layers = getattr(
        target_text_config,
        "num_hidden_layers",
        target_model_config.get_total_num_hidden_layers(),
    )
    if target_layer_ids[0] < 0 or target_layer_ids[-1] >= target_num_layers:
        raise ValueError(
            "Qwen3-Omni DSpark target_layer_ids must be zero-based text-layer "
            f"indices in [0, {target_num_layers - 1}]; got {target_layer_ids}."
        )

    if _get_qwen3_dspark_value(draft_hf_config, "use_aux_hidden_state") is not True:
        raise ValueError(
            "Qwen3-Omni DSpark requires use_aux_hidden_state=true because Omni "
            "conditioning is supplied through thinker hidden states."
        )

    markov_rank = getattr(draft_hf_config, "markov_rank", None)
    if (
        not isinstance(markov_rank, int)
        or isinstance(markov_rank, bool)
        or markov_rank <= 0
    ):
        raise ValueError(
            "Qwen3-Omni DSpark requires a positive integer markov_rank in the "
            "draft config."
        )
    markov_head_type = getattr(draft_hf_config, "markov_head_type", None)
    if markov_head_type != "vanilla":
        raise ValueError(
            "Qwen3-Omni DSpark currently requires markov_head_type='vanilla'; "
            f"got {markov_head_type!r}."
        )

    sample_from_anchor = getattr(draft_hf_config, "sample_from_anchor", True)
    bonus_anchor = getattr(draft_hf_config, "dspark_bonus_anchor", False)
    if sample_from_anchor is not True or bonus_anchor is not False:
        raise ValueError(
            "Qwen3-Omni DSpark requires sample_from_anchor=true and "
            "dspark_bonus_anchor=false."
        )

    target_vocab_size = getattr(
        target_text_config,
        "vocab_size",
        target_model_config.get_vocab_size(),
    )
    draft_input_vocab_size = getattr(draft_hf_config, "vocab_size", None)
    if (
        not isinstance(draft_input_vocab_size, int)
        or isinstance(draft_input_vocab_size, bool)
        or draft_input_vocab_size <= 0
    ):
        raise ValueError(
            "Qwen3-Omni DSpark input vocab_size must be a positive integer; "
            f"got {draft_input_vocab_size}."
        )
    draft_output_vocab_size = getattr(draft_hf_config, "draft_vocab_size", None)
    if draft_output_vocab_size is None:
        draft_output_vocab_size = draft_input_vocab_size
    if (
        not isinstance(draft_output_vocab_size, int)
        or isinstance(draft_output_vocab_size, bool)
        or not 0 < draft_output_vocab_size <= target_vocab_size
    ):
        raise ValueError(
            "Qwen3-Omni DSpark draft_vocab_size must be a positive integer no "
            f"larger than target vocab_size ({target_vocab_size}); got "
            f"{draft_output_vocab_size}."
        )

    noise_token_id = _get_nested_config_value(
        draft_hf_config, "dflash_config", "mask_token_id"
    )
    for name in (
        "mask_token_id",
        "dspark_noise_token_id",
        "pard_token",
        "ptd_token_id",
    ):
        if noise_token_id is None:
            noise_token_id = getattr(draft_hf_config, name, None)
        if noise_token_id is not None:
            break
    if (
        not isinstance(noise_token_id, int)
        or isinstance(noise_token_id, bool)
        or not 0 <= noise_token_id < draft_input_vocab_size
    ):
        raise ValueError(
            "Qwen3-Omni DSpark requires a valid mask/noise token id within the "
            f"draft input vocabulary [0, {draft_input_vocab_size - 1}]."
        )

    rope_configs = (
        getattr(draft_hf_config, "rope_parameters", None),
        getattr(draft_hf_config, "rope_scaling", None),
    )
    has_mrope = getattr(draft_hf_config, "mrope_section", None) is not None
    has_mrope = has_mrope or any(
        isinstance(rope_config, Mapping) and "mrope_section" in rope_config
        for rope_config in rope_configs
    )
    if has_mrope:
        raise ValueError(
            "Qwen3-Omni DSpark draft checkpoints must use logical 1-D RoPE and "
            "must not define mrope_section."
        )


@config
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    enforce_eager: bool | None = None
    """Override the default enforce_eager from model_config"""
    # General speculative decoding control
    num_speculative_tokens: int = Field(default=None, gt=0)  # type: ignore[assignment]
    """The number of speculative tokens, if provided. It will default to the
    number in the draft model config if present, otherwise, it is required."""
    model: str | None = None
    """The name of the draft model, eagle head, or additional weights, if
    provided."""
    method: SpeculativeMethod | None = None
    """The name of the speculative method to use. If users provide and set the
    `model` param, the speculative method type will be detected automatically
    if possible, if `model` param is not provided, the method name must be
    provided.

    If using `ngram` method, the related configuration `prompt_lookup_max` and
    `prompt_lookup_min` should be considered."""
    draft_tensor_parallel_size: int | None = Field(default=None, ge=1)
    """The degree of the tensor parallelism for the draft model. Can only be 1
    or the same as the target model's tensor parallel size."""
    tensor_parallel_size: int | None = None
    """Users should pass "draft_tensor_parallel_size". This parameter's purpose is to
    warn users when they mistakenly provide the wrong argument."""

    # Draft model configuration
    quantization: me_quant.QuantizationMethods | str | None = None
    """Quantization method that was used to quantize the draft model weights.
    If `None`, we assume the model weights are not quantized. Note that it only
    takes effect when using the draft model-based speculative method."""
    moe_backend: MoEBackend | None = None
    """MoE backend to use for the draft model. When `None`, the draft model
    inherits the target model's `--moe-backend` setting. Useful when the
    drafter and generator require different MoE kernels (e.g. quantized
    generator with unquantized drafter)."""
    attention_backend: AttentionBackendEnum | None = None
    """Attention backend to use for the draft model. When `None`, the backend is
    automatically selected. Useful when the drafter requires a different attention
    backend (e.g. DFlash needs a backend that supports non-causal attention)."""
    kv_cache_dtype: CacheDType | None = None
    """KV cache dtype for the draft model. When `None`, the draft inherits the
    target model's `--kv-cache-dtype`."""
    max_model_len: int | None = Field(default=None, ge=1)
    """The maximum model length of the draft model. Used when testing the
    ability to skip speculation for some sequences."""
    revision: str | None = None
    """The specific model version to use for the draft model. It can be a
    branch name, a tag name, or a commit id. If unspecified, will use the
    default version."""
    code_revision: str | None = None
    """The specific revision to use for the draft model code on Hugging Face
    Hub. It can be a branch name, a tag name, or a commit id. If unspecified,
    will use the default version."""

    # Advanced control
    disable_padded_drafter_batch: bool = False
    """Disable input padding for speculative decoding. If set to True,
    speculative input batches can contain sequences of different lengths,
    which may only be supported by certain attention backends. This currently
    only affects the EAGLE method of speculation."""
    use_local_argmax_reduction: bool = False
    """Use vocab-parallel local argmax instead of all-gathering full logits
    for draft token generation. Reduces communication from O(vocab_size) to
    O(2 * tp_size) per token. Only applies to greedy draft selection in
    non-tree speculation."""

    use_heterogeneous_vocab: bool = False
    """Allow draft and target models to use different vocabularies.
    When enabled, builds a token-level intersection at init and constrains
    draft logits to shared tokens only (TLI algorithm). Requires
    method='draft_model'."""

    # Ngram proposer configuration
    prompt_lookup_max: int | None = Field(default=None, ge=1)
    """Maximum size of ngram token window when using Ngram proposer, required
    when method is set to ngram."""
    prompt_lookup_min: int | None = Field(default=None, ge=1)
    """Minimum size of ngram token window when using Ngram proposer, if
    provided. Defaults to 1."""

    # Alternative drafting strategies
    parallel_drafting: bool = False
    """Enable parallel drafting, where all speculative tokens are generated
    in parallel rather than sequentially. This can improve performance but
    requires the speculative model be trained to support parallel drafting.
    Only compatible with EAGLE and draft model methods."""

    # required configuration params passed from engine
    target_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the target model."""
    target_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """The parallel configuration for the target model."""

    # dynamic speculative decoding control
    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]] | None = None
    """Batch-size schedule used to dynamically choose speculative-token count.

    Each entry is ``(range_start, range_end, num_speculative_tokens)`` with an
    inclusive batch-size range.
    """

    # params generated in the post-init stage
    draft_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the draft model initialized internal."""
    draft_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """The parallel configuration for the draft model initialized internal."""

    # Suffix decoding configuration
    suffix_decoding_max_tree_depth: int = 24
    """The maximum depth of the suffix decoding global and prompt trees. The
    tree depth limits the sum of the prefix match and speculation lengths."""

    suffix_decoding_max_cached_requests: int = 10000
    """The maximum number of requests to cache in the global suffix tree. If
    exceeded, will trigger eviction in FIFO order. If set to 0, the global
    suffix tree is disabled and past responses are not cached (prompt trees
    are still used)."""

    suffix_decoding_max_spec_factor: float = 1.0
    """The maximum spec factor for suffix decoding. The spec factor controls
    speculation lengths based on the prefix match length: max_spec_tokens =
    max_spec_factor * prefix_match_length."""

    suffix_decoding_min_token_prob: float = 0.1
    """The minimum token probability for suffix decoding. Will only speculate
    tokens with estimated probability (based on frequency counts) greater than
    or equal to this value."""

    draft_load_config: LoadConfig | None = None
    """Load config for the draft model. If not specified, will use the load
    config from the target model."""

    rejection_sample_method: RejectionSampleMethod = "standard"
    """The rejection sampling method to use. 'standard' uses probabilistic
    rejection sampling (with or without cached draft logits, controlled by
    draft_sample_method). 'synthetic' accepts draft tokens with a decaying
    probability calibrated to synthetic_acceptance_rate. 'block' uses block
    verification (Sun et al.), which jointly verifies the draft tokens as a
    block instead of one at a time."""

    synthetic_acceptance_rates: list[float] | None = None
    """Per-position *unconditional* acceptance rates for synthetic rejection
    sampling. Position i's entry is the marginal probability that the first
    i+1 draft tokens are all accepted; the list must have length
    num_speculative_tokens, each entry in [0, 1], and be monotonically
    non-increasing. Only valid when rejection_sample_method is 'synthetic'.
    Mutually exclusive with synthetic_acceptance_length."""

    synthetic_acceptance_length: float | None = None
    """Target mean acceptance length for synthetic rejection sampling, in
    [1, num_speculative_tokens + 1]. Resolved internally to
    synthetic_acceptance_rates. Only valid when rejection_sample_method is 'synthetic'.
    Mutually exclusive with synthetic_acceptance_rates."""

    enable_adaptive_verification: bool = False
    """Whether to adaptively size the draft-verification budget from per-request
    confidence. Currently only supported for method="dspark"."""

    @staticmethod
    def _acceptance_length_to_rates(length: float, n: int) -> list[float]:
        """Mean acceptance length to unconditional per-position rates, using
        the minimum-variance schedule."""
        num_drafts = length - 1  # expected number of accepted draft tokens
        num_full = int(num_drafts)
        return (
            [1.0] * num_full + [num_drafts - num_full] + [0.0] * (n - num_full - 1)
        )[:n]

    @staticmethod
    def _resolve_synthetic_acceptance_rates(
        n: int,
        rates: list[float] | None,
        length: float | None,
    ) -> list[float]:
        """Return per-position unconditional acceptance rates from exactly one
        of `rates` or `length` (validates range, length, and monotonicity)."""
        if (rates is None) == (length is None):
            raise ValueError(
                "rejection_sample_method='synthetic' requires exactly one of "
                "synthetic_acceptance_rates or synthetic_acceptance_length."
            )
        if rates is not None:
            if len(rates) != n:
                raise ValueError(
                    f"synthetic_acceptance_rates must have length {n}, got {rates}."
                )
            if not all(0.0 <= r <= 1.0 for r in rates):
                raise ValueError(
                    f"synthetic_acceptance_rates entries must be in [0, 1], "
                    f"got {rates}."
                )
            if any(rates[i] > rates[i - 1] for i in range(1, n)):
                raise ValueError(
                    f"synthetic_acceptance_rates must be non-increasing, got {rates}."
                )
            return list(rates)
        assert length is not None
        if not 1.0 <= length <= float(n + 1):
            raise ValueError(
                f"synthetic_acceptance_length must be in [1, {n + 1}], got {length}."
            )
        return SpeculativeConfig._acceptance_length_to_rates(length, n)

    draft_sample_method: DraftSampleMethod = "greedy"
    """How the draft model samples tokens. 'greedy' always picks the argmax
    token, and the draft probabilities are treated as one-hot during rejection
    sampling. 'probabilistic' samples stochastically from the draft
    distribution and uses the full draft logits for the probability ratio test
    during rejection sampling. This comes at the cost of additional GPU memory
    usage."""

    dspark_draft_topk: int | None = Field(default=None, ge=1)
    """For Qwen3 DSpark drafting, evaluate the Markov projection only for the
    top-k base-logit candidates. Requires draft tensor parallel size 1."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        # Eagle3 and extract_hidden_states affect the computation graph because
        # they return intermediate hidden states in addition to the final hidden state.
        uses_aux_hidden_states = self.method in (
            "eagle3",
            "extract_hidden_states",
            "dflash",
            "dspark",
        )
        factors.append(uses_aux_hidden_states)

        if uses_aux_hidden_states and self.draft_model_config is not None:
            factors.append(self.draft_model_config.compute_hash())

            # The specific layers used also affect the computation graph.
            layer_ids = getattr(
                self.draft_model_config.hf_config,
                "eagle_aux_hidden_state_layer_ids",
                None,
            )
            if layer_ids is not None:
                # Convert to tuple to make it hashable
                factors.append(tuple(layer_ids))

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        initial_architecture = hf_config.architectures[0]
        use_v32_mtp = hf_config.model_type in ("deepseek_v32", "glm_moe_dsa")
        if hf_config.model_type == "dots3_note":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            mtp_layer_types = getattr(hf_config, "mtp_layer_types", None)
            if mtp_layer_types is None:
                mtp_layer_types = ["sliding_attention"] * n_predict
            if len(mtp_layer_types) != n_predict:
                raise ValueError(
                    "mtp_layer_types must have one entry per MTP layer: "
                    f"got {len(mtp_layer_types)} for {n_predict} layers"
                )
            hf_config.layer_types = [*hf_config.layer_types, *mtp_layer_types]
            hf_config.model_type = "dots3_note_mtp"
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Dots3NoteMTPModel"]}
            )
        if hf_config.model_type in (
            "deepseek_v3",
            "deepseek_v32",
            "glm_moe_dsa",
        ):
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": [
                        "DeepseekV32MTPModel" if use_v32_mtp else "DeepSeekMTPModel"
                    ],
                }
            )
        if hf_config.model_type == "deepseek_v4":
            hf_config.model_type = "deepseek_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["DeepSeekV4MTPModel"]}
            )
        if hf_config.model_type in ("pangu_ultra_moe"):
            hf_config.model_type = "pangu_ultra_moe_mtp"
        if hf_config.model_type == "pangu_ultra_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["OpenPanguMTPModel"]}
            )

        if hf_config.model_type == "kimi_k3":
            # Kimi-K3 keeps the text-model fields (incl. the MTP layer count)
            # nested under ``text_config`` (a KimiLinearConfig).
            text_config = getattr(hf_config, "text_config", hf_config)
            n_predict = getattr(text_config, "num_nextn_predict_layers", None)
            hf_config.model_type = "kimi_k3_mtp"
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["KimiK3MTPModel"]}
            )

        if hf_config.architectures[0] == "MiMoForCausalLM":
            hf_config.model_type = "mimo_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["MiMoMTPModel"],
                }
            )

        if (arch := hf_config.architectures[0]) in (
            "MiMoV2ForCausalLM",
            "MiMoV2OmniForCausalLM",
        ):
            from vllm.model_executor.models.mimo_v2_mtp import (
                _MIMO_V2_PRO_NUM_MTP_LAYERS,
            )

            mtp_arch_maps = {
                "MiMoV2ForCausalLM": "MiMoV2MTPModel",
                "MiMoV2OmniForCausalLM": "MiMoV2OmniMTPModel",
            }

            hf_config.model_type = "mimo_v2_mtp"
            # vLLM currently supports only the first MiMo-V2 MTP layer.
            n_predict = _MIMO_V2_PRO_NUM_MTP_LAYERS
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "num_nextn_predict_layers": n_predict,
                    "architectures": [mtp_arch_maps[arch]],
                }
            )

        if hf_config.architectures[0] == "MiMoV2FlashForCausalLM":
            from vllm.model_executor.models.mimo_v2_mtp import (
                _MIMO_V2_FLASH_NUM_MTP_LAYERS,
            )

            hf_config.model_type = "mimo_v2_mtp"
            # vLLM currently supports only the first MiMo-V2 MTP layer.
            n_predict = _MIMO_V2_FLASH_NUM_MTP_LAYERS
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "num_nextn_predict_layers": n_predict,
                    "architectures": ["MiMoV2MTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeForCausalLM":
            hf_config.model_type = "glm4_moe_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeMTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeLiteForCausalLM":
            hf_config.model_type = "glm4_moe_lite_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeLiteMTPModel"],
                }
            )

        if hf_config.architectures[0] == "GlmOcrForConditionalGeneration":
            hf_config.model_type = "glm_ocr_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["GlmOcrMTPModel"],
                }
            )

        if hf_config.model_type == "ernie4_5_moe":
            hf_config.model_type = "ernie_mtp"
        if hf_config.model_type == "ernie_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ErnieMTPModel"]}
            )

        if hf_config.architectures[0] in (
            "NemotronH_Super_Omni_Reasoning_V3",
            "NemotronH_Omni_Reasoning_V3",
        ):
            # Promote VLM's text_config so MTP detection below fires correctly
            hf_config = hf_config.text_config

        if (
            hf_config.model_type in {"nemotron_h", "nemotron_h_puzzle"}
            and hasattr(hf_config, "num_nextn_predict_layers")
            and hf_config.num_nextn_predict_layers > 0
        ):
            # Check if this is an MTP variant
            hf_config.model_type = "nemotron_h_mtp"
        if hf_config.model_type == "nemotron_h_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["NemotronHMTPModel"]}
            )

        if hf_config.model_type == "qwen3_next":
            hf_config.model_type = "qwen3_next_mtp"
        if hf_config.model_type == "qwen3_next_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]}
            )

        architectures = getattr(hf_config, "architectures", []) or []
        if initial_architecture == "BailingMoeV3ForCausalLM":
            hf_config.model_type = "bailing_hybrid_v3_mtp"
        elif (
            hf_config.model_type == "bailing_hybrid"
            or "BailingMoeV2_5ForCausalLM" in architectures
        ):
            hf_config.model_type = "bailing_hybrid_mtp"
        if hf_config.model_type == "bailing_hybrid_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["BailingMoeV25MTPModel"],
                }
            )
        if hf_config.model_type == "bailing_hybrid_v3_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["BailingMoeV3MTPModel"],
                }
            )

        if hf_config.model_type == "exaone_moe":
            hf_config.model_type = "exaone_moe_mtp"
        if hf_config.model_type == "exaone_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ExaoneMoeMTP"]}
            )
        if "exaone4_5" in hf_config.model_type:
            hf_config.model_type = "exaone4_5_mtp"
        if hf_config.model_type == "exaone4_5_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Exaone4_5_MTP"]}
            )
        if hf_config.model_type in (
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_5_text",
            "qwen3_5_moe_text",
        ):
            # Checkpoints that ship only the text config resolve to the
            # `qwen3_5_text` / `qwen3_5_moe_text` model types and carry the
            # same `mtp_num_hidden_layers` field as the multimodal ones.
            is_moe = hf_config.model_type in ("qwen3_5_moe", "qwen3_5_moe_text")
            hf_config.model_type = "qwen3_5_mtp"
            n_predict = getattr(hf_config, "mtp_num_hidden_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"],
                }
            )
        if hf_config.model_type in ("intern_s2_preview", "interns2_mobius"):
            is_mobius = hf_config.model_type == "interns2_mobius"
            text_config = getattr(hf_config, "text_config", None)
            is_moe = is_mobius or (
                getattr(text_config, "model_type", None) == "qwen3_5_moe_text"
            )
            if is_mobius:
                assert text_config is not None
                text_config.model_type = "qwen3_5_moe_text"
                text_config.num_experts = text_config.mtp_num_experts
                text_config.num_experts_per_tok = text_config.mtp_num_experts_per_tok
            hf_config.model_type = "qwen3_5_mtp"
            n_predict = getattr(text_config, "mtp_num_hidden_layers", None)
            architecture = (
                "InternS2MobiusMTP"
                if is_mobius
                else ("Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP")
            )
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": [architecture],
                }
            )
        if hf_config.model_type in ("longcat_flash", "longcat_flash_ngram"):
            hf_config.model_type = "longcat_flash_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["LongCatFlashMTPModel"]}
            )

        if hf_config.model_type in ("step3p5", "step3p7") or hf_config.architectures[
            0
        ] in ("Step3p5ForCausalLM", "Step3p7ForConditionalGeneration"):
            quantization_config = getattr(hf_config, "quantization_config", None)
            hf_config = getattr(hf_config, "text_config", hf_config)
            if (
                quantization_config is not None
                and getattr(hf_config, "quantization_config", None) is None
            ):
                hf_config.update({"quantization_config": quantization_config})
            hf_config.model_type = "step3p5_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update({"n_predict": n_predict, "architectures": ["Step3p5MTP"]})

        if initial_architecture == "MistralLarge3ForCausalLM":
            hf_config.update({"architectures": ["EagleMistralLarge3ForCausalLM"]})

        if hf_config.model_type == "hy_v3":
            hf_config.model_type = "hy_v3_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["HYV3MTPModel"]}
            )

        if hf_config.model_type == "hy_v4":
            hf_config.model_type = "hy_v4_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["HYV4MTPModel"]}
            )

        if hf_config.model_type in ("inkling_mm_model", "inkling_model"):
            mtp_config = getattr(hf_config, "mtp_config", None) or {}
            hf_config = getattr(hf_config, "text_config", hf_config)
            checkpoint_depths = mtp_config.get("num_nextn_predict_layers", 0)
            if checkpoint_depths < 1:
                raise ValueError("The Inkling checkpoint does not contain MTP weights")
            hf_config.model_type = "inkling_mtp"
            hf_config.update(
                {
                    "n_predict": checkpoint_depths,
                    "num_nextn_predict_layers": checkpoint_depths,
                    "chain_hidden_post_norm": mtp_config.get(
                        "chain_hidden_post_norm", False
                    ),
                    "local_layer_ids": mtp_config.get("local_layer_ids", []),
                    "architectures": ["InklingMTPModel"],
                }
            )

        if hf_config.model_type in ("gemma4_assistant", "gemma4_unified_assistant"):
            hf_config.model_type = "gemma4_mtp"
            text_config = getattr(hf_config, "text_config", hf_config)
            # The assistant runs all decoder layers in a single forward
            # call to produce one draft token, so n_predict=1.
            # num_kv_shared_layers must be 0: cross-model KV sharing is
            # set up by the proposer after model construction.
            if hasattr(text_config, "num_kv_shared_layers"):
                text_config.num_kv_shared_layers = 0
            hf_config.update({"n_predict": 1, "architectures": ["Gemma4MTPModel"]})

        if (
            hf_config.model_type == "minimax_m3_vl"
            or initial_architecture == "MiniMaxM3SparseForConditionalGeneration"
        ):
            # MTP modules live on the language model of this VL checkpoint, so
            # promote text_config before rewriting it into an MTP config.
            quantization_config = getattr(hf_config, "quantization_config", None)
            hf_config = getattr(hf_config, "text_config", hf_config)
            if (
                quantization_config is not None
                and getattr(hf_config, "quantization_config", None) is None
            ):
                hf_config.update({"quantization_config": quantization_config})
            hf_config.model_type = "minimax_m3_mtp"
            n_predict = getattr(hf_config, "num_mtp_modules", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["MiniMaxM3MTP"]}
            )
        elif (
            hf_config.model_type == "minimax_m3_mtp"
            or initial_architecture == "MiniMaxM3MTP"
        ):
            # Standalone MTP checkpoints already use a flat MTP config with no
            # VL wrapper / text_config to promote, so just normalize the
            # architecture and derive n_predict from num_mtp_modules.
            n_predict = getattr(hf_config, "num_mtp_modules", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["MiniMaxM3MTP"]}
            )
        if hf_config.model_type == "glm5_next":
            hf_config.model_type = "glm5_next_mtp"
            n_predict = hf_config.num_nextn_predict_layers
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Glm5NextMTPModel"]}
            )

        return hf_config

    @staticmethod
    def _apply_composed_hf_override(
        target_hf_overrides: Callable[[PretrainedConfig], PretrainedConfig],
        hf_config: PretrainedConfig,
    ) -> PretrainedConfig:
        hf_config = SpeculativeConfig.hf_config_override(hf_config)
        return target_hf_overrides(hf_config)

    @staticmethod
    def compose_draft_hf_overrides(
        target_hf_overrides: HfOverrides | None,
    ) -> Callable[[PretrainedConfig], PretrainedConfig]:
        """Build the ``hf_overrides`` for the draft ``ModelConfig``.

        Callable overrides on the target are config-to-config transforms
        (e.g. test harnesses shrinking ``num_hidden_layers``) and must also
        reach the draft config — otherwise a draft belonging to a large
        target is instantiated at full size even when the target is shrunk.
        Dict overrides are target-specific key patches and are not applied
        to the draft.

        The composed override must stay picklable: the draft ``ModelConfig``
        is sent to spawned engine-core processes, so a local closure would
        fail with ``Can't get local object`` during pickling. Bind the
        target via ``functools.partial`` over a module-referenceable static
        method instead.
        """
        if not callable(target_hf_overrides):
            return SpeculativeConfig.hf_config_override

        return functools.partial(
            SpeculativeConfig._apply_composed_hf_override, target_hf_overrides
        )

    @staticmethod
    def _is_custom_proposer_path(model: str | None) -> bool:
        """True if ``model`` is a dotted import path (e.g. ``pkg.MyProposer``)."""
        if model is None:
            return False
        if model.startswith(("http://", "https://", "file://")):
            return False
        if "/" in model:
            return False
        parts = model.split(".")
        return len(parts) >= 2 and all(part.isidentifier() for part in parts)

    def __post_init__(self):
        # Note: "method" is a new parameter that helps to extend the
        # configuration of non-model-based proposers, and the "model" parameter
        # will be used to set the draft model, eagle head, or additional weight
        # when needed. If users do not specify "method", the speculative method
        # will be detected automatically if possible. If the speculative method
        # can not be detected, it will be considered as the "draft_model" by
        # default.

        # infer method from user args
        if self.method is None and SpeculativeConfig._is_custom_proposer_path(
            self.model
        ):
            self.method = "custom_class"
        elif self.method is None:
            if self.model in ("ngram", "[ngram]"):
                self.method = "ngram"
            else:
                self.method = "draft_model"

        if self.method in get_args(MTPModelTypes) and self.method != "mtp":
            logger.warning(
                "method `%s` is deprecated and replaced with mtp.", self.method
            )
            self.method = "mtp"

        if self.model is None and self.num_speculative_tokens is not None:
            if self.method == "mtp":
                if self.target_model_config is None:
                    raise ValueError("target_model_config must be present for mtp")
                # use the draft model from the same model:
                # Use model_weights if set (e.g. runai_streamer replaces
                # model with a local cache dir but keeps the original path
                # in model_weights), so the draft model can load weights
                # from the same source as the target model.
                self.model = (
                    self.target_model_config.model_weights
                    or self.target_model_config.model
                )
                # Align the quantization of draft model for cases such as
                # --quantization fp8 with a bf16 checkpoint.
                if not self.quantization:
                    self.quantization = self.target_model_config.quantization
            elif self.method == "dspark":
                # DeepSeek DSpark can ship the weights inside the target checkpoint
                if self.target_model_config is None:
                    raise ValueError("target_model_config must be present for dspark")
                self.model = self.target_model_config.model
                if not self.quantization:
                    self.quantization = self.target_model_config.quantization
            elif self.method in ("ngram", "[ngram]"):
                self.model = "ngram"
            elif self.method == "ngram_gpu":
                self.model = "ngram_gpu"
            elif self.method == "suffix":
                self.model = "suffix"
            elif self.method == "extract_hidden_states":
                self.model = "extract_hidden_states"
            elif self.method == "custom_class":
                # method was set explicitly, but model should already contain the
                # custom module path. If not, this is a configuration error.
                if self.model is None:
                    raise ValueError(
                        "method='custom_class' requires 'model' to contain the "
                        "custom proposer module path (e.g., 'my_module.MyProposer')."
                    )
            else:
                raise ValueError(
                    "num_speculative_tokens was provided but without speculative model."
                )

        if self.method in ("ngram", "[ngram]"):
            self.method = "ngram"

        if self.method in ("ngram", "ngram_gpu"):
            # Set default values if not provided
            if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                if self.prompt_lookup_max is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                if self.prompt_lookup_min is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}"
                )

            # TODO: current we still need extract vocab_size from target model
            # config, in future, we may try refactor it out, and set
            # draft related config as None here.
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        elif self.method == "suffix":
            self._validate_suffix_decoding()
        elif self.method == "custom_class":
            # Custom class proposer does not need a draft model.
            # It will dynamically load the user-provided class at runtime.
            logger.warning_once(
                "Using a custom class-based proposer backend. This is an "
                "experimental feature and the proposer interface is subject to "
                "breaking changes in future vLLM releases."
            )
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        elif self.method == "extract_hidden_states":
            from vllm.transformers_utils.configs.extract_hidden_states import (
                ExtractHiddenStatesConfig,
            )

            # ExtractHiddenStatesModel is instantiated manually in load_model()
            # We just need to store the target model config for KV cache shape info
            self.model = "extract_hidden_states"
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if hasattr(self.draft_model_config, "hf_config"):
                hf_config = self.draft_model_config.hf_config.to_dict()
            elif (
                isinstance(self.draft_model_config, dict)
                and "hf_config" in self.draft_model_config
            ):
                hf_config = self.draft_model_config["hf_config"]
            else:
                hf_config = {}

            self.draft_model_config = copy.copy(self.target_model_config)
            self.draft_model_config.hf_config = ExtractHiddenStatesConfig(
                self.draft_model_config.hf_config, **hf_config
            )
            self.update_arch_()
            self.draft_parallel_config = self.target_parallel_config

        else:
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if self.model is not None:
                # Old-format Medusa checkpoints (e.g. FasterDecoding/medusa-*)
                # lack a model_type key in config.json, so AutoConfig cannot
                # detect them. When the method is explicitly "medusa", inject
                # model_type so MedusaConfig.from_pretrained is used instead.
                draft_hf_overrides: HfOverrides
                if self.method == "medusa":
                    draft_hf_overrides = {"model_type": "medusa"}
                else:
                    # Compose any callable hf_overrides set on the target so the
                    # draft config receives the same transform (e.g. the test
                    # shrink). Dict overrides stay target-only.
                    draft_hf_overrides = SpeculativeConfig.compose_draft_hf_overrides(
                        self.target_model_config.hf_overrides
                    )
                self.draft_model_config = ModelConfig(
                    model=self.model,
                    runner="draft",
                    tokenizer=(
                        self.model
                        if self.use_heterogeneous_vocab
                        else self.target_model_config.tokenizer
                    ),
                    tokenizer_mode=self.target_model_config.tokenizer_mode,
                    trust_remote_code=self.target_model_config.trust_remote_code,
                    allowed_local_media_path=self.target_model_config.allowed_local_media_path,
                    allowed_media_domains=self.target_model_config.allowed_media_domains,
                    dtype=self.target_model_config.dtype,
                    seed=self.target_model_config.seed,
                    revision=self.revision,
                    code_revision=self.code_revision,
                    tokenizer_revision=self.target_model_config.tokenizer_revision,
                    max_model_len=self.max_model_len,  # type: ignore[arg-type]
                    spec_target_max_model_len=self.target_model_config.max_model_len,
                    quantization=self.quantization,
                    enforce_eager=self.target_model_config.enforce_eager,
                    max_logprobs=self.target_model_config.max_logprobs,
                    hf_overrides=draft_hf_overrides,
                    config_format=self.target_model_config.config_format,
                )

                # Old-format Medusa checkpoints (e.g. FasterDecoding/medusa-*)
                # omit vocab_size in config.json, so MedusaConfig falls back to
                # its default (32001). Align with the target model's vocab size
                # to avoid shape mismatches when loading LM-head weights.
                if self.method == "medusa":
                    target_vocab = self.target_model_config.hf_config.vocab_size
                    draft_hf = self.draft_model_config.hf_config
                    if draft_hf.vocab_size != target_vocab:
                        draft_hf.vocab_size = target_vocab
                        draft_hf.truncated_vocab_size = target_vocab

                # Automatically detect the method
                if self.method in ("eagle", "eagle3", "dflash", "dspark"):
                    pass
                # examples:
                # yuhuili/EAGLE-LLaMA3-Instruct-8B
                # yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
                # AngelSlim/Qwen3-8B_eagle3
                # deepseek-ai/dspark_qwen3_8b_block7
                elif "eagle-" in self.draft_model_config.model.lower():
                    self.method = "eagle"
                elif "eagle3" in self.draft_model_config.model.lower():
                    self.method = "eagle3"
                elif (
                    "dflash" in self.draft_model_config.model.lower()
                    or "MuseGlimmerAssistantModel"
                    in self.draft_model_config.architectures
                ):
                    self.method = "dflash"
                elif (
                    "dspark" in self.draft_model_config.model.lower()
                    or "Qwen3DSparkModel" in self.draft_model_config.architectures
                    or _QWEN3_OMNI_DSPARK_ARCHITECTURE
                    in self.draft_model_config.architectures
                    or "Gemma4DSparkModel" in self.draft_model_config.architectures
                    or (
                        "DSparkDraftModel" in self.draft_model_config.architectures
                        and self.draft_model_config.hf_config.model_type == "qwen3"
                    )
                ):
                    self.method = "dspark"
                elif self.draft_model_config.hf_config.model_type == "medusa":
                    self.method = "medusa"
                elif self.draft_model_config.hf_config.model_type == "mlp_speculator":
                    self.method = "mlp_speculator"
                elif self.draft_model_config.hf_config.model_type in get_args(
                    MTPModelTypes
                ):
                    self.method = "mtp"
                    if (
                        self.num_speculative_tokens > 1
                        and self.draft_model_config.hf_config.model_type
                        not in ("step3p5_mtp", "inkling_mtp")
                    ):
                        logger.warning(
                            "Enabling num_speculative_tokens > 1 will run "
                            "multiple times of forward on same MTP layer"
                            ",which may result in lower acceptance rate"
                        )
                elif self.method == "draft_model":
                    pass
                else:
                    raise NotImplementedError(
                        f"Unsupported speculative method: '{self.method}'"
                    )

                if self.method in ("eagle", "eagle3"):
                    # EAGLE drafts share the target's positional space; a
                    # draft checkpoint with a smaller max_position_embeddings
                    # than the target under-sizes its rotary cache (#48894).
                    SpeculativeConfig._maybe_override_draft_max_position_embeddings(
                        self.draft_model_config.hf_config,
                        self.target_model_config.max_model_len,
                    )

                # Replace hf_config for EAGLE draft_model
                if self.method in ("eagle", "eagle3", "dflash"):
                    from vllm.transformers_utils.configs.eagle import EAGLEConfig
                    from vllm.transformers_utils.configs.speculators import (
                        SpeculatorsConfig,
                    )

                    if isinstance(
                        self.draft_model_config.hf_config,
                        (EAGLEConfig, SpeculatorsConfig),
                    ):
                        pass
                    else:
                        eagle_config = EAGLEConfig(
                            self.draft_model_config.hf_config,
                            method=self.method,
                            model_type="eagle",
                        )
                        self.draft_model_config.hf_config = eagle_config
                        self.update_arch_()

                if (
                    self.method == "dspark"
                    and "DSparkDraftModel" in self.draft_model_config.architectures
                    and self.draft_model_config.hf_config.model_type == "qwen3"
                ):
                    self.draft_model_config.hf_config.architectures = [
                        "Qwen3DSparkModel"
                    ]
                    self.update_arch_()
                elif self.method == "dspark" and (
                    "Qwen3DSparkModel" not in self.draft_model_config.architectures
                    and _QWEN3_OMNI_DSPARK_ARCHITECTURE
                    not in self.draft_model_config.architectures
                    and "Gemma4DSparkModel" not in self.draft_model_config.architectures
                    and "K3DSparkModel" not in self.draft_model_config.architectures
                ):
                    # DeepSeek-V4 DSpark reuses the full DeepSeek-V4 config
                    # and its weights ship in the target checkpoint.
                    self.draft_model_config.hf_config.model_type = "deepseek_v4"
                    self.draft_model_config.hf_config.architectures = [
                        "DSparkDraftModel"
                    ]
                    self.draft_model_config.quantization = (
                        self.target_model_config.quantization
                    )
                    self.update_arch_()
                elif (
                    self.method == "dspark"
                    and "Gemma4DSparkModel" in self.draft_model_config.architectures
                ):
                    # Normalize the self-contained Gemma4 draft's config keys to
                    # the DSpark conventions.
                    hf = self.draft_model_config.hf_config
                    if (
                        getattr(hf, "dspark_target_layer_ids", None) is None
                        and getattr(hf, "target_layer_ids", None) is not None
                    ):
                        hf.dspark_target_layer_ids = hf.target_layer_ids
                    if (
                        getattr(hf, "n_predict", None) is None
                        and getattr(hf, "block_size", None) is not None
                    ):
                        hf.n_predict = hf.block_size

                if self.method in ("dflash", "dspark"):
                    self.parallel_drafting = True

                if self.num_speculative_tokens is not None and hasattr(
                    self.draft_model_config.hf_config, "num_lookahead_tokens"
                ):
                    self.draft_model_config.hf_config.num_lookahead_tokens = (
                        self.num_speculative_tokens
                    )

                n_predict = getattr(
                    self.draft_model_config.hf_config, "n_predict", None
                )
                if n_predict is not None:
                    if self.num_speculative_tokens is None:
                        # Default to max value defined in draft model config.
                        self.num_speculative_tokens = n_predict
                    elif (
                        self.num_speculative_tokens > n_predict
                        and self.num_speculative_tokens % n_predict != 0
                    ):
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
                            f"num_speculative_tokens:{self.num_speculative_tokens}"
                            f" must be divisible by {n_predict=}"
                        )

                if self.num_speculative_tokens is None:
                    raise ValueError(
                        "A speculative model was provided, but "
                        "`num_speculative_tokens` was not provided"
                    )

                if self.dspark_draft_topk is not None and self.method != "dspark":
                    raise ValueError("dspark_draft_topk is only supported by DSpark")

                dspark_draft_topk = None
                if self.method == "dspark":
                    hf_config = self.draft_model_config.hf_config
                    dspark_draft_topk = self.dspark_draft_topk
                    if dspark_draft_topk is None:
                        dspark_draft_topk = getattr(
                            hf_config, "dspark_draft_topk", None
                        )
                    if dspark_draft_topk is not None:
                        draft_vocab_size = (
                            getattr(hf_config, "draft_vocab_size", None)
                            or hf_config.vocab_size
                        )
                        if not 1 <= dspark_draft_topk <= draft_vocab_size:
                            raise ValueError(
                                "dspark_draft_topk must be between 1 and the "
                                f"draft vocabulary size ({draft_vocab_size})"
                            )
                        if (
                            "Qwen3DSparkModel"
                            not in self.draft_model_config.architectures
                            and _QWEN3_OMNI_DSPARK_ARCHITECTURE
                            not in self.draft_model_config.architectures
                        ):
                            raise ValueError(
                                "dspark_draft_topk is only supported by "
                                "Qwen3DSparkModel and Qwen3OmniDSparkModel"
                            )
                        hf_config.dspark_draft_topk = dspark_draft_topk

                    assert self.target_model_config is not None
                    _validate_qwen3_omni_dspark(
                        self.target_model_config,
                        self.draft_model_config,
                        self.num_speculative_tokens,
                    )

                self.draft_tensor_parallel_size = (
                    SpeculativeConfig._verify_and_get_draft_tp(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size,
                        self.draft_model_config.hf_config,
                    )
                )
                self.draft_model_config.max_model_len = (
                    SpeculativeConfig._maybe_override_draft_max_model_len(
                        self.max_model_len,
                        self.draft_model_config.max_model_len,
                        self.target_model_config.max_model_len,
                    )
                )

                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config, self.draft_tensor_parallel_size
                    )
                )

        if self.method != "dspark" and self.enable_adaptive_verification:
            raise ValueError("Adaptive verification only supported with DSpark")

        return self

    def _validate_suffix_decoding(self):
        if not has_arctic_inference():
            raise ImportError(
                "Arctic Inference is required for suffix decoding. "
                "Install via `pip install arctic-inference==0.1.1`."
            )
        if self.num_speculative_tokens is None:
            # Suffix decoding decides the actual number of speculative tokens
            # dynamically and treats num_speculative_tokens as a maximum limit.
            self.num_speculative_tokens = self.suffix_decoding_max_tree_depth
            logger.warning(
                "Defaulted num_speculative_tokens to %s for suffix decoding.",
                self.num_speculative_tokens,
            )
        # Validate values
        if self.suffix_decoding_max_tree_depth < 1:
            raise ValueError(
                f"suffix_decoding_max_tree_depth="
                f"{self.suffix_decoding_max_tree_depth} must be >= 1"
            )
        if self.suffix_decoding_max_cached_requests < 0:
            raise ValueError(
                f"suffix_decoding_max_cached_requests="
                f"{self.suffix_decoding_max_cached_requests} must be >= 0"
            )
        if self.suffix_decoding_max_spec_factor < 0:
            raise ValueError(
                f"suffix_decoding_max_spec_factor="
                f"{self.suffix_decoding_max_spec_factor} must be >= 0"
            )
        if not 0 <= self.suffix_decoding_min_token_prob <= 1:
            raise ValueError(
                f"suffix_decoding_min_token_prob="
                f"{self.suffix_decoding_min_token_prob} must be in [0, 1]"
            )

    @staticmethod
    def _maybe_override_draft_max_model_len(
        speculative_max_model_len: int | None,
        draft_max_model_len: int,
        target_max_model_len: int,
    ) -> int:
        """Determine the max sequence len for the draft model. This is usually
        the draft_max_model_len, but may be the target_max_model_len if it is
        less than the draft_max_model_len, or may be speculative_max_model_len
        if it is specified.

        This is necessary so that sequences do not exceed the capacity of the
        draft model or the target model.

        speculative_max_model_len is mainly used for testing that sequences can
        skip speculation.
        """

        if speculative_max_model_len is not None:
            if speculative_max_model_len > draft_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {draft_max_model_len=}"
                )

            if speculative_max_model_len > target_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {target_max_model_len=}"
                )

            return speculative_max_model_len

        result = min(
            draft_max_model_len,
            target_max_model_len,
        )
        if result != draft_max_model_len:
            logger.info(
                "Overriding draft model max model len from %d to %d",
                draft_max_model_len,
                result,
            )
        return result

    @staticmethod
    def _maybe_override_draft_max_position_embeddings(
        draft_hf_config: PretrainedConfig,
        target_max_model_len: int,
    ) -> None:
        """Raise an EAGLE draft's max_position_embeddings up to the target's.

        The proposer feeds the draft positions up to the target's
        max_model_len, while max_position_embeddings sizes the draft's
        rotary cos_sin_cache. A smaller checkpoint value (e.g. 2048 for
        yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) makes that cache gather go
        out of bounds (#48894).

        Args:
            draft_hf_config: The draft model's HF config, mutated in place.
            target_max_model_len: The target model's max_model_len.
        """
        draft_max_position_embeddings = getattr(
            draft_hf_config, "max_position_embeddings", None
        )
        if (
            draft_max_position_embeddings is None
            or draft_max_position_embeddings >= target_max_model_len
        ):
            return
        logger.info(
            "Overriding draft model max_position_embeddings from %d to the "
            "target model's max_model_len (%d); EAGLE drafts share the "
            "target's positional space.",
            draft_max_position_embeddings,
            target_max_model_len,
        )
        draft_hf_config.max_position_embeddings = target_max_model_len

    @staticmethod
    def _verify_and_get_draft_tp(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int | None,
        draft_hf_config: PretrainedConfig,
    ) -> int:
        """
        Verifies and adjusts the tensor parallel size for a draft model
        specified using speculative_draft_tensor_parallel_size.
        """
        # If speculative_draft_tensor_parallel_size is unset then set it
        # appropriately else verify that it is set correctly.
        if speculative_draft_tensor_parallel_size is None:
            if draft_hf_config.model_type == "mlp_speculator":
                speculative_draft_tensor_parallel_size = 1
                if target_parallel_config.tensor_parallel_size > 1:
                    logger.warning(
                        "%s cannot currently be run with tp>1; "
                        "setting speculative_draft_tensor_parallel_size=1",
                        draft_hf_config.model_type,
                    )
            else:
                speculative_draft_tensor_parallel_size = (
                    target_parallel_config.tensor_parallel_size
                )
        elif speculative_draft_tensor_parallel_size not in (
            1,
            target_parallel_config.tensor_parallel_size,
        ):
            raise ValueError(
                f"{speculative_draft_tensor_parallel_size=} cannot be "
                f"other value than 1 or target model tensor_parallel_size"
            )
        return speculative_draft_tensor_parallel_size

    def update_arch_(self):
        """
        EagleConfig and ExtractHiddenStatesConfig update architectures, so update all
        architectures-related fields in self.draft_model_config
        """
        self.draft_model_config.hf_text_config = get_hf_text_config(
            self.draft_model_config.hf_config
        )
        self.draft_model_config.model_arch_config = (
            self.draft_model_config.get_model_arch_config()
        )
        model_info, arch = self.draft_model_config.registry.inspect_model_cls(
            self.draft_model_config.architectures,
            self.draft_model_config,
        )
        self.draft_model_config._model_info = model_info
        self.draft_model_config._architecture = arch

    @staticmethod
    def create_draft_parallel_config(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int,
    ) -> ParallelConfig:
        """Create a parallel config for use by the draft worker.

        This is mostly a copy of the target parallel config, except the tp_size.
        """
        draft_parallel_config = ParallelConfig(
            pipeline_parallel_size=target_parallel_config.pipeline_parallel_size,
            tensor_parallel_size=speculative_draft_tensor_parallel_size,
            distributed_executor_backend=target_parallel_config.distributed_executor_backend,
            max_parallel_loading_workers=target_parallel_config.max_parallel_loading_workers,
            disable_custom_all_reduce=target_parallel_config.disable_custom_all_reduce,
            ray_workers_use_nsight=target_parallel_config.ray_workers_use_nsight,
            placement_group=target_parallel_config.placement_group,
        )

        return draft_parallel_config

    @field_validator("attention_backend", mode="before")
    @classmethod
    def _parse_attention_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            if value.lower() == "auto":
                return None
            return AttentionBackendEnum[value.upper()]
        return value

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        if self.tensor_parallel_size is not None:
            raise ValueError(
                "'tensor_parallel_size' is not a valid argument in the "
                "speculative_config. Please pass 'draft_tensor_parallel_size' instead."
            )

        if self.num_speculative_tokens is None:
            raise ValueError(
                "num_speculative_tokens must be provided with "
                "speculative model unless the draft model config contains an "
                "n_predict parameter."
            )

        if self.num_speculative_tokens <= 0:
            raise ValueError(
                "Expected num_speculative_tokens to be greater "
                f"than zero ({self.num_speculative_tokens})."
            )

        if self.rejection_sample_method == "synthetic":
            # Consolidate to per-position rates
            self.synthetic_acceptance_rates = self._resolve_synthetic_acceptance_rates(
                self.num_speculative_tokens,
                self.synthetic_acceptance_rates,
                self.synthetic_acceptance_length,
            )
            self.synthetic_acceptance_length = None
        elif (
            self.synthetic_acceptance_rates is not None
            or self.synthetic_acceptance_length is not None
        ):
            raise ValueError(
                "synthetic_acceptance_rates / synthetic_acceptance_length "
                "are only valid with rejection_sample_method='synthetic'."
            )

        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config
            )

        if self.use_heterogeneous_vocab and not self.uses_draft_model():
            raise ValueError(
                "use_heterogeneous_vocab only works with method='draft_model'"
            )

        if self.use_heterogeneous_vocab and self.draft_sample_method != "greedy":
            raise ValueError(
                "use_heterogeneous_vocab currently only supports greedy draft "
                "sampling. Set draft_sample_method='greedy' (the default) or "
                "omit it."
            )

        if not self.use_heterogeneous_vocab:
            self.verify_equal_vocab_size_if_draft_model()
        return self

    def verify_equal_vocab_size_if_draft_model(self):
        if (
            self.method == "draft_model"
            and self.target_model_config is not None
            and self.draft_model_config is not None
        ):
            target_vocab_size = self.target_model_config.get_vocab_size()
            draft_vocab_size = self.draft_model_config.get_vocab_size()
            if target_vocab_size != draft_vocab_size:
                raise ValueError(
                    f"Target and draft model should have the same vocabulary size. "
                    f"Target model vocab_size={target_vocab_size}. "
                    f"Draft model vocab_size={draft_vocab_size}. "
                    f"Using models with different tokenizers can cause out-of-bounds "
                    f"errors during speculative decoding."
                )

    @property
    def max_num_new_slots_for_drafting(self) -> int:
        """Return the maximum additional drafting slots per request.

        The scheduler budget already includes one query slot per decoding request.
        Let K be ``num_speculative_tokens``. Standard configurations require:

        ==================== ============= ======== ================
        Algorithm            Method        Parallel Additional slots
        ==================== ============= ======== ================
        EAGLE3               eagle3        No       0
        P-EAGLE              eagle3        Yes      K - 1
        DFlash               dflash        Yes      K
        DSpark               dspark        Yes      K - 1
        MTP                  mtp           No       0
        N-gram               ngram         No       0
        Draft model          draft_model   No       1
        PARD                 draft_model   Yes      K
        ==================== ============= ======== ================
        """
        num_draft_tokens = self.num_speculative_tokens

        if self.use_dflash():
            # DFlash uses one bonus query followed by K mask queries.
            return num_draft_tokens

        if self.parallel_drafting:
            if self.uses_draft_model():
                # PARD does not shift the existing input, so all K query
                # positions require additional slots.
                return num_draft_tokens

            # The existing query is reused; only masked queries need new slots.
            return num_draft_tokens - 1

        if self.uses_draft_model():
            # The autoregressive draft-model input retains one unsliced token.
            return 1

        return 0

    def use_gemma4_mtp(self) -> bool:
        return (
            self.method == "mtp"
            and self.draft_model_config is not None
            and getattr(self.draft_model_config.hf_config, "model_type", None)
            == "gemma4_mtp"
        )

    def use_step3p5_mtp(self) -> bool:
        return (
            self.method == "mtp"
            and self.draft_model_config is not None
            and getattr(self.draft_model_config.hf_config, "model_type", None)
            == "step3p5_mtp"
        )

    def use_eagle(self) -> bool:
        # NOTE: This method is usually a stand-in for "speculative decoding using
        # target model hidden states"
        # TODO(ben): Refactor this so the naming is clearer
        return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")

    def use_dflash(self) -> bool:
        return self.method == "dflash"

    def use_dspark(self) -> bool:
        return self.method == "dspark"

    def uses_dynamic_speculative_decoding(self) -> bool:
        return self.num_speculative_tokens_per_batch_size is not None

    def uses_draft_model(self) -> bool:
        return self.method == "draft_model"

    def uses_extract_hidden_states(self) -> bool:
        return self.method == "extract_hidden_states"

    def use_ngram_gpu(self) -> bool:
        return self.method == "ngram_gpu"

    def use_multi_module_mtp(self) -> bool:
        if self.method != "mtp" or self.draft_model_config is None:
            return False
        num_mtp_layers = getattr(
            self.draft_model_config.hf_config, "num_nextn_predict_layers", 1
        )
        return min(num_mtp_layers, self.num_speculative_tokens) > 1

    def __repr__(self) -> str:
        method = self.method
        model = (
            None
            if method
            in (
                "ngram",
                "suffix",
                "extract_hidden_states",
                "custom_class",
            )
            else self.draft_model_config.model
        )
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {model=}, {num_spec_tokens=})"
