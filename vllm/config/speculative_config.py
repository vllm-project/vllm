# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import SkipValidation
from pydantic.dataclasses import dataclass
from transformers import PretrainedConfig

import vllm.envs as envs
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationMethods

if TYPE_CHECKING:
    from vllm.config.model_config import ModelConfig
    from vllm.config.parallel_config import ParallelConfig

logger = init_logger(__name__)

SpeculativeMethod = Literal["ngram", "eagle", "eagle3", "medusa",
                            "mlp_speculator", "draft_model", "deepseek_mtp"]
SpeculativeAcceptanceMethod = Literal["rejection_sampler",
                                      "typical_acceptance_sampler"]


@config
@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    # General speculative decoding control
    num_speculative_tokens: SkipValidation[int] = None  # type: ignore
    """The number of speculative tokens, if provided. It will default to the
    number in the draft model config if present, otherwise, it is required."""
    model: Optional[str] = None
    """The name of the draft model, eagle head, or additional weights, if
    provided."""
    method: Optional[SpeculativeMethod] = None
    """The name of the speculative method to use. If users provide and set the
    `model` param, the speculative method type will be detected automatically
    if possible, if `model` param is not provided, the method name must be
    provided.

    If using `ngram` method, the related configuration `prompt_lookup_max` and
    `prompt_lookup_min` should be considered."""
    acceptance_method: SpeculativeAcceptanceMethod = "rejection_sampler"
    """The method to use for accepting draft tokens:\n
    - "rejection_sampler" maps to `RejectionSampler`.\n
    - "typical_acceptance_sampler" maps to `TypicalAcceptanceSampler`.

    If using `typical_acceptance_sampler`, the related configuration
    `posterior_threshold` and `posterior_alpha` should be considered."""
    draft_tensor_parallel_size: Optional[int] = None
    """The degree of the tensor parallelism for the draft model. Can only be 1
    or the same as the target model's tensor parallel size."""
    disable_logprobs: bool = True
    """If set to True, token log probabilities are not returned during
    speculative decoding. If set to False, token log probabilities are returned
    according to the log probability settings in SamplingParams."""

    # Draft model configuration
    quantization: Optional[QuantizationMethods] = None
    """Quantization method that was used to quantize the draft model weights.
    If `None`, we assume the model weights are not quantized. Note that it only
    takes effect when using the draft model-based speculative method."""
    max_model_len: Optional[int] = None
    """The maximum model length of the draft model. Used when testing the
    ability to skip speculation for some sequences."""
    revision: Optional[str] = None
    """The specific model version to use for the draft model. It can be a
    branch name, a tag name, or a commit id. If unspecified, will use the
    default version."""
    code_revision: Optional[str] = None
    """The specific revision to use for the draft model code on Hugging Face
    Hub. It can be a branch name, a tag name, or a commit id. If unspecified,
    will use the default version."""

    # Advanced control
    disable_mqa_scorer: bool = False
    """Disable the MQA scorer and fall back to batch expansion for scoring
    proposals."""
    disable_by_batch_size: Optional[int] = None
    """Disable speculative decoding for new incoming requests when the number
    of enqueued requests is larger than this value, if provided."""

    # Ngram proposer configuration
    prompt_lookup_max: Optional[int] = None
    """Maximum size of ngram token window when using Ngram proposer, required
    when method is set to ngram."""
    prompt_lookup_min: Optional[int] = None
    """Minimum size of ngram token window when using Ngram proposer, if
    provided. Defaults to 1."""

    # Typical acceptance sampler configuration
    posterior_threshold: Optional[float] = None
    """A threshold value that sets a lower bound on the posterior probability
    of a token in the target model for it to be accepted. This threshold is
    used only when we use the `TypicalAcceptanceSampler` for token acceptance.
    """
    posterior_alpha: Optional[float] = None
    """Scaling factor for entropy-based threshold, applied when using
    `TypicalAcceptanceSampler`."""

    speculative_token_tree: Optional[str] = None
    """Specifies the tree structure for speculative token generation.
    """
    # required configuration params passed from engine
    target_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the target model."""
    target_parallel_config: SkipValidation[
        ParallelConfig] = None  # type: ignore
    """The parallel configuration for the target model."""
    enable_chunked_prefill: SkipValidation[bool] = None  # type: ignore
    """Whether vLLM is configured to use chunked prefill or not. Used for
    raising an error since it's not yet compatible with speculative decode."""
    disable_log_stats: SkipValidation[bool] = None  # type: ignore
    """Whether to disable the periodic printing of stage times in speculative
    decoding."""

    # params generated in the post-init stage
    draft_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the draft model initialized internal."""
    draft_parallel_config: SkipValidation[
        ParallelConfig] = None  # type: ignore
    """The parallel configuration for the draft model initialized internal."""

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
        # Eagle3 affects the computation graph because it returns intermediate
        # hidden states in addition to the final hidden state.
        factors.append(self.method == "eagle3")
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str

    @classmethod
    def from_dict(cls, dict_value: dict) -> SpeculativeConfig:
        """Parse the CLI value for the speculative config."""
        return cls(**dict_value)

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        if hf_config.model_type == "deepseek_v3":
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update({
                "n_predict": n_predict,
                "architectures": ["DeepSeekMTPModel"]
            })

        if hf_config.architectures[0] == "MiMoForCausalLM":
            hf_config.model_type = "mimo_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update({
                "num_hidden_layers": 0,
                "n_predict": n_predict,
                "architectures": ["MiMoMTPModel"]
            })
            return hf_config

        return hf_config

    def __post_init__(self):

        # Note: "method" is a new parameter that helps to extend the
        # configuration of non-model-based proposers, and the "model" parameter
        # will be used to set the draft model, eagle head, or additional weight
        # when needed. If users do not specify "method", the speculative method
        # will be detected automatically if possible. If the speculative method
        # can not be detected, it will be considered as the "draft_model" by
        # default.

        if self.model is None and self.num_speculative_tokens is not None:
            # TODO(Shangming): Refactor mtp configuration logic when supporting
            # mtp acceleration for more models besides deepseek_v3
            if self.target_model_config and \
                (self.target_model_config.hf_text_config.model_type \
                        == "deepseek_v3" or
                    self.target_model_config.hf_text_config.model_type \
                        == "mimo"):
                # use the draft model from the same model:
                self.model = self.target_model_config.model
            elif self.method in ("ngram", "[ngram]"):
                self.model = "ngram"
            else:
                raise ValueError("num_speculative_tokens was provided without "
                                 "speculative model.")

        # Automatically configure the method for ngram when "model" is used
        # instead of "method"
        if self.method is None and (self.model is not None
                                    and self.model in ("ngram", "[ngram]")):
            self.method = "ngram"

        if self.method in ("ngram", "[ngram]"):
            # Unified to "ngram" internally
            self.method = "ngram"
            # Set default values if not provided
            if (self.prompt_lookup_min is None
                    and self.prompt_lookup_max is None):
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                assert self.prompt_lookup_max is not None
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                assert self.prompt_lookup_min is not None
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min < 1:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must be > 0")
            if self.prompt_lookup_max < 1:
                raise ValueError(
                    f"prompt_lookup_max={self.prompt_lookup_max} must be > 0")
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}")

            # TODO: current we still need extract vocab_size from target model
            # config, in future, we may try refactor it out, and set
            # draft related config as None here.
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        else:
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if self.model is not None:
                self.draft_model_config = ModelConfig(
                    model=self.model,
                    task="draft",
                    tokenizer=self.target_model_config.tokenizer,
                    tokenizer_mode=self.target_model_config.tokenizer_mode,
                    trust_remote_code=self.target_model_config.
                    trust_remote_code,
                    allowed_local_media_path=self.target_model_config.
                    allowed_local_media_path,
                    dtype=self.target_model_config.dtype,
                    seed=self.target_model_config.seed,
                    revision=self.revision,
                    code_revision=self.code_revision,
                    tokenizer_revision=self.target_model_config.
                    tokenizer_revision,
                    spec_target_max_model_len=self.target_model_config.
                    max_model_len,
                    quantization=self.quantization,
                    enforce_eager=self.target_model_config.enforce_eager,
                    max_seq_len_to_capture=self.target_model_config.
                    max_seq_len_to_capture,
                    max_logprobs=self.target_model_config.max_logprobs,
                    hf_overrides=SpeculativeConfig.hf_config_override,
                )

                # Automatically detect the method
                if self.method in ('eagle', 'eagle3'):
                    pass
                elif "eagle-" in self.draft_model_config.model.lower() or \
                        "eagle3-" in self.draft_model_config.model.lower():
                    self.method = "eagle"
                elif self.draft_model_config.hf_config.model_type == "medusa":
                    self.method = "medusa"
                elif (self.draft_model_config.hf_config.model_type ==
                      "mlp_speculator"):
                    self.method = "mlp_speculator"
                elif (self.draft_model_config.hf_config.model_type ==
                      "deepseek_mtp"):
                    self.method = "deepseek_mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                                "All Deepseek MTP models only have " \
                                "one layer. Might need some code changes " \
                                "to support multiple layers."
                            )
                else:
                    self.method = "draft_model"

                # Replace hf_config for EAGLE draft_model
                if self.method in ("eagle", "eagle3"):
                    if self.enable_chunked_prefill and not envs.VLLM_USE_V1:
                        raise ValueError(
                            "Chunked prefill and EAGLE are not compatible "
                            "when using V0.")

                    from vllm.transformers_utils.configs.eagle import (
                        EAGLEConfig)
                    if isinstance(self.draft_model_config.hf_config,
                                  EAGLEConfig):
                        pass
                    else:
                        eagle_config = EAGLEConfig(
                            self.draft_model_config.hf_config,
                            method=self.method)
                        self.draft_model_config.hf_config = eagle_config

                if (self.num_speculative_tokens is not None
                        and hasattr(self.draft_model_config.hf_config,
                                    "num_lookahead_tokens")):
                    self.draft_model_config.hf_config.num_lookahead_tokens = \
                    self.num_speculative_tokens

                n_predict = getattr(self.draft_model_config.hf_config,
                                    "n_predict", None)
                if n_predict is not None:
                    if self.num_speculative_tokens is None:
                        # Default to max value defined in draft model config.
                        self.num_speculative_tokens = n_predict
                    elif self.num_speculative_tokens > n_predict and \
                            self.num_speculative_tokens % n_predict != 0:
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
                            f"num_speculative_tokens:{self.num_speculative_tokens}"
                            f" must be divisible by {n_predict=}")

                self.draft_tensor_parallel_size = \
                    SpeculativeConfig._verify_and_get_draft_tp(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size,
                        self.draft_model_config.hf_config
                )

                self.draft_model_config.max_model_len = (
                    SpeculativeConfig._maybe_override_draft_max_model_len(
                        self.max_model_len,
                        self.draft_model_config.max_model_len,
                        self.target_model_config.max_model_len,
                    ))

                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size))

        if self.acceptance_method == "typical_acceptance_sampler":
            if self.posterior_threshold is None:
                self.posterior_threshold = 0.09
            if self.posterior_alpha is None:
                self.posterior_alpha = 0.3

        self._verify_args()

    @staticmethod
    def _maybe_override_draft_max_model_len(
        speculative_max_model_len: Optional[int],
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
                raise ValueError(f"{speculative_max_model_len=} cannot be "
                                 f"larger than {draft_max_model_len=}")

            if speculative_max_model_len > target_max_model_len:
                raise ValueError(f"{speculative_max_model_len=} cannot be "
                                 f"larger than {target_max_model_len=}")

            return speculative_max_model_len

        return min(
            draft_max_model_len,
            target_max_model_len,
        )

    @staticmethod
    def _verify_and_get_draft_tp(
            target_parallel_config: ParallelConfig,
            speculative_draft_tensor_parallel_size: Optional[int],
            draft_hf_config: PretrainedConfig) -> int:
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
                        draft_hf_config.model_type)
            else:
                speculative_draft_tensor_parallel_size = \
                    target_parallel_config.tensor_parallel_size
        elif speculative_draft_tensor_parallel_size not in (
                1, target_parallel_config.tensor_parallel_size):
            raise ValueError(
                f"{speculative_draft_tensor_parallel_size=} cannot be "
                f"other value than 1 or target model tensor_parallel_size")
        return speculative_draft_tensor_parallel_size

    @staticmethod
    def create_draft_parallel_config(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int,
    ) -> ParallelConfig:
        """Create a parallel config for use by the draft worker.

        This is mostly a copy of the target parallel config, except the tp_size.
        """
        draft_parallel_config = ParallelConfig(
            pipeline_parallel_size=target_parallel_config.
            pipeline_parallel_size,
            tensor_parallel_size=speculative_draft_tensor_parallel_size,
            distributed_executor_backend=target_parallel_config.
            distributed_executor_backend,
            max_parallel_loading_workers=target_parallel_config.
            max_parallel_loading_workers,
            disable_custom_all_reduce=target_parallel_config.
            disable_custom_all_reduce,
            ray_workers_use_nsight=target_parallel_config.
            ray_workers_use_nsight,
            placement_group=target_parallel_config.placement_group,
        )

        return draft_parallel_config

    def _verify_args(self) -> None:
        if self.num_speculative_tokens is None:
            raise ValueError(
                "num_speculative_tokens must be provided with "
                "speculative model unless the draft model config contains an "
                "n_predict parameter.")

        if self.num_speculative_tokens <= 0:
            raise ValueError("Expected num_speculative_tokens to be greater "
                             f"than zero ({self.num_speculative_tokens}).")

        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config)
            # Validate and set draft token acceptance related settings.

        if self.acceptance_method is None:
            raise ValueError("acceptance_method is not set. "
                             "Expected values are rejection_sampler or "
                             "typical_acceptance_sampler.")

        if (self.acceptance_method != 'rejection_sampler'
                and self.acceptance_method != 'typical_acceptance_sampler'):
            raise ValueError(
                "Expected acceptance_method to be either "
                "rejection_sampler or typical_acceptance_sampler. Instead it "
                f"is {self.acceptance_method}")

        if self.acceptance_method == "typical_acceptance_sampler" and (
            (self.posterior_threshold is not None
             and self.posterior_threshold < 0) or
            (self.posterior_alpha is not None and self.posterior_alpha < 0)):
            raise ValueError(
                "Expected the posterior_threshold and posterior_alpha of "
                "typical_acceptance_sampler to be > 0. "
                "Instead found posterior_threshold = "
                f"{self.posterior_threshold} and posterior_alpha = "
                f"{self.posterior_alpha}")

        if (self.disable_by_batch_size is not None
                and self.disable_by_batch_size < 2):
            raise ValueError("Expect the batch size threshold of disabling "
                             "speculative decoding is > 1, but got "
                             f"{self.disable_by_batch_size=}")

        if self.method == "eagle3" and self.target_model_config and \
            "llama" not in self.target_model_config.hf_text_config.model_type:
            raise ValueError(
                "Eagle3 is only supported for Llama models. "
                f"Got {self.target_model_config.hf_text_config.model_type=}")

    @property
    def num_lookahead_slots(self) -> int:
        """The number of additional slots the scheduler should allocate per
        step, in addition to the slots allocated for each known token.

        This is equal to the number of speculative tokens, as each speculative
        token must be scored.
        """
        return self.num_speculative_tokens

    def use_eagle(self) -> bool:
        return self.method in ("eagle", "eagle3", "deepseek_mtp")

    def __repr__(self) -> str:
        method = self.method
        model = None if method == "ngram" else self.draft_model_config.model
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {model=}, {num_spec_tokens=})"
