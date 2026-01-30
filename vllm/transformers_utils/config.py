# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from dataclasses import asdict
from functools import cache, partial
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, TypeAlias

import huggingface_hub
from huggingface_hub import get_safetensors_metadata
from packaging.version import Version
from transformers import GenerationConfig, PretrainedConfig
from transformers.models.auto.image_processing_auto import get_image_processor_config
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.utils import CONFIG_NAME as HF_CONFIG_NAME

from vllm import envs
from vllm.logger import init_logger
from vllm.transformers_utils.utils import parse_safetensors_file_metadata

from .config_parser_base import ConfigParserBase
from .gguf_utils import (
    check_gguf_file,
    is_gguf,
    is_remote_gguf,
    split_remote_gguf,
)
from .repo_utils import (
    file_or_path_exists,
    get_hf_file_to_dict,
    list_repo_files,
    try_get_local_file,
    with_retry,
)

try:
    # Transformers v5
    from transformers.configuration_utils import ALLOWED_ATTENTION_LAYER_TYPES
except ImportError:
    # Transformers v4
    from transformers.configuration_utils import (
        ALLOWED_LAYER_TYPES as ALLOWED_ATTENTION_LAYER_TYPES,
    )


if envs.VLLM_USE_MODELSCOPE:
    from modelscope import AutoConfig
else:
    from transformers import AutoConfig

MISTRAL_CONFIG_NAME = "params.json"

logger = init_logger(__name__)


class LazyConfigDict(dict):
    def __getitem__(self, key):
        if isinstance(value := super().__getitem__(key), type):
            return value

        import vllm.transformers_utils.configs as configs

        return getattr(configs, value)


_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = LazyConfigDict(
    afmoe="AfmoeConfig",
    bagel="BagelConfig",
    chatglm="ChatGLMConfig",
    deepseek_vl_v2="DeepseekVLV2Config",
    deepseek_v32="DeepseekV3Config",
    flex_olmo="FlexOlmoConfig",
    funaudiochat="FunAudioChatConfig",
    hunyuan_vl="HunYuanVLConfig",
    isaac="IsaacConfig",
    kimi_linear="KimiLinearConfig",
    kimi_vl="KimiVLConfig",
    kimi_k25="KimiK25Config",
    RefinedWeb="RWConfig",  # For tiiuae/falcon-40b(-instruct)
    RefinedWebModel="RWConfig",  # For tiiuae/falcon-7b(-instruct)
    jais="JAISConfig",
    mlp_speculator="MLPSpeculatorConfig",
    medusa="MedusaConfig",
    midashenglm="MiDashengLMConfig",
    eagle="EAGLEConfig",
    speculators="SpeculatorsConfig",
    nemotron="NemotronConfig",
    olmo3="Olmo3Config",
    ovis="OvisConfig",
    ultravox="UltravoxConfig",
    vibevoice="VibeVoiceASRConfig",
    step3_vl="Step3VLConfig",
    step3_text="Step3TextConfig",
    qwen3_asr="Qwen3ASRConfig",
    qwen3_next="Qwen3NextConfig",
    lfm2_moe="Lfm2MoeConfig",
    tarsier2="Tarsier2Config",
)

_CONFIG_ATTRS_MAPPING: dict[str, str] = {
    "llm_config": "text_config",
}

_AUTO_CONFIG_KWARGS_OVERRIDES: dict[str, dict[str, Any]] = {
    "internvl_chat": {"has_no_defaults_at_init": True},
    "Llama_Nemotron_Nano_VL": {"attn_implementation": "eager"},
    "NVLM_D": {"has_no_defaults_at_init": True},
}


def is_rope_parameters_nested(rope_parameters: dict[str, Any]) -> bool:
    """Check if rope_parameters is nested by layer types."""
    # Cannot be nested if rope_parameters is empty
    if not rope_parameters:
        return False
    return set(rope_parameters.keys()).issubset(ALLOWED_ATTENTION_LAYER_TYPES)


class HFConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        kwargs["local_files_only"] = huggingface_hub.constants.HF_HUB_OFFLINE
        config_dict, _ = PretrainedConfig.get_config_dict(
            model,
            revision=revision,
            code_revision=code_revision,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        # Use custom model class if it's in our registry
        model_type = config_dict.get("model_type")
        if model_type is None:
            model_type = (
                "speculators"
                if config_dict.get("speculators_config") is not None
                else model_type
            )
        # Allow hf_overrides to override model_type before checking _CONFIG_REGISTRY
        if (hf_overrides := kwargs.pop("hf_overrides", None)) is not None:
            model_type = hf_overrides.get("model_type", model_type)

        if model_type in _CONFIG_REGISTRY:
            config_class = _CONFIG_REGISTRY[model_type]
            config = config_class.from_pretrained(
                model,
                revision=revision,
                code_revision=code_revision,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        else:
            try:
                kwargs = _maybe_update_auto_config_kwargs(kwargs, model_type=model_type)
                config = AutoConfig.from_pretrained(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    code_revision=code_revision,
                    **kwargs,
                )
            except ValueError as e:
                if (
                    not trust_remote_code
                    and "requires you to execute the configuration file" in str(e)
                ):
                    err_msg = (
                        "Failed to load the model config. If the model "
                        "is a custom model not yet available in the "
                        "HuggingFace transformers library, consider setting "
                        "`trust_remote_code=True` in LLM or using the "
                        "`--trust-remote-code` flag in the CLI."
                    )
                    raise RuntimeError(err_msg) from e
                else:
                    raise e
        config = _maybe_remap_hf_config_attrs(config)
        return config_dict, config


class MistralConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        # This function loads a params.json config which
        # should be used when loading models in mistral format
        config_dict = _download_mistral_config_file(model, revision)
        if (
            max_position_embeddings := config_dict.get("max_position_embeddings")
        ) is None:
            max_position_embeddings = _maybe_retrieve_max_pos_from_hf(
                model, revision, **kwargs
            )
            config_dict["max_position_embeddings"] = max_position_embeddings

        from vllm.transformers_utils.configs.mistral import adapt_config_dict

        # Get missing fields from HF config if available
        try:
            hf_config_dict, _ = PretrainedConfig.get_config_dict(
                model,
                revision=revision,
                code_revision=code_revision,
                **kwargs,
            )
        except OSError:  # Not found
            hf_config_dict = {}

        config = adapt_config_dict(config_dict, defaults=hf_config_dict)

        # Mistral configs may define sliding_window as list[int]. Convert it
        # to int and add the layer_types list[str] to make it HF compatible
        if (sliding_window := getattr(config, "sliding_window", None)) and isinstance(
            sliding_window, list
        ):
            pattern_repeats = config.num_hidden_layers // len(sliding_window)
            layer_types = sliding_window * pattern_repeats
            config.layer_types = [
                "full_attention" if layer_type is None else "sliding_attention"
                for layer_type in layer_types
            ]
            config.sliding_window = next(filter(None, sliding_window), None)

        return config_dict, config


_CONFIG_FORMAT_TO_CONFIG_PARSER: dict[str, type[ConfigParserBase]] = {
    "hf": HFConfigParser,
    "mistral": MistralConfigParser,
}

ConfigFormat = Literal[
    "auto",
    "hf",
    "mistral",
]


def get_config_parser(config_format: str) -> ConfigParserBase:
    """Get the config parser for a given config format."""
    if config_format not in _CONFIG_FORMAT_TO_CONFIG_PARSER:
        raise ValueError(f"Unknown config format `{config_format}`.")
    return _CONFIG_FORMAT_TO_CONFIG_PARSER[config_format]()


def register_config_parser(config_format: str):
    """Register a customized vllm config parser.
     When a config format is not supported by vllm, you can register a customized
    config parser to support it.
     Args:
         config_format (str): The config parser format name.
     Examples:

         >>> from vllm.transformers_utils.config import (get_config_parser,
                                                         register_config_parser)
         >>> from vllm.transformers_utils.config_parser_base import ConfigParserBase
         >>>
         >>> @register_config_parser("custom_config_parser")
         ... class CustomConfigParser(ConfigParserBase):
         ...     def parse(
         ...         self,
         ...         model: Union[str, Path],
         ...         trust_remote_code: bool,
         ...         revision: str | None = None,
         ...         code_revision: str | None = None,
         ...         **kwargs,
         ...     ) -> tuple[dict, PretrainedConfig]:
         ...         raise NotImplementedError
         >>>
         >>> type(get_config_parser("custom_config_parser"))
         <class 'CustomConfigParser'>
    """  # noqa: E501

    def _wrapper(config_parser_cls):
        if config_format in _CONFIG_FORMAT_TO_CONFIG_PARSER:
            logger.warning(
                "Config format `%s` is already registered, and will be "
                "overwritten by the new parser class `%s`.",
                config_format,
                config_parser_cls,
            )
        if not issubclass(config_parser_cls, ConfigParserBase):
            raise ValueError(
                "The config parser must be a subclass of `ConfigParserBase`."
            )
        _CONFIG_FORMAT_TO_CONFIG_PARSER[config_format] = config_parser_cls
        logger.info(
            "Registered config parser `%s` with config format `%s`",
            config_parser_cls,
            config_format,
        )
        return config_parser_cls

    return _wrapper


def set_default_rope_theta(config: PretrainedConfig, default_theta: float) -> None:
    """Some models may have no rope_theta in their config but still use RoPE.
    This function sets a default rope_theta if it's missing."""
    if getattr(config, "rope_parameters", None) is None:
        config.rope_parameters = {"rope_type": "default"}
    if "rope_theta" not in config.rope_parameters:
        config.rope_parameters["rope_theta"] = default_theta


def patch_rope_parameters(config: PretrainedConfig) -> None:
    """Provide backwards compatibility for RoPE."""
    from vllm.config.utils import getattr_iter

    # Older custom models may use non-standard field names
    # which need patching for both Transformers v4 and v5.
    names = ["rope_theta", "rotary_emb_base"]
    rope_theta = getattr_iter(config, names, None, warn=True)
    names = ["partial_rotary_factor", "rotary_pct", "rotary_emb_fraction"]
    partial_rotary_factor = getattr_iter(config, names, None, warn=True)
    ompe = getattr(config, "original_max_position_embeddings", None)

    if Version(version("transformers")) < Version("5.0.0"):
        # Transformers v4 installed, legacy config fields may be present
        if (rope_scaling := getattr(config, "rope_scaling", None)) is not None:
            config.rope_parameters = rope_scaling
        if (
            rope_theta is not None
            or partial_rotary_factor is not None
            or ompe is not None
        ) and not getattr(config, "rope_parameters", None):
            config.rope_parameters = {"rope_type": "default"}
        # Patch legacy fields into rope_parameters
        if rope_theta is not None:
            config.rope_parameters["rope_theta"] = rope_theta
        if partial_rotary_factor is not None:
            config.rope_parameters["partial_rotary_factor"] = partial_rotary_factor
        if ompe is not None:
            config.rope_parameters["original_max_position_embeddings"] = ompe
    elif rope_theta is not None or getattr(config, "rope_parameters", None):
        # Transformers v5 installed
        # Patch these fields in case they used non-standard names
        if rope_theta is not None:
            config.rope_theta = rope_theta
        if partial_rotary_factor is not None:
            config.partial_rotary_factor = partial_rotary_factor
        # Standardize and validate RoPE parameters
        config.standardize_rope_params()
        config.validate_rope()

    # No RoPE parameters to patch
    if getattr(config, "rope_parameters", None) is None:
        return

    # Handle nested rope_parameters in interleaved sliding attention models
    if is_rope_parameters_nested(config.rope_parameters):
        for rope_parameters_layer_type in config.rope_parameters.values():
            patch_rope_parameters_dict(rope_parameters_layer_type)
    else:
        patch_rope_parameters_dict(config.rope_parameters)


def patch_rope_parameters_dict(rope_parameters: dict[str, Any]) -> None:
    if "rope_type" in rope_parameters and "type" in rope_parameters:
        rope_type = rope_parameters["rope_type"]
        rope_type_legacy = rope_parameters["type"]
        if (rope_type_legacy == "su" and rope_type == "longrope") or (
            rope_type_legacy == "mrope" and rope_type == "default"
        ):
            pass  # No action needed
        elif rope_type != rope_type_legacy:
            raise ValueError(
                f"Found conflicts between 'rope_type={rope_type}' (modern "
                f"field) and 'type={rope_type_legacy}' (legacy field). "
                "You should only specify one of them."
            )

    if "rope_type" not in rope_parameters and "type" in rope_parameters:
        rope_parameters["rope_type"] = rope_parameters["type"]
        logger.info("Replacing legacy 'type' key with 'rope_type'")

    if "rope_type" not in rope_parameters:
        raise ValueError("rope_parameters should have a 'rope_type' key")

    if rope_parameters["rope_type"] == "su":
        rope_parameters["rope_type"] = "longrope"
        logger.warning("Replacing legacy rope_type 'su' with 'longrope'")
    elif rope_parameters["rope_type"] == "mrope":
        if "mrope_section" not in rope_parameters:
            raise ValueError(
                "Legacy rope_type 'mrope' requires 'mrope_section' in rope_parameters"
            )
        rope_parameters["rope_type"] = "default"
        logger.warning("Replacing legacy rope_type 'mrope' with 'default'")


def _uses_mrope(config: PretrainedConfig) -> bool:
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is None:
        return False

    return "mrope_section" in rope_parameters


def uses_mrope(config: PretrainedConfig) -> bool:
    """Detect if the model with this config uses M-ROPE."""
    return (
        _uses_mrope(config)
        or _uses_mrope(config.get_text_config())
        or thinker_uses_mrope(config)
    )


def thinker_uses_mrope(config: PretrainedConfig) -> bool:
    """Detect if the model contains a thinker config and it uses M-ROPE."""
    thinker_config = getattr(config, "thinker_config", None)
    if thinker_config is None:
        return False

    thinker_text_config = getattr(thinker_config, "text_config", None)
    if thinker_text_config is None:
        return False

    return uses_mrope(thinker_text_config)


def uses_xdrope_dim(config: PretrainedConfig) -> int:
    """Detect if the model with this config uses XD-ROPE."""
    xdrope_section = getattr(config, "xdrope_section", None)
    if xdrope_section is not None and isinstance(xdrope_section, list):
        return len(xdrope_section)
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return 0

    if isinstance(rope_scaling, dict) and "xdrope_section" in rope_scaling:
        xdrope_section = rope_scaling["xdrope_section"]
        if xdrope_section is not None and isinstance(xdrope_section, list):
            return len(xdrope_section)

    return 0


def is_encoder_decoder(config: PretrainedConfig) -> bool:
    """Detect if the model with this config is used as an encoder/decoder."""

    def _is_encoder_decoder(config: PretrainedConfig) -> bool:
        return getattr(config, "is_encoder_decoder", False)

    return _is_encoder_decoder(config) or _is_encoder_decoder(config.get_text_config())


def is_interleaved(config: PretrainedConfig) -> bool:
    """
    Detect if the model with this config is used with interleaved attention.
    """
    text_config = config.get_text_config()
    if layer_types := getattr(text_config, "layer_types", None):
        return len(set(layer_types)) > 1
    return False


def _maybe_update_auto_config_kwargs(kwargs: dict[str, Any], model_type: str):
    """
    Update kwargs for AutoConfig initialization based on model_type
    """
    if model_type in _AUTO_CONFIG_KWARGS_OVERRIDES:
        kwargs.update(_AUTO_CONFIG_KWARGS_OVERRIDES[model_type])
    return kwargs


def _maybe_remap_hf_config_attrs(config: PretrainedConfig) -> PretrainedConfig:
    """Remap config attributes to match the expected names."""
    for old_attr, new_attr in _CONFIG_ATTRS_MAPPING.items():
        if hasattr(config, old_attr):
            if not hasattr(config, new_attr):
                config.update({new_attr: getattr(config, old_attr)})
            logger.debug("Remapped config attribute '%s' to '%s'", old_attr, new_attr)
    return config


def maybe_override_with_speculators(
    model: str,
    tokenizer: str | None,
    trust_remote_code: bool,
    revision: str | None = None,
    vllm_speculative_config: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[str, str | None, dict[str, Any] | None]:
    """
    Resolve model configuration when speculators are detected.

    Checks if the provided model is a speculators model and if so, extracts
    the target model configuration and builds the speculative config.

    Args:
        model: Model name or path
        tokenizer: Tokenizer name or path
        trust_remote_code: Whether to trust remote code
        revision: Model revision
        vllm_speculative_config: Existing vLLM speculative config

    Returns:
        Tuple of (resolved_model, resolved_tokenizer, speculative_config)
    """
    if check_gguf_file(model):
        kwargs["gguf_file"] = Path(model).name
        gguf_model_repo = Path(model).parent
    elif is_remote_gguf(model):
        repo_id, _ = split_remote_gguf(model)
        gguf_model_repo = Path(repo_id)
    else:
        gguf_model_repo = None
    kwargs["local_files_only"] = huggingface_hub.constants.HF_HUB_OFFLINE
    config_dict, _ = PretrainedConfig.get_config_dict(
        model if gguf_model_repo is None else gguf_model_repo,
        revision=revision,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    speculators_config = config_dict.get("speculators_config")

    if speculators_config is None:
        # No speculators config found, return original values
        return model, tokenizer, vllm_speculative_config

    # Speculators format detected - process overrides
    from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

    speculative_config = SpeculatorsConfig.extract_vllm_speculative_config(
        config_dict=config_dict
    )

    # Set the draft model to the speculators model
    speculative_config["model"] = model

    # Override model and tokenizer with the verifier model from config
    verifier_model = speculators_config["verifier"]["name_or_path"]
    model = tokenizer = verifier_model

    return model, tokenizer, speculative_config


def get_config(
    model: str | Path,
    trust_remote_code: bool,
    revision: str | None = None,
    code_revision: str | None = None,
    config_format: str | ConfigFormat = "auto",
    hf_overrides_kw: dict[str, Any] | None = None,
    hf_overrides_fn: Callable[[PretrainedConfig], PretrainedConfig] | None = None,
    **kwargs,
) -> PretrainedConfig:
    # Separate model folder from file path for GGUF models

    _is_gguf = is_gguf(model)
    _is_remote_gguf = is_remote_gguf(model)
    if _is_gguf:
        if check_gguf_file(model):
            # Local GGUF file
            kwargs["gguf_file"] = Path(model).name
            model = Path(model).parent
        elif _is_remote_gguf:
            # Remote GGUF - extract repo_id from repo_id:quant_type format
            # The actual GGUF file will be downloaded later by GGUFModelLoader
            # Keep model as repo_id:quant_type for download, but use repo_id for config
            model, _ = split_remote_gguf(model)

    if config_format == "auto":
        try:
            # First check for Mistral to avoid defaulting to
            # Transformers implementation.
            if file_or_path_exists(model, MISTRAL_CONFIG_NAME, revision=revision):
                config_format = "mistral"
            elif (_is_gguf and not _is_remote_gguf) or file_or_path_exists(
                model, HF_CONFIG_NAME, revision=revision
            ):
                config_format = "hf"
            # Remote GGUF models must have config.json in repo,
            # otherwise the config can't be parsed correctly.
            # FIXME(Isotr0py): Support remote GGUF repos without config.json
            elif _is_remote_gguf and not file_or_path_exists(
                model, HF_CONFIG_NAME, revision=revision
            ):
                err_msg = (
                    "Could not find config.json for remote GGUF model repo. "
                    "To load remote GGUF model through `<repo_id>:<quant_type>`, "
                    "ensure your model has config.json (HF format) file. "
                    "Otherwise please specify --hf-config-path <original_repo> "
                    "in engine args to fetch config from unquantized hf model."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
            else:
                raise ValueError(
                    "Could not detect config format for no config file found. "
                    "With config_format 'auto', ensure your model has either "
                    "config.json (HF format) or params.json (Mistral format). "
                    "Otherwise please specify your_custom_config_format "
                    "in engine args for customized config parser."
                )

        except Exception as e:
            error_message = (
                "Invalid repository ID or local directory specified:"
                " '{model}'.\nPlease verify the following requirements:\n"
                "1. Provide a valid Hugging Face repository ID.\n"
                "2. Specify a local directory that contains a recognized "
                "configuration file.\n"
                "   - For Hugging Face models: ensure the presence of a "
                "'config.json'.\n"
                "   - For Mistral models: ensure the presence of a "
                "'params.json'.\n"
            ).format(model=model)

            raise ValueError(error_message) from e

    config_parser = get_config_parser(config_format)
    config_dict, config = config_parser.parse(
        model,
        trust_remote_code=trust_remote_code,
        revision=revision,
        code_revision=code_revision,
        hf_overrides=hf_overrides_kw,
        **kwargs,
    )

    # Patching defaults for GGUF models
    if _is_gguf:
        # Some models have different default values between GGUF and HF.
        def apply_gguf_default(key: str, gguf_default: Any):
            """
            Apply GGUF defaults unless explicitly configured.

            This function reads/writes external `config` and `config_dict`.
            If the specified `key` is not in `config_dict` (i.e. not explicitly
            configured and the default HF value is used), it updates the
            corresponding `config` value to `gguf_default`.
            """
            if key not in config_dict:
                config.update({key: gguf_default})

        # Apply architecture-specific GGUF defaults.
        if config.model_type in {"qwen3_moe"}:
            # Qwen3 MoE: norm_topk_prob is always true.
            # Note that, this parameter is always false (HF default) on Qwen2 MoE.
            apply_gguf_default("norm_topk_prob", True)

    # Special architecture mapping check for GGUF models
    if _is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})

    # Architecture mapping for models without explicit architectures field
    if not config.architectures:
        if config.model_type not in MODEL_MAPPING_NAMES:
            logger.warning(
                "Model config does not have a top-level 'architectures' field: "
                "expecting `hf_overrides={'architectures': ['...']}` to be passed "
                "in engine args."
            )
        else:
            model_type = MODEL_MAPPING_NAMES[config.model_type]
            config.update({"architectures": [model_type]})

    # ModelOpt 0.31.0 and after saves the quantization config in the model
    # config file.
    quantization_config = config_dict.get("quantization_config", None)

    # ModelOpt 0.29.0 and before saves the quantization config in a separate
    # "hf_quant_config.json" in the same directory as the model config file.
    if quantization_config is None and file_or_path_exists(
        model, "hf_quant_config.json", revision
    ):
        quantization_config = get_hf_file_to_dict(
            "hf_quant_config.json", model, revision
        )

    if quantization_config is not None:
        config.quantization_config = quantization_config
        # auto-enable DeepGEMM UE8M0 if model config requests it
        scale_fmt = quantization_config.get("scale_fmt", None)
        if scale_fmt in ("ue8m0",):
            if not envs.is_set("VLLM_USE_DEEP_GEMM_E8M0"):
                os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "1"
                logger.info_once(
                    (
                        "Detected quantization_config.scale_fmt=%s; "
                        "enabling UE8M0 for DeepGEMM."
                    ),
                    scale_fmt,
                )
            elif not envs.VLLM_USE_DEEP_GEMM_E8M0:
                logger.warning_once(
                    (
                        "Model config requests UE8M0 "
                        "(quantization_config.scale_fmt=%s), but "
                        "VLLM_USE_DEEP_GEMM_E8M0=0 is set; "
                        "UE8M0 for DeepGEMM disabled."
                    ),
                    scale_fmt,
                )

    if hf_overrides_kw:
        logger.debug("Overriding HF config with %s", hf_overrides_kw)
        config.update(hf_overrides_kw)
    if hf_overrides_fn:
        logger.debug("Overriding HF config with %s", hf_overrides_fn)
        config = hf_overrides_fn(config)

    # Exhaustively patch RoPE parameters everywhere they might be
    patch_rope_parameters(config)
    patch_rope_parameters(config.get_text_config())
    SubConfigs: TypeAlias = dict[str, PretrainedConfig]
    sub_configs: SubConfigs | None = getattr(config, "sub_configs", None)
    if sub_configs:
        for sub_config in sub_configs:
            patch_rope_parameters(getattr(config, sub_config))

    if trust_remote_code:
        maybe_register_config_serialize_by_value()

    return config


@cache
def get_pooling_config(
    model: str,
    revision: str | None = "main",
) -> dict[str, Any] | None:
    """
    This function gets the pooling and normalize
    config from the model - only applies to
    sentence-transformers models.

    Args:
        model: The name of the Hugging Face model.
        revision: The specific version of the model to use.
            Defaults to 'main'.

    Returns:
        A dictionary containing the pooling type and whether
            normalization is used, or None if no pooling configuration is found.
    """
    if is_remote_gguf(model):
        model, _ = split_remote_gguf(model)

    modules_file_name = "modules.json"

    modules_dict = None
    if file_or_path_exists(
        model=model, config_name=modules_file_name, revision=revision
    ):
        modules_dict = get_hf_file_to_dict(modules_file_name, model, revision)

    if modules_dict is None:
        return None

    logger.info("Found sentence-transformers modules configuration.")

    pooling = next(
        (
            item
            for item in modules_dict
            if item["type"] == "sentence_transformers.models.Pooling"
        ),
        None,
    )
    normalize = bool(
        next(
            (
                item
                for item in modules_dict
                if item["type"] == "sentence_transformers.models.Normalize"
            ),
            False,
        )
    )

    if pooling:
        from vllm.config.pooler import SEQ_POOLING_TYPES, TOK_POOLING_TYPES

        pooling_file_name = "{}/config.json".format(pooling["path"])
        pooling_dict = get_hf_file_to_dict(pooling_file_name, model, revision) or {}

        logger.info("Found pooling configuration.")

        config: dict[str, Any] = {"use_activation": normalize}
        for key, val in pooling_dict.items():
            if val is True:
                pooling_type = parse_pooling_type(key)
                if pooling_type in SEQ_POOLING_TYPES:
                    config["seq_pooling_type"] = pooling_type
                elif pooling_type in TOK_POOLING_TYPES:
                    config["tok_pooling_type"] = pooling_type
                else:
                    logger.debug("Skipping unrelated field: %r=%r", key, val)

        return config

    return None


def parse_pooling_type(pooling_name: str):
    if "pooling_mode_" in pooling_name:
        pooling_name = pooling_name.replace("pooling_mode_", "")

    if "_" in pooling_name:
        pooling_name = pooling_name.split("_", 1)[0]

    if "lasttoken" in pooling_name:
        pooling_name = "last"

    return pooling_name.upper()


@cache
def get_sentence_transformer_tokenizer_config(
    model: str | Path, revision: str | None = "main"
):
    """
    Returns the tokenization configuration dictionary for a
    given Sentence Transformer BERT model.

    Parameters:
    - model (str|Path): The name of the Sentence Transformer
    BERT model.
    - revision (str, optional): The revision of the m
    odel to use. Defaults to 'main'.

    Returns:
    - dict: A dictionary containing the configuration parameters
    for the Sentence Transformer BERT model.
    """
    sentence_transformer_config_files = [
        "sentence_bert_config.json",
        "sentence_roberta_config.json",
        "sentence_distilbert_config.json",
        "sentence_camembert_config.json",
        "sentence_albert_config.json",
        "sentence_xlm-roberta_config.json",
        "sentence_xlnet_config.json",
    ]
    encoder_dict = None

    for config_file in sentence_transformer_config_files:
        if (
            try_get_local_file(model=model, file_name=config_file, revision=revision)
            is not None
        ):
            encoder_dict = get_hf_file_to_dict(config_file, model, revision)
            if encoder_dict:
                break

    if not encoder_dict and not Path(model).is_absolute():
        try:
            # If model is on HuggingfaceHub, get the repo files
            repo_files = list_repo_files(model, revision=revision)
        except Exception:
            repo_files = []

        for config_name in sentence_transformer_config_files:
            if config_name in repo_files:
                encoder_dict = get_hf_file_to_dict(config_name, model, revision)
                if encoder_dict:
                    break

    if not encoder_dict:
        return None

    logger.info("Found sentence-transformers tokenize configuration.")

    if all(k in encoder_dict for k in ("max_seq_length", "do_lower_case")):
        return encoder_dict
    return None


def maybe_register_config_serialize_by_value() -> None:
    """Try to register HF model configuration class to serialize by value

    If trust_remote_code is set, and the model's config file specifies an
    `AutoConfig` class, then the config class is typically an instance of
    a custom class imported from the HF modules cache.

    Examples:

    >>> from transformers import AutoConfig
    >>> klass = AutoConfig.from_pretrained(
    ...     "meta-llama/Meta-Llama-3-8B", trust_remote_code=True
    ... )
    >>> klass.__class__  # transformers.models.llama.configuration_llama.LlamaConfig
    >>> import transformers_modules  # error, not initialized
    >>> klass = AutoConfig.from_pretrained(
    ...     "deepseek-ai/DeepSeek-V2.5", trust_remote_code=True
    ... )
    >>> import transformers_modules  # success, initialized
    >>> klass.__class__  # transformers_modules.deepseek-ai.DeepSeek-V2.5.98b11844770b2c3ffc18b175c758a803640f4e77.configuration_deepseek.DeepseekV2Config

    In the DeepSeek example, the config class is an instance of a custom
    class that is not serializable by default. This class will not be
    importable in spawned workers, and won't exist at all on
    other nodes, which breaks serialization of the config.

    In this function we tell the cloudpickle serialization library to pass
    instances of these generated classes by value instead of by reference,
    i.e. the class definition is serialized along with its data so that the
    class module does not need to be importable on the receiving end.

    See: https://github.com/cloudpipe/cloudpickle?tab=readme-ov-file#overriding-pickles-serialization-mechanism-for-importable-constructs
    """  # noqa
    try:
        import transformers_modules

        transformers_modules_available = True
    except ImportError:
        transformers_modules_available = False

    try:
        import multiprocessing
        import pickle

        import cloudpickle

        from vllm.config import VllmConfig

        # Register multiprocessing reducers to handle cross-process
        # serialization of VllmConfig objects that may contain custom configs
        # from transformers_modules
        def _reduce_config(config: VllmConfig):
            return (pickle.loads, (cloudpickle.dumps(config),))

        multiprocessing.reducer.register(VllmConfig, _reduce_config)

        # Register transformers_modules with cloudpickle if available
        if transformers_modules_available:
            cloudpickle.register_pickle_by_value(transformers_modules)

            # ray vendors its own version of cloudpickle
            from vllm.v1.executor.ray_utils import ray

            if ray:
                ray.cloudpickle.register_pickle_by_value(transformers_modules)

    except Exception as e:
        logger.warning(
            "Unable to register remote classes used by"
            " trust_remote_code with by-value serialization. This may"
            " lead to a later error. If remote code is not needed"
            " remove `--trust-remote-code`",
            exc_info=e,
        )


def get_hf_image_processor_config(
    model: str | Path,
    hf_token: bool | str | None = None,
    revision: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    # ModelScope does not provide an interface for image_processor
    if envs.VLLM_USE_MODELSCOPE:
        return dict()
    # Separate model folder from file path for GGUF models
    if check_gguf_file(model):
        model = Path(model).parent
    elif is_remote_gguf(model):
        model, _ = split_remote_gguf(model)
    return get_image_processor_config(
        model, token=hf_token, revision=revision, **kwargs
    )


def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    text_config = config.get_text_config()

    if text_config is not config and not hasattr(text_config, "num_attention_heads"):
        raise ValueError(
            "The text_config extracted from the model config does not have "
            "`num_attention_heads` attribute. This indicates a mismatch "
            "between the model config and vLLM's expectations. Please "
            "ensure that the model config is compatible with vLLM."
        )

    return text_config


def try_get_generation_config(
    model: str,
    trust_remote_code: bool,
    revision: str | None = None,
    config_format: str | ConfigFormat = "auto",
) -> GenerationConfig | None:
    # GGUF files don't have generation_config.json - their config is embedded
    # in the file header. Skip all filesystem lookups to avoid re-reading the
    # memory-mapped file, which can hang in multi-process scenarios when the
    # EngineCore process already has the file mapped.
    if is_gguf(model):
        return None

    try:
        return GenerationConfig.from_pretrained(
            model,
            revision=revision,
        )
    except OSError:  # Not found
        try:
            config = get_config(
                model,
                trust_remote_code=trust_remote_code,
                revision=revision,
                config_format=config_format,
            )
            return GenerationConfig.from_model_config(config)
        except OSError:  # Not found
            return None


def try_get_safetensors_metadata(
    model: str,
    *,
    revision: str | None = None,
):
    get_safetensors_metadata_partial = partial(
        get_safetensors_metadata, model, revision=revision
    )

    try:
        return with_retry(
            get_safetensors_metadata_partial, "Error retrieving safetensors"
        )
    except Exception:
        return None


def try_get_tokenizer_config(
    pretrained_model_name_or_path: str | os.PathLike,
    trust_remote_code: bool,
    revision: str | None = None,
) -> dict[str, Any] | None:
    try:
        return get_tokenizer_config(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    except Exception:
        return None


@cache
def try_get_dense_modules(
    model: str | Path,
    revision: str | None = None,
) -> list[dict[str, Any]] | None:
    try:
        modules = get_hf_file_to_dict("modules.json", model, revision)
        if not modules:
            return None

        if isinstance(modules, dict):
            modules = modules.get("modules", [])

        dense_modules = [
            m for m in modules if m.get("type") == "sentence_transformers.models.Dense"
        ]
        if not dense_modules:
            return None

        layer_configs = []
        for module in dense_modules:
            folder = module.get("path", "")

            config_path = f"{folder}/config.json" if folder else "config.json"
            layer_config = get_hf_file_to_dict(config_path, model, revision)
            if not layer_config:
                continue
            layer_config["folder"] = folder
            layer_configs.append(layer_config)
        return layer_configs
    except Exception:
        return None


def get_safetensors_params_metadata(
    model: str,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    """
    Get the safetensors metadata for remote model repository.
    """
    full_metadata = {}
    if (model_path := Path(model)).exists():
        safetensors_to_check = model_path.glob("*.safetensors")
        full_metadata = {
            param_name: info
            for file_path in safetensors_to_check
            if file_path.is_file()
            for param_name, info in parse_safetensors_file_metadata(file_path).items()
        }
    else:
        repo_mt = try_get_safetensors_metadata(model, revision=revision)
        if repo_mt and (files_mt := repo_mt.files_metadata):
            full_metadata = {
                param_name: asdict(info)
                for file_mt in files_mt.values()
                for param_name, info in file_mt.tensors.items()
            }
    return full_metadata


def _download_mistral_config_file(model, revision) -> dict:
    config_file_name = "params.json"
    config_dict = get_hf_file_to_dict(config_file_name, model, revision)
    if config_dict is None:
        raise ValueError(
            f"Failed to load mistral '{config_file_name}' config for model "
            f"{model}. Please check if the model is a mistral-format model "
            f"and if the config file exists."
        )
    assert isinstance(config_dict, dict)
    return config_dict


def _maybe_retrieve_max_pos_from_hf(model, revision, **kwargs) -> int:
    max_position_embeddings = 128_000
    try:
        trust_remote_code_val = kwargs.get("trust_remote_code", False)
        hf_config = get_config(
            model=model,
            trust_remote_code=trust_remote_code_val,
            revision=revision,
            config_format="hf",
        )
        if hf_value := hf_config.get_text_config().max_position_embeddings:
            max_position_embeddings = hf_value
    except Exception as e:
        logger.warning(
            "The params.json file is missing 'max_position_embeddings'"
            " and could not get a value from the HF config."
            " Defaulting to 128000",
            exc_info=e,
        )

    return max_position_embeddings
