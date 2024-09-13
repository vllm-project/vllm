import contextlib
import enum
import json
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import huggingface_hub
from huggingface_hub import (file_exists, hf_hub_download,
                             try_to_load_from_cache)
from transformers import GenerationConfig, PretrainedConfig
from transformers.models.auto.image_processing_auto import (
    get_image_processor_config)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
from transformers.utils import CONFIG_NAME as HF_CONFIG_NAME

from vllm.envs import VLLM_USE_MODELSCOPE
from vllm.logger import init_logger
# yapf conflicts with isort for this block
# yapf: disable
from vllm.transformers_utils.configs import (ChatGLMConfig, DbrxConfig,
                                             EAGLEConfig, ExaoneConfig,
                                             GraniteConfig, InternVLChatConfig,
                                             JAISConfig, MedusaConfig,
                                             MLPSpeculatorConfig, MPTConfig,
                                             NemotronConfig, RWConfig,
                                             UltravoxConfig)
# yapf: enable
from vllm.transformers_utils.utils import check_gguf_file

if VLLM_USE_MODELSCOPE:
    from modelscope import AutoConfig
else:
    from transformers import AutoConfig

MISTRAL_CONFIG_NAME = "params.json"

logger = init_logger(__name__)

_CONFIG_REGISTRY: Dict[str, Type[PretrainedConfig]] = {
    "chatglm": ChatGLMConfig,
    "dbrx": DbrxConfig,
    "mpt": MPTConfig,
    "RefinedWeb": RWConfig,  # For tiiuae/falcon-40b(-instruct)
    "RefinedWebModel": RWConfig,  # For tiiuae/falcon-7b(-instruct)
    "jais": JAISConfig,
    "mlp_speculator": MLPSpeculatorConfig,
    "medusa": MedusaConfig,
    "eagle": EAGLEConfig,
    "exaone": ExaoneConfig,
    "internvl_chat": InternVLChatConfig,
    "nemotron": NemotronConfig,
    "ultravox": UltravoxConfig,
    # Granite can be removed from here once we have upgraded to
    # transformers 4.45+
    "granite": GraniteConfig,
}

for name, cls in _CONFIG_REGISTRY.items():
    with contextlib.suppress(ValueError):
        AutoConfig.register(name, cls)


class ConfigFormat(str, enum.Enum):
    AUTO = "auto"
    HF = "hf"
    MISTRAL = "mistral"


def file_or_path_exists(model: Union[str, Path], config_name, revision,
                        token) -> bool:
    if Path(model).exists():
        return (Path(model) / config_name).is_file()

    # Offline mode support: Check if config file is cached already
    cached_filepath = try_to_load_from_cache(repo_id=model,
                                             filename=config_name,
                                             revision=revision)
    if isinstance(cached_filepath, str):
        # The config file exists in cache- we can continue trying to load
        return True

    # NB: file_exists will only check for the existence of the config file on
    # hf_hub. This will fail in offline mode.
    try:
        return file_exists(model, config_name, revision=revision, token=token)
    except huggingface_hub.errors.OfflineModeIsEnabled:
        # Don't raise in offline mode, all we know is that we don't have this
        # file cached.
        return False


def get_config(
    model: Union[str, Path],
    trust_remote_code: bool,
    revision: Optional[str] = None,
    code_revision: Optional[str] = None,
    rope_scaling: Optional[dict] = None,
    rope_theta: Optional[float] = None,
    config_format: ConfigFormat = ConfigFormat.AUTO,
    **kwargs,
) -> PretrainedConfig:
    # Separate model folder from file path for GGUF models

    is_gguf = check_gguf_file(model)
    if is_gguf:
        kwargs["gguf_file"] = Path(model).name
        model = Path(model).parent

    if config_format == ConfigFormat.AUTO:
        if is_gguf or file_or_path_exists(model,
                                          HF_CONFIG_NAME,
                                          revision=revision,
                                          token=kwargs.get("token")):
            config_format = ConfigFormat.HF
        elif file_or_path_exists(model,
                                 MISTRAL_CONFIG_NAME,
                                 revision=revision,
                                 token=kwargs.get("token")):
            config_format = ConfigFormat.MISTRAL
        else:
            # If we're in offline mode and found no valid config format, then
            # raise an offline mode error to indicate to the user that they
            # don't have files cached and may need to go online.
            # This is conveniently triggered by calling file_exists().
            file_exists(model,
                        HF_CONFIG_NAME,
                        revision=revision,
                        token=kwargs.get("token"))

            raise ValueError(f"No supported config format found in {model}")

    if config_format == ConfigFormat.HF:
        config_dict, _ = PretrainedConfig.get_config_dict(
            model, revision=revision, code_revision=code_revision, **kwargs)

        # Use custom model class if it's in our registry
        model_type = config_dict.get("model_type")
        if model_type in _CONFIG_REGISTRY:
            config_class = _CONFIG_REGISTRY[model_type]
            config = config_class.from_pretrained(model,
                                                  revision=revision,
                                                  code_revision=code_revision)
        else:
            try:
                config = AutoConfig.from_pretrained(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    code_revision=code_revision,
                    **kwargs,
                )
            except ValueError as e:
                if (not trust_remote_code
                        and "requires you to execute the configuration file"
                        in str(e)):
                    err_msg = (
                        "Failed to load the model config. If the model "
                        "is a custom model not yet available in the "
                        "HuggingFace transformers library, consider setting "
                        "`trust_remote_code=True` in LLM or using the "
                        "`--trust-remote-code` flag in the CLI.")
                    raise RuntimeError(err_msg) from e
                else:
                    raise e

    elif config_format == ConfigFormat.MISTRAL:
        config = load_params_config(model, revision)
    else:
        raise ValueError(f"Unsupported config format: {config_format}")

    # Special architecture mapping check for GGUF models
    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(
                f"Can't get gguf config for {config.model_type}.")
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})

    for key, value in [
        ("rope_scaling", rope_scaling),
        ("rope_theta", rope_theta),
    ]:
        if value is not None:
            logger.info(
                "Updating %s from %r to %r",
                key,
                getattr(config, key, None),
                value,
            )
            config.update({key: value})

    return config


def load_params_config(model, revision) -> PretrainedConfig:
    # This function loads a params.json config which
    # should be used when loading models in mistral format

    config_file_name = "params.json"

    config_path = Path(model) / config_file_name

    if not config_path.is_file():
        config_path = Path(
            hf_hub_download(model, config_file_name, revision=revision))

    with open(config_path, "r") as file:
        config_dict = json.load(file)

    config_mapping = {
        "dim": "hidden_size",
        "norm_eps": "rms_norm_eps",
        "n_kv_heads": "num_key_value_heads",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "hidden_dim": "intermediate_size",
    }

    def recurse_elems(elem: Any):
        if isinstance(elem, dict):
            config_dict = {}
            for key, value in elem.items():
                key = config_mapping.get(key, key)
                config_dict[key] = recurse_elems(value)
            return PretrainedConfig(**config_dict)
        else:
            return elem

    config_dict["model_type"] = config_dict.get("model_type", "transformer")
    config_dict["hidden_act"] = config_dict.get("activation", "silu")
    config_dict["tie_word_embeddings"] = config_dict.get(
        "tie_embeddings", False)
    config_dict["max_seq_len"] = config_dict.get("max_seq_len", 128_000)
    config_dict["max_position_embeddings"] = config_dict.get(
        "max_position_embeddings", 128_000)

    if config_dict.get("moe") is not None:
        config_dict["architectures"] = ["MixtralForCausalLM"]
    else:
        config_dict["architectures"] = ["MistralForCausalLM"]

    if config_dict.get("vision_encoder") is not None:
        multimodal_config = config_dict.pop("vision_encoder")

        config_dict = {
            "text_config": config_dict,
            "vision_config": multimodal_config
        }
        config_dict["architectures"] = ["PixtralForConditionalGeneration"]
        config_dict["model_type"] = "pixtral"

    config = recurse_elems(config_dict)
    return config


def get_hf_image_processor_config(
    model: Union[str, Path],
    revision: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    # ModelScope does not provide an interface for image_processor
    if VLLM_USE_MODELSCOPE:
        return dict()
    # Separate model folder from file path for GGUF models
    if check_gguf_file(model):
        model = Path(model).parent
    return get_image_processor_config(model, revision=revision, **kwargs)


def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    if hasattr(config, "text_config"):
        # The code operates under the assumption that text_config should have
        # `num_attention_heads` (among others). Assert here to fail early
        # if transformers config doesn't align with this assumption.
        assert hasattr(config.text_config, "num_attention_heads")
        return config.text_config
    else:
        return config


def try_get_generation_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
) -> Optional[GenerationConfig]:
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
            )
            return GenerationConfig.from_model_config(config)
        except OSError:  # Not found
            return None
