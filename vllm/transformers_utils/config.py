import enum
import json
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import huggingface_hub
from huggingface_hub import (file_exists, hf_hub_download,
                             try_to_load_from_cache)
from huggingface_hub.utils import (EntryNotFoundError, LocalEntryNotFoundError,
                                   RepositoryNotFoundError,
                                   RevisionNotFoundError)
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
                                             H2OVLChatConfig,
                                             InternVLChatConfig, JAISConfig,
                                             MedusaConfig, MllamaConfig,
                                             MLPSpeculatorConfig, MPTConfig,
                                             NemotronConfig, NVLM_D_Config,
                                             RWConfig, SolarConfig,
                                             UltravoxConfig)
# yapf: enable
from vllm.transformers_utils.utils import check_gguf_file

if VLLM_USE_MODELSCOPE:
    from modelscope import AutoConfig
else:
    from transformers import AutoConfig

MISTRAL_CONFIG_NAME = "params.json"

logger = init_logger(__name__)

_CONFIG_REGISTRY_OVERRIDE_HF: Dict[str, Type[PretrainedConfig]] = {
    "mllama": MllamaConfig
}

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
    "h2ovl_chat": H2OVLChatConfig,
    "internvl_chat": InternVLChatConfig,
    "nemotron": NemotronConfig,
    "NVLM_D": NVLM_D_Config,
    "solar": SolarConfig,
    "ultravox": UltravoxConfig,
    **_CONFIG_REGISTRY_OVERRIDE_HF
}


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


def patch_rope_scaling(config: PretrainedConfig) -> None:
    """Provide backwards compatibility for RoPE."""
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        patch_rope_scaling(text_config)

    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None:
        patch_rope_scaling_dict(rope_scaling)


def patch_rope_scaling_dict(rope_scaling: Dict[str, Any]) -> None:
    if "rope_type" not in rope_scaling and "type" in rope_scaling:
        rope_scaling["rope_type"] = rope_scaling["type"]
        logger.info("Replacing legacy 'type' key with 'rope_type'")

    if "rope_type" not in rope_scaling:
        raise ValueError("rope_scaling should have a 'rope_type' key")

    if rope_scaling["rope_type"] == "su":
        rope_scaling["rope_type"] = "longrope"
        logger.warning("Replacing legacy rope_type 'su' with 'longrope'")
    elif rope_scaling["rope_type"] == "mrope":
        assert "mrope_section" in rope_scaling
        rope_scaling["rope_type"] = "default"
        logger.warning("Replacing legacy rope_type 'mrope' with 'default'")


def uses_mrope(config: PretrainedConfig) -> bool:
    """Detect if the model with this config uses M-ROPE."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return False

    return "mrope_section" in rope_scaling


def is_encoder_decoder(config: PretrainedConfig) -> bool:
    """Detect if the model with this config is used as an encoder/decoder."""
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return is_encoder_decoder(text_config)

    return getattr(config, "is_encoder_decoder", False)


def get_config(
    model: Union[str, Path],
    trust_remote_code: bool,
    revision: Optional[str] = None,
    code_revision: Optional[str] = None,
    config_format: ConfigFormat = ConfigFormat.AUTO,
    token: Optional[str] = None,
    **kwargs,
) -> PretrainedConfig:
    # Separate model folder from file path for GGUF models

    is_gguf = check_gguf_file(model)
    if is_gguf:
        kwargs["gguf_file"] = Path(model).name
        model = Path(model).parent

    if config_format == ConfigFormat.AUTO:
        if is_gguf or file_or_path_exists(
                model, HF_CONFIG_NAME, revision=revision, token=token):
            config_format = ConfigFormat.HF
        elif file_or_path_exists(model,
                                 MISTRAL_CONFIG_NAME,
                                 revision=revision,
                                 token=token):
            config_format = ConfigFormat.MISTRAL
        else:
            # If we're in offline mode and found no valid config format, then
            # raise an offline mode error to indicate to the user that they
            # don't have files cached and may need to go online.
            # This is conveniently triggered by calling file_exists().
            file_exists(model, HF_CONFIG_NAME, revision=revision, token=token)

            raise ValueError(f"No supported config format found in {model}")

    if config_format == ConfigFormat.HF:
        config_dict, _ = PretrainedConfig.get_config_dict(
            model,
            revision=revision,
            code_revision=code_revision,
            token=token,
            **kwargs,
        )

        # Use custom model class if it's in our registry
        model_type = config_dict.get("model_type")
        if model_type in _CONFIG_REGISTRY:
            config_class = _CONFIG_REGISTRY[model_type]
            config = config_class.from_pretrained(
                model,
                revision=revision,
                code_revision=code_revision,
                token=token,
                **kwargs,
            )
        else:
            try:
                config = AutoConfig.from_pretrained(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    code_revision=code_revision,
                    token=token,
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
        config = load_params_config(model, revision, token=token, **kwargs)
    else:
        raise ValueError(f"Unsupported config format: {config_format}")

    # Special architecture mapping check for GGUF models
    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(
                f"Can't get gguf config for {config.model_type}.")
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})

    patch_rope_scaling(config)

    if trust_remote_code:
        maybe_register_config_serialize_by_value()

    return config


def get_hf_file_to_dict(file_name: str,
                        model: Union[str, Path],
                        revision: Optional[str] = 'main',
                        token: Optional[str] = None):
    """
    Downloads a file from the Hugging Face Hub and returns 
    its contents as a dictionary.

    Parameters:
    - file_name (str): The name of the file to download.
    - model (str): The name of the model on the Hugging Face Hub.
    - revision (str): The specific version of the model. 
    - token (str): The Hugging Face authentication token.

    Returns:
    - config_dict (dict): A dictionary containing 
    the contents of the downloaded file.
    """
    file_path = Path(model) / file_name

    if file_or_path_exists(model=model,
                           config_name=file_name,
                           revision=revision,
                           token=token):

        if not file_path.is_file():
            try:
                hf_hub_file = hf_hub_download(model,
                                              file_name,
                                              revision=revision)
            except (RepositoryNotFoundError, RevisionNotFoundError,
                    EntryNotFoundError, LocalEntryNotFoundError) as e:
                logger.debug("File or repository not found in hf_hub_download",
                             e)
                return None
            file_path = Path(hf_hub_file)

        with open(file_path) as file:
            return json.load(file)
    return None


def get_pooling_config(model: str,
                       revision: Optional[str] = 'main',
                       token: Optional[str] = None):
    """
    This function gets the pooling and normalize 
    config from the model - only applies to 
    sentence-transformers models. 

    Args:
        model (str): The name of the Hugging Face model.
        revision (str, optional): The specific version 
        of the model to use. Defaults to 'main'.

    Returns:
        dict: A dictionary containing the pooling 
        type and whether normalization is used.
    """

    modules_file_name = "modules.json"
    modules_dict = get_hf_file_to_dict(modules_file_name, model, revision,
                                       token)

    if modules_dict is None:
        return None

    pooling = next((item for item in modules_dict
                    if item["type"] == "sentence_transformers.models.Pooling"),
                   None)
    normalize = bool(
        next((item for item in modules_dict
              if item["type"] == "sentence_transformers.models.Normalize"),
             False))

    if pooling:

        pooling_file_name = "{}/config.json".format(pooling["path"])
        pooling_dict = get_hf_file_to_dict(pooling_file_name, model, revision,
                                           token)
        pooling_type_name = next(
            (item for item, val in pooling_dict.items() if val is True), None)

        if pooling_type_name is not None:
            pooling_type_name = get_pooling_config_name(pooling_type_name)

        return {"pooling_type": pooling_type_name, "normalize": normalize}

    return None


def get_pooling_config_name(pooling_name: str) -> Union[str, None]:
    if "pooling_mode_" in pooling_name:
        pooling_name = pooling_name.replace("pooling_mode_", "")

    if "_" in pooling_name:
        pooling_name = pooling_name.split("_")[0]

    if "lasttoken" in pooling_name:
        pooling_name = "last"

    supported_pooling_types = ['LAST', 'ALL', 'CLS', 'STEP', 'MEAN']
    pooling_type_name = pooling_name.upper()

    try:
        if pooling_type_name in supported_pooling_types:
            return pooling_type_name
    except NotImplementedError as e:
        logger.debug("Pooling type not supported", e)
        return None
    return None


def get_sentence_transformer_tokenizer_config(model: str,
                                              revision: Optional[str] = 'main',
                                              token: Optional[str] = None):
    """
    Returns the tokenization configuration dictionary for a 
    given Sentence Transformer BERT model.

    Parameters:
    - model (str): The name of the Sentence Transformer 
    BERT model.
    - revision (str, optional): The revision of the m
    odel to use. Defaults to 'main'.
    - token (str): A Hugging Face access token.

    Returns:
    - dict: A dictionary containing the configuration parameters 
    for the Sentence Transformer BERT model.
    """
    for config_name in [
            "sentence_bert_config.json",
            "sentence_roberta_config.json",
            "sentence_distilbert_config.json",
            "sentence_camembert_config.json",
            "sentence_albert_config.json",
            "sentence_xlm-roberta_config.json",
            "sentence_xlnet_config.json",
    ]:
        encoder_dict = get_hf_file_to_dict(config_name, model, revision, token)
        if encoder_dict:
            break

    if not encoder_dict:
        return None

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
        >>> klass = AutoConfig.from_pretrained('meta-llama/Meta-Llama-3-8B', trust_remote_code=True)
        >>> klass.__class__ # transformers.models.llama.configuration_llama.LlamaConfig
        >>> import transformers_modules # error, not initialized
        >>> klass = AutoConfig.from_pretrained('deepseek-ai/DeepSeek-V2.5', trust_remote_code=True)
        >>> import transformers_modules # success, initialized
        >>> klass.__class__ # transformers_modules.deepseek-ai.DeepSeek-V2.5.98b11844770b2c3ffc18b175c758a803640f4e77.configuration_deepseek.DeepseekV2Config

        In the DeepSeek example, the config class is an instance of a custom
        class that is not serializable by default. This class will not be
        importable in spawned workers, and won't exist at all on
        other nodes, which breaks serialization of the config.

        In this function we tell the cloudpickle serialization library to pass
        instances of these generated classes by value instead of by reference,
        i.e. the class definition is serialized along with its data so that the
        class module does not need to be importable on the receiving end.

        See: https://github.com/cloudpipe/cloudpickle?tab=readme-ov-file#overriding-pickles-serialization-mechanism-for-importable-constructs
    """ # noqa
    try:
        import transformers_modules
    except ImportError:
        # the config does not need trust_remote_code
        return

    try:
        import cloudpickle
        cloudpickle.register_pickle_by_value(transformers_modules)

        # ray vendors its own version of cloudpickle
        from vllm.executor.ray_utils import ray
        if ray:
            ray.cloudpickle.register_pickle_by_value(transformers_modules)

        # multiprocessing uses pickle to serialize arguments when using spawn
        # Here we get pickle to use cloudpickle to serialize config objects
        # that contain instances of the custom config class to avoid
        # serialization problems if the generated module (and model) has a `.`
        # in its name
        import multiprocessing
        import pickle

        from vllm.config import VllmConfig

        def _reduce_config(config: VllmConfig):
            return (pickle.loads, (cloudpickle.dumps(config), ))

        multiprocessing.reducer.register(VllmConfig, _reduce_config)

    except Exception as e:
        logger.warning(
            "Unable to register remote classes used by"
            " trust_remote_code with by-value serialization. This may"
            " lead to a later error. If remote code is not needed"
            " remove `--trust-remote-code`",
            exc_info=e)


def load_params_config(model: Union[str, Path],
                       revision: Optional[str],
                       token: Optional[str] = None,
                       **kwargs) -> PretrainedConfig:
    # This function loads a params.json config which
    # should be used when loading models in mistral format

    config_file_name = "params.json"

    config_dict = get_hf_file_to_dict(config_file_name, model, revision, token)
    assert isinstance(config_dict, dict)

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

    config_dict.update(kwargs)

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
