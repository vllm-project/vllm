# SPDX-License-Identifier: Apache-2.0

import enum
import json
import os
import time
from functools import cache
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Type, Union

import huggingface_hub
from huggingface_hub import hf_hub_download
from huggingface_hub import list_repo_files as hf_list_repo_files
from huggingface_hub import try_to_load_from_cache
from huggingface_hub.utils import (EntryNotFoundError, HfHubHTTPError,
                                   HFValidationError, LocalEntryNotFoundError,
                                   RepositoryNotFoundError,
                                   RevisionNotFoundError)
from torch import nn
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
from vllm.transformers_utils.configs import (ChatGLMConfig, Cohere2Config,
                                             DbrxConfig, DeepseekVLV2Config,
                                             EAGLEConfig, ExaoneConfig,
                                             H2OVLChatConfig,
                                             InternVLChatConfig, JAISConfig,
                                             MedusaConfig, MllamaConfig,
                                             MLPSpeculatorConfig, MPTConfig,
                                             NemotronConfig, NVLM_D_Config,
                                             Olmo2Config, RWConfig,
                                             SolarConfig, Telechat2Config,
                                             UltravoxConfig)
# yapf: enable
from vllm.transformers_utils.utils import check_gguf_file
from vllm.utils import resolve_obj_by_qualname

if VLLM_USE_MODELSCOPE:
    from modelscope import AutoConfig
else:
    from transformers import AutoConfig

MISTRAL_CONFIG_NAME = "params.json"
HF_TOKEN = os.getenv('HF_TOKEN', None)

logger = init_logger(__name__)

_CONFIG_REGISTRY_OVERRIDE_HF: Dict[str, Type[PretrainedConfig]] = {
    "mllama": MllamaConfig
}

_CONFIG_REGISTRY: Dict[str, Type[PretrainedConfig]] = {
    "chatglm": ChatGLMConfig,
    "cohere2": Cohere2Config,
    "dbrx": DbrxConfig,
    "deepseek_vl_v2": DeepseekVLV2Config,
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
    "olmo2": Olmo2Config,
    "solar": SolarConfig,
    "telechat": Telechat2Config,
    "ultravox": UltravoxConfig,
    **_CONFIG_REGISTRY_OVERRIDE_HF
}


class ConfigFormat(str, enum.Enum):
    AUTO = "auto"
    HF = "hf"
    MISTRAL = "mistral"


def with_retry(func: Callable[[], Any],
               log_msg: str,
               max_retries: int = 2,
               retry_delay: int = 2):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error("%s: %s", log_msg, e)
                raise
            logger.error("%s: %s, retrying %d of %d", log_msg, e, attempt + 1,
                         max_retries)
            time.sleep(retry_delay)
            retry_delay *= 2


# @cache doesn't cache exceptions
@cache
def list_repo_files(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
    token: Union[str, bool, None] = None,
) -> list[str]:

    def lookup_files() -> list[str]:
        # directly list files if model is local
        if (local_path := Path(repo_id)).exists():
            return [
                str(file.relative_to(local_path))
                for file in local_path.rglob('*') if file.is_file()
            ]
        # if model is remote, use hf_hub api to list files
        try:
            if VLLM_USE_MODELSCOPE:
                from vllm.transformers_utils.utils import (
                    modelscope_list_repo_files)
                return modelscope_list_repo_files(repo_id,
                                                  revision=revision,
                                                  token=token)
            return hf_list_repo_files(repo_id,
                                      revision=revision,
                                      repo_type=repo_type,
                                      token=token)
        except huggingface_hub.errors.OfflineModeIsEnabled:
            # Don't raise in offline mode,
            # all we know is that we don't have this
            # file cached.
            return []

    return with_retry(lookup_files, "Error retrieving file list")


def file_exists(
    repo_id: str,
    file_name: str,
    *,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    token: Union[str, bool, None] = None,
) -> bool:
    file_list = list_repo_files(repo_id,
                                repo_type=repo_type,
                                revision=revision,
                                token=token)
    return file_name in file_list


# In offline mode the result can be a false negative
def file_or_path_exists(model: Union[str, Path], config_name: str,
                        revision: Optional[str]) -> bool:
    if (local_path := Path(model)).exists():
        return (local_path / config_name).is_file()

    # Offline mode support: Check if config file is cached already
    cached_filepath = try_to_load_from_cache(repo_id=model,
                                             filename=config_name,
                                             revision=revision)
    if isinstance(cached_filepath, str):
        # The config file exists in cache- we can continue trying to load
        return True

    # NB: file_exists will only check for the existence of the config file on
    # hf_hub. This will fail in offline mode.

    # Call HF to check if the file exists
    return file_exists(str(model),
                       config_name,
                       revision=revision,
                       token=HF_TOKEN)


def patch_rope_scaling(config: PretrainedConfig) -> None:
    """Provide backwards compatibility for RoPE."""
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        patch_rope_scaling(text_config)

    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None:
        patch_rope_scaling_dict(rope_scaling)


def patch_rope_scaling_dict(rope_scaling: Dict[str, Any]) -> None:
    if "rope_type" in rope_scaling and "type" in rope_scaling:
        rope_type = rope_scaling["rope_type"]
        rope_type_legacy = rope_scaling["type"]
        if rope_type != rope_type_legacy:
            raise ValueError(
                f"Found conflicts between 'rope_type={rope_type}' (modern "
                f"field) and 'type={rope_type_legacy}' (legacy field). "
                "You should only specify one of them.")

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
    **kwargs,
) -> PretrainedConfig:
    # Separate model folder from file path for GGUF models

    is_gguf = check_gguf_file(model)
    if is_gguf:
        kwargs["gguf_file"] = Path(model).name
        model = Path(model).parent

    if config_format == ConfigFormat.AUTO:
        if is_gguf or file_or_path_exists(
                model, HF_CONFIG_NAME, revision=revision):
            config_format = ConfigFormat.HF
        elif file_or_path_exists(model, MISTRAL_CONFIG_NAME,
                                 revision=revision):
            config_format = ConfigFormat.MISTRAL
        else:
            raise ValueError(f"No supported config format found in {model}.")

    if config_format == ConfigFormat.HF:
        config_dict, _ = PretrainedConfig.get_config_dict(
            model,
            revision=revision,
            code_revision=code_revision,
            token=HF_TOKEN,
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
                token=HF_TOKEN,
                **kwargs,
            )
        else:
            try:
                config = AutoConfig.from_pretrained(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    code_revision=code_revision,
                    token=HF_TOKEN,
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
        config = load_params_config(model, revision, token=HF_TOKEN, **kwargs)
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


def try_get_local_file(model: Union[str, Path],
                       file_name: str,
                       revision: Optional[str] = 'main') -> Optional[Path]:
    file_path = Path(model) / file_name
    if file_path.is_file():
        return file_path
    else:
        try:
            cached_filepath = try_to_load_from_cache(repo_id=model,
                                                     filename=file_name,
                                                     revision=revision)
            if isinstance(cached_filepath, str):
                return Path(cached_filepath)
        except HFValidationError:
            ...
    return None


def get_hf_file_to_dict(file_name: str,
                        model: Union[str, Path],
                        revision: Optional[str] = 'main'):
    """
    Downloads a file from the Hugging Face Hub and returns
    its contents as a dictionary.

    Parameters:
    - file_name (str): The name of the file to download.
    - model (str): The name of the model on the Hugging Face Hub.
    - revision (str): The specific version of the model.

    Returns:
    - config_dict (dict): A dictionary containing
    the contents of the downloaded file.
    """

    file_path = try_get_local_file(model=model,
                                   file_name=file_name,
                                   revision=revision)

    if file_path is None:
        try:
            hf_hub_file = hf_hub_download(model, file_name, revision=revision)
        except huggingface_hub.errors.OfflineModeIsEnabled:
            return None
        except (RepositoryNotFoundError, RevisionNotFoundError,
                EntryNotFoundError, LocalEntryNotFoundError) as e:
            logger.debug("File or repository not found in hf_hub_download", e)
            return None
        except HfHubHTTPError as e:
            logger.warning(
                "Cannot connect to Hugging Face Hub. Skipping file "
                "download for '%s':",
                file_name,
                exc_info=e)
            return None
        file_path = Path(hf_hub_file)

    if file_path is not None and file_path.is_file():
        with open(file_path) as file:
            return json.load(file)

    return None


@cache
def get_pooling_config(model: str, revision: Optional[str] = 'main'):
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

    modules_dict = None
    if file_or_path_exists(model=model,
                           config_name=modules_file_name,
                           revision=revision):
        modules_dict = get_hf_file_to_dict(modules_file_name, model, revision)

    if modules_dict is None:
        return None

    logger.info("Found sentence-transformers modules configuration.")

    pooling = next((item for item in modules_dict
                    if item["type"] == "sentence_transformers.models.Pooling"),
                   None)
    normalize = bool(
        next((item for item in modules_dict
              if item["type"] == "sentence_transformers.models.Normalize"),
             False))

    if pooling:

        pooling_file_name = "{}/config.json".format(pooling["path"])
        pooling_dict = get_hf_file_to_dict(pooling_file_name, model, revision)
        pooling_type_name = next(
            (item for item, val in pooling_dict.items() if val is True), None)

        if pooling_type_name is not None:
            pooling_type_name = get_pooling_config_name(pooling_type_name)

        logger.info("Found pooling configuration.")
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


@cache
def get_sentence_transformer_tokenizer_config(model: str,
                                              revision: Optional[str] = 'main'
                                              ):
    """
    Returns the tokenization configuration dictionary for a
    given Sentence Transformer BERT model.

    Parameters:
    - model (str): The name of the Sentence Transformer
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
        if try_get_local_file(model=model,
                              file_name=config_file,
                              revision=revision) is not None:
            encoder_dict = get_hf_file_to_dict(config_file, model, revision)
            if encoder_dict:
                break

    if not encoder_dict and not model.startswith("/"):
        try:
            # If model is on HuggingfaceHub, get the repo files
            repo_files = list_repo_files(model,
                                         revision=revision,
                                         token=HF_TOKEN)
        except Exception:
            repo_files = []

        for config_name in sentence_transformer_config_files:
            if config_name in repo_files:
                encoder_dict = get_hf_file_to_dict(config_name, model,
                                                   revision)
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


def load_params_config(model: Union[str, Path], revision: Optional[str],
                       **kwargs) -> PretrainedConfig:
    # This function loads a params.json config which
    # should be used when loading models in mistral format

    config_file_name = "params.json"

    config_dict = get_hf_file_to_dict(config_file_name, model, revision)
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

            return config_dict
        else:
            return elem

    config_dict["model_type"] = config_dict.get("model_type", "transformer")
    config_dict["hidden_act"] = config_dict.get("activation", "silu")
    config_dict["tie_word_embeddings"] = config_dict.get(
        "tie_embeddings", False)
    config_dict["max_seq_len"] = config_dict.get("max_seq_len", 128_000)
    config_dict["max_position_embeddings"] = config_dict.get(
        "max_position_embeddings", 128_000)

    if config_dict.get("quantization") is not None:
        quantization = config_dict.get("quantization", {})
        if quantization.get("qformat_weight") == "fp8_e4m3":
            # This maps to the FP8 static per-tensor quantization scheme
            quantization_config = {
                "quant_method": "fp8",
                "activation_scheme": "static"
            }
        else:
            raise ValueError(
                f"Found unknown quantization='{quantization}' in config")

        config_dict["quantization_config"] = quantization_config

    config_type: Literal["text",
                         "multimodal"] = "multimodal" if config_dict.get(
                             "vision_encoder") is not None else "text"

    if config_dict.get("moe") is not None:
        config_dict["architectures"] = ["MixtralForCausalLM"]
    else:
        config_dict["architectures"] = ["MistralForCausalLM"]

    if config_type == "multimodal":
        multimodal_config = config_dict.pop("vision_encoder")

        config_dict = {
            "text_config": config_dict,
            "vision_config": multimodal_config
        }
        config_dict["architectures"] = ["PixtralForConditionalGeneration"]
        config_dict["model_type"] = "pixtral"

    config_dict.update(kwargs)

    config_dict = recurse_elems(config_dict)

    # transform to HF config format
    if config_type == "multimodal":
        config_dict["text_config"] = PretrainedConfig(
            **config_dict["text_config"])
        config_dict["vision_config"] = PretrainedConfig(
            **config_dict["vision_config"])

    return PretrainedConfig(**config_dict)


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


def get_cross_encoder_activation_function(config: PretrainedConfig):
    if (hasattr(config, "sbert_ce_default_activation_function")
            and config.sbert_ce_default_activation_function is not None):

        function_name = config.sbert_ce_default_activation_function
        assert function_name.startswith("torch.nn.modules."), \
            "Loading of activation functions is restricted to " \
            "torch.nn.modules for security reasons"
        return resolve_obj_by_qualname(function_name)()
    else:
        return nn.Sigmoid() if config.num_labels == 1 else nn.Identity()
