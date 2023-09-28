from typing import Optional

from transformers import AutoConfig, PretrainedConfig

from vllm.transformers_utils.configs import *  # pylint: disable=wildcard-import

_CONFIG_REGISTRY = {
    "mpt": MPTConfig,
    "baichuan": BaiChuanConfig,
    "aquila": AquilaConfig,
    "qwen": QWenConfig,
    "RefinedWeb": RWConfig,  # For tiiuae/falcon-40b(-instruct)
    "RefinedWebModel": RWConfig,  # For tiiuae/falcon-7b(-instruct)
}


def get_config(model: str,
               trust_remote_code: bool,
               revision: Optional[str] = None) -> PretrainedConfig:
    # NOTE: Because the Mistral model in HF hub does not have
    # `configuration_mistral.py`, we cannot use `AutoConfig` to load the
    # config. Instead, we use `MistralConfig` directly.
    # NOTE: This is a hack. This does not work for local models.
    # FIXME: Remove this once the Mistral model is available in the stable
    # version of HF transformers.
    if "mistral" in model.lower():
        return MistralConfig.from_pretrained(model, revision=revision)

    try:
        config = AutoConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision)
    except ValueError as e:
        if (not trust_remote_code and
                "requires you to execute the configuration file" in str(e)):
            err_msg = (
                "Failed to load the model config. If the model is a custom "
                "model not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e
    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        config = config_class.from_pretrained(model, revision=revision)
    return config
