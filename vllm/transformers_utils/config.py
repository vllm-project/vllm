from transformers import AutoConfig, PretrainedConfig

from vllm.transformers_utils.configs import *  # pylint: disable=wildcard-import

_CONFIG_REGISTRY = {
    "mpt": MPTConfig,
}


def get_config(model: str) -> PretrainedConfig:
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        config = config_class.from_pretrained(model)
    return config
