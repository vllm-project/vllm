from typing import Optional

from transformers import AutoConfig, PretrainedConfig

from vllm.transformers_utils.configs import *

_CONFIG_REGISTRY = {
    "chatglm": ChatGLMConfig,
    "mpt": MPTConfig,
    "RefinedWeb": RWConfig,  # For tiiuae/falcon-40b(-instruct)
    "RefinedWebModel": RWConfig,  # For tiiuae/falcon-7b(-instruct)
    "starcoder2": Starcoder2Config,
}


def get_config(model: str,
               trust_remote_code: bool,
               revision: Optional[str] = None,
               code_revision: Optional[str] = None) -> PretrainedConfig:
    # FIXME(woosuk): This is a temporary fix for StarCoder2.
    # Remove this when the model is supported by HuggingFace transformers.
    if "bigcode" in model and "starcoder2" in model:
        config_class = _CONFIG_REGISTRY["starcoder2"]
        config = config_class.from_pretrained(model,
                                              revision=revision,
                                              code_revision=code_revision)
        return config

    try:
        config = AutoConfig.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision)
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
        config = config_class.from_pretrained(model,
                                              revision=revision,
                                              code_revision=code_revision)
    return config
