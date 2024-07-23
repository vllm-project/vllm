import contextlib
from typing import Dict, Optional, Type

from transformers import GenerationConfig, PretrainedConfig

from vllm.envs import VLLM_USE_MODELSCOPE
from vllm.logger import init_logger
from vllm.transformers_utils.configs import (ChatGLMConfig, DbrxConfig,
                                             JAISConfig, MedusaConfig,
                                             MLPSpeculatorConfig, MPTConfig,
                                             RWConfig)

if VLLM_USE_MODELSCOPE:
    from modelscope import AutoConfig
else:
    from transformers import AutoConfig

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
}

for name, cls in _CONFIG_REGISTRY.items():
    with contextlib.suppress(ValueError):
        AutoConfig.register(name, cls)


def get_config(model: str,
               trust_remote_code: bool,
               revision: Optional[str] = None,
               code_revision: Optional[str] = None,
               rope_scaling: Optional[dict] = None,
               rope_theta: Optional[float] = None) -> PretrainedConfig:
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
    for key, value in [("rope_scaling", rope_scaling),
                       ("rope_theta", rope_theta)]:
        if value is not None:
            logger.info("Updating %s from %r to %r", key,
                        getattr(config, key, None), value)
            config.update({key: value})
    return config


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
