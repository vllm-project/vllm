from transformers import AutoConfig, PretrainedConfig
from vllm.transformers_utils.configs import *


def get_config(model: str) -> PretrainedConfig:
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    if config.model_type == "RefinedWeb":
        config = RWConfig.from_pretrained(model)
    return config
