from transformers import AutoConfig, PretrainedConfig


def get_config(model: str) -> PretrainedConfig:
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    return config
