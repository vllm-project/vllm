from transformers import AutoConfig, PretrainedConfig
from typing import Optional

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
               rope_scaling: Optional[dict],
               revision: Optional[str] = None) -> PretrainedConfig:

    def _rope_scaling_validation():
        """
        Validate the `rope_scaling` configuration.
        """
        if rope_scaling is None:
            return

        if not isinstance(rope_scaling, dict) or len(rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with",
                "with two fields, `type` and `factor`, "
                f"got {rope_scaling}")
        rope_scaling_type = rope_scaling.get("type", None)
        rope_scaling_factor = rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in [
                "linear", "dynamic"
        ]:
            raise ValueError(
                "`rope_scaling`'s name field must be one ",
                f"of ['linear', 'dynamic'], got {rope_scaling_type}")
        if rope_scaling_factor is None or not isinstance(
                rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(
                "`rope_scaling`'s factor field must be an float > 1,",
                f" got {rope_scaling_factor}")

    try:
        _rope_scaling_validation()
        config = AutoConfig.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            rope_scaling=rope_scaling,
            revision=revision)
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
