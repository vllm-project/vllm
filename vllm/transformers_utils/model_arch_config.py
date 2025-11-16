# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from pathlib import Path
from typing import Any

from transformers.utils import CONFIG_NAME as HF_CONFIG_NAME

from vllm.config.model_arch import ModelArchitectureConfig
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    MISTRAL_CONFIG_NAME,
    ConfigFormat,
    file_or_path_exists,
)
from vllm.transformers_utils.model_arch_config_parser import (
    HFModelArchConfigParser,
    ModelArchConfigParserBase,
)
from vllm.transformers_utils.utils import (
    check_gguf_file,
)

logger = init_logger(__name__)


_CONFIG_FORMAT_TO_MODEL_ARCH_CONFIG_PARSER: dict[
    str, type[ModelArchConfigParserBase]
] = {
    "hf": HFModelArchConfigParser,
}
SUPPORTED_ARCHITECTURES: list[str] = ["LlamaForCausalLM"]


def get_model_arch_config_parser(config_format: str) -> ModelArchConfigParserBase:
    """Get the model architecture config parser for a given config format."""
    if config_format not in _CONFIG_FORMAT_TO_MODEL_ARCH_CONFIG_PARSER:
        raise ValueError(f"Unknown config format `{config_format}`.")
    return _CONFIG_FORMAT_TO_MODEL_ARCH_CONFIG_PARSER[config_format]()


def get_model_arch_config(
    model: str | Path,
    trust_remote_code: bool,
    revision: str | None = None,
    code_revision: str | None = None,
    config_format: str | ConfigFormat = "auto",
    model_arch_overrides_kw: dict[str, Any] | None = None,
    model_arch_overrides_fn: Callable[
        ["ModelArchitectureConfig"], "ModelArchitectureConfig"
    ]
    | None = None,
    **kwargs,
) -> "ModelArchitectureConfig":
    # Separate model folder from file path for GGUF models
    is_gguf = check_gguf_file(model)
    kwargs["is_gguf"] = is_gguf

    if config_format == "auto":
        try:
            if is_gguf or file_or_path_exists(model, HF_CONFIG_NAME, revision=revision):
                config_format = "hf"
            elif file_or_path_exists(model, MISTRAL_CONFIG_NAME, revision=revision):
                config_format = "mistral"
            else:
                raise ValueError(
                    "Could not detect config format for no config file found. "
                    "With config_format 'auto', ensure your model has either "
                    "config.json (HF format) or params.json (Mistral format). "
                    "Otherwise please specify your_custom_config_format "
                    "in engine args for customized config parser."
                )

        except Exception as e:
            error_message = (
                "Invalid repository ID or local directory specified:"
                " '{model}'.\nPlease verify the following requirements:\n"
                "1. Provide a valid Hugging Face repository ID.\n"
                "2. Specify a local directory that contains a recognized "
                "configuration file.\n"
                "   - For Hugging Face models: ensure the presence of a "
                "'config.json'.\n"
                "   - For Mistral models: ensure the presence of a "
                "'params.json'.\n"
                "3. For GGUF: pass the local path of the GGUF checkpoint.\n"
                "   Loading GGUF from a remote repo directly is not yet "
                "supported.\n"
            ).format(model=model)

            raise ValueError(error_message) from e

    config_parser = get_model_arch_config_parser(config_format)
    config_dict, config = config_parser.parse(
        model,
        trust_remote_code=trust_remote_code,
        revision=revision,
        code_revision=code_revision,
        **kwargs,
    )

    if model_arch_overrides_kw:
        logger.debug("Overriding model arch config with %s", model_arch_overrides_kw)
        for key, value in model_arch_overrides_kw.items():
            setattr(config, key, value)
    if model_arch_overrides_fn:
        logger.debug("Overriding model arch config with %s", model_arch_overrides_fn)
        config = model_arch_overrides_fn(config)

    return config
