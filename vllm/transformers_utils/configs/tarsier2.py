# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import PretrainedConfig, Qwen2VLConfig
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLTextConfig


def _resolve_text_config(
    text_config: "PretrainedConfig | dict",
) -> "PretrainedConfig":
    """Unwrap a potentially nested text config to a Qwen2VLTextConfig.

    Tarsier2's config.json stores a full Qwen2VLConfig as the text_config
    field. In Transformers v5, Qwen2VLConfig was split so that text model
    attributes (e.g. num_attention_heads) live on an inner Qwen2VLTextConfig,
    not on Qwen2VLConfig itself. This helper digs through that nesting and
    returns a config object that directly exposes num_attention_heads.
    """
    if isinstance(text_config, dict):
        inner = text_config.get("text_config")
        if inner is not None:
            return _resolve_text_config(inner)
        return Qwen2VLTextConfig(**text_config)

    if isinstance(text_config, PretrainedConfig):
        if hasattr(text_config, "num_attention_heads"):
            return text_config
        inner = getattr(text_config, "text_config", None)
        if inner is not None:
            return _resolve_text_config(inner)

    return text_config


class Tarsier2Config(Qwen2VLConfig):
    """
    Tarsier2's config.json reports ``model_type: "llava"``, which causes
    ``AutoConfig.from_pretrained`` to instantiate ``LlavaConfig`` with a
    ``Qwen2VLConfig`` as its text_config. In Transformers v5, ``Qwen2VLConfig``
    was split into ``Qwen2VLConfig`` + ``Qwen2VLTextConfig``, so attributes
    like ``num_attention_heads`` no longer exist directly on ``Qwen2VLConfig``.

    This class is registered under model_type ``"tarsier2"`` (and also under
    ``"llava"`` to intercept the on-disk model_type) so that the config is
    loaded as a plain ``Qwen2VLConfig`` equivalent. ``get_text_config`` is
    overridden to unwrap any remaining nesting and always return a
    ``Qwen2VLTextConfig`` with the text model attributes present.
    """

    model_type = "tarsier2"

    def get_text_config(self, **kwargs) -> PretrainedConfig:
        text_config = super().get_text_config(**kwargs)
        return _resolve_text_config(text_config)
