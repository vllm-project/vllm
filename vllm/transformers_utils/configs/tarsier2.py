# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import Qwen2VLConfig


class Tarsier2Config(Qwen2VLConfig):
    """
    Config class for Tarsier2 models.

    Tarsier2's config.json reports ``model_type: "llava"``, which causes
    ``AutoConfig.from_pretrained`` to instantiate ``LlavaConfig`` instead of
    ``Qwen2VLConfig``. ``LlavaConfig.get_text_config()`` returns a
    ``Qwen2VLConfig`` sub-object that does not expose ``num_attention_heads``
    directly (it is delegated to the inner ``Qwen2VLTextConfig``), triggering
    the ``ValueError`` in ``get_hf_text_config``.

    Registering this subclass and loading via ``config_class.from_pretrained``
    (see ``HFConfigParser.parse``) bypasses the ``model_type`` mismatch.
    ``Qwen2VLConfig.__post_init__`` then correctly converts the flat
    ``text_config`` dict into a ``Qwen2VLTextConfig`` with all fields intact
    (``num_attention_heads=28``, ``hidden_size=3584``, etc.).
    """

    model_type = "tarsier2"
