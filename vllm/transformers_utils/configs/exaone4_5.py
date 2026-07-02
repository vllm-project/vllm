# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config shim for EXAONE 4.5 checkpoints.

LG-AI EXAONE 4.5 releases publish ``text_config.layer_types`` with one more
entry than ``text_config.num_hidden_layers``; the trailing entry describes the
multi-token-prediction (MTP) head that ships alongside the transformer stack.

On ``transformers`` v5 the inner text config is built as ``Exaone4Config``
(``Exaone4_5_Config.__post_init__`` remaps the ``exaone4_5_text`` model type to
``exaone4``). ``Exaone4Config`` has no concept of the MTP head, so its strict
``validate_layer_type`` check rejects the config before vLLM reaches the model
loader::

    ValueError: `num_hidden_layers` (64) must be equal to the number of
    `layer_types` (65)

The MTP head is consumed separately through ``num_nextn_predict_layers`` (see
``Exaone4_5MTP``), so dropping the trailing ``layer_types`` entry only unblocks
the base config validation and leaves MTP behavior unchanged.

This is a temporary compatibility shim, not the canonical fix. The trailing
``layer_types`` entry is a model-card issue and is better corrected at the
source: see the LG-AI model-repo discussion
(https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B-FP8/discussions/2). Once the
published config sets ``len(layer_types) == num_hidden_layers``, the guard below
never fires and this shim can be removed.
"""

from __future__ import annotations

from transformers.models.exaone4_5 import Exaone4_5_Config as _BaseExaone4_5Config


class Exaone4_5Config(_BaseExaone4_5Config):
    """``Exaone4_5_Config`` that trims the trailing MTP ``layer_types`` entry.

    The trim is applied only when ``len(layer_types) == num_hidden_layers + 1``
    so that genuinely malformed configs are left untouched and surface their
    own errors instead of being silently masked.
    """

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        text_config = config_dict.get("text_config")
        if isinstance(text_config, dict):
            num_hidden_layers = text_config.get("num_hidden_layers")
            layer_types = text_config.get("layer_types")
            if (
                isinstance(num_hidden_layers, int)
                and isinstance(layer_types, list)
                and len(layer_types) == num_hidden_layers + 1
            ):
                config_dict = {
                    **config_dict,
                    "text_config": {
                        **text_config,
                        "layer_types": layer_types[:num_hidden_layers],
                    },
                }
        return super().from_dict(config_dict, **kwargs)


__all__ = ["Exaone4_5Config"]
