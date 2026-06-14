# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""EXAONE 4.5 model configuration shim.

The LG-AI EXAONE 4.5 release config (e.g. ``LGAI-EXAONE/EXAONE-4.5-33B-FP8``)
publishes ``text_config.layer_types`` of length ``text_config.num_hidden_layers
+ 1``: the extra entry corresponds to the MTP head registered alongside the
regular transformer stack. Transformers v5's strict ``validate_layer_type``
rejects this +1 mismatch when instantiating the inner ``Exaone4Config`` (the
``exaone4_5_text`` model type is remapped to ``exaone4``, which has no notion
of the MTP layer). The MTP layer is consumed separately by the
speculative-decode path via ``num_nextn_predict_layers`` /
``_num_mtp_layers``; trimming the trailing layer_types entry before the inner
config is built is safe for the base model and lets the base + MTP path load
on mainline transformers.
"""

from __future__ import annotations

from transformers.models.exaone4_5 import (
    Exaone4_5_Config as _UpstreamExaone4_5_Config,
)


class Exaone4_5Config(_UpstreamExaone4_5_Config):
    """Drop the trailing MTP-head entry from ``text_config.layer_types``.

    Only applies the trim when the length is exactly ``num_hidden_layers + 1``;
    any other shape is left untouched so unrelated config bugs are not masked.
    """

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        text_config = config_dict.get("text_config")
        if isinstance(text_config, dict):
            n_layers = text_config.get("num_hidden_layers")
            layer_types = text_config.get("layer_types")
            if (
                isinstance(n_layers, int)
                and isinstance(layer_types, list)
                and len(layer_types) == n_layers + 1
            ):
                config_dict = {
                    **config_dict,
                    "text_config": {
                        **text_config,
                        "layer_types": layer_types[:n_layers],
                    },
                }
        return super().from_dict(config_dict, **kwargs)


__all__ = ["Exaone4_5Config"]
