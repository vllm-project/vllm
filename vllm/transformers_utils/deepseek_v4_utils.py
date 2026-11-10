# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any


def get_compress_ratios(config: Any) -> list[int]:
    """Per-layer compress ratios (0 = sliding window), MTP layers included.

    Reads the legacy `compress_ratios` of DeepSeek-V4.1 configs, or derives it
    from `layer_types` of the upstream `DeepseekV4Config`, which drops the MTP
    layers. Returns an empty list for other configs.
    """
    if (compress_ratios := getattr(config, "compress_ratios", None)) is not None:
        return compress_ratios
    if (compress_rates := getattr(config, "compress_rates", None)) is None:
        return []
    ratios = [compress_rates.get(t, 0) for t in config.layer_types]
    return ratios + [0] * config.num_nextn_predict_layers
