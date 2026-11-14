# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Digest of the checkpoint a model was loaded from."""

from vllm.config import LoadConfig, ModelConfig
from vllm.config.utils import hash_factors


def path_checkpoint_digest(
    model_config: ModelConfig, load_config: LoadConfig
) -> dict[str, str]:
    factors: dict[str, object] = {
        "model": model_config.model,
        "model_weights": model_config.model_weights,
        "revision": model_config.revision,
        "load_format": str(load_config.load_format),
    }
    return {"weights": hash_factors(factors), "weights_source": "path"}
