# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hw-agnostic FP8 KV-cache method."""

from __future__ import annotations

from vllm.model_executor.hw_agnostic.quantization.fp8_config import Fp8Config
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod


class Fp8KVCacheMethod(BaseKVCacheMethod):
    """Hw-agnostic FP8 KV-cache scaling."""

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
