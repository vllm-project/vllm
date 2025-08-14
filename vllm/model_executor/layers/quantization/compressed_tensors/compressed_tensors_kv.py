# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod


class CompressedTensorsKVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from compressed-tensors
    checkpoints.
    """

    def __init__(self, quant_config: CompressedTensorsConfig):
        self.validate_kv_cache_scheme(quant_config.kv_cache_scheme)
        super().__init__(quant_config)

    @staticmethod
    def validate_kv_cache_scheme(kv_cache_scheme: Optional[dict[str, Any]]):
        """
        Validator for the kv cache scheme. Useful for controlling the
        kv cache quantization schemes, that are being supported in vLLM
        :param kv_cache_scheme: the compressed-tensors kv cache scheme
        """
        if kv_cache_scheme is None:
            return

        type_ = kv_cache_scheme.get("type")
        num_bits = kv_cache_scheme.get("num_bits")

        if type_ != "float" and num_bits != 8:
            raise NotImplementedError(
                "Currently supported kv cache quantization is "
                "num_bits=8, type=float, however "
                f"received num_bits={num_bits}, type={type_}")

        strategy = kv_cache_scheme.get("strategy")
        if strategy != "tensor":
            raise NotImplementedError(
                "Only support per-tensor scaling factor "
                "for compressed-tensors KV cache. "
                f"Expected strategy: tensor, found strategy: {strategy}")

        is_symmetric = kv_cache_scheme.get("symmetric")
        if not is_symmetric:
            raise NotImplementedError(
                "Only support symmetric scaling factor "
                "for compressed-tensors KV cache. "
                f"However found symmetric: {is_symmetric}")
