# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization config for the GLM-5.3-Flash attention projections, which load
as BF16 in every checkpoint."""

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)


def bf16_projection_quant_config(
    quant_config: QuantizationConfig | None,
) -> OnlineQuantizationConfig | None:
    """Quant config for the KDA / MLA attention projections.

    The KDA and indexer projections are BF16 in every checkpoint. The MLA
    projections are BF16 in the NVFP4 checkpoint and block-FP8 in the FP8
    checkpoint, which is dequantized to BF16 at load. The checkpoint's own
    method therefore never applies to them; only online quantization that the
    user requested explicitly with ``--quantization-config`` does:

    * composed with a quantized checkpoint (any form): that online config;
    * pure online quantization (BF16 checkpoint): only when it names
      ``targets``. The ``--quantization mxfp8`` / ``fp8_per_*`` shorthands keep
      these projections BF16.

    Otherwise None. Layers matching no online target stay unquantized.
    """
    if isinstance(quant_config, OnlineQuantizationConfig):
        return quant_config if quant_config.args.targets is not None else None
    return getattr(quant_config, "online_quantization_config", None)
