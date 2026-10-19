# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache fake quantization (quantize-dequantize) for accuracy studies.

When ``VLLM_KV_CACHE_FAKE_QUANT`` is set, K and V of every attention layer are
QDQ'd right after RoPE and before they are written to a plain BF16/FP16 cache
(``--kv-cache-dtype auto``), so the cache and the current attention call both
see the quantized values.

- ``nvfp4``: vLLM's NVFP4 emulation (``ref_nvfp4_quant_dequant``): E2M1 values,
  one E4M3 scale per 16 values along head_dim, global scale 1.0.
- ``mxfp4``: Quark ``qdq_mxfp4`` (via ``mxfp4_utils``): block 32, E8M0 "even" scale.
"""

from functools import cache

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

KV_FAKE_QUANT_FORMATS = ("mxfp4", "nvfp4")
_NVFP4_BLOCK_SIZE = 16


@cache
def get_kv_fake_quant_format() -> str | None:
    """The KV fake-quant format for all attention layers, or None when off."""
    fmt = envs.VLLM_KV_CACHE_FAKE_QUANT
    if not fmt:
        return None
    if fmt not in KV_FAKE_QUANT_FORMATS:
        raise ValueError(
            f"Unsupported VLLM_KV_CACHE_FAKE_QUANT={fmt!r}, "
            f"expected one of {KV_FAKE_QUANT_FORMATS}"
        )
    logger.info("KV cache fake quant enabled: %s", fmt)
    return fmt


def _kv_fake_quant(x: torch.Tensor, fmt: str) -> torch.Tensor:
    if fmt == "nvfp4":
        from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
            ref_nvfp4_quant_dequant,
        )

        global_scale = torch.ones(1, dtype=torch.float32, device=x.device)
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        return ref_nvfp4_quant_dequant(x2d, global_scale, _NVFP4_BLOCK_SIZE).view(
            x.shape
        )
    if fmt == "mxfp4":
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
            _quant_dequant_mxfp4,
        )

        return _quant_dequant_mxfp4(x.contiguous(), "even")
    raise ValueError(f"Unsupported KV fake-quant format {fmt!r}")


def _kv_fake_quant_fake(x: torch.Tensor, fmt: str) -> torch.Tensor:
    return torch.empty_like(x)


# A custom op keeps the Triton / Quark kernels opaque to torch.compile.
direct_register_custom_op(
    op_name="kv_cache_fake_quant",
    op_func=_kv_fake_quant,
    fake_impl=_kv_fake_quant_fake,
)
kv_cache_fake_quant = torch.ops.vllm.kv_cache_fake_quant
