# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load-time decoding of CSF-compressed NVFP4 block scales.

CSF checkpoints (``weight_scale_encoding: "csf"``, codec
``byte-window4-fixed-stream-u24-exceptions/1``) store each E4M3 block-scale
matrix as two tensors:

- ``<name>.nvfp4_csf_fixed`` (uint8, ``[rows // 16, 16 * (1 + cols // 2)]``):
  per 16-row slab, one base byte per row followed by the rows' 4-bit codes,
  two per byte, low nibble first. A scale byte is ``base + code``.
- ``<name>.nvfp4_csf_exceptions`` (uint32): sorted words whose low 24 bits are
  a flat ``row * cols + col`` position and whose high 8 bits replace the
  scale byte there.

The decoded bytes equal the uncompressed checkpoint's E4M3 scales exactly, so
every NVFP4 kernel consumes them unchanged.
"""

from collections.abc import Iterable, Iterator

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

CSF_FIXED_SUFFIX = ".nvfp4_csf_fixed"
CSF_EXCEPTIONS_SUFFIX = ".nvfp4_csf_exceptions"


def decode_nvfp4_csf_scale(
    fixed: torch.Tensor, exceptions: torch.Tensor
) -> torch.Tensor:
    """Rebuild one ``[rows, cols]`` float8_e4m3fn block-scale matrix."""
    if fixed.dtype != torch.uint8 or fixed.dim() != 2 or fixed.shape[1] % 16:
        raise ValueError(f"Invalid CSF fixed stream {fixed.dtype} {fixed.shape}")
    if exceptions.dtype != torch.uint32:
        raise ValueError(f"Invalid CSF exceptions dtype {exceptions.dtype}")
    rows = fixed.shape[0] * 16
    cols = (fixed.shape[1] // 16 - 1) * 2
    fixed = fixed.cpu()
    bases = fixed[:, :16].reshape(rows, 1).to(torch.int16)
    if bool((bases > 240).any()):
        raise ValueError("CSF row bases must be in 0..240")
    packed = fixed[:, 16:].reshape(rows, cols // 2)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(rows, cols)
    out = (bases + codes.to(torch.int16)).reshape(-1)
    words = exceptions.cpu().reshape(-1).view(torch.int32).to(torch.int64)
    words &= 0xFFFFFFFF
    positions = words & 0xFFFFFF
    if positions.numel() and int(positions.max()) >= rows * cols:
        raise ValueError("CSF exception position out of range")
    out[positions] = (words >> 24).to(torch.int16)
    return out.to(torch.uint8).view(torch.float8_e4m3fn).reshape(rows, cols)


def decode_csf_scale_streams(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Replace CSF stream pairs with decoded ``<name>`` E4M3 scale tensors."""
    pending: dict[str, dict[str, torch.Tensor]] = {}
    decoded = 0
    for name, tensor in weights:
        if name.endswith(CSF_FIXED_SUFFIX):
            base, part = name[: -len(CSF_FIXED_SUFFIX)], "fixed"
        elif name.endswith(CSF_EXCEPTIONS_SUFFIX):
            base, part = name[: -len(CSF_EXCEPTIONS_SUFFIX)], "exceptions"
        else:
            yield name, tensor
            continue
        parts = pending.setdefault(base, {})
        parts[part] = tensor
        if len(parts) == 2:
            del pending[base]
            yield base, decode_nvfp4_csf_scale(parts["fixed"], parts["exceptions"])
            decoded += 1
    if pending:
        raise ValueError(f"CSF scale streams without a partner: {sorted(pending)[:4]}")
    if decoded:
        logger.info("Decoded %d CSF expert scale tensors at load", decoded)
