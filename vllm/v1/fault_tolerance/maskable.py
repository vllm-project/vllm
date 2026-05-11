# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FTMaskBuffer — capability mixin for FT-capable collective backends.

Some collective backends (DeepEP-LL, NIXL-EP, FT NCCL) expose a per-peer
mask buffer that the kernel auto-sets when a peer is detected to have
faulted. ``query_mask`` reads which peers are flagged; ``clean_mask``
resets the buffer.

This is a CAPABILITY mixin, not a base class. It lives separately from
``BaseDeviceCommunicator`` / ``All2AllManagerBase`` so that non-FT-capable
backends don't carry methods they can never implement, and so callers can
ask "does this backend support FT masking?" via ``isinstance(x, FTMaskBuffer)``.

See vLLM FT design doc §7.2 for the lifecycle (when ``query_mask`` is
called and when ``clean_mask`` is called).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class FTMaskBuffer(Protocol):
    """Capability contract for backends that expose an FT mask buffer.

    Implemented today by ``DeepEPLLAll2AllManager`` and
    ``NixlEPAll2AllManager``. FT NCCL collectives (cf.
    ``vllm/distributed/eplb/ft_nccl_wrapper.py``) are the natural future
    implementer.

    Any caller that needs the mask should gate on
    ``isinstance(x, FTMaskBuffer)`` rather than calling the methods
    unconditionally and catching ``NotImplementedError``.
    """

    def query_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Read which peers are currently flagged as faulty into ``mask``.

        ``mask`` is a 1-D ``int32`` tensor with length equal to the EP
        group's world size. The kernel writes 1 for any peer it has
        detected as faulty, 0 otherwise.
        """
        ...

    def clean_mask(self) -> None:
        """Reset the buffer to zeros. Used during retry/recovery once the
        EP group is otherwise quiesced."""
        ...
