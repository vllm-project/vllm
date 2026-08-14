# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM-owned wrapper over AITER's ``CustomAllreduce``.

vLLM's ``CudaCommunicator`` stores one of these as ``aiter_ar_comm`` (when
``VLLM_ROCM_USE_AITER_CUSTOM_AR`` is set) so the plain allreduce and
the fused allreduce+RMSNorm path share a single AITER instance with its IPC buffers.

"""

import torch
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)


class AiterCustomAllreduce:
    # Default IPC buffer size for AITER's CustomAllreduce.
    MAX_SIZE: int = 8192 * 1024 * 8 * 2

    @classmethod
    def effective_max_size(cls) -> int:
        """
        Max input byte size eligible for AITER custom allreduce.
        """
        return cls.MAX_SIZE // 2

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int | None = None,
    ):
        from aiter.dist.device_communicators.custom_all_reduce import (
            CustomAllreduce as _AiterCustomAllreduce,
        )

        if max_size is None:
            max_size = self.MAX_SIZE

        self._impl = _AiterCustomAllreduce(group, device, max_size=max_size)

    @property
    def aiter_ca(self):
        return self._impl

    @property
    def disabled(self) -> bool:
        return self._impl.disabled

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        return self._impl.should_custom_ar(inp)

    def custom_all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        return self._impl.custom_all_reduce(inp)

    def capture(self):
        return self._impl.capture()

    def close(self) -> None:
        self._impl.close()

    @property
    def supports_dynamic_hidden_dim(self) -> bool:
        """Aiter's fused_allreduce_rmsnorm kernel dispatches on hidden_dim.
        Before aiter v0.1.12 the launcher was template-specialized on HIDDEN_DIM
        and silently no-op'd for sizes outside {512, 1024, 2048, 4096}. From v0.1.12
        hidden_dim is a runtime argument. Older builds are detected via
        AiterCustomAllreduce.supports_dynamic_hidden_dim; This function is used to
        skip fusion for unsupported sizes on them.
        Ref (old kernel): https://github.com/ROCm/aiter/blob/6a0e7b26ccf33164785531212cc2ec2cde0b9243/csrc/include/custom_all_reduce.cuh#L2590
        """
        return hasattr(self._impl, "_pool")

    @staticmethod
    def build_supports_per_group_quant() -> bool:
        """True if the running AITER build exposes the per-group AR+RMS+quant
        kernel (added in ROCm/aiter PR #2823).

        The pattern registration in ``RocmAiterAllReduceFusionPass`` keys off
        this so vLLM degrades to the AR+RMS-only fusion when run against an
        older aiter that lacks the per-group launcher.
        """
        from aiter.dist.device_communicators.custom_all_reduce import (
            CustomAllreduce as _AiterCustomAllreduce,
        )

        return hasattr(_AiterCustomAllreduce, "fused_ar_rms_per_group_quant")

    # TODO(frida-andersson): drop once vLLM pins AITER >= 0.1.14 (ROCm/aiter#2823).
    @property
    def supports_per_group_quant(self) -> bool:
        return self.build_supports_per_group_quant()
