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
    def effective_max_size(cls) -> int | None:
        """Max input byte size eligible for AITER custom allreduce.

        Returns ``None`` when custom AR is disabled entirely
        (``AITER_CUSTOM_AR_MAX_SIZE=0``), so callers skip the fusion instead of
        building a zero-length compile range.

        This has to agree with what AITER accepts at runtime, which is
        ``min(_car_max_size, max_size / 2)``. ``_car_max_size`` comes from
        ``AITER_CUSTOM_AR_MAX_SIZE`` and defaults to 64 MiB independently of
        the registered pool, so the two can disagree.

        The compile range of the fused allreduce+RMSNorm pass is derived from
        this value. If it exceeds what ``should_custom_ar()`` will accept, the
        pass gets compiled for message sizes AITER rejects,
        ``custom_fused_ar_rms()`` returns ``None``, and
        ``_rocm_aiter_fused_allreduce_rmsnorm_impl`` asserts on it -- the
        server fails to start instead of falling back to RCCL.
        """
        two_shot_limit = cls.MAX_SIZE // 2
        try:
            # Private, but it is the only place the env var semantics live;
            # duplicating them here would be the more fragile option.
            from aiter.dist.device_communicators.custom_all_reduce import (
                _resolve_car_max_size,
            )
        except ImportError:
            return two_shot_limit
        car_max_size = _resolve_car_max_size(cls.MAX_SIZE)
        if car_max_size <= 0:
            # AITER_CUSTOM_AR_MAX_SIZE=0 turns custom AR off for every size.
            return None
        return min(two_shot_limit, car_max_size)

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

    def should_custom_ag(self, inp: torch.Tensor) -> bool:
        return self._impl.should_custom_ag(inp)

    def should_custom_rs(self, inp: torch.Tensor, dim: int) -> bool:
        return self._impl.should_custom_rs(inp, dim)

    def custom_all_gather(self, inp: torch.Tensor, dim: int = 0) -> torch.Tensor | None:
        return self._impl.custom_all_gather(inp, dim=dim)

    def custom_reduce_scatter(
        self, inp: torch.Tensor, out: torch.Tensor, dim: int = 0
    ) -> torch.Tensor | None:
        return self._impl.custom_reduce_scatter(inp, out, dim=dim)

    def use_1stage_fused_ar_rms(self, inp: torch.Tensor) -> bool:
        """Whether AITER's fused allreduce+RMSNorm runs as its one-stage kernel.

        Mirrors the launcher contract of aiter's ``fused_allreduce_rmsnorm``
        (csrc/include/custom_all_reduce.cuh): rows of 16-byte packs, at most
        1024 packs per row, at most 80 tokens, and the byte cap of the
        one-stage custom allreduce for this TP size and topology. Outside it
        the fused op runs the two-stage variant (cross-device reduce-scatter
        + local norm), which is slower than an explicit ``all_reduce`` + norm,
        so callers that can fall back should require this. Capture-static:
        depends only on shape, dtype, TP size and topology.
        """
        hidden_dim = inp.shape[-1]
        # Token cap first: prefill-sized inputs leave here with one comparison.
        if inp.numel() // hidden_dim > 80:
            return False
        if inp.dtype not in (torch.bfloat16, torch.float16):
            return False
        pack_size = 16 // inp.element_size()
        if hidden_dim % pack_size != 0 or hidden_dim // pack_size > 1024:
            return False
        ca = self._impl
        world_size = ca.world_size
        if world_size == 2:
            return True
        if not ca.fully_connected:
            return False
        total_bytes = inp.numel() * inp.element_size()
        if world_size <= 4:
            return total_bytes < 256 * 1024
        if world_size <= 8:
            return total_bytes < 128 * 1024
        return False

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
