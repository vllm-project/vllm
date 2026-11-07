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
        """Max input byte size eligible for AITER custom allreduce."""
        return cls.MAX_SIZE // 2

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int | None = None,
        *,
        register_graph_buffers: bool = True,
    ):
        from aiter.dist.device_communicators.custom_all_reduce import (
            CustomAllreduce as _AiterCustomAllreduce,
        )

        if max_size is None:
            max_size = self.MAX_SIZE

        kwargs = {}
        if not register_graph_buffers:
            kwargs["enable_register_for_capturing"] = False
        self._impl = _AiterCustomAllreduce(group, device, max_size=max_size, **kwargs)
        self._group = group
        # Inboxes and call counters of AITER's Gluon all-reduce, which does the
        # plain all-reduce on gfx1250 with 4 ranks.
        self._gluon: tuple[tuple[int, ...], torch.Tensor] | None = None
        if not self._impl.disabled:
            self._gluon = self._init_gluon()

    def _init_gluon(self) -> tuple[tuple[int, ...], torch.Tensor] | None:
        """Allocate the Gluon all-reduce state, or return None where it cannot
        run. All ranks of the group call it."""
        from vllm.distributed.parallel_state import in_the_same_node_as
        from vllm.platforms.rocm import on_gfx1250

        if not on_gfx1250() or self._impl.world_size != 4:
            return None
        try:
            from aiter.dist.device_communicators.custom_all_reduce import (
                _should_use_vmm,
            )
            from aiter.ops.triton.comms.all_reduce import all_reduce_init
        except ImportError:
            logger.info_once(
                "AITER Gluon all-reduce is disabled: it needs AITER v0.1.28 or later.",
                scope="global",
            )
            return None
        # The inboxes are shared with hipIpc, which AITER uses on gfx1250 from
        # ROCm 7.14.
        if _should_use_vmm(True) or not all(
            in_the_same_node_as(self._group, source_rank=0)
        ):
            logger.info_once(
                "AITER Gluon all-reduce is disabled: it needs hipIpc "
                "(ROCm 7.14 or later) and all ranks on one node.",
                scope="global",
            )
            return None
        # The inbox of each rank is 16 times the largest input.
        gluon = all_reduce_init(self._group, self.effective_max_size())
        logger.info_once(
            "AITER custom all-reduce uses the Gluon kernel for bf16 inputs.",
            scope="global",
        )
        return gluon

    def _should_gluon(self, inp: torch.Tensor) -> bool:
        return (
            self._gluon is not None
            and inp.dtype == torch.bfloat16
            and inp.is_contiguous()
            and inp.numel() % 4 == 0
            and inp.data_ptr() % 16 == 0
            and inp.nbytes <= self.effective_max_size()
        )

    @property
    def aiter_ca(self):
        return self._impl

    @property
    def disabled(self) -> bool:
        return self._impl.disabled

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        return self._impl.should_custom_ar(inp)

    def custom_all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        if self._should_gluon(inp):
            from aiter.ops.triton.comms.all_reduce import all_reduce

            assert self._gluon is not None
            return all_reduce(inp, self._group, *self._gluon)
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
        if self._gluon is not None:
            from aiter.ops.triton.comms.all_reduce import all_reduce_free

            all_reduce_free(self._group, self._gluon[0])
            self._gluon = None
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
