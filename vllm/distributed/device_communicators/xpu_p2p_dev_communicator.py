# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys

import torch
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

from .xpu_p2p_communicator import XpuP2pCommunicator

logger = init_logger(__name__)


class XpuP2pDevCommunicator(XpuP2pCommunicator):
    """XpuP2pCommunicator variant with device-side synchronization.

    The parent's host-side protocol blocks the CPU for the whole collective
    (q.wait() + /dev/shm spin, latency == CPU time). Here a single kernel
    per collective stages the local input into the peer-visible buffer,
    handshakes through per-workgroup flags in device memory (the analogue
    of the CUDA custom all-reduce Signal/RankSignals) and reduces; the host
    only enqueues and moves on. Measured on 2x Arc B70, 10 KiB bf16:
    6.8us/call pipelined (vs 17.2us host-sync p2p, 56.0us oneCCL), and the
    all_reduce() call itself costs 4.4us of CPU against a 27.3us completion
    - the host is free for the difference, which host-side sync can never
    give (its call blocks for 19.6us of the 42.1us).

    The kernel is OpenCL C compiled at runtime by the driver (SYCL
    kernel_compiler extension), so no SYCL device compiler is needed - the
    same g++-only JIT as the parent. If the runtime cannot compile it, or
    any rank fails the smoke test, this class degrades to the parent's
    host-sync protocol, and beyond that to oneCCL.
    """

    # The flag page appended to the exported staging region: one uint32 per
    # workgroup on its own cache line, 64 workgroups max.
    _EXTRA_SHARED_BYTES = 4096
    _MAX_WGS = 64
    _WG_SIZE = 256

    # The kernel's vectorized PCIe reads reach ~12 GB/s, below the ~28 GB/s
    # of the copy engine the host-sync parent uses, so the parent overtakes
    # this path for large messages (measured bf16: 100 KiB 12.9us dev vs
    # 16.6us host-sync, 256 KiB 25.0us vs 20.7us, 1 MiB 85.5us vs 54.1us).
    # Above this cap all_reduce falls through to the parent.
    _DEV_MAX_BYTES = 128 * 1024

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        use_all2all: bool = False,
    ):
        super().__init__(
            cpu_group, device, device_group, unique_name, use_all2all=use_all2all
        )
        self._dev_ready = False
        try:
            self._open_dev()
        except Exception:
            logger.warning(
                "Failed to set up device-sync all-reduce for %s, using %s",
                unique_name,
                "host-sync p2p" if self._p2p_ready else "oneCCL",
                exc_info=True,
            )
            self._dev_ready = False
        if self._dev_ready:
            logger.info("Using device-synchronized p2p all-reduce for %s", unique_name)

    def _open_dev(self) -> None:
        # Device-side signaling still needs everything the parent set up
        # (exported staging buffer, mapped peer pointer).
        if not self._all_ranks_ok(self._p2p_ready):
            return

        ok = False
        try:
            self._dev_ext = self._load_dev_ext()
            self._dev_ext.init(torch.xpu.current_stream().sycl_queue)
            # The flag page lives at the tail of the exported region, so
            # the peer's flag page is visible at the same offset of the
            # mapped peer buffer. Flags are monotonic sequence numbers and
            # must start below any seq the kernels will use.
            flags = self._stage[2 * self.MAX_BYTES :]
            flags.zero_()
            torch.xpu.synchronize()
            self._my_flags = self._stage.data_ptr() + 2 * self.MAX_BYTES
            base = self._peer_stage  # parent: peer's staging base pointer
            self._peer_flags = base + 2 * self.MAX_BYTES
            ok = True
        except Exception:
            logger.warning(
                "device-sync kernel unavailable for %s",
                self.unique_name,
                exc_info=True,
            )
        # Both ranks must have zeroed their flags before either launches a
        # kernel that signals into them.
        if not self._all_ranks_ok(ok):
            return

        # Smoke test through the real kernel before trusting the path.
        ok = False
        try:
            t = torch.full((64,), float(self.rank_in_group + 1), device=self.device)
            out = self._dev_all_reduce(t)
            torch.xpu.synchronize()
            ok = bool((out == 3.0).all())
        except Exception:
            logger.warning(
                "device-sync smoke test failed for %s",
                self.unique_name,
                exc_info=True,
            )
        if self._all_ranks_ok(ok):
            self._dev_ready = True

    @staticmethod
    def _load_dev_ext():
        from torch.utils.cpp_extension import load

        prefix = sys.prefix
        return load(
            name="vllm_xpu_zeipc_dev",
            sources=[os.path.join(os.path.dirname(__file__), "xpu_zeipc_dev.cpp")],
            extra_cflags=[
                "-std=c++17",
                "-Wno-deprecated-declarations",
                "-DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING",
            ],
            extra_include_paths=[os.path.join(prefix, "include")],
            extra_ldflags=[
                f"-L{os.path.join(prefix, 'lib')}",
                f"-Wl,-rpath,{os.path.join(prefix, 'lib')}",
                "-lsycl",
            ],
            with_cuda=False,
        )

    _DTYPE_CODE = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}

    def _launch_cfg(self, numel: int) -> tuple[int, int]:
        # Both ranks must compute the identical grid (numel matches by
        # collective contract); chunk is 8-aligned for the vector loop.
        nwg = max(1, min(self._MAX_WGS, (numel + 2047) // 2048))
        chunk = (((numel + nwg - 1) // nwg) + 7) // 8 * 8
        return (numel + chunk - 1) // chunk, chunk

    def _dev_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if not input_.is_contiguous():
            input_ = input_.contiguous()
        # Same parity double-buffering as the parent; the slot-reuse safety
        # argument holds unchanged because a rank's kernel s+1 signals only
        # after its kernel s (including every read of the peer slot) has
        # completed on its in-order queue.
        seq = self._seq + 1
        parity = seq & 1
        off = parity * self.MAX_BYTES
        numel = input_.numel()
        nwg, chunk = self._launch_cfg(numel)
        output = torch.empty_like(input_)
        self._dev_ext.launch(
            self._DTYPE_CODE[input_.dtype],
            output.data_ptr(),
            input_.data_ptr(),
            self._stage.data_ptr() + off,
            self._peer_stage + off,
            self._my_flags,
            self._peer_flags,
            seq,
            numel,
            chunk,
            nwg,
            self._WG_SIZE,
        )
        self._seq = seq
        return output

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if (
            self._dev_ready
            and input_.dtype in self._SUPPORTED_DTYPES
            and input_.nbytes <= self._DEV_MAX_BYTES
        ):
            return self._dev_all_reduce(input_)
        return super().all_reduce(input_)
