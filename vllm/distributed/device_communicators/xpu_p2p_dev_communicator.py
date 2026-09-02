# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys

import torch
from torch.distributed import ProcessGroup

from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

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

    _MAX_WGS = 64
    _WG_SIZE = 256

    # Eager crossover. The kernel's vectorized PCIe reads reach ~12 GB/s,
    # below the ~28 GB/s of the copy engine the host-sync parent uses, so
    # the parent overtakes this path for large messages (measured bf16:
    # 100 KiB 12.9us dev vs 16.6us host-sync, 256 KiB 25.0us vs 20.7us,
    # 1 MiB 85.5us vs 54.1us). Above this, eager calls fall through to the
    # parent.
    _DEV_EAGER_BYTES = 128 * 1024

    # Capacity. Under XPU graph capture the kernel is the only recordable
    # path (see all_reduce), so it must hold the largest all-reduce a
    # captured batch issues: max_cudagraph_capture_size tokens x hidden x
    # 2 bytes, 8 MiB at 512 x 8192. The communicator cannot derive that
    # bound itself (which tensors a model all-reduces, and how capture
    # sizes scale with speculative tokens, is the model's business), so
    # it is fixed and all_reduce raises if a capture exceeds it. The
    # kernel's PCIe read rate is flat at ~12 GB/s of payload from 512 KiB
    # up to this size (measured bf16 under replay: 1.31 MB 116us, 4 MiB
    # 340us, 8 MiB 696us; oneCCL 350us, 998us, 2053us).
    _DEV_SLOT_BYTES = 8 << 20

    # All-gather capacity. The input is this rank's shard, half the size
    # of the all-reduce input at the same token count: T x hidden bytes for
    # the hidden-state gathers (4 MiB at 512 x 8192), num_reqs x vocab
    # bytes for the drafter's captured logits gather (248 KB at one
    # sequence and a 248k vocab).
    # The kernel beats oneCCL at every size up to the slot (measured bf16,
    # peer bytes: 16 KiB 10.6us vs 40.7us, 1 MiB 87.6us vs 315.9us, 4 MiB
    # 338us vs 1225us), so eager dispatch has no crossover: kernel up to
    # the slot, oneCCL above. A copy-engine path would win from 128 KiB up
    # (14.6us vs 16.2us; 52us vs 88us at 1 MiB) but there is none for
    # all-gather and those sizes only occur in prefill.
    _AG_SLOT_BYTES = 4 << 20

    # Appended to the parent's exported staging region, in this order: a
    # pair of staging slots for the all-reduce kernel, a pair for the
    # all-gather kernel, then per kernel one page of per-workgroup
    # handshake flags (uint32 each on its own cache line, written by the
    # peer) and one page of per-workgroup sequence counters (same indexing,
    # written only by this rank's kernels). The two kernels share nothing:
    # each has its own slots and its own counter, so their interleaving
    # needs no reasoning.
    _PAGE_BYTES = 4096
    _EXTRA_SHARED_BYTES = 2 * _DEV_SLOT_BYTES + 2 * _AG_SLOT_BYTES + 4 * _PAGE_BYTES

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
        self._ag_ready = False
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
            self._dev_ext.init(current_stream().sycl_queue)
            # Same offsets on both ranks, so the peer's slots and flags are
            # visible at the same offsets of the mapped peer buffer. Flags
            # and counters are monotonic sequence numbers and must start
            # below any seq the kernels will use.
            slots_off = 2 * self.MAX_BYTES
            ag_slots_off = slots_off + 2 * self._DEV_SLOT_BYTES
            flags_off = ag_slots_off + 2 * self._AG_SLOT_BYTES
            counters_off = flags_off + self._PAGE_BYTES
            ag_flags_off = counters_off + self._PAGE_BYTES
            ag_counters_off = ag_flags_off + self._PAGE_BYTES
            self._stage[flags_off:].zero_()
            torch.xpu.synchronize()
            mine, peer = self._stage.data_ptr(), self._peer_stage
            self._dev_slots = mine + slots_off
            self._peer_dev_slots = peer + slots_off
            self._my_flags = mine + flags_off
            self._peer_flags = peer + flags_off
            self._counters = mine + counters_off
            self._ag_slots = mine + ag_slots_off
            self._peer_ag_slots = peer + ag_slots_off
            self._ag_my_flags = mine + ag_flags_off
            self._ag_peer_flags = peer + ag_flags_off
            self._ag_counters = mine + ag_counters_off
            ok = True
        except Exception:
            logger.warning(
                "device-sync kernel unavailable for %s",
                self.unique_name,
                exc_info=True,
            )
        # Both ranks must have zeroed their flags before either launches a
        # kernel that signals into them.  Counters are rank-local, but the
        # kernel derives every seq from them, so zeroing them at the same
        # point keeps both ranks' seqs identical from the first launch.
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
        if not self._all_ranks_ok(ok):
            return
        self._dev_ready = True

        # The all-gather kernel is separately optional: if only it fails,
        # the all-reduce path stays and all_gather degrades to oneCCL.
        ok = False
        try:
            t = torch.full((64,), float(self.rank_in_group + 1), device=self.device)
            out = self._dev_all_gather(t, 0)
            torch.xpu.synchronize()
            ok = bool((out[:64] == 1.0).all()) and bool((out[64:] == 2.0).all())
        except Exception:
            logger.warning(
                "device-sync all-gather smoke test failed for %s",
                self.unique_name,
                exc_info=True,
            )
        if self._all_ranks_ok(ok):
            self._ag_ready = True

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

    def _launch_cfg(self, numel: int, vec: int = 8) -> tuple[int, int]:
        # Both ranks must compute the identical grid (numel matches by
        # collective contract); chunk is aligned to the vector width.
        nwg = max(1, min(self._MAX_WGS, (numel + 2047) // 2048))
        chunk = (((numel + nwg - 1) // nwg) + vec - 1) // vec * vec
        return (numel + chunk - 1) // chunk, chunk

    def _dev_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if not input_.is_contiguous():
            input_ = input_.contiguous()
        numel = input_.numel()
        nwg, chunk = self._launch_cfg(numel)
        output = torch.empty_like(input_)
        # Every argument is a function of the tensor alone; the sequence
        # number and slot parity come from a device-side counter, so a
        # launch recorded into an XPU graph stays correct on replay.
        # vLLM's cached current_stream() is the queue its graph capture
        # records on (cuda_graph.py passes it explicitly), and it costs a
        # TLS read where torch.xpu.current_stream() builds a Stream object
        # per call (1.5us, a fifth of this collective).
        self._dev_ext.launch(
            current_stream().sycl_queue,
            self._DTYPE_CODE[input_.dtype],
            output.data_ptr(),
            input_.data_ptr(),
            self._dev_slots,
            self._peer_dev_slots,
            self._my_flags,
            self._peer_flags,
            self._counters,
            numel,
            chunk,
            self._DEV_SLOT_BYTES // input_.element_size(),
            nwg,
            self._WG_SIZE,
        )
        return output

    def _dev_all_gather(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        if not input_.is_contiguous():
            input_ = input_.contiguous()
        input_size = input_.size()
        output = torch.empty(
            (2 * input_size[0],) + input_size[1:],
            dtype=input_.dtype,
            device=input_.device,
        )
        nbytes = input_.nbytes
        if nbytes:
            nwg, chunk = self._launch_cfg(nbytes, vec=16)
            self._dev_ext.launch_ag(
                current_stream().sycl_queue,
                output.data_ptr(),
                input_.data_ptr(),
                self._ag_slots,
                self._peer_ag_slots,
                self._ag_my_flags,
                self._ag_peer_flags,
                self._ag_counters,
                nbytes,
                chunk,
                self._AG_SLOT_BYTES,
                self.rank_in_group * nbytes,
                (1 - self.rank_in_group) * nbytes,
                nwg,
                self._WG_SIZE,
            )
        # Same concat-then-move layout as the base class.
        output = output.reshape((2,) + input_size).movedim(0, dim)
        return output.reshape(
            input_size[:dim] + (2 * input_size[dim],) + input_size[dim + 1 :]
        )

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += input_.dim()
        if self._ag_ready and input_.nbytes <= self._AG_SLOT_BYTES:
            return self._dev_all_gather(input_, dim)
        if not torch.xpu.is_current_stream_capturing():
            return super().all_gather(input_, dim)
        # oneCCL rejects graph recording; see all_reduce.
        if not self._ag_ready:
            reason = "device-sync all-gather unavailable"
        else:
            reason = (
                f"{input_.nbytes} bytes exceeds the {self._AG_SLOT_BYTES} "
                "byte staging slot; lower max_cudagraph_capture_size (or "
                "max_num_seqs), or set VLLM_XPU_ENABLE_XPU_GRAPH=0"
            )
        raise RuntimeError(
            f"XPU custom all-gather ({self.unique_name}) cannot be "
            f"graph-captured: {reason}"
        )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        dev_ok = self._dev_ready and input_.dtype in self._SUPPORTED_DTYPES
        # Eager: dispatch by speed.
        if dev_ok and input_.nbytes <= self._DEV_EAGER_BYTES:
            return self._dev_all_reduce(input_)
        if not torch.xpu.is_current_stream_capturing():
            return super().all_reduce(input_)
        # Capturing: dispatch by capability. Neither fallback can be
        # recorded (the parent's protocol blocks the host on a spin loop,
        # oneCCL rejects graph recording); both would run eagerly during
        # capture and be missing from the replay, which is silent wrong
        # output. A slower kernel beats that; a raise beats it too.
        if dev_ok and input_.nbytes <= self._DEV_SLOT_BYTES:
            return self._dev_all_reduce(input_)
        if not self._dev_ready:
            reason = "device-sync path unavailable"
        elif input_.dtype not in self._SUPPORTED_DTYPES:
            reason = f"dtype {input_.dtype} unsupported"
        else:
            reason = (
                f"{input_.nbytes} bytes exceeds the {self._DEV_SLOT_BYTES} "
                "byte staging slot; lower max_cudagraph_capture_size (or "
                "max_num_seqs), or set VLLM_XPU_ENABLE_XPU_GRAPH=0"
            )
        raise RuntimeError(
            f"XPU custom all-reduce ({self.unique_name}) cannot be "
            f"graph-captured: {reason}"
        )
