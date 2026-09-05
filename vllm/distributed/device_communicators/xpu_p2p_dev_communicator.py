# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

from .xpu_p2p_communicator import XpuP2pCommunicator, _ops_available, xpu_p2p

logger = init_logger(__name__)

_COLLECTIVE_OPS = (
    "xpu_p2p_signal_page_bytes",
    "xpu_p2p_all_reduce",
    "xpu_p2p_all_gather",
)


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

    The kernels ship in vllm_xpu_kernels as the xpu_p2p_all_reduce and
    xpu_p2p_all_gather ops. If a build does not carry them, or any rank
    fails the smoke test, this class degrades to the parent's host-sync
    protocol, and beyond that to oneCCL.
    """

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

    def _extra_shared_bytes(self) -> int:
        # Appended to the parent's exported staging region, in this order: a
        # pair of staging slots for the all-reduce kernel, a pair for the
        # all-gather kernel, then per kernel one page of per-workgroup
        # handshake flags (uint32 each on its own cache line, written by the
        # peer) and one page of per-workgroup sequence counters (same
        # indexing, written only by this rank's kernels). The two kernels
        # share nothing: each has its own slots and its own counter, so their
        # interleaving needs no reasoning.
        #
        # The kernels report the page size rather than this side fixing it:
        # it follows their maximum workgroup count, and a page sized smaller
        # than they expect would be overrun. A build without them reserves
        # nothing and leaves _open_dev to degrade to the host-sync path.
        if not _ops_available(*_COLLECTIVE_OPS):
            return 0
        return (
            2 * self._DEV_SLOT_BYTES
            + 2 * self._AG_SLOT_BYTES
            + 4 * xpu_p2p.signal_page_bytes()
        )

    def _open_dev(self) -> None:
        have_ops = _ops_available(*_COLLECTIVE_OPS)
        if not have_ops:
            logger.debug(
                "device-sync all-reduce disabled for %s: this "
                "vllm_xpu_kernels build does not provide the p2p collectives",
                self.unique_name,
            )
        # Device-side signaling still needs everything the parent set up
        # (exported staging buffer, mapped peer pointer). Both ranks must
        # agree before either launches: the kernel handshake spins with no
        # timeout, so a rank that skipped it would leave the other spinning.
        if not self._all_ranks_ok(self._p2p_ready and have_ops):
            return

        ok = False
        try:
            page_bytes = xpu_p2p.signal_page_bytes()
            # Same offsets on both ranks, so the peer's slots and flags are
            # visible at the same offsets of the mapped peer buffer. Flags
            # and counters are monotonic sequence numbers and must start
            # below any seq the kernels will use.
            slots_off = 2 * self.MAX_BYTES
            ag_slots_off = slots_off + 2 * self._DEV_SLOT_BYTES
            flags_off = ag_slots_off + 2 * self._AG_SLOT_BYTES
            counters_off = flags_off + page_bytes
            ag_flags_off = counters_off + page_bytes
            ag_counters_off = ag_flags_off + page_bytes
            self._stage[flags_off:].zero_()
            torch.xpu.synchronize()
            # XPU device addresses sit above 2**63, which torch's `int`
            # schema type cannot carry; as_fptr is the two's-complement
            # reinterpretation the ops undo. Done once here rather than per
            # call, which would put a Python frame in a 6.8us collective.
            mine, peer = self._stage.data_ptr(), self._peer_stage
            fptr = xpu_p2p.as_fptr
            self._dev_slots = fptr(mine + slots_off)
            self._peer_dev_slots = fptr(peer + slots_off)
            self._my_flags = fptr(mine + flags_off)
            self._peer_flags = fptr(peer + flags_off)
            self._counters = fptr(mine + counters_off)
            self._ag_slots = fptr(mine + ag_slots_off)
            self._peer_ag_slots = fptr(peer + ag_slots_off)
            self._ag_my_flags = fptr(mine + ag_flags_off)
            self._ag_peer_flags = fptr(peer + ag_flags_off)
            self._ag_counters = fptr(mine + ag_counters_off)
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

    def _dev_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if not input_.is_contiguous():
            input_ = input_.contiguous()
        output = torch.empty_like(input_)
        # Every argument is a function of the tensor alone; the sequence
        # number and slot parity come from a device-side counter, so a
        # launch recorded into an XPU graph stays correct on replay. The op
        # submits to the queue behind torch's current stream, which under
        # capture is the recording one.
        torch.ops._xpu_C.xpu_p2p_all_reduce(
            output,
            input_,
            self._dev_slots,
            self._peer_dev_slots,
            self._my_flags,
            self._peer_flags,
            self._counters,
            self._DEV_SLOT_BYTES,
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
        # An empty input is a no-op inside the op, on both ranks alike.
        torch.ops._xpu_C.xpu_p2p_all_gather(
            output,
            input_,
            self._ag_slots,
            self._peer_ag_slots,
            self._ag_my_flags,
            self._ag_peer_flags,
            self._ag_counters,
            self._AG_SLOT_BYTES,
            self.rank_in_group,
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
