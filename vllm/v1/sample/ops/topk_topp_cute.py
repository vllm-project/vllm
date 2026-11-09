# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k/top-p masking with the CuTe DSL filter kernel (topk_topp_cute_kernel).

A JIT port of the core path of the CUDA filter kernel, with the same contract
as apply_top_k_top_p_cuda: FP32 logits, per-row device k and p, exactly k
tokens with ties broken by the lowest index, SM90 and SM100.
"""

import functools
from dataclasses import dataclass

import torch

from vllm.platforms import current_platform

_STATE: dict[int, "_DeviceState"] = {}


@functools.cache
def topk_topp_cute_available(device_index: int = 0) -> bool:
    """Whether the GPU and the CuTe DSL support the kernel."""
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability(device_index)
    if capability is None or not (
        capability.major == 9 or (capability.major == 10 and capability.minor in (0, 3))
    ):
        return False
    try:
        import cutlass.cute  # noqa: F401
    except ImportError:
        return False
    return True


def apply_top_k_top_p_cute(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Mask logits outside the per-row top-k, then outside the top-p nucleus.

    Args:
        logits: [batch_size, vocab_size] float32 CUDA tensor.
        k: [batch_size] top-k per row, or None. k >= vocab_size disables it.
        p: [batch_size] top-p per row, or None. p >= 1 disables it.

    Returns:
        The masked logits. They alias the input unless its rows were not
        contiguous.

    """
    if k is None and p is None:
        return logits
    if logits.stride(1) != 1:
        logits = logits.contiguous()
    batch, vocab = logits.shape
    if batch == 0:
        return logits
    state = _state(logits.device)
    flags = int(k is not None) | int(p is not None) << 1
    if vocab % 4 == 0 and logits.stride(0) % 4 == 0 and logits.data_ptr() % 16 == 0:
        flags |= 4
    k = state.unused_k if k is None else k.to(torch.int32).contiguous()
    p = state.unused_p if p is None else p.to(torch.float32).contiguous()
    cluster, grid, splits, depth = state.plan.launch(batch, vocab)
    state.kernels[cluster](logits, k, p, state.workspace, flags, grid, splits, depth)
    return logits


def reserve_topk_topp_cute_workspace(device: torch.device) -> None:
    """Compile the kernels and allocate the workspace before CUDA graph
    capture, during memory profiling."""
    _state(device)


# Kernel constants, without importing the CuTe DSL (checked in _DeviceState).
STAGE, SHALLOW = 4096, 3


@dataclass(frozen=True)
class _Plan:
    sms: int
    clusters: dict[int, int]  # co-resident clusters per cluster size
    max_splits: int
    deep: int  # ring stages of one CTA per SM

    def launch(self, batch: int, vocab: int) -> tuple[bool, int, int, int]:
        """(cluster, grid, splits, ring stages). Both kernels are persistent
        over rows, and each CTA owns workspace slot blockIdx.x, so the grid
        stays within the 2 * SMs slots."""
        splits = self.choose_splits(batch, vocab)
        deep = batch * splits <= self.sms
        ctas = self.sms if deep else 2 * self.sms
        grid = min(batch, ctas // splits) * splits
        return splits > 1, grid, splits, self.deep if deep else SHALLOW

    def choose_splits(self, batch: int, vocab: int) -> int:
        """Splits per row: clusters stay co-resident and keep a stage per CTA."""

        def fits(splits: int) -> bool:
            return splits <= self.max_splits and vocab // splits >= STAGE

        # Deep rings while the CTAs fill at most half the SMs.
        splits = 1
        while (
            fits(splits * 2)
            and batch <= self.clusters[splits * 2]
            and batch * splits * 2 <= self.sms // 2
        ):
            splits *= 2
        # Up to one row per SM, small clusters with shallow rings still cut
        # the per-row latency of long rows.
        if splits == 1 and vocab >= 16 * STAGE and fits(2) and batch <= self.sms:
            splits = 4 if fits(4) and batch * 4 * 8 <= 2 * self.sms * 7 else 2
        return splits


class _DeviceState:
    def __init__(self, device: torch.device):
        import vllm.v1.sample.ops.topk_topp_cute_kernel as kn

        assert (STAGE, SHALLOW) == (kn.STAGE, kn.SHALLOW)
        self.unused_k = torch.zeros(1, dtype=torch.int32, device=device)
        self.unused_p = torch.ones(1, dtype=torch.float32, device=device)
        props = torch.cuda.get_device_properties(device)
        sms = props.multi_processor_count
        smem = props.shared_memory_per_block_optin - kn.RING_OFFSET
        deep = min(kn.MAX_STAGES, smem // (kn.STAGE * 4))
        self.plan = _plan(_index(device), sms, deep)
        self.kernels = {cluster: _compile(cluster) for cluster in (False, True)}
        # At most two CTAs per SM run at once, each with its own overflow store.
        self.workspace = torch.empty(
            2 * sms * 2 * kn.GCAP, dtype=torch.uint32, device=device
        )


def _plan(device_index: int, sms: int, deep: int) -> _Plan:
    """Co-resident clusters of each size, for CTAs with deep rings (one per
    SM)."""
    from cutlass.utils import HardwareInfo

    hw = HardwareInfo(device_index)
    clusters = {1: sms}
    for splits in (2, 4, 8, 16):
        active = hw.get_max_active_clusters(splits)
        if active <= 0:
            break
        clusters[splits] = active
    return _Plan(sms, clusters, max(clusters), deep)


def _compile(cluster: bool):
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Uint32
    from quack.compile_utils import make_fake_tensor

    import vllm.v1.sample.ops.topk_topp_cute_kernel as kn

    batch = cute.sym_int()
    x = make_fake_tensor(Float32, (batch, cute.sym_int()), divisibility=1)
    ks = make_fake_tensor(Int32, (cute.sym_int(),), divisibility=1)
    ps = make_fake_tensor(Float32, (cute.sym_int(),), divisibility=1)
    ws = make_fake_tensor(Uint32, (cute.sym_int(),), divisibility=4)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kn.launch,
        x,
        ks,
        ps,
        ws,
        Int32(0),
        Int32(1),
        Int32(1),
        Int32(1),
        cluster,
        stream,
        options="--enable-tvm-ffi",
    )


def _state(device: torch.device) -> _DeviceState:
    index = _index(device)
    state = _STATE.get(index)
    if state is None:
        with torch.accelerator.device_index(index):
            state = _DeviceState(torch.device("cuda", index))
        _STATE[index] = state
    return state


def _index(device: torch.device) -> int:
    if device.index is None:
        return torch.accelerator.current_device_index()
    return device.index
