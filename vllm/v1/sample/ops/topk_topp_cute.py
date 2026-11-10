# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k/top-p masking with the CuTe DSL filter kernel (topk_topp_cute_kernel).

A JIT kernel over hierarchical maxima with the contract of
apply_top_k_top_p_cuda: FP32 logits, per-row device k and p, exactly k tokens
with ties broken by the lowest index, SM90 and SM100.
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
    splits = state.plan.choose_splits(batch, vocab)
    state.kernels[splits > 1](logits, k, p, flags, splits, _smem(vocab, splits))
    return logits


def reserve_topk_topp_cute_workspace(device: torch.device) -> None:
    """Compile the kernels before CUDA graph capture."""
    _state(device)


# Kernel constants, without importing the CuTe DSL (checked in _DeviceState).
ROUND, MAX_SPLITS = 8192, 16


@dataclass(frozen=True)
class _Plan:
    sms: int
    clusters: dict[int, int]  # co-resident clusters per size, one CTA per SM
    smem_per_sm: int
    max_smem: int

    def choose_splits(self, batch: int, vocab: int) -> int:
        """CTAs per row: clusters while all CTAs stay resident at two per SM,
        all clusters fit at once (else a second wave doubles the time), and
        each CTA streams a round at least."""
        splits = 1
        while (
            splits < MAX_SPLITS
            and batch * splits * 2 <= 2 * self.sms
            and vocab // (splits * 2) >= ROUND
            and batch <= self.active(vocab, splits * 2)
        ):
            splits *= 2
        while _smem(vocab, splits) > self.max_smem and splits < MAX_SPLITS:
            splits *= 2
        return splits

    def active(self, vocab: int, splits: int) -> int:
        per_sm = 2 if 2 * (_smem(vocab, splits) + 1024) <= self.smem_per_sm else 1
        return self.clusters.get(splits, 0) * per_sm


@functools.cache
def _smem(vocab: int, splits: int) -> int:
    import vllm.v1.sample.ops.topk_topp_cute_kernel as kn

    return kn.layout(vocab, splits, splits > 1)[-1]


class _DeviceState:
    def __init__(self, device: torch.device):
        import vllm.v1.sample.ops.topk_topp_cute_kernel as kn

        assert (ROUND, MAX_SPLITS) == (kn.ROUND, kn.MAX_SPLITS)
        self.unused_k = torch.zeros(1, dtype=torch.int32, device=device)
        self.unused_p = torch.ones(1, dtype=torch.float32, device=device)
        props = torch.cuda.get_device_properties(device)
        self.plan = _plan(
            _index(device),
            props.multi_processor_count,
            props.shared_memory_per_multiprocessor,
            props.shared_memory_per_block_optin,
        )
        self.kernels = {cluster: _compile(cluster) for cluster in (False, True)}


def _plan(device_index: int, sms: int, smem_per_sm: int, max_smem: int) -> _Plan:
    from cutlass.utils import HardwareInfo

    hw = HardwareInfo(device_index)
    clusters = {1: sms}
    for splits in (2, 4, 8, 16):
        active = hw.get_max_active_clusters(splits)
        if active <= 0:
            break
        clusters[splits] = active
    return _Plan(sms, clusters, smem_per_sm, max_smem)


def _compile(cluster: bool):
    import cutlass.cute as cute
    from cutlass import Float32, Int32
    from quack.compile_utils import make_fake_tensor

    import vllm.v1.sample.ops.topk_topp_cute_kernel as kn

    x = make_fake_tensor(Float32, (cute.sym_int(), cute.sym_int()), divisibility=1)
    ks = make_fake_tensor(Int32, (cute.sym_int(),), divisibility=1)
    ps = make_fake_tensor(Float32, (cute.sym_int(),), divisibility=1)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kn.launch,
        x,
        ks,
        ps,
        Int32(0),
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
