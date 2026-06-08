# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MoRI EP kernel auto-selection (``MoriAll2AllManager._select_kernel``).

Validates the SGLang #18437 port and the high-throughput / low-latency split:

* ``moriep_low_latency``                -> ``AsyncLL`` (regardless of node count)
* ``moriep_high_throughput`` intra-node -> ``IntraNode`` (with fp4 block tuning)
* ``moriep_high_throughput`` inter-node -> ``InterNodeV1LL`` when
  ``max_num_tokens_per_dp_rank`` <= ``VLLM_ROCM_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD``,
  else ``InterNodeV1``.

The heavy ``__init__`` (registers a torch process group, calls ``mori.shmem``)
is bypassed via ``__new__``; ``mori`` is stubbed so the test runs without ROCm.
"""

from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec

import pytest
import torch


def _install_mori_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a minimal ``mori`` stub exposing just the kernel-type enum."""
    stub = types.ModuleType("mori")
    stub.__spec__ = ModuleSpec("mori", loader=None)
    stub_ops = types.ModuleType("mori.ops")
    stub_ops.__spec__ = ModuleSpec("mori.ops", loader=None)

    class KernelType:
        IntraNode = "IntraNode"
        InterNodeV1 = "InterNodeV1"
        InterNodeV1LL = "InterNodeV1LL"
        AsyncLL = "AsyncLL"

    stub_ops.EpDispatchCombineKernelType = KernelType
    stub_ops.EpDispatchCombineOp = type("EpDispatchCombineOp", (), {})
    stub.ops = stub_ops
    monkeypatch.setitem(sys.modules, "mori", stub)
    monkeypatch.setitem(sys.modules, "mori.ops", stub_ops)


@pytest.fixture
def mgr(monkeypatch: pytest.MonkeyPatch):
    """A bare ``MoriAll2AllManager`` with all GPU / mori machinery stubbed."""
    _install_mori_stub(monkeypatch)
    # ``_select_kernel`` asserts running on gfx942/gfx950 -- force gfx942.
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx942", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: False)

    from vllm.distributed.device_communicators.all2all import MoriAll2AllManager

    m = MoriAll2AllManager.__new__(MoriAll2AllManager)
    return m


def _kernel(mgr, *, quant_dtype=torch.bfloat16, max_tokens: int):
    """Return just the kernel_type from the ``_select_kernel`` tuple."""
    return mgr._select_kernel(quant_dtype, max_tokens)[0]


# -- low latency backend -----------------------------------------------------


def test_low_latency_always_async_ll(mgr):
    mgr._all2all_backend = "moriep_low_latency"
    for internode in (False, True):
        mgr.internode = internode
        for n_tokens in (1, 256, 257, 4096):
            assert _kernel(mgr, max_tokens=n_tokens) == "AsyncLL", (
                internode,
                n_tokens,
            )


# -- high throughput, intra-node ---------------------------------------------


def test_intra_node_unaffected_by_threshold(mgr, monkeypatch):
    """When ``internode=False`` the threshold is irrelevant: always IntraNode."""
    mgr._all2all_backend = "moriep_high_throughput"
    mgr.internode = False
    monkeypatch.setenv(
        "VLLM_ROCM_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD", "256"
    )
    for n_tokens in (1, 127, 128, 255, 256, 257, 4096):
        assert _kernel(mgr, max_tokens=n_tokens) == "IntraNode", n_tokens


def test_intra_node_fp4_tuning_changes_block_config(mgr):
    """FP4 intra-node tuning differs from the bf16 path and across the
    128-token crossover (mirrors SGLang ``get_ep_dispatch_configs``)."""
    mgr._all2all_backend = "moriep_high_throughput"
    mgr.internode = False

    fp4 = torch.float4_e2m1fn_x2
    kt, warp_small, block_small, _ = mgr._select_kernel(fp4, 64)
    assert (kt, warp_small, block_small) == ("IntraNode", 5, 225)
    kt, warp_big, block_big, _ = mgr._select_kernel(fp4, 256)
    assert (kt, warp_big, block_big) == ("IntraNode", 16, 256)
    # bf16 intra-node uses its own block tuning.
    kt, warp_bf16, block_bf16, _ = mgr._select_kernel(torch.bfloat16, 256)
    assert (kt, warp_bf16, block_bf16) == ("IntraNode", 16, 80)


# -- high throughput, inter-node switch (SGLang #18437) ----------------------


@pytest.mark.parametrize(
    "threshold, n_tokens, expected",
    [
        (256, 1, "InterNodeV1LL"),
        (256, 255, "InterNodeV1LL"),
        (256, 256, "InterNodeV1LL"),  # boundary: <= threshold -> LL
        (256, 257, "InterNodeV1"),
        (256, 4096, "InterNodeV1"),
        (128, 128, "InterNodeV1LL"),
        (128, 129, "InterNodeV1"),
        (0, 1, "InterNodeV1"),
        (1_000_000, 4096, "InterNodeV1LL"),
    ],
)
def test_inter_node_switch(mgr, monkeypatch, threshold, n_tokens, expected):
    mgr._all2all_backend = "moriep_high_throughput"
    mgr.internode = True
    monkeypatch.setenv(
        "VLLM_ROCM_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD", str(threshold)
    )
    assert _kernel(mgr, max_tokens=n_tokens) == expected


def test_inter_node_default_threshold_is_256(mgr, monkeypatch):
    """Unset env var -> default = 256 (matches SGLang #18437)."""
    mgr._all2all_backend = "moriep_high_throughput"
    mgr.internode = True
    monkeypatch.delenv(
        "VLLM_ROCM_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD", raising=False
    )
    assert _kernel(mgr, max_tokens=256) == "InterNodeV1LL"
    assert _kernel(mgr, max_tokens=257) == "InterNodeV1"
