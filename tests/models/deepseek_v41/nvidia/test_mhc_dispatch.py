# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which DeepSeek-V4.1 mHC boundaries take the SM80 CuTe DSL kernels: all
of them on SM80, those of up to `SM90_MHC_MAX_TOKENS` rows on SM90, none
elsewhere; which deployments fuse the TP all-reduce into the boundaries, and
up to how many rows on SM90; which SM80 deployments reduce at the boundaries
with the two-shot kernel, and up to how many rows, with the MoE's top-k
reduction where the MoE leaves it open."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.nvidia.ops import mhc
from vllm.models.deepseek_v41.nvidia.specialized_decode import ampere


@pytest.fixture
def capability(monkeypatch):
    def set_capability(value: int, cuda: bool = True) -> None:
        monkeypatch.setattr(ampere, "is_ampere", lambda: cuda and value == 80)
        monkeypatch.setattr(
            mhc,
            "current_platform",
            SimpleNamespace(
                is_cuda=lambda: cuda, is_device_capability=lambda c: c == value
            ),
        )
        mhc.ampere_mhc_max_tokens.cache_clear()

    yield set_capability
    mhc.ampere_mhc_max_tokens.cache_clear()


@pytest.mark.parametrize(
    ("value", "cuda", "expected"),
    [
        (80, True, mhc.AMPERE_MHC_MAX_TOKENS),
        (90, True, mhc.SM90_MHC_MAX_TOKENS),
        (100, True, 0),
        # A ROCm GPU can also report capability 9.x.
        (90, False, 0),
    ],
)
def test_ampere_mhc_max_tokens(capability, value, cuda, expected):
    capability(value, cuda)
    assert mhc.ampere_mhc_max_tokens() == expected
    assert mhc.use_ampere_mhc() == (expected > 0)


@pytest.mark.parametrize("value", [80, 90])
def test_boundary_rows(capability, value):
    capability(value)
    norm_weight = torch.ones(4096)

    def takes(rows: int) -> bool:
        return mhc._ampere_mhc(torch.empty(rows, 4, 4096), norm_weight)

    limit = mhc.ampere_mhc_max_tokens()
    assert takes(1) and takes(limit)
    assert not takes(limit + 1)
    assert not mhc._ampere_mhc(torch.empty(1, 4, 4096), None)


@pytest.mark.parametrize(
    ("value", "tp", "multicast", "expected"),
    [
        # H100 at TP8, with NVLink multicast.
        (90, 8, 1, True),
        (90, 4, 1, False),
        (90, 8, 0, False),
        # GB300 keeps its TP4 overlap path.
        (103, 4, 1, True),
        (103, 8, 1, False),
        # A100 has no NVLink multicast.
        (80, 8, 1, False),
    ],
)
def test_supports_mhc_all_reduce(monkeypatch, value, tp, multicast, expected):
    """Which TP deployments fuse the all-reduce into the mHC boundaries."""
    monkeypatch.setattr(
        mhc,
        "current_platform",
        SimpleNamespace(
            is_device_capability=lambda c: c == value,
            is_device_capability_family=lambda f: f == value // 10 * 10,
        ),
    )
    comm = SimpleNamespace(
        ca_comm=SimpleNamespace(mnnvl_lamport_ag_multicast_ptr=multicast)
    )
    monkeypatch.setattr(
        mhc, "get_tp_group", lambda: SimpleNamespace(device_communicator=comm)
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, enable_expert_parallel=False
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(hidden_size=5120, hc_mult=4)
        ),
    )
    assert mhc.supports_mhc_all_reduce(config) == expected


@pytest.mark.parametrize(
    ("value", "tp", "custom", "expected"),
    [
        # A100 at TP8, 4 and 2 with vLLM's custom all-reduce.
        (80, 8, "on", True),
        (80, 4, "on", True),
        (80, 2, "on", True),
        (80, 6, "on", False),
        # Without full peer access the custom all-reduce is off or absent.
        (80, 8, "disabled", False),
        (80, 8, "partial", False),
        (80, 8, None, False),
        # H100 fuses the all-reduce into the boundaries instead.
        (90, 8, "on", False),
    ],
)
def test_supports_ampere_all_reduce(monkeypatch, value, tp, custom, expected):
    """Which TP deployments reduce at the SM80 boundaries with their own
    two-shot kernel."""
    monkeypatch.setattr(
        mhc,
        "current_platform",
        SimpleNamespace(is_device_capability=lambda c: c == value),
    )
    ca_comm = None
    if custom is not None:
        ca_comm = SimpleNamespace(
            disabled=custom == "disabled", fully_connected=custom != "partial"
        )
    comm = SimpleNamespace(ca_comm=ca_comm)
    monkeypatch.setattr(
        mhc, "get_tp_group", lambda: SimpleNamespace(device_communicator=comm)
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, enable_expert_parallel=False
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(hidden_size=5120)),
    )
    assert mhc.supports_ampere_all_reduce(config) == expected


@pytest.mark.parametrize("over", [False, True])
def test_sm80_boundary_all_reduce(capability, monkeypatch, over):
    """SM80 boundaries that take the all-reduce fuse the two-shot kernel with
    the post-mix and collapse for the rows it takes, then compute the next
    coefficients; above them, the TP group's all-reduce and the SM80 mHC
    kernels."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import (
        mhc as ampere_mhc,
    )

    capability(80)
    rows = 3
    calls: list[str] = []

    def all_reduce(x):
        calls.append("group")
        return x

    def two_shot(x):
        calls.append("two_shot")
        return x

    def post_collapse(x, streams, *args):
        calls.append("fused")
        return streams, x, x

    def coefficients(*args, **kwargs):
        calls.append("coefficients")
        return (torch.zeros(rows, 4),) * 3

    two_shot.supports = lambda x: not over  # type: ignore[attr-defined]
    two_shot.post_collapse = post_collapse  # type: ignore[attr-defined]
    monkeypatch.setattr(
        mhc, "get_tp_group", lambda: SimpleNamespace(all_reduce=all_reduce)
    )
    monkeypatch.setattr(mhc, "_ampere_all_reduce", two_shot)
    monkeypatch.setattr(mhc, "_all_reduce_mhc", None)
    monkeypatch.setattr(
        ampere_mhc, "mhc_post_pre", lambda *a, **k: calls.append("post_pre")
    )
    monkeypatch.setattr(ampere_mhc, "mhc_coefficients", coefficients)
    mhc.mhc_shifted_post_pre(
        torch.zeros(rows, 4096),
        torch.zeros(rows, 4, 4096),
        torch.zeros(rows, 4, 1),
        torch.zeros(rows, 4, 4),
        None,
        None,
        None,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        pre_mix=torch.zeros(rows, 4),
        norm_weight=torch.ones(4096),
        stream=SimpleNamespace(),
        reduce_results=True,
    )
    assert calls == (["group", "post_pre"] if over else ["fused", "coefficients"])


def test_sm80_boundary_finalize(capability, monkeypatch):
    """On SM80 a MoE may leave its top-k reduction open up to the two-shot
    kernel's rows; the next boundary's kernel then finalizes it with the
    all-reduce, post-mix and collapse, and the next coefficients follow."""
    from vllm.model_executor.layers.fused_moe.moe_output import (
        MoEOutput,
        UnfinalizedMoEOutput,
    )
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import (
        mhc as ampere_mhc,
    )

    capability(80)
    rows, top_k = 3, 6
    calls: list[str] = []

    def finalize(routes, route_weights, shared, streams, *args):
        calls.append("finalize")
        assert routes.shape == (rows * top_k, 4096)
        assert route_weights.shape == (rows, top_k)
        return streams, shared, shared

    def coefficients(*args, **kwargs):
        calls.append("coefficients")
        return (torch.zeros(rows, 4),) * 3

    two_shot = SimpleNamespace(capacity=128, finalize_post_collapse=finalize)
    monkeypatch.setattr(mhc, "_ampere_all_reduce", two_shot)
    monkeypatch.setattr(mhc, "_all_reduce_mhc", None)
    monkeypatch.setattr(ampere_mhc, "mhc_coefficients", coefficients)
    assert mhc.mhc_all_reduce_max_tokens() == 128
    routed = UnfinalizedMoEOutput(
        gemm2_permuted=torch.zeros(rows * top_k, 4096),
        expert_weights=torch.zeros(rows, top_k),
        expanded_idx_to_permuted_idx=torch.zeros(rows, top_k, dtype=torch.int32),
        route_major=True,
    )
    mhc.mhc_shifted_post_pre(
        MoEOutput(routed=routed, shared_output=torch.zeros(rows, 4096)),
        torch.zeros(rows, 4, 4096),
        torch.zeros(rows, 4, 1),
        torch.zeros(rows, 4, 4),
        None,
        None,
        None,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        pre_mix=torch.zeros(rows, 4),
        norm_weight=torch.ones(4096),
        stream=SimpleNamespace(),
        reduce_results=True,
    )
    assert calls == ["finalize", "coefficients"]


@pytest.mark.parametrize("over", [False, True])
def test_sm90_fused_all_reduce_rows(capability, monkeypatch, over):
    """SM90 boundaries in FULL graphs fuse the all-reduce up to its one-shot
    rows; larger boundaries of the SM80 kernels all-reduce first."""
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere import (
        mhc as ampere_mhc,
    )

    capability(90)
    limit = mhc.mhc_all_reduce_max_tokens()
    assert limit == mhc.SM90_MHC_ALL_REDUCE_MAX_TOKENS < mhc.SM90_MHC_MAX_TOKENS
    rows = limit + 1 if over else limit
    residual = torch.zeros(rows, 4, 4096)
    calls: list[str] = []

    def all_reduce(x):
        calls.append("all_reduce")
        return x

    def post_pre(*args, **kwargs):
        calls.append("post_pre")

    def fused(*args):
        calls.append("fused")
        return residual, residual[:, 0]

    monkeypatch.setattr(
        mhc, "get_tp_group", lambda: SimpleNamespace(all_reduce=all_reduce)
    )
    monkeypatch.setattr(ampere_mhc, "mhc_post_pre", post_pre)
    monkeypatch.setattr(mhc, "_all_reduce_mhc", fused)
    monkeypatch.setattr(
        ampere_mhc, "mhc_coefficients", lambda *a, **k: (torch.zeros(rows, 4),) * 3
    )
    mhc.mhc_shifted_post_pre(
        torch.zeros(rows, 4096),
        residual,
        torch.zeros(rows, 4, 1),
        torch.zeros(rows, 4, 4),
        None,
        None,
        None,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        pre_mix=torch.zeros(rows, 4),
        norm_weight=torch.ones(4096),
        stream=SimpleNamespace(),
        reduce_results=True,
    )
    assert calls == (["all_reduce", "post_pre"] if over else ["fused"])
