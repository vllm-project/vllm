# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The hyper-connection read of Qwen4Exp's specialized decode (`hc_read`,
CuTe DSL kernels) against the stock read (`GatedResidual`) at every kind of
read site (the MTP draft layer's first read, which combines the projected
embedding without injection logits, included), its CUDA graph replay, and its
warm-up; with the tensor-core kernels of TP1 reads (`tensor_cores`) also
against an FP32 reference next to the stock read's error, and their gate."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig
from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(90),
    reason="the specialized decode's read kernels require SM90+",
)

if current_platform.is_cuda():
    import cutlass.cute as cute

    from vllm.models.qwen4_exp.nvidia.ops.hc import grouped_gemma_rmsnorm
    from vllm.models.qwen4_exp.nvidia.specialized_decode.hc import (
        down,
        down_tc,
        read,
        up_mix,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.hc.read import (
        TC_MIN_ROWS,
        hc_project,
        hc_read,
        tensor_core_reads_unsupported,
        use_tensor_core_reads,
        warm_up_hc_read,
    )
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.decode import (
        stock_project,
        stock_read,
    )

STREAMS = 4
HIDDEN = 2560
LOW_RANK = 320
CONFIG = HyperConnectionConfig(
    hc_count=STREAMS,
    hidden_size=HIDDEN,
    params_dtype=torch.bfloat16,
    hc_lowrank=LOW_RANK,
    rms_norm_eps=1e-6,
    hc_per_branch_norm=True,
)
# Row counts that leave MMA columns of hc_up_mix's CTAs empty: a short last
# group, or a second CTA with fewer groups than the first.
PADDED_ROWS = [4, 12, 17, 20, 24, 33, 36, 40, 49, 52, 56]
# Reads that take the tensor-core kernels (tcgen05, SM100 family) at TP1:
# the smallest (read.TC_MIN_ROWS, checked below), partial tiles and the full
# 128-row tile.
TC_ROWS = [52, 65, 80, 96, 128]
TENSOR_CORES = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="the tensor-core read kernels require tcgen05 (SM100 family)",
)


def _hyper_connection(use_combine: bool, seed: int) -> GatedResidual:
    """A decoder layer's hyper-connection (`use_combine`) or the final
    mixer, with random weights of the checkpoint's scale."""
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        hc = GatedResidual(CONFIG, use_combine=use_combine)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    down = (
        hc.input_mix_weight_down_block_inject.weight
        if use_combine
        else hc.input_mix_weight_down.weight
    )
    for weight, scale in (
        (hc.hc_norm.weight, 0.1),
        (down, 0.02),
        (hc.input_mix_weight_up.weight, 0.05),
    ):
        weight.data.copy_(
            torch.randn(weight.shape, generator=generator, device="cuda") * scale
        )
    if use_combine:
        down.data[LOW_RANK + STREAMS :] = 0  # the merged weight's padding rows
    return hc


@pytest.fixture
def layer_hc(dist_init) -> GatedResidual:
    return _hyper_connection(use_combine=True, seed=0)


@pytest.fixture
def final_hc(dist_init) -> GatedResidual:
    return _hyper_connection(use_combine=False, seed=1)


def _inputs(rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Streams, a block output and its injection logits (a column view of a
    down projection's output, as the decode passes them)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(
            shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )

    injection = randn(rows, LOW_RANK + STREAMS + 12)[:, LOW_RANK : LOW_RANK + STREAMS]
    return randn(rows, STREAMS * HIDDEN), randn(rows, HIDDEN), injection


@pytest.mark.parametrize(
    "rows, tensor_cores",
    [(rows, False) for rows in (1, 4, 16, 33, 48, 64)]
    + [pytest.param(rows, True, marks=TENSOR_CORES) for rows in TC_ROWS],
)
@pytest.mark.parametrize("site", ["mix", "combine", "draft", "final"])
def test_read_matches_the_stock_read(rows, tensor_cores, site, layer_hc, final_hc):
    """The streams are the stock ones bit for bit; the block input (FP32 gate
    where the stock gate is BF16) and the injection logits (accumulation
    order) within the tolerances below."""
    streams, block_output, injection = _inputs(rows, seed=rows)
    hc = final_hc if site == "final" else layer_hc
    pending, pending_injection = {
        "mix": (None, None),
        "draft": (block_output, None),
    }.get(site, (block_output, injection))

    expected = stock_read(hc, streams, pending, pending_injection)
    actual = hc_read(hc, streams, pending, pending_injection, tensor_cores)

    assert torch.equal(actual[0], expected[0])
    block_input, expected_block_input = actual[1].float(), expected[1].float()
    cosine = F.cosine_similarity(
        block_input.flatten(), expected_block_input.flatten(), dim=0
    )
    assert cosine >= 0.99999
    max_diff = (block_input - expected_block_input).abs().max()
    assert max_diff <= 1e-2 * expected_block_input.abs().max()
    if site == "final":
        assert actual[2] is None and expected[2] is None
    else:
        # At most one BF16 rounding step apart.
        torch.testing.assert_close(actual[2], expected[2], rtol=1.6e-2, atol=1e-3)


@pytest.mark.parametrize("rows", PADDED_ROWS)
@pytest.mark.parametrize("padding", ["last row", "nan"])
def test_up_mix_padding_columns_do_not_reach_real_rows(rows, padding, layer_hc):
    """`hc_up_mix` leaves the MMA columns past the last row unstaged. Its
    real rows equal a call on inputs padded to the CTAs' staged rows: with
    copies of the last row (what those columns held when they were staged)
    and with NaN (whatever stale shared memory holds)."""
    groups = -(-rows // up_mix.GROUP_ROWS)
    per = up_mix.groups_per_cta(rows)
    staged = -(-groups // per) * per * up_mix.GROUP_ROWS
    assert staged > rows
    generator = torch.Generator(device="cuda").manual_seed(rows)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(
            shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )

    # low is a column view of the down projection's [rows, 336] output.
    down_output, xn = randn(staged, LOW_RANK + 16), randn(staged, STREAMS * HIDDEN)
    up = layer_hc.input_mix_weight_up.weight
    actual = up_mix.hc_up_mix(down_output[:rows, :LOW_RANK], xn[:rows], up)
    for tensor in (down_output, xn):
        tensor[rows:] = tensor[rows - 1] if padding == "last row" else float("nan")
    expected = up_mix.hc_up_mix(down_output[:, :LOW_RANK], xn, up)[:rows]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "rows, tensor_cores",
    [(rows, False) for rows in (*PADDED_ROWS, 64)]
    + [pytest.param(rows, True, marks=TENSOR_CORES) for rows in TC_ROWS],
)
def test_reads_replay_bitwise_in_a_cuda_graph(rows, tensor_cores, layer_hc, final_hc):
    streams, block_output, _ = _inputs(rows, seed=rows)

    def reads() -> tuple[torch.Tensor, ...]:
        mixed, first_input, injection = hc_read(
            layer_hc, streams, None, None, tensor_cores
        )
        combined, second_input, injection = hc_read(
            layer_hc, mixed, block_output, injection, tensor_cores
        )
        final_streams, final_input, _ = hc_read(
            final_hc, combined, block_output, injection, tensor_cores
        )
        return first_input, second_input, injection, final_streams, final_input

    eager = reads()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = reads()
    graph.replay()
    torch.accelerator.synchronize()
    for replayed, expected in zip(captured, eager):
        assert torch.equal(replayed, expected)


@pytest.mark.parametrize(
    "max_rows, tensor_cores",
    [(64, False), pytest.param(128, True, marks=TENSOR_CORES)],
)
def test_warm_up_compiles_every_read(
    max_rows, tensor_cores, layer_hc, final_hc, monkeypatch
):
    for module in (down, up_mix, down_tc):
        module._compiled.cache_clear()
    warm_up_hc_read(layer_hc, final_hc, max_rows, tensor_cores)

    def no_compile(*args: object, **kwargs: object) -> None:
        raise AssertionError("a read compiled after the warm-up")

    monkeypatch.setattr(cute, "compile", no_compile)
    for rows in range(1, max_rows + 1):
        streams, block_output, injection = _inputs(rows, seed=0)
        hc_read(layer_hc, streams, None, None, tensor_cores)
        streams, _, injection = hc_read(
            layer_hc, streams, block_output, injection, tensor_cores
        )
        hc_read(final_hc, streams, block_output, injection, tensor_cores)


def _relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """The largest error over the reference's largest magnitude."""
    return ((actual.float() - reference).abs().max() / reference.abs().max()).item()


@TENSOR_CORES
@pytest.mark.parametrize("rows", TC_ROWS)
@pytest.mark.parametrize("kind", ["layer", "final"])
def test_tensor_core_projections_against_fp32(rows, kind, layer_hc, final_hc):
    """The tensor-core projections' block input and injection logits are as
    close to an FP32 reference as the stock projections' (within half a BF16
    step of the largest value), at a decoder layer's read and at the final
    mixer."""
    hc = final_hc if kind == "final" else layer_hc
    streams, _, _ = _inputs(rows, seed=rows + 1)
    xn = grouped_gemma_rmsnorm(streams, hc.hc_norm.weight, 1e-6, STREAMS)
    down_weight = (
        hc.input_mix_weight_down_block_inject.weight
        if hc.use_combine
        else hc.input_mix_weight_down.weight
    ).float()
    outputs = LOW_RANK + (STREAMS if hc.use_combine else 0)
    projected = xn.float() @ down_weight[:outputs].T
    low = F.silu(projected[:, :LOW_RANK] / STREAMS)
    gate = low @ hc.input_mix_weight_up.weight.float().T
    mixed = torch.sigmoid(gate) * xn.float()
    reference = mixed.view(rows, STREAMS, HIDDEN).mean(dim=1)

    block_input, injection = hc_project(hc, xn, tensor_cores=True)
    stock_input, stock_injection = stock_project(hc, xn)

    error = _relative_error(block_input, reference)
    assert error <= _relative_error(stock_input, reference) + 2**-9, error
    cosine = F.cosine_similarity(
        block_input.float().flatten(), reference.flatten(), dim=0
    )
    assert cosine >= 0.99999
    if hc.use_combine:
        assert injection is not None and stock_injection is not None
        reference_injection = projected[:, LOW_RANK:outputs]
        assert _relative_error(injection, reference_injection) <= (
            _relative_error(stock_injection, reference_injection) + 2**-9
        )
    else:
        assert injection is None


@TENSOR_CORES
def test_tensor_cores_take_the_reads_from_the_bound(layer_hc, monkeypatch):
    """With `tensor_cores` the projections of `TC_MIN_ROWS` rows and more
    take the tensor-core kernels and smaller ones the warp-MMA kernels;
    without it every read takes the warp-MMA kernels."""
    assert TC_ROWS[0] == TC_MIN_ROWS
    calls: list[str] = []
    for name in ("hc_down", "hc_down_tc", "hc_up_mix", "hc_up_gate_mix"):
        kernel = getattr(read, name)

        def recording(*args, kernel=kernel, name=name, **kwargs):
            calls.append(name)
            return kernel(*args, **kwargs)

        monkeypatch.setattr(read, name, recording)
    for rows, tensor_cores, expected in (
        (TC_MIN_ROWS - 1, True, ["hc_down", "hc_up_mix"]),
        (TC_MIN_ROWS, True, ["hc_down_tc", "hc_up_gate_mix"]),
        (128, True, ["hc_down_tc", "hc_up_gate_mix"]),
        (64, False, ["hc_down", "hc_up_mix"]),
    ):
        calls.clear()
        xn = torch.zeros(rows, STREAMS * HIDDEN, dtype=torch.bfloat16, device="cuda")
        hc_project(layer_hc, xn, tensor_cores=tensor_cores)
        assert calls == expected, (rows, tensor_cores, calls)


@TENSOR_CORES
def test_tensor_core_reads_gate(monkeypatch):
    """TP1 decodes whose steps reach `TC_MIN_ROWS` rows take the
    tensor-core reads; other TP sizes, smaller decodes and devices without
    the shared memory or the 16-CTA clusters the kernels need do not."""
    device = torch.accelerator.current_device_index()
    assert tensor_core_reads_unsupported(device) is None
    assert use_tensor_core_reads(128, 1, device)
    assert use_tensor_core_reads(TC_MIN_ROWS, 1, device)
    assert not use_tensor_core_reads(TC_MIN_ROWS - 1, 1, device)
    assert not use_tensor_core_reads(128, 4, device)

    class NoClusters:
        def __init__(self, device_id: int = 0) -> None:
            pass

        def get_max_active_clusters(self, cluster_size: int) -> int:
            return 0

    with monkeypatch.context() as patch:
        patch.setattr(read.cutlass.utils, "HardwareInfo", NoClusters)
        assert tensor_core_reads_unsupported(device) == "clusters of 16 CTAs"
        assert not use_tensor_core_reads(128, 1, device)
    real = torch.cuda.get_device_properties(device)
    small = SimpleNamespace(
        major=real.major, minor=real.minor, shared_memory_per_block_optin=100 * 1024
    )
    with monkeypatch.context() as patch:
        patch.setattr(read.torch.cuda, "get_device_properties", lambda index: small)
        missing = tensor_core_reads_unsupported(device)
        assert missing is not None and "shared memory" in missing
        assert not use_tensor_core_reads(128, 1, device)
