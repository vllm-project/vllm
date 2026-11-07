# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests of the Qwen4Exp exact IPC sequence-parallel collectives.

With `--enable-hc-sp-aiter-ipc` the decoder layer's four SP collectives go through
AITER's IPC reduce-scatter and all-gather instead of RCCL, and
`--enable-hc-sp-quick-reduce` takes the same pair for the steps below the INT4
size gate. The results are exact either way, so the 2-GPU checks assert that the
AITER path was really taken (a spy on its entry points, including during CUDA
graph capture): an RCCL fallback would pass any numerical bound. The kernels of a
replayed graph are not checked here: a torch profile around ``graph.replay()``
records no event in this test process; a server trace is the evidence for them.
The selection and the gate are CPU tests at the end of the file.
"""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("AITER IPC collectives are ROCm-only", allow_module_level=True)

from tests.utils import multi_gpu_test
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import ParallelConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.device_communicators.aiter_custom_all_reduce import (
    AiterCustomAllreduce,
)
from vllm.distributed.parallel_state import graph_capture
from vllm.forward_context import set_forward_context
from vllm.logger import _print_info_once, _print_warning_once
from vllm.models.common.ops import sequence_parallel as common_sp
from vllm.models.qwen4_exp.amd import model as amd_model
from vllm.models.qwen4_exp.amd.ops import sequence_parallel as amd_sp
from vllm.utils.network_utils import get_open_port

from .test_sp_quick_reduce import (
    NMSE_BOUND,
    NMSE_FLOOR,
    _aiter_has_sp_quick_reduce,
    _forward_layer,
    _nmse,
    _partials,
    _require_quick_reduce,
    _run,
)
from .test_tpsp import _make_config, _make_decoder

# Exact collectives differ from the fp32 sum by bf16 rounding only (about 3e-6).
NMSE_BOUND_EXACT = 1e-5


def _require_aiter() -> AiterCustomAllreduce:
    comm = rocm_aiter_ops.get_aiter_allreduce()
    assert comm is not None and not comm.disabled, (
        "the TP group has no enabled AITER custom all-reduce: needs "
        "VLLM_ROCM_USE_AITER=1 and its custom all-reduce"
    )
    return comm


class _CallSpy:
    """Records, in order, the calls that reach the attributes of `target` named by
    `methods` ({recorded name: attribute name}); the calls still run."""

    def __init__(self, target, **methods: str) -> None:
        self.target = target
        self.methods = methods
        self.calls: list[str] = []

    @property
    def counts(self) -> dict[str, int]:
        return {name: self.calls.count(name) for name in self.methods}

    def __enter__(self):
        self._stack = ExitStack()
        for name, attr in self.methods.items():
            original = getattr(self.target, attr)

            def wrapper(*args, _name=name, _original=original, **kwargs):
                self.calls.append(_name)
                return _original(*args, **kwargs)

            self._stack.enter_context(mock.patch.object(self.target, attr, wrapper))
        return self

    def __exit__(self, *exc):
        self._stack.close()


def _aiter_spy() -> _CallSpy:
    """The calls that reach AITER's IPC reduce-scatter and all-gather."""
    return _CallSpy(
        AiterCustomAllreduce,
        reduce_scatter="custom_reduce_scatter",
        all_gather="custom_all_gather",
    )


def _flydsl_spy(kernels) -> _CallSpy:
    """The launches of the FlyDSL INT4 reduce-scatter and all-gather."""
    return _CallSpy(kernels, reduce_scatter="reduce_scatter", all_gather="all_gather")


def _reference_block(x: torch.Tensor, rank: int) -> torch.Tensor:
    padded = x.shape[0] + x.shape[0] % 2  # an odd tensor is padded to the group size
    reference = tensor_model_parallel_all_reduce(x.float())
    reference = torch.nn.functional.pad(reference, (0, 0, 0, padded - x.shape[0]))
    chunk = padded // 2
    return reference[rank * chunk : (rank + 1) * chunk]


def _check_eager(rank: int, config: VllmConfig) -> None:
    """Eager calls (the copy-in path), decode-sized and prefill-sized, odd rows
    padded as the common path pads them."""
    _require_aiter()
    for rows in (64, 2048, 63, 2063):
        x = _partials(rows, rank)
        with _aiter_spy() as spy:
            out = amd_sp.sp_reduce_scatter_ipc(x)
        assert spy.counts == {"reduce_scatter": 1, "all_gather": 0}, (rows, spy.counts)
        block = _reference_block(x, rank)
        assert out.shape == block.shape, (rows, out.shape)
        assert _nmse(out, block) < NMSE_BOUND_EXACT, rows
        with _aiter_spy() as spy:
            gathered = amd_sp.sp_all_gather_ipc(out)
        assert spy.counts == {"reduce_scatter": 0, "all_gather": 1}, (rows, spy.counts)
        torch.testing.assert_close(
            gathered, tensor_model_parallel_all_gather(out, dim=0), atol=0, rtol=0
        )


def _check_graph_replay(rank: int, config: VllmConfig) -> None:
    """The path production takes: a CUDA graph captured under `graph_capture()`
    in the forward context the V2 runner builds for FULL graphs, replayed with
    fresh values in the captured input buffer."""
    _require_aiter()
    rows = 128
    device = torch.device(f"cuda:{rank}")
    x = _partials(rows, rank)
    amd_sp.sp_all_gather_ipc(amd_sp.sp_reduce_scatter_ipc(x))  # eager warm-up
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with (
        set_forward_context(
            None,
            config,
            num_tokens=rows,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=None,
        ),
        _aiter_spy() as spy,
        graph_capture(device) as capture,
        torch.cuda.graph(graph, stream=capture.stream),
    ):
        reduced = amd_sp.sp_reduce_scatter_ipc(x)
        gathered = amd_sp.sp_all_gather_ipc(reduced)
    assert spy.counts == {"reduce_scatter": 1, "all_gather": 1}, spy.counts
    previous = None
    for seed in (1, 2, 3):
        x.copy_(_partials(rows, rank, seed=seed))
        graph.replay()
        torch.cuda.synchronize()
        block = _reference_block(x, rank)
        assert _nmse(reduced, block) < NMSE_BOUND_EXACT, seed
        torch.testing.assert_close(
            gathered, tensor_model_parallel_all_gather(reduced, dim=0), atol=0, rtol=0
        )
        assert previous is None or not torch.equal(previous, reduced), seed
        previous = reduced.clone()


def _check_fallbacks(rank: int, config: VllmConfig) -> None:
    """No usable AITER communicator, or one that declines the shape: the common
    exact path, and AITER is not called."""
    _require_aiter()
    x = _partials(256, rank)
    shard = _partials(128, rank, seed=200)
    expected_rs = common_sp.sp_reduce_scatter(x)
    expected_ag = common_sp.sp_all_gather(shard)
    for patched in (
        (mock.patch.object(rocm_aiter_ops, "get_aiter_allreduce", return_value=None),),
        (
            mock.patch.object(
                AiterCustomAllreduce,
                "disabled",
                new_callable=mock.PropertyMock,
                return_value=True,
            ),
        ),
        (
            mock.patch.object(
                AiterCustomAllreduce, "should_custom_rs", mock.Mock(return_value=False)
            ),
            mock.patch.object(
                AiterCustomAllreduce, "should_custom_ag", mock.Mock(return_value=False)
            ),
        ),
    ):
        with ExitStack() as stack, _aiter_spy() as spy:
            for patch in patched:
                stack.enter_context(patch)
            torch.testing.assert_close(
                amd_sp.sp_reduce_scatter_ipc(x), expected_rs, atol=0, rtol=0
            )
            torch.testing.assert_close(
                amd_sp.sp_all_gather_ipc(shard), expected_ag, atol=0, rtol=0
            )
        assert spy.counts == {"reduce_scatter": 0, "all_gather": 0}, spy.counts


def _check_compiled_ops(rank: int, config: VllmConfig) -> None:
    """The opaque ops decide per call, so one trace serves every token count."""
    _require_aiter()

    def collectives(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gathered = amd_sp.sp_all_gather_ipc(amd_sp.sp_shard(x) * 2)
        reduced = amd_sp.sp_all_gather_ipc(amd_sp.sp_reduce_scatter_ipc(x))
        return gathered[: x.shape[0]], reduced[: x.shape[0]]

    compiled = torch.compile(
        collectives,
        fullgraph=True,
        options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
    )
    traced = _partials(2100, rank)
    torch._dynamo.mark_dynamic(traced, 0)
    compiled(traced)
    for rows in (2063, 64, 2100):
        x = _partials(rows, rank, seed=300 + rows)
        with _aiter_spy() as spy:
            results = compiled(x)
        assert spy.counts == {"reduce_scatter": 1, "all_gather": 2}, (rows, spy.counts)
        for got, want in zip(results, collectives(x)):
            torch.testing.assert_close(got, want, atol=0, rtol=0)


def _fresh_decoder(rank: int):
    # A second layer in one config would register its layer names twice.
    config = _make_config(rank, hc_sp=True)
    config.model_config.hf_text_config.ple_layer_ids = []
    return _make_decoder(config, rank)


def _check_layer_sites(rank: int, config: VllmConfig) -> None:
    """Which pair a layer's four sites use: the default config, whose flags are
    off, keeps the common pair; the run's flag (`--enable-hc-sp-aiter-ipc`) selects
    the IPC pair; without HC SP neither opt-in is asked. The selection logic itself
    is covered by the CPU tests below."""
    layer = _fresh_decoder(rank)  # flags off by default
    assert layer.sp_all_gather is common_sp.sp_all_gather
    assert layer.sp_reduce_scatter is amd_sp.sp_reduce_scatter
    config.model_config.hf_text_config.ple_layer_ids = []
    layer = _make_decoder(config, rank)  # `--enable-hc-sp-aiter-ipc`
    assert layer.sp_all_gather is amd_sp.sp_all_gather_ipc
    assert layer.sp_reduce_scatter is amd_sp.sp_reduce_scatter_ipc
    # Without HC SP (as at TP=1, where both flags are ignored) neither opt-in is
    # asked, so neither warns.
    with (
        mock.patch.object(
            amd_model, "is_hc_sequence_parallel_enabled", return_value=False
        ),
        mock.patch.object(common_sp, "use_quick_sp", side_effect=AssertionError),
        mock.patch.object(amd_sp, "use_sp_aiter_ipc", side_effect=AssertionError),
    ):
        layer = _fresh_decoder(rank)
    assert layer.sp_all_gather is common_sp.sp_all_gather
    assert layer.sp_reduce_scatter is amd_sp.sp_reduce_scatter


def _check_layer_forward(rank: int, config: VllmConfig) -> None:
    """A layer built with the run's flag (`--enable-hc-sp-aiter-ipc`, or
    `--enable-hc-sp-quick-reduce` alone: its 5 tokens are below the INT4 size gate)
    reaches AITER at its four sites: two reduce-scatters and two all-gathers per
    forward."""
    _require_aiter()
    config.model_config.hf_text_config.ple_layer_ids = []
    layer = _make_decoder(config, rank)
    with _aiter_spy() as spy:
        _forward_layer(layer, config)
    assert spy.counts == {"reduce_scatter": 2, "all_gather": 2}, spy.counts


def _check_quantized_routes(rank: int, config: VllmConfig) -> None:
    """With `--enable-hc-sp-quick-reduce` alone, shapes below the INT4 size gate take
    AITER's exact IPC collectives and above it the FlyDSL kernels, never both: the gate
    pads each rank's block to 16 rows and needs the full tensor to be at least
    8 MiB, so 1,632 rows are below it and 1,633 / 1,634 above (a 1,633-row tensor
    is padded to 1,634 first)."""
    _require_aiter()
    kernels = _require_quick_reduce()._flydsl_int4
    for rows, int4 in (
        (600, False),
        (1632, False),
        (1633, True),
        (1634, True),
        (2063, True),
    ):
        x = _partials(rows, rank)
        with _flydsl_spy(kernels) as flydsl, _aiter_spy() as spy:
            reduced = amd_sp.sp_reduce_scatter_quantized(x)
            gathered = amd_sp.sp_all_gather_quantized(reduced)
        block = _reference_block(x, rank)
        if int4:
            assert flydsl.calls == ["reduce_scatter", "all_gather"], rows
            assert spy.counts == {"reduce_scatter": 0, "all_gather": 0}, rows
            assert NMSE_FLOOR < _nmse(reduced, block) < NMSE_BOUND, rows
        else:
            assert flydsl.calls == [], rows
            assert spy.counts == {"reduce_scatter": 1, "all_gather": 1}, rows
            assert _nmse(reduced, block) < NMSE_BOUND_EXACT, rows
            torch.testing.assert_close(
                gathered,
                tensor_model_parallel_all_gather(reduced, dim=0),
                atol=0,
                rtol=0,
            )


@pytest.fixture
def sp_aiter_ipc_env(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")


@pytest.mark.parametrize(
    "check",
    [
        _check_eager,
        _check_graph_replay,
        _check_fallbacks,
        _check_compiled_ops,
        _check_layer_sites,
        _check_layer_forward,
    ],
    ids=lambda check: check.__name__.removeprefix("_check_"),
)
@multi_gpu_test(num_gpus=2)
def test_sp_aiter_ipc(sp_aiter_ipc_env, check) -> None:
    flags = {"enable_hc_sp_aiter_ipc": True}
    mp.spawn(_run, args=(get_open_port(), check, flags), nprocs=2)


@multi_gpu_test(num_gpus=2)
def test_sp_quick_flag_alone_reaches_aiter_ipc(sp_aiter_ipc_env, monkeypatch) -> None:
    """The INT4 flag implies the IPC pair below its size gate, with or without an
    AITER whose FlyDSL INT4 serves SP."""
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", "INT4")
    flags = {"enable_hc_sp_quick_reduce": True}
    mp.spawn(_run, args=(get_open_port(), _check_layer_forward, flags), nprocs=2)


@pytest.mark.skipif(
    not _aiter_has_sp_quick_reduce(),
    reason="needs an AITER whose FlyDSL QuickAllReduceInt4 has reduce_scatter "
    "and all_gather",
)
@multi_gpu_test(num_gpus=2)
def test_sp_quantized_routes(sp_aiter_ipc_env, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", "INT4")
    mp.spawn(_run, args=(get_open_port(), _check_quantized_routes), nprocs=2)


# The selection and the gate, on the CPU: the fakes play the platform and AITER's
# communicator. They live here, not in test_quick_all_reduce.py, because the CI step
# that runs this file is the one that depends on the code they test.
EXACT_PAIR = (common_sp.sp_all_gather, amd_sp.sp_reduce_scatter)
IPC_PAIR = (amd_sp.sp_all_gather_ipc, amd_sp.sp_reduce_scatter_ipc)
INT4_PAIR = (amd_sp.sp_all_gather_quantized, amd_sp.sp_reduce_scatter_quantized)


def _parallel_only(**flags: bool) -> SimpleNamespace:
    return SimpleNamespace(parallel_config=ParallelConfig(**flags))


def _platform(monkeypatch, cdna: int = 4, comm: bool = True) -> None:
    """A GPU of the given CDNA version whose TP group has (or lacks) AITER's custom
    all-reduce; the once-only logs start afresh."""
    monkeypatch.setattr("vllm.platforms.rocm.get_cdna_version", lambda: cdna)
    monkeypatch.setattr(amd_sp, "_aiter_comm", lambda: object() if comm else None)
    _print_info_once.cache_clear()
    _print_warning_once.cache_clear()


@pytest.mark.parametrize(
    "quick_flag, ipc_flag, int4_served, ipc_served, expected",
    [
        pytest.param(False, False, False, True, EXACT_PAIR, id="no_flags"),
        pytest.param(False, True, False, True, IPC_PAIR, id="ipc_flag"),
        pytest.param(False, True, False, False, EXACT_PAIR, id="ipc_flag_unserved"),
        pytest.param(True, False, True, True, INT4_PAIR, id="quick_flag"),
        pytest.param(True, True, True, True, INT4_PAIR, id="quick_over_ipc"),
        pytest.param(True, False, False, True, IPC_PAIR, id="quick_without_int4"),
        # the pair's own chain then runs RCCL below the gate
        pytest.param(True, False, True, False, INT4_PAIR, id="quick_without_ipc"),
        pytest.param(True, False, False, False, EXACT_PAIR, id="quick_nothing_served"),
    ],
)
def test_decoder_layer_sp_collectives(
    monkeypatch, quick_flag, ipc_flag, int4_served, ipc_served, expected
):
    _platform(monkeypatch, comm=ipc_served)
    monkeypatch.setattr(
        common_sp, "use_quick_sp", lambda config: quick_flag and int4_served
    )
    config = _parallel_only(
        enable_hc_sp_quick_reduce=quick_flag, enable_hc_sp_aiter_ipc=ipc_flag
    )

    assert amd_sp.decoder_layer_sp_collectives(config, hc_sp=True) == expected


def test_decoder_layer_sp_collectives_ask_no_opt_in_without_hc_sp(monkeypatch):
    monkeypatch.setattr(
        common_sp, "use_quick_sp", mock.Mock(side_effect=AssertionError)
    )
    monkeypatch.setattr(
        amd_sp, "use_sp_aiter_ipc", mock.Mock(side_effect=AssertionError)
    )
    config = _parallel_only(enable_hc_sp_quick_reduce=True, enable_hc_sp_aiter_ipc=True)

    assert amd_sp.decoder_layer_sp_collectives(config, hc_sp=False) == EXACT_PAIR


@pytest.mark.parametrize(
    "flag, below_gate, message",
    [
        pytest.param(
            "enable_hc_sp_aiter_ipc",
            False,
            "for the Qwen4Exp decoder-layer sequence-parallel collectives.",
            id="ipc_flag",
        ),
        pytest.param(
            "enable_hc_sp_quick_reduce",
            True,
            "for the Qwen4Exp decoder-layer sequence-parallel collectives below the "
            "INT4 size gate.",
            id="quick_flag",
        ),
    ],
)
def test_use_sp_aiter_ipc_serves_either_flag(
    monkeypatch, caplog_vllm, flag, below_gate, message
):
    _platform(monkeypatch)
    config = _parallel_only(**{flag: True})

    assert amd_sp.use_sp_aiter_ipc(config, below_int4_gate=below_gate) is True
    assert amd_sp.use_sp_aiter_ipc(config, below_int4_gate=below_gate) is True

    # the e2e flag gate counts the shared prefix; once per process
    assert (
        caplog_vllm.text.count(
            "Using AITER IPC reduce-scatter and all-gather " + message
        )
        == 1
    )
    assert "needs gfx950" not in caplog_vllm.text


@pytest.mark.parametrize(
    "flag, below_gate, cdna, comm, message",
    [
        pytest.param(
            "enable_hc_sp_aiter_ipc",
            False,
            3,
            True,
            "--enable-hc-sp-aiter-ipc needs gfx950",
            id="ipc_flag_gfx942",
        ),
        pytest.param(
            "enable_hc_sp_aiter_ipc",
            False,
            4,
            False,
            "--enable-hc-sp-aiter-ipc needs gfx950",
            id="ipc_flag_no_communicator",
        ),
        pytest.param(
            "enable_hc_sp_quick_reduce",
            True,
            4,
            False,
            "--enable-hc-sp-quick-reduce needs gfx950",
            id="quick_flag_no_communicator",
        ),
    ],
)
def test_use_sp_aiter_ipc_warns_once_when_it_cannot_serve(
    monkeypatch, caplog_vllm, flag, below_gate, cdna, comm, message
):
    _platform(monkeypatch, cdna=cdna, comm=comm)
    config = _parallel_only(**{flag: True})

    assert amd_sp.use_sp_aiter_ipc(config, below_int4_gate=below_gate) is False
    assert amd_sp.use_sp_aiter_ipc(config, below_int4_gate=below_gate) is False

    assert caplog_vllm.text.count(message) == 1
    assert caplog_vllm.text.count("VLLM_ROCM_USE_AITER_CUSTOM_AR") == 1
    assert caplog_vllm.text.count("below the INT4 size gate stay on RCCL") == int(
        below_gate
    )
    assert "Using AITER IPC" not in caplog_vllm.text


@pytest.mark.parametrize(
    "quick_flag, expect_gate_wording",
    [
        pytest.param(True, True, id="int4_served"),
        pytest.param(False, False, id="ipc_only"),
    ],
)
def test_decoder_layer_sp_collectives_name_the_int4_gate_only_with_int4(
    monkeypatch, caplog_vllm, quick_flag, expect_gate_wording
):
    _platform(monkeypatch)
    monkeypatch.setattr(common_sp, "use_quick_sp", lambda config: quick_flag)
    config = _parallel_only(
        enable_hc_sp_quick_reduce=quick_flag, enable_hc_sp_aiter_ipc=not quick_flag
    )

    amd_sp.decoder_layer_sp_collectives(config, hc_sp=True)

    assert ("below the INT4 size gate" in caplog_vllm.text) is expect_gate_wording


def test_use_sp_aiter_ipc_is_off_without_a_flag(monkeypatch, caplog_vllm):
    _platform(monkeypatch)

    assert amd_sp.use_sp_aiter_ipc(_parallel_only()) is False
    assert (
        "AITER IPC" not in caplog_vllm.text and "needs gfx950" not in caplog_vllm.text
    )
