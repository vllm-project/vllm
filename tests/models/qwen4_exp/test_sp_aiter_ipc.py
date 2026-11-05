# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-GPU tests of the Qwen4Exp exact IPC sequence-parallel collectives.

With `--enable-hc-sp-aiter-ipc` the decoder layer's four SP collectives go through
AITER's IPC reduce-scatter and all-gather instead of RCCL, and with
`--enable-hc-sp-quick-reduce` as well they serve the shapes below the INT4 size
gate. The results are exact either way, so the checks assert that the AITER path
was really taken (a spy on its entry points, including during CUDA graph
capture): an RCCL fallback would pass any numerical bound. The kernels of a
replayed graph are not checked here: a torch profile around ``graph.replay()``
records no event in this test process; a server trace is the evidence for them.
"""

import itertools
from contextlib import ExitStack
from unittest import mock

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("AITER IPC collectives are ROCm-only", allow_module_level=True)

from tests.utils import multi_gpu_test
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.config.utils import replace
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.device_communicators.aiter_custom_all_reduce import (
    AiterCustomAllreduce,
)
from vllm.distributed.parallel_state import graph_capture
from vllm.forward_context import set_forward_context
from vllm.models.common.ops import sequence_parallel as common_sp
from vllm.models.qwen4_exp.amd import model as amd_model
from vllm.models.qwen4_exp.amd.ops import sequence_parallel as amd_sp
from vllm.utils.network_utils import get_open_port
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

from .test_sp_quick_reduce import (
    NMSE_BOUND,
    NMSE_FLOOR,
    _aiter_has_sp_quick_reduce,
    _forward_layer,
    _require_quick_reduce,
)
from .test_tpsp import _make_config, _make_decoder

HIDDEN = 2560  # the width Qwen4Exp runs at
# Exact collectives differ from the fp32 sum by bf16 rounding only (about 3e-6).
NMSE_BOUND_EXACT = 1e-5


def _nmse(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.float(), ref.float()
    return float(((got - ref) ** 2).sum() / (ref**2).sum())


def _partials(rows: int, rank: int, seed: int = 100) -> torch.Tensor:
    """Per-rank partial sums with distinct scales, so a rank mix-up shows."""
    generator = torch.Generator(device="cuda").manual_seed(seed + rank)
    x = torch.randn(
        rows, HIDDEN, dtype=torch.float32, device="cuda", generator=generator
    )
    return (x * (1.0 + 0.5 * rank)).to(torch.bfloat16)


def _require_aiter() -> AiterCustomAllreduce:
    comm = rocm_aiter_ops.get_aiter_allreduce()
    assert comm is not None and not comm.disabled, (
        "the TP group has no enabled AITER custom all-reduce: needs "
        "VLLM_ROCM_USE_AITER=1 and its custom all-reduce"
    )
    return comm


class _Spy:
    """Counts the calls that reach AITER's reduce-scatter and all-gather."""

    def __init__(self) -> None:
        self.calls = {"reduce_scatter": 0, "all_gather": 0}

    def __enter__(self):
        self._patches = []
        for name, attr in (
            ("reduce_scatter", "custom_reduce_scatter"),
            ("all_gather", "custom_all_gather"),
        ):
            original = getattr(AiterCustomAllreduce, attr)

            def wrapper(this, *args, _name=name, _original=original, **kwargs):
                self.calls[_name] += 1
                return _original(this, *args, **kwargs)

            patch = mock.patch.object(AiterCustomAllreduce, attr, wrapper)
            patch.start()
            self._patches.append(patch)
        return self

    def __exit__(self, *exc):
        for patch in self._patches:
            patch.stop()


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
        with _Spy() as spy:
            out = amd_sp.sp_reduce_scatter_ipc(x)
        assert spy.calls == {"reduce_scatter": 1, "all_gather": 0}, (rows, spy.calls)
        block = _reference_block(x, rank)
        assert out.shape == block.shape, (rows, out.shape)
        assert _nmse(out, block) < NMSE_BOUND_EXACT, rows
        with _Spy() as spy:
            gathered = amd_sp.sp_all_gather_ipc(out)
        assert spy.calls == {"reduce_scatter": 0, "all_gather": 1}, (rows, spy.calls)
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
        _Spy() as spy,
        graph_capture(device) as capture,
        torch.cuda.graph(graph, stream=capture.stream),
    ):
        reduced = amd_sp.sp_reduce_scatter_ipc(x)
        gathered = amd_sp.sp_all_gather_ipc(reduced)
    assert spy.calls == {"reduce_scatter": 1, "all_gather": 1}, spy.calls
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
        with ExitStack() as stack, _Spy() as spy:
            for patch in patched:
                stack.enter_context(patch)
            torch.testing.assert_close(
                amd_sp.sp_reduce_scatter_ipc(x), expected_rs, atol=0, rtol=0
            )
            torch.testing.assert_close(
                amd_sp.sp_all_gather_ipc(shard), expected_ag, atol=0, rtol=0
            )
        assert spy.calls == {"reduce_scatter": 0, "all_gather": 0}, spy.calls


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
        with _Spy() as spy:
            results = compiled(x)
        assert spy.calls == {"reduce_scatter": 1, "all_gather": 2}, (rows, spy.calls)
        for got, want in zip(results, collectives(x)):
            torch.testing.assert_close(got, want, atol=0, rtol=0)


def _fresh_decoder(rank: int):
    # A second layer in one config would register its layer names twice.
    config = _make_config(rank, hc_sp=True)
    config.model_config.hf_text_config.ple_layer_ids = []
    return _make_decoder(config, rank)


def _check_layer_sites(rank: int, config: VllmConfig) -> None:
    """Which pair a layer's four sites use. The flag the run was started with
    (IPC) selects the IPC pair; the default config, whose flags are off, keeps
    the common pair; each (quick, ipc) state picks its own pair."""
    layer = _fresh_decoder(rank)  # flags off by default
    assert layer.sp_all_gather is common_sp.sp_all_gather
    assert layer.sp_reduce_scatter is amd_sp.sp_reduce_scatter
    config.model_config.hf_text_config.ple_layer_ids = []
    layer = _make_decoder(config, rank)  # `--enable-hc-sp-aiter-ipc`
    assert layer.sp_all_gather is amd_sp.sp_all_gather_ipc
    assert layer.sp_reduce_scatter is amd_sp.sp_reduce_scatter_ipc
    for quick, ipc in itertools.product((False, True), repeat=2):
        with (
            mock.patch.object(amd_model, "use_quick_sp", return_value=quick),
            mock.patch.object(amd_model, "use_sp_aiter_ipc", return_value=ipc),
        ):
            layer = _fresh_decoder(rank)
        want = amd_sp.DECODER_LAYER_SP_COLLECTIVES[quick, ipc]
        assert (layer.sp_all_gather, layer.sp_reduce_scatter) == want, (quick, ipc)
    assert len(set(amd_sp.DECODER_LAYER_SP_COLLECTIVES.values())) == 4
    # Without HC SP (as at TP=1, where both flags are ignored) neither opt-in is
    # asked, so neither warns.
    with (
        mock.patch.object(
            amd_model, "is_hc_sequence_parallel_enabled", return_value=False
        ),
        mock.patch.object(amd_model, "use_quick_sp", side_effect=AssertionError),
        mock.patch.object(amd_model, "use_sp_aiter_ipc", side_effect=AssertionError),
    ):
        layer = _fresh_decoder(rank)
    want = amd_sp.DECODER_LAYER_SP_COLLECTIVES[False, False]
    assert (layer.sp_all_gather, layer.sp_reduce_scatter) == want


def _check_layer_forward(rank: int, config: VllmConfig) -> None:
    """A layer built with the flag reaches AITER at its four sites: two
    reduce-scatters and two all-gathers per forward."""
    _require_aiter()
    config.model_config.hf_text_config.ple_layer_ids = []
    layer = _make_decoder(config, rank)
    with _Spy() as spy:
        _forward_layer(layer, config)
    assert spy.calls == {"reduce_scatter": 2, "all_gather": 2}, spy.calls


class _FlydslSpy:
    """Records the FlyDSL INT4 reduce-scatter / all-gather launches, in order."""

    def __init__(self, kernels) -> None:
        self.kernels = kernels
        self.launches: list[str] = []

    def __enter__(self):
        self._patches = []
        for name in ("reduce_scatter", "all_gather"):
            original = getattr(self.kernels, name)

            def wrapper(*args, _name=name, _original=original, **kwargs):
                self.launches.append(_name)
                return _original(*args, **kwargs)

            patch = mock.patch.object(self.kernels, name, wrapper)
            patch.start()
            self._patches.append(patch)
        return self

    def __exit__(self, *exc):
        for patch in self._patches:
            patch.stop()


def _check_quantized_ipc_routes(rank: int, config: VllmConfig) -> None:
    """With both flags a pair of shapes below the INT4 size gate takes AITER's
    exact IPC collectives and above it the FlyDSL kernels, never both: the gate
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
        with _FlydslSpy(kernels) as flydsl, _Spy() as spy:
            reduced = amd_sp.sp_reduce_scatter_quantized_ipc(x)
            gathered = amd_sp.sp_all_gather_quantized_ipc(reduced)
        block = _reference_block(x, rank)
        if int4:
            assert flydsl.launches == ["reduce_scatter", "all_gather"], rows
            assert spy.calls == {"reduce_scatter": 0, "all_gather": 0}, rows
            assert NMSE_FLOOR < _nmse(reduced, block) < NMSE_BOUND, rows
        else:
            assert flydsl.launches == [], rows
            assert spy.calls == {"reduce_scatter": 1, "all_gather": 1}, rows
            assert _nmse(reduced, block) < NMSE_BOUND_EXACT, rows
            torch.testing.assert_close(
                gathered,
                tensor_model_parallel_all_gather(reduced, dim=0),
                atol=0,
                rtol=0,
            )


def _run(rank: int, port: int, check, quick_reduce: bool = False) -> None:
    """Initialize rank-local runtime state and run one check."""
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    config = _make_config(rank, hc_sp=True)
    config.parallel_config = replace(
        config.parallel_config,
        enable_hc_sp_aiter_ipc=True,
        enable_hc_sp_quick_reduce=quick_reduce,
    )
    init_distributed_environment(
        world_size=2,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        init_workspace_manager(torch.device(f"cuda:{rank}"))
        with set_current_vllm_config(config):
            initialize_model_parallel(tensor_model_parallel_size=2)
            with torch.device(f"cuda:{rank}"), torch.inference_mode():
                torch.set_default_dtype(torch.bfloat16)
                check(rank, config)
    finally:
        reset_workspace_manager()
        cleanup_dist_env_and_memory()


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
    mp.spawn(_run, args=(get_open_port(), check), nprocs=2)


@pytest.mark.skipif(
    not _aiter_has_sp_quick_reduce(),
    reason="needs an AITER whose FlyDSL QuickAllReduceInt4 has reduce_scatter "
    "and all_gather",
)
@multi_gpu_test(num_gpus=2)
def test_sp_quantized_ipc_routes(sp_aiter_ipc_env, monkeypatch) -> None:
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", "INT4")
    mp.spawn(_run, args=(get_open_port(), _check_quantized_ipc_routes, True), nprocs=2)
