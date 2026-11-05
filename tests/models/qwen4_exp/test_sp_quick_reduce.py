# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-GPU tests of the INT4 QuickReduce sequence-parallel collectives.

`--enable-hc-sp-quick-reduce` routes the decoder layer's four SP collectives
through AITER's FlyDSL reduce-scatter and all-gather. The kernels are lossy:
the checks bound the error of the INT4 codec and require that the quantized
path was really taken, because an exact fallback would pass any bound.
"""

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("INT4 QuickReduce is ROCm-only", allow_module_level=True)

from tests.utils import multi_gpu_test
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.utils import replace
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.quick_all_reduce import QuickAllReduce
from vllm.forward_context import set_forward_context
from vllm.models.common.ops import sequence_parallel as common_sp
from vllm.models.qwen4_exp.amd.ops import sequence_parallel as amd_sp
from vllm.utils.network_utils import get_open_port
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

from .test_tpsp import _make_config, _make_decoder


def _aiter_has_sp_quick_reduce() -> bool:
    try:
        from aiter.ops.flydsl import QuickAllReduceInt4
    except ImportError:
        return False
    return all(
        hasattr(QuickAllReduceInt4, name)
        for name in ("reduce_scatter", "all_gather", "compile_sp")
    )


pytestmark = pytest.mark.skipif(
    not _aiter_has_sp_quick_reduce(),
    reason="needs an AITER whose FlyDSL QuickAllReduceInt4 has reduce_scatter "
    "and all_gather",
)


HIDDEN = 2560  # the width the FlyDSL kernels were validated at
# Normalized MSE against the exact fp32 sum: 6e-3 for normal data and 1e-2 for
# heavy tails were measured on the FlyDSL codec. Exact collectives give about
# 1e-6 (bf16 rounding only), so the floor tells the INT4 path from a fallback.
NMSE_FLOOR = 1e-3
NMSE_BOUND = 1.5e-2
# The same ratio per row: a row lost at the padding seam or taken from the wrong
# block gives 1.0, which the all-rows NMSE hides (6 lost rows of 1,030 add 6e-3).
ROW_NMSE_BOUND = 0.1


def _nmse(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.float(), ref.float()
    return float(((got - ref) ** 2).sum() / (ref**2).sum())


def _max_row_nmse(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.float(), ref.float()
    energy = (ref**2).sum(1).clamp_min(1e-12)  # an odd tensor's padded row is zeros
    return float((((got - ref) ** 2).sum(1) / energy).max())


def _partials(rows: int, rank: int, seed: int = 100) -> torch.Tensor:
    """Per-rank partial sums with distinct scales, so a rank mix-up shows."""
    generator = torch.Generator(device="cuda").manual_seed(seed + rank)
    x = torch.randn(
        rows, HIDDEN, dtype=torch.float32, device="cuda", generator=generator
    )
    return (x * (1.0 + 0.5 * rank)).to(torch.bfloat16)


def _require_quick_reduce() -> QuickAllReduce:
    communicator = get_tp_group().device_communicator
    assert isinstance(communicator, CudaCommunicator)
    qr_comm = communicator.qr_comm
    assert (
        qr_comm is not None and not qr_comm.disabled and qr_comm.sp_quick_reduce_enabled
    ), (
        "FlyDSL sequence-parallel QuickReduce is not active: needs "
        "VLLM_ROCM_USE_AITER=1, VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4, "
        "--enable-hc-sp-quick-reduce and an AITER with reduce_scatter/all_gather"
    )
    return qr_comm


def _assert_reduce_scatter(rank: int, rows: int) -> None:
    x = _partials(rows, rank)
    padded = rows + rows % 2  # the odd tensors are padded to the group size first
    chunk = padded // 2
    reference = tensor_model_parallel_all_reduce(x.float())
    reference = torch.nn.functional.pad(reference, (0, 0, 0, padded - rows))
    out = amd_sp.sp_reduce_scatter_quantized(x)
    assert out.shape == (chunk, HIDDEN), (rows, out.shape)
    block = reference[rank * chunk : (rank + 1) * chunk]
    error = _nmse(out, block)
    assert NMSE_FLOOR < error < NMSE_BOUND, (rows, error)
    assert _max_row_nmse(out, block) < ROW_NMSE_BOUND, rows


def _assert_all_gather(rank: int, chunk: int) -> None:
    shard = _partials(chunk, rank, seed=200)
    exact = common_sp.sp_all_gather(shard)
    out = amd_sp.sp_all_gather_quantized(shard)
    assert out.shape == (2 * chunk, HIDDEN), (chunk, out.shape)
    error = _nmse(out, exact)
    assert NMSE_FLOOR < error < NMSE_BOUND, (chunk, error)
    assert _max_row_nmse(out, exact) < ROW_NMSE_BOUND, chunk
    # The codec runs on every block, the rank's own included, so all ranks hold
    # the same tensor.
    gathered = tensor_model_parallel_all_gather(out.contiguous(), dim=0)
    assert torch.equal(gathered[: out.shape[0]], gathered[out.shape[0] :])


def _check_reduce_scatter(rank: int, config: VllmConfig) -> None:
    _require_quick_reduce()
    # A block must be a multiple of 16 rows at this width (80 KiB: the kernels
    # need whole 32 KiB tiles over both blocks). 2060 and 2063 (padded to 2064)
    # leave 1030 and 1032 rows per rank and take the padding path; 2048 and 2080
    # need none, and 2080 is 325 tiles: an odd count, so half-tile blocks.
    for rows in (2060, 2048, 2063, 2080):
        _assert_reduce_scatter(rank, rows)


def _check_all_gather(rank: int, config: VllmConfig) -> None:
    _require_quick_reduce()
    for chunk in (1030, 1024, 1040):  # padded, whole tiles, odd tile count
        _assert_all_gather(rank, chunk)


def _check_small_blocks(rank: int, config: VllmConfig) -> None:
    """With the minimum size lowered: full tensors of 5, 15 and 25 tiles (5
    tiles per 32 rows at this width), whose blocks are 2.5, 7.5 and 12.5 tiles."""
    _require_quick_reduce()
    for rows in (32, 96, 160):
        _assert_reduce_scatter(rank, rows)
    for chunk in (16, 48, 80):
        _assert_all_gather(rank, chunk)


def _check_exact_fallbacks(rank: int, config: VllmConfig) -> None:
    """Shapes and dtypes the kernels do not take give the exact result."""
    _require_quick_reduce()
    small = _partials(128, rank)  # 0.65 MB full tensor, below the 8 MB minimum
    assert torch.equal(
        amd_sp.sp_reduce_scatter_quantized(small), common_sp.sp_reduce_scatter(small)
    )
    shard = small[:64].contiguous()
    assert torch.equal(
        amd_sp.sp_all_gather_quantized(shard), common_sp.sp_all_gather(shard)
    )
    half = _partials(2060, rank).half()
    assert torch.equal(
        amd_sp.sp_reduce_scatter_quantized(half), common_sp.sp_reduce_scatter(half)
    )
    half_shard = half[:1030].contiguous()
    assert torch.equal(
        amd_sp.sp_all_gather_quantized(half_shard), common_sp.sp_all_gather(half_shard)
    )


def _check_compiled_ops(rank: int, config: VllmConfig) -> None:
    """The opaque ops decide per call, so one trace serves every token count."""
    _require_quick_reduce()

    def collectives(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gathered = amd_sp.sp_all_gather_quantized(amd_sp.sp_shard(x) * 2)
        reduced = amd_sp.sp_all_gather_quantized(amd_sp.sp_reduce_scatter_quantized(x))
        return gathered[: x.shape[0]], reduced[: x.shape[0]]

    compiled = torch.compile(
        collectives,
        fullgraph=True,
        options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
    )
    traced = _partials(2100, rank)
    torch._dynamo.mark_dynamic(traced, 0)
    compiled(traced)
    # 2063 and 2100 take the INT4 kernels; 600 rows are below the minimum size
    # and take the exact path inside the same compiled graph.
    for rows in (2063, 600, 2100):
        x = _partials(rows, rank, seed=300 + rows)
        for got, want in zip(compiled(x), collectives(x)):
            torch.testing.assert_close(got, want, atol=0, rtol=0)


def _forward_layer(layer, config: VllmConfig) -> None:
    num_tokens = 5
    torch.manual_seed(30)
    hidden = torch.randn(num_tokens, 256)
    with (
        set_current_vllm_config(config),
        set_forward_context(None, config, num_tokens=num_tokens),
    ):
        layer(
            common_sp.sp_shard(hidden),
            None,
            None,
            torch.arange(num_tokens),
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
        )


def _check_layer_sites(rank: int, config: VllmConfig) -> None:
    """A layer built with the flag on uses the quantized pair at its four sites;
    with the flag off it keeps the exact pair."""
    # A second layer in one config would register its layer names twice.
    exact_config = _make_config(rank, hc_sp=True)  # the flag is off by default
    exact_config.model_config.hf_text_config.ple_layer_ids = []
    exact_layer = _make_decoder(exact_config, rank)
    assert exact_layer.sp_all_gather is common_sp.sp_all_gather
    assert exact_layer.sp_reduce_scatter is amd_sp.sp_reduce_scatter
    config.model_config.hf_text_config.ple_layer_ids = []
    layer = _make_decoder(config, rank)
    assert layer.sp_all_gather is amd_sp.sp_all_gather_quantized
    assert layer.sp_reduce_scatter is amd_sp.sp_reduce_scatter_quantized

    calls: list[str] = []

    def record(name, collective):
        def wrapper(x):
            calls.append(name)
            return collective(x)

        return wrapper

    layer.sp_all_gather = record("all_gather", layer.sp_all_gather)
    layer.sp_reduce_scatter = record("reduce_scatter", layer.sp_reduce_scatter)
    _forward_layer(layer, config)
    # all-gather and reduce-scatter around the attention block, then around the MoE
    assert calls == ["all_gather", "reduce_scatter", "all_gather", "reduce_scatter"]


def _check_layer_launches(rank: int, config: VllmConfig) -> None:
    """With the minimum size lowered, the layer's four sites reach the FlyDSL
    kernels: the tiny layer tensors would otherwise stay on the exact path."""
    kernels = _require_quick_reduce()._flydsl_int4
    config.model_config.hf_text_config.ple_layer_ids = []
    layer = _make_decoder(config, rank)
    launches: list[str] = []

    def spy(name):
        kernel = getattr(kernels, name)

        def wrapper(*args, **kwargs):
            launches.append(name)
            return kernel(*args, **kwargs)

        setattr(kernels, name, wrapper)

    spy("reduce_scatter")
    spy("all_gather")
    _forward_layer(layer, config)
    assert launches == ["all_gather", "reduce_scatter", "all_gather", "reduce_scatter"]


def _run(rank: int, port: int, check, flags: dict[str, bool] | None = None) -> None:
    """Initialize rank-local runtime state and run one check, with the parallel
    flags `flags` (default: `--enable-hc-sp-quick-reduce`)."""
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    config = _make_config(rank, hc_sp=True)
    config.parallel_config = replace(
        config.parallel_config, **(flags or {"enable_hc_sp_quick_reduce": True})
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
def sp_quick_reduce_env(monkeypatch):
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", "INT4")


@pytest.mark.parametrize(
    "check",
    [
        _check_reduce_scatter,
        _check_all_gather,
        _check_exact_fallbacks,
        _check_compiled_ops,
        _check_layer_sites,
    ],
    ids=lambda check: check.__name__.removeprefix("_check_"),
)
@multi_gpu_test(num_gpus=2)
def test_sp_quick_reduce(sp_quick_reduce_env, check) -> None:
    mp.spawn(_run, args=(get_open_port(), check), nprocs=2)


@pytest.mark.parametrize(
    "check",
    [_check_layer_launches, _check_small_blocks],
    ids=lambda check: check.__name__.removeprefix("_check_"),
)
@multi_gpu_test(num_gpus=2)
def test_sp_quick_reduce_small_sizes(sp_quick_reduce_env, monkeypatch, check) -> None:
    """Checks that need the minimum size lowered, which the others rely on."""
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", "0")
    mp.spawn(_run, args=(get_open_port(), check), nprocs=2)
