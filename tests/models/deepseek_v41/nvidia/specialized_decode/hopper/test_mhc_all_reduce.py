# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The SM90 mHC boundary with the fused all-reduce, at TP8.

On H100 the boundaries of up to `SM90_MHC_ALL_REDUCE_MAX_TOKENS` rows
inside FULL graphs reduce the TP-partial sublayer output in `AllReduceMHC`
(the Lamport all-reduce fused with the mHC post-mix, collapse and RMSNorm) and
compute the next coefficients with the SM80 kernel on the side stream. Each
check runs one rank late, so a missing wait reads a peer's stale data:

* at 1, 6 and 12 rows, the streams bitwise against the all-reduce in rank
  order and the separate mHC post, the layer input within one BF16 ulp
  (the fused RMSNorm sums squares in another order), the coefficients and
  the stream mean bitwise against the SM80 kernels on those streams;
* a MoE output left unfinalized, bitwise as the plain call on its sum;
* one CUDA graph of the boundary, replayed bitwise.

Needs eight SM90 GPUs with NVLink multicast; about a minute.
"""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

TP = 8
HIDDEN, HC, TOP_K = 5120, 4, 6
LATE_CYCLES = 20_000
REPLAYS = 200


def _late(rank: int, step: int) -> None:
    """Delay one rank per step, a different one each step."""
    if rank == step % TP:
        torch.cuda._sleep(LATE_CYCLES)


def _boundary(rows: int):
    """The replicated inputs of one boundary, equal on every rank."""
    generator = torch.Generator(device="cuda").manual_seed(rows)

    def randn(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=generator, device="cuda").to(dtype)

    return dict(
        residual=randn(rows, HC, HIDDEN),
        post=torch.rand(rows, HC, 1, generator=generator, device="cuda"),
        comb=randn(rows, HC, HC, dtype=torch.float32) * 0.1,
        pre=torch.rand(rows, HC, generator=generator, device="cuda"),
        norm_weight=randn(HIDDEN),
        fn=randn(HC * (HC + 2), HC * HIDDEN, dtype=torch.float32) * 0.02,
        scale=torch.tensor([0.5, 0.25, 1.0], device="cuda"),
        base=randn(HC * (HC + 2), dtype=torch.float32),
    )


def _partial(rows: int, rank: int) -> torch.Tensor:
    """This rank's partial sum, with +0/-0 pairs that look like the Lamport
    sentinel."""
    generator = torch.Generator(device="cuda").manual_seed(97 * rank + rows)
    x = torch.randn(rows, HIDDEN, generator=generator, device="cuda").bfloat16()
    x[:, :16] = 0
    x[:, 9:16:2] = -0.0
    return x


def _moe_partial(rows: int, rank: int):
    """This rank's MoE output left unfinalized (route-major GEMM2 rows) and
    its finalized sum: one FP32 FMA per route in route order, the shared
    output added, one rounding."""
    from vllm.model_executor.layers.fused_moe.moe_output import (
        MoEOutput,
        route_major_moe_output,
    )

    generator = torch.Generator(device="cuda").manual_seed(7 + rank + 13 * rows)
    routes = torch.randn(rows, TOP_K, HIDDEN, generator=generator, device="cuda")
    routes = routes.bfloat16()
    weights = torch.rand(rows, TOP_K, generator=generator, device="cuda")
    shared = torch.randn(rows, HIDDEN, generator=generator, device="cuda").bfloat16()
    ids = torch.zeros(rows, TOP_K, dtype=torch.int32, device="cuda")
    acc = torch.zeros(rows, HIDDEN, device="cuda")
    for k in range(TOP_K):
        torch.addcmul(acc, routes[:, k].float(), weights[:, k : k + 1], out=acc)
    finalized = (acc + shared.float()).bfloat16()
    unfinalized = route_major_moe_output(routes, weights, ids, None)
    return MoEOutput(unfinalized, shared_output=shared), finalized


def _run(x, b, stream, capture_aux=False):
    from vllm.models.deepseek_v41.nvidia.ops import mhc

    return mhc.mhc_shifted_post_pre(
        x,
        b["residual"],
        b["post"],
        b["comb"],
        b["fn"],
        b["scale"],
        b["base"],
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        pre_mix=b["pre"],
        norm_weight=b["norm_weight"],
        norm_eps=1e-6,
        capture_aux=capture_aux,
        stream=stream,
        reduce_results=True,
    )


def _check(rank: int) -> None:
    from tests.distributed.test_custom_all_reduce import (
        _bf16_ulps,
        _unfused_all_reduce_mhc,
    )
    from vllm.distributed import get_tp_group
    from vllm.models.deepseek_v41.nvidia.ops import mhc
    from vllm.models.deepseek_v41.nvidia.specialized_decode.ampere.mhc import (
        mhc_coefficients,
    )

    side = torch.cuda.Stream()
    step = 0
    for rows in (1, 6, mhc.SM90_MHC_ALL_REDUCE_MAX_TOKENS):
        b = _boundary(rows)
        x = _partial(rows, rank)
        _late(rank, step := step + 1)
        streams, post, comb, layer_input, pre, aux = _run(x, b, side, True)
        torch.cuda.current_stream().wait_stream(side)

        peers = get_tp_group().all_gather(x, dim=0).view(TP, rows, HIDDEN)
        want_streams, want_input = _unfused_all_reduce_mhc(
            peers,
            b["residual"],
            b["post"].squeeze(-1),
            b["comb"],
            b["pre"],
            b["norm_weight"],
            1e-6,
        )
        assert torch.equal(streams, want_streams), rows
        assert _bf16_ulps(layer_input, want_input) <= 1, rows
        want_post, want_comb, want_pre = mhc_coefficients(
            None,
            want_streams,
            None,
            None,
            b["fn"],
            b["scale"],
            b["base"],
            1e-6,
            1e-6,
            1e-6,
            2.0,
            20,
        )
        assert torch.equal(post, want_post.unsqueeze(-1)), rows
        assert torch.equal(comb, want_comb), rows
        assert torch.equal(pre, want_pre), rows
        assert torch.equal(aux, want_streams.mean(dim=1)), rows

        # A MoE that left its top-k reduction open (route-major rows, as the
        # SM90 Marlin experts leave them) reduces as the plain call on the
        # finalized sum.
        moe_output, finalized = _moe_partial(rows, rank)
        _late(rank, step := step + 1)
        want = _run(finalized, b, side)
        torch.cuda.current_stream().wait_stream(side)
        _late(rank, step := step + 1)
        got = _run(moe_output, b, side)
        torch.cuda.current_stream().wait_stream(side)
        for g, w in zip(got, want):
            assert torch.equal(g, w), f"finalize, {rows} rows"

    # One graph of the boundary, replayed with one rank late each time.
    rows = 6
    b = _boundary(rows)
    x = _partial(rows, rank)
    want = _run(x, b, side)
    torch.cuda.current_stream().wait_stream(side)
    want = [t.clone() for t in want]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = _run(x, b, side)
        torch.cuda.current_stream().wait_stream(side)
    for replay in range(REPLAYS):
        _late(rank, replay)
        graph.replay()
        for g, w in zip(got, want):
            assert torch.equal(g, w), replay
    torch.accelerator.synchronize()


def _worker(rank: int, port: str, skip_path: str) -> None:
    from tests.utils import init_test_distributed_environment
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import cleanup_dist_env_and_memory, get_tp_group
    from vllm.models.deepseek_v41.nvidia.ops import mhc

    torch.accelerator.set_device_index(rank)
    with set_current_vllm_config(VllmConfig()), torch.inference_mode():
        init_test_distributed_environment(TP, 1, rank, port, local_rank=rank)
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=TP, enable_expert_parallel=False
            ),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    hidden_size=HIDDEN, hc_mult=HC, num_experts_per_tok=TOP_K
                )
            ),
        )
        supported = torch.tensor([int(mhc.supports_mhc_all_reduce(config))])
        torch.distributed.all_reduce(
            supported, torch.distributed.ReduceOp.MIN, get_tp_group().cpu_group
        )
        if not supported.item():
            if rank == 0:
                with open(skip_path, "w") as f:
                    f.write("no NVLink multicast for the fused all-reduce + mHC")
            cleanup_dist_env_and_memory()
            return
        mhc.init_mhc_all_reduce(config)
        mhc.prepare_ampere_mhc(HIDDEN)
        _check(rank)
        cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=8)
@pytest.mark.skipif(not current_platform.is_device_capability(90), reason="needs SM90")
def test_sm90_mhc_all_reduce():
    if torch.accelerator.device_count() < TP:
        pytest.skip("needs eight GPUs")
    from vllm.utils.network_utils import get_open_port

    with tempfile.TemporaryDirectory() as tmp:
        skip_path = os.path.join(tmp, "skipped")
        torch.multiprocessing.spawn(
            _worker, args=(str(get_open_port()), skip_path), nprocs=TP
        )
        if os.path.exists(skip_path):
            with open(skip_path) as f:
                pytest.skip(f.read())
