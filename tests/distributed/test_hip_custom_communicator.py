# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import ray
import torch
import torch.distributed as dist

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.hip_custom_communicator import (
    AllReduceAddAttnResRmsNormConfig,
    AllReduceAddRmsNormConfig,
    AllReduceConfig,
    HipCustomCommunicator,
)
from vllm.distributed.parallel_state import get_tp_group, graph_capture
from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="HipCustomCommunicator is built for gfx950 only",
)

HIDDEN = 7168
NUM_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 512, 2048]
MAX_TOKENS = 2048
DTYPES = [torch.bfloat16, torch.float16]
EPS = 1e-5
ATTN_RES_EPS = 1e-6
NUM_BLOCKS = 4
INT4_RELATIVE_RMS = 0.02
INT4_ROW_RELATIVE_RMS = 0.1
TP_SIZES = [2, 4, 8]
REFUSED_AT_SIZE = (
    "staging_too_small",
    "partials_too_small",
    "unfenced_too_small",
    "grid_not_resident",
    "launch_out_of_range",
)
QUANTIZATIONS = ["exact", "int4"]


def _on_ranks(monkeypatch, tp_size, quantization, body):
    """Run `body(comm, group, device, rank)` on every rank of a TP group."""
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    @ray.remote(num_gpus=1, max_calls=1)
    def ranks(monkeypatch, tp_size, pp_size, rank, port):
        with monkeypatch.context() as m:
            for var in [
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
            ]:
                m.delenv(var, raising=False)
            device = torch.device(f"cuda:{rank}")
            torch.accelerator.set_device_index(device)
            init_test_distributed_environment(tp_size, pp_size, rank, port)
            ensure_model_parallel_initialized(tp_size, pp_size)
            group = get_tp_group()
            dist.all_reduce(torch.zeros(1, device=device), group=group.device_group)
            torch.accelerator.synchronize()
            comm = HipCustomCommunicator(
                group.cpu_group,
                group.device_group,
                device,
                max_rows=MAX_TOKENS,
                max_cols=HIDDEN,
                quantization=quantization,
            )
            assert not comm.disabled, comm.lifecycle
            try:
                body(comm, group, device, rank)
            finally:
                comm.close()

    multi_process_parallel(monkeypatch, tp_size, 1, ranks)


def _int4_sum(xs, config):
    """The sum an INT4 config computes: each rank's input through the format,
    summed in fp32, and the sum through it again where a two-shot gathers it."""

    def through_int4(x):
        groups = x.float().reshape(x.shape[0], -1, 32)
        largest = groups.abs().amax(dim=-1, keepdim=True)
        negative = (groups == -largest).any(dim=-1, keepdim=True)
        extreme = torch.where(negative, -largest, largest)
        scale = (extreme / -8).clamp(-65504, 65504).half().float()
        inverse = torch.where(scale != 0, 1 / scale, torch.zeros_like(scale))
        levels = torch.round(groups * inverse + 8).clamp(0, 15)
        return (scale * (levels - 8)).reshape(x.shape)

    summed = sum(through_int4(x) for x in xs)
    if config.algorithm.startswith("two_shot"):
        summed = through_int4(summed)
    return summed


def _compare(got, want, config, where, exact=False):
    """For an INT4 config, close to the format's own answer, overall and in each
    row; otherwise equal, or close."""
    if config.quantization == "int4":
        diff = (got.float() - want.float()).reshape(got.shape[0], -1)
        ref = want.float().reshape(got.shape[0], -1)
        rms = diff.square().mean().sqrt() / ref.square().mean().sqrt()
        row_rms = diff.square().mean(-1).sqrt() / ref.square().mean(-1).sqrt()
        assert rms <= INT4_RELATIVE_RMS, (where, config, rms.item())
        worst = row_rms.nan_to_num(0.0).max()
        assert worst <= INT4_ROW_RELATIVE_RMS, (where, config, worst.item())
    elif exact:
        assert torch.equal(got, want), (where, config)
    else:
        torch.testing.assert_close(
            got, want, atol=2e-2, rtol=2e-2, msg=str((where, config))
        )


@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("quantization", QUANTIZATIONS)
def test_all_reduce_correctness(monkeypatch, tp_size, quantization):
    def body(comm, group, device, rank):
        for rows in NUM_TOKENS:
            for dtype in DTYPES:
                xs = [
                    torch.randint(
                        -8,
                        8,
                        (rows, HIDDEN),
                        generator=torch.Generator(device=device).manual_seed(r),
                        device=device,
                    ).to(dtype)
                    for r in range(comm.arguments.world)
                ]
                x = xs[rank]
                summed = x.clone()
                dist.all_reduce(summed, group=group.device_group)
                assert comm.should_allreduce(x)

                q = comm.arguments.quantization
                forced = [
                    comm.kernel_config("all_reduce", a, rows, HIDDEN, q)
                    for a in AllReduceConfig.algorithms
                ]
                for config in [None, *(c for c in forced if c is not None)]:
                    try:
                        got = comm.all_reduce(x, config=config)
                    except RuntimeError as e:
                        if config is None or not str(e).startswith(REFUSED_AT_SIZE):
                            raise
                        continue
                    used = config or comm.tuned("all_reduce", rows, HIDDEN, None)
                    if used.quantization == "int4":
                        want = _int4_sum(xs, used).to(dtype)
                    else:
                        want = summed
                    _compare(got, want, used, (rows, dtype), exact=True)

    _on_ranks(monkeypatch, tp_size, quantization, body)


@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("quantization", QUANTIZATIONS)
def test_all_reduce_add_rms_norm_correctness(monkeypatch, tp_size, quantization):
    def body(comm, group, device, rank):
        from vllm.ir.ops import fused_add_rms_norm, rms_norm

        for rows in NUM_TOKENS:
            for dtype in DTYPES:
                for with_residual in [False, True]:
                    xs = [
                        torch.randint(
                            -8,
                            8,
                            (rows, HIDDEN),
                            generator=torch.Generator(device=device).manual_seed(r),
                            device=device,
                        ).to(dtype)
                        for r in range(comm.arguments.world)
                    ]
                    x = xs[rank]
                    g = torch.Generator(device=device).manual_seed(0)
                    residual = torch.randint(
                        -8, 8, (rows, HIDDEN), generator=g, device=device
                    ).to(dtype)
                    weight = torch.empty(HIDDEN, dtype=dtype, device=device)
                    weight.normal_(mean=1.0, std=0.1, generator=g)

                    exact_sum = x.clone()
                    dist.all_reduce(exact_sum, group=group.device_group)
                    r = residual if with_residual else None
                    assert comm.should_allreduce_add_rms_norm(x, weight, r)

                    q = comm.arguments.quantization
                    forced = [
                        comm.kernel_config(
                            "all_reduce_add_rms_norm", a, rows, HIDDEN, q
                        )
                        for a in AllReduceAddRmsNormConfig.algorithms
                    ]
                    for config in [None, *(c for c in forced if c is not None)]:
                        got = comm.all_reduce_add_rms_norm(
                            x, weight, EPS, residual=r, config=config
                        )
                        got = got if with_residual else (got,)
                        used = config or comm.tuned(
                            "all_reduce_add_rms_norm", rows, HIDDEN, with_residual
                        )
                        if used.quantization == "int4":
                            summed = _int4_sum(xs, used).to(dtype)
                        else:
                            summed = exact_sum
                        if with_residual:
                            want = fused_add_rms_norm(
                                summed, residual.clone(), weight, EPS
                            )
                        else:
                            want = (rms_norm(summed, weight, EPS),)
                        for g_, w_ in zip(got, want, strict=True):
                            _compare(g_, w_, used, (rows, dtype, with_residual))

    _on_ranks(monkeypatch, tp_size, quantization, body)


@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("quantization", QUANTIZATIONS)
def test_all_reduce_add_attn_res_rms_norm_correctness(
    monkeypatch, tp_size, quantization
):
    def body(comm, group, device, rank):
        from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

        dtype = torch.bfloat16
        for rows in NUM_TOKENS:
            for with_prefix, write_index in [(True, -1), (False, NUM_BLOCKS)]:
                xs = [
                    torch.randint(
                        -8,
                        8,
                        (rows, HIDDEN),
                        generator=torch.Generator(device=device).manual_seed(r),
                        device=device,
                    ).to(dtype)
                    for r in range(comm.arguments.world)
                ]
                x = xs[rank]
                g = torch.Generator(device=device).manual_seed(0)
                prefix = (
                    torch.randint(-8, 8, (rows, HIDDEN), generator=g, device=device).to(
                        dtype
                    )
                    if with_prefix
                    else None
                )
                blocks = torch.randint(
                    -8, 8, (rows, 10, HIDDEN), generator=g, device=device
                ).to(dtype)
                norm_weight = torch.empty(HIDDEN, dtype=dtype, device=device)
                norm_weight.normal_(mean=1.0, std=0.1, generator=g)
                qk_weight = torch.empty(HIDDEN, dtype=dtype, device=device)
                qk_weight.normal_(mean=0.0, std=0.1, generator=g)
                out_weight = torch.empty(HIDDEN, dtype=dtype, device=device)
                out_weight.normal_(mean=1.0, std=0.1, generator=g)

                exact_sum = x.clone()
                dist.all_reduce(exact_sum, group=group.device_group)
                assert comm.should_allreduce_add_attn_res_rms_norm(
                    x,
                    prefix,
                    blocks,
                    norm_weight,
                    qk_weight,
                    out_weight,
                    NUM_BLOCKS,
                    write_index,
                )

                q = comm.arguments.quantization
                forced = [
                    comm.kernel_config(
                        "all_reduce_add_attn_res_rms_norm", a, rows, HIDDEN, q
                    )
                    for a in AllReduceAddAttnResRmsNormConfig.algorithms
                ]
                for config in [None, *(c for c in forced if c is not None)]:
                    got_blocks = blocks.clone()
                    got_prefix, got_out = comm.all_reduce_add_attn_res_rms_norm(
                        x,
                        prefix,
                        got_blocks,
                        norm_weight,
                        qk_weight,
                        out_weight,
                        NUM_BLOCKS,
                        write_index,
                        ATTN_RES_EPS,
                        EPS,
                        config=config,
                    )
                    used = config or comm.tuned(
                        "all_reduce_add_attn_res_rms_norm", rows, HIDDEN, None
                    )
                    if used.quantization == "int4":
                        summed = _int4_sum(xs, used).to(dtype)
                    else:
                        summed = exact_sum
                    want_prefix = (
                        prefix.clone() if prefix is not None else summed.clone()
                    )
                    want_blocks = blocks.clone()
                    want_out = attn_res(
                        want_prefix,
                        summed if prefix is not None else None,
                        want_blocks,
                        norm_weight,
                        qk_weight,
                        out_weight,
                        NUM_BLOCKS,
                        write_index,
                        ATTN_RES_EPS,
                        EPS,
                    )
                    where = (rows, with_prefix)
                    _compare(got_out, want_out, used, where)
                    _compare(got_prefix, want_prefix, used, where)
                    _compare(got_blocks, want_blocks, used, where)

    _on_ranks(monkeypatch, tp_size, quantization, body)


@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("quantization", QUANTIZATIONS)
def test_hip_custom_communicator_capture(monkeypatch, tp_size, quantization):
    """Each op captured in a CUDA graph replays to the right answer on new input."""

    def body(comm, group, device, rank):
        from vllm.ir.ops import fused_add_rms_norm, rms_norm
        from vllm.models.kimi_k3.amd.ops.attn_res import attn_res

        for rows in NUM_TOKENS:
            for dtype in DTYPES:
                attn = dtype == torch.bfloat16
                xs = [
                    torch.randint(
                        -8,
                        8,
                        (rows, HIDDEN),
                        generator=torch.Generator(device=device).manual_seed(r),
                        device=device,
                    ).to(dtype)
                    for r in range(comm.arguments.world)
                ]
                x = xs[rank]
                g = torch.Generator(device=device).manual_seed(0)
                residual = torch.randint(
                    -8, 8, (rows, HIDDEN), generator=g, device=device
                ).to(dtype)
                prefix = torch.randint(
                    -8, 8, (rows, HIDDEN), generator=g, device=device
                ).to(dtype)
                blocks = torch.randint(
                    -8, 8, (rows, 10, HIDDEN), generator=g, device=device
                ).to(dtype)
                write_blocks = blocks.clone()
                weight = torch.empty(HIDDEN, dtype=dtype, device=device)
                weight.normal_(mean=1.0, std=0.1, generator=g)
                qk_weight = torch.empty(HIDDEN, dtype=dtype, device=device)
                qk_weight.normal_(mean=0.0, std=0.1, generator=g)
                attn_args = (weight, qk_weight, weight, NUM_BLOCKS)

                graph = torch.cuda.CUDAGraph()
                with comm.capture():
                    comm.all_reduce(x)
                    comm.all_reduce_add_rms_norm(x, weight, EPS)
                    comm.all_reduce_add_rms_norm(x, weight, EPS, residual=residual)
                    if attn:
                        comm.all_reduce_add_attn_res_rms_norm(
                            x, prefix, blocks, *attn_args, -1, ATTN_RES_EPS, EPS
                        )
                        comm.all_reduce_add_attn_res_rms_norm(
                            x,
                            None,
                            write_blocks,
                            *attn_args,
                            NUM_BLOCKS,
                            ATTN_RES_EPS,
                            EPS,
                        )
                    with torch.cuda.graph(graph):
                        reduced = comm.all_reduce(x)
                        normed = comm.all_reduce_add_rms_norm(x, weight, EPS)
                        added, added_residual = comm.all_reduce_add_rms_norm(
                            x, weight, EPS, residual=residual
                        )
                        if attn:
                            attn_prefix, attn_out = (
                                comm.all_reduce_add_attn_res_rms_norm(
                                    x,
                                    prefix,
                                    blocks,
                                    *attn_args,
                                    -1,
                                    ATTN_RES_EPS,
                                    EPS,
                                )
                            )
                            write_prefix, write_out = (
                                comm.all_reduce_add_attn_res_rms_norm(
                                    x,
                                    None,
                                    write_blocks,
                                    *attn_args,
                                    NUM_BLOCKS,
                                    ATTN_RES_EPS,
                                    EPS,
                                )
                            )

                later = [t + 1 for t in xs]
                x.copy_(later[rank])
                exact_sum = x.clone()
                dist.all_reduce(exact_sum, group=group.device_group)
                ar = comm.tuned("all_reduce", rows, HIDDEN, None)
                norm = comm.tuned("all_reduce_add_rms_norm", rows, HIDDEN, False)
                add = comm.tuned("all_reduce_add_rms_norm", rows, HIDDEN, True)
                ra = comm.tuned("all_reduce_add_attn_res_rms_norm", rows, HIDDEN, None)
                ar_sum, norm_sum, add_sum, ra_sum = (
                    _int4_sum(later, c).to(dtype)
                    if c.quantization == "int4"
                    else exact_sum
                    for c in (ar, norm, add, ra)
                )
                want_normed = rms_norm(norm_sum, weight, EPS)
                want_added, want_residual = fused_add_rms_norm(
                    add_sum, residual.clone(), weight, EPS
                )
                if attn:
                    want_prefix = prefix.clone()
                    want_attn = attn_res(
                        want_prefix,
                        ra_sum,
                        blocks.clone(),
                        *attn_args,
                        -1,
                        ATTN_RES_EPS,
                        EPS,
                    )
                    want_write_prefix = ra_sum.clone()
                    want_write_blocks = write_blocks.clone()
                    want_write = attn_res(
                        want_write_prefix,
                        None,
                        want_write_blocks,
                        *attn_args,
                        NUM_BLOCKS,
                        ATTN_RES_EPS,
                        EPS,
                    )
                graph.replay()
                torch.accelerator.synchronize()

                _compare(reduced, ar_sum, ar, (rows, dtype), exact=True)
                _compare(normed, want_normed, norm, (rows, dtype))
                _compare(added, want_added, add, (rows, dtype))
                _compare(added_residual, want_residual, add, (rows, dtype))
                if attn:
                    _compare(attn_out, want_attn, ra, rows)
                    _compare(attn_prefix, want_prefix, ra, rows)
                    _compare(write_out, want_write, ra, rows)
                    _compare(write_prefix, want_write_prefix, ra, rows)
                    _compare(write_blocks, want_write_blocks, ra, rows)

    _on_ranks(monkeypatch, tp_size, quantization, body)


@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("quantization", QUANTIZATIONS)
def test_hip_custom_communicator_independence(monkeypatch, tp_size, quantization):
    """Outputs must not depend on what their memory held before."""

    def body(comm, group, device, rank):
        dtype = torch.bfloat16
        failures = []
        for cols in [HIDDEN, HIDDEN // 2]:
            for rows in [8, 256, MAX_TOKENS]:
                g = torch.Generator(device=device).manual_seed(rank)
                x = torch.randn((rows, cols), generator=g, device=device)
                g.manual_seed(0)
                x[:, torch.randperm(cols, generator=g, device=device)[:8]] *= 50
                x = x.to(dtype)
                residual = torch.randn((rows, cols), generator=g, device=device)
                residual = residual.to(dtype)
                prefix = torch.randn((rows, cols), generator=g, device=device) * 4
                prefix = prefix.to(dtype)
                blocks = torch.randn((rows, 10, cols), generator=g, device=device)
                blocks = blocks.to(dtype)
                weight = torch.empty(cols, dtype=dtype, device=device)
                weight.normal_(mean=1.0, std=0.1, generator=g)
                qk_weight = torch.empty(cols, dtype=dtype, device=device)
                qk_weight.normal_(mean=0.0, std=0.1, generator=g)

                for variant, op, algorithms in [
                    ("all_reduce", "all_reduce", AllReduceConfig.algorithms),
                    (
                        "rms_norm",
                        "all_reduce_add_rms_norm",
                        AllReduceAddRmsNormConfig.algorithms,
                    ),
                    (
                        "add_rms_norm",
                        "all_reduce_add_rms_norm",
                        AllReduceAddRmsNormConfig.algorithms,
                    ),
                    (
                        "attn_res",
                        "all_reduce_add_attn_res_rms_norm",
                        AllReduceAddAttnResRmsNormConfig.algorithms,
                    ),
                    (
                        "attn_res_block_write",
                        "all_reduce_add_attn_res_rms_norm",
                        AllReduceAddAttnResRmsNormConfig.algorithms,
                    ),
                ]:
                    q = comm.arguments.quantization
                    forced = [
                        comm.kernel_config(op, a, rows, cols, q) for a in algorithms
                    ]
                    for config in [None, *(c for c in forced if c is not None)]:
                        answers = []
                        for fill in [float("nan"), 1e4]:
                            poison = [torch.full_like(x, fill) for _ in range(4)]
                            del poison
                            try:
                                if variant == "all_reduce":
                                    got = [comm.all_reduce(x, config=config)]
                                elif variant == "rms_norm":
                                    got = [
                                        comm.all_reduce_add_rms_norm(
                                            x, weight, EPS, config=config
                                        )
                                    ]
                                elif variant == "add_rms_norm":
                                    got = list(
                                        comm.all_reduce_add_rms_norm(
                                            x,
                                            weight,
                                            EPS,
                                            residual=residual,
                                            config=config,
                                        )
                                    )
                                else:
                                    with_prefix = variant == "attn_res"
                                    written = blocks.clone()
                                    got = [
                                        written,
                                        *comm.all_reduce_add_attn_res_rms_norm(
                                            x,
                                            prefix if with_prefix else None,
                                            written,
                                            weight,
                                            qk_weight,
                                            weight,
                                            NUM_BLOCKS,
                                            -1 if with_prefix else NUM_BLOCKS,
                                            ATTN_RES_EPS,
                                            EPS,
                                            config=config,
                                        ),
                                    ]
                            except RuntimeError as e:
                                if not str(e).startswith(REFUSED_AT_SIZE):
                                    raise
                                break
                            answers.append([t.clone() for t in got])
                        torch.accelerator.synchronize()
                        where = (variant, config, rows, cols)
                        if len(answers) == 2:
                            for nan_filled, finite_filled in zip(*answers):
                                if not torch.isfinite(nan_filled).all():
                                    failures.append(("not finite", *where))
                                elif not torch.equal(nan_filled, finite_filled):
                                    failures.append(("differs", *where))
        assert not failures, "\n".join(map(str, failures))

    _on_ranks(monkeypatch, tp_size, quantization, body)


def test_hip_custom_communicator_refusal(monkeypatch):
    def body(comm, group, device, rank):
        past = torch.ones(16 * MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
        assert not comm.should_allreduce(past)

        three_d = torch.ones(2, 4, HIDDEN, dtype=torch.bfloat16, device=device)
        with pytest.raises(RuntimeError, match="^not_two_d:"):
            comm.all_reduce(three_d)

        strided = torch.ones(4, 2 * HIDDEN, dtype=torch.bfloat16, device=device)
        x = strided[:, :HIDDEN]
        if comm.should_allreduce(x):
            want = x.clone()
            dist.all_reduce(want, group=group.device_group)
            assert torch.equal(comm.all_reduce(x), want)

        x = torch.ones(4, HIDDEN, dtype=torch.bfloat16, device=device)
        want = x.clone()
        dist.all_reduce(want, group=group.device_group)
        graph = torch.cuda.CUDAGraph()
        with (
            pytest.raises(RuntimeError, match="^ranks_disagree:"),
            comm.capture(),
            torch.cuda.graph(graph),
        ):
            for _ in range(rank + 1):
                comm.all_reduce(x)
        assert not comm.disabled
        with pytest.raises(RuntimeError, match="^broken: "):
            comm.should_allreduce(x)
        with pytest.raises(RuntimeError, match="^broken: "):
            comm.all_reduce(x)
        comm.close()
        again = HipCustomCommunicator(
            group.cpu_group,
            group.device_group,
            device,
            max_rows=MAX_TOKENS,
            max_cols=HIDDEN,
            quantization="exact",
        )
        try:
            assert torch.equal(again.all_reduce(x), want)
        finally:
            again.close()

    _on_ranks(monkeypatch, 2, "exact", body)


@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("quantization", QUANTIZATIONS)
def test_hip_custom_communicator_routing(monkeypatch, tp_size, quantization):
    """The tuned table picks a config for every op at both ends of the arena."""

    def body(comm, group, device, rank):
        for op, residual in [
            ("all_reduce", None),
            ("all_reduce_add_rms_norm", False),
            ("all_reduce_add_rms_norm", True),
            ("all_reduce_add_attn_res_rms_norm", None),
        ]:
            for rows in [1, MAX_TOKENS]:
                assert comm.tuned(op, rows, HIDDEN, residual) is not None, (op, rows)

    _on_ranks(monkeypatch, tp_size, quantization, body)


def test_tensor_model_parallel_all_reduce_routing(monkeypatch):
    """VLLM's TP all-reduce runs ours when it is on, eagerly and in graph_capture."""
    if torch.accelerator.device_count() < 2:
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_USE_HIP_CUSTOM_COMMUNICATOR", "1")
    monkeypatch.setenv("VLLM_ROCM_HIP_CUSTOM_COMMUNICATOR_QUANTIZATION", "NONE")

    @ray.remote(num_gpus=1, max_calls=1)
    def ranks(monkeypatch, tp_size, pp_size, rank, port):
        import tempfile

        from transformers import LlamaConfig

        from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config

        with monkeypatch.context() as m, tempfile.TemporaryDirectory() as model:
            for var in [
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
            ]:
                m.delenv(var, raising=False)
            # vLLM sizes the communicator from the served model, as in serving.
            LlamaConfig(
                hidden_size=HIDDEN,
                intermediate_size=512,
                num_hidden_layers=1,
                num_attention_heads=56,
                num_key_value_heads=8,
                vocab_size=256,
            ).save_pretrained(model)
            served = VllmConfig(
                model_config=ModelConfig(
                    model=model, skip_tokenizer_init=True, dtype="bfloat16"
                )
            )
            device = torch.device(f"cuda:{rank}")
            torch.accelerator.set_device_index(device)
            with set_current_vllm_config(served):
                init_test_distributed_environment(tp_size, pp_size, rank, port)
                ensure_model_parallel_initialized(tp_size, pp_size)
            group = get_tp_group()
            ours = group.device_communicator.hip_custom_communicator
            assert ours is not None and not ours.disabled

            x = torch.randint(
                -8,
                8,
                (16, HIDDEN),
                generator=torch.Generator(device=device).manual_seed(rank),
                device=device,
            ).to(torch.bfloat16)
            want = x.clone()
            dist.all_reduce(want, group=group.device_group)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as profile:
                got = tensor_model_parallel_all_reduce(x)
                torch.accelerator.synchronize()
            assert torch.equal(got, want)
            kernels = [e.name for e in profile.events() if e.device_type.name == "CUDA"]
            assert any("hip_custom_communicator" in k for k in kernels), kernels

            graph = torch.cuda.CUDAGraph()
            with graph_capture(device=device) as context:
                assert ours.lifecycle.status == "capturing"
                with torch.cuda.graph(graph, stream=context.stream):
                    out = tensor_model_parallel_all_reduce(x)
            assert ours.lifecycle.status == "open"
            x.add_(1)
            want = x.clone()
            dist.all_reduce(want, group=group.device_group)
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.equal(out, want)

    multi_process_parallel(monkeypatch, 2, 1, ranks)
