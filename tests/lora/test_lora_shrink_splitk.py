# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the LoRA shrink deterministic split-K path.

The shrink op reads VLLM_LORA_DETERMINISTIC_SPLIT_K and VLLM_BATCH_INVARIANT
once at import, so every case runs in a fresh interpreter whose environment
is set before start-up.
"""

import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="LoRA shrink kernel requires CUDA"
)

S8 = {
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_LORA_DETERMINISTIC_SPLIT_K": "8",
    "VLLM_LORA_ENABLE_DUAL_STREAM": "0",
}
S1 = {**S8, "VLLM_LORA_DETERMINISTIC_SPLIT_K": "0"}
BI_OFF = {**S1, "VLLM_BATCH_INVARIANT": "0"}
DEVICE = "cuda:0"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(autouse=True)
def cleanup_fixture():
    """GPU work runs in subprocesses; skip conftest teardown (needs ray)."""
    yield


@pytest.fixture(autouse=True)
def dynamo_reset():
    yield


def _subprocess(env: dict[str, str], code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, **env},
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _run_case(env: dict[str, str], case: str, *args) -> None:
    code = f"import runpy; runpy.run_path({__file__!r})[{case!r}](*{args!r})"
    proc = _subprocess(env, code)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _lora_data(m, k, rank, nslices, num_loras, dtype, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    inputs = torch.randn((m, k), dtype=dtype, device=DEVICE, generator=g)
    weights = [
        torch.randn((num_loras, rank, k), dtype=dtype, device=DEVICE, generator=g)
        for _ in range(nslices)
    ]
    mapping = torch.randint(
        0, num_loras, (m,), dtype=torch.int32, device=DEVICE, generator=g
    )
    return inputs, weights, mapping


def _meta(mapping, max_loras, meta=None):
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

    if meta is None:
        meta = LoRAKernelMeta.make(max_loras, mapping.numel(), device=DEVICE)
    meta.prepare_tensors(mapping)
    return meta


def _new_out(weights, m, fill=0.0):
    shape = (len(weights), m, weights[0].size(1))
    return torch.full(shape, fill, dtype=torch.float32, device=DEVICE)


def _shrink(inputs, weights, meta, out, scaling):
    from vllm.lora.ops.triton_ops import lora_shrink
    from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT

    _LORA_A_PTR_DICT.clear()
    args = meta.meta_args(token_nums=inputs.size(0), specialize_active_lora=False)
    lora_shrink(inputs, weights, out, *args, scaling)
    return out


def _assert_matches_fp64(out, inputs, weights, mapping, scaling):
    x, ids = inputs.double().cpu(), mapping.cpu()
    ref = torch.zeros(out.shape, dtype=torch.float64)
    for s, w in enumerate(weights):
        for lid in ids[ids >= 0].unique().tolist():
            rows = ids == lid
            ref[s, rows] = scaling * (x[rows] @ w[lid].double().cpu().T)
    out = out.double().cpu()
    assert torch.isfinite(out).all()
    assert torch.all(out[:, ids < 0] == 0)
    torch.testing.assert_close(out, ref, rtol=0.005, atol=0.005)


class _LaunchRecorder:
    def __init__(self, name, kernel, sink):
        self.name, self.kernel, self.sink = name, kernel, sink

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def record(*args, **kwargs):
            bound = dict(zip(self.kernel.arg_names, args))
            self.sink.append((self.name, {**bound, **kwargs}))
            return launch(*args, **kwargs)

        return record


def _case_numerics(k, rank, nslices, dtype, scaling):
    dtype = getattr(torch, dtype)
    inputs, weights, mapping = _lora_data(17, k, rank, nslices, 2, dtype, seed=0)
    out = _shrink(inputs, weights, _meta(mapping, 2), _new_out(weights, 17), scaling)
    _assert_matches_fp64(out, inputs, weights, mapping, scaling)


def _case_dispatch(split_k):
    import vllm.lora.ops.triton_ops.lora_shrink_op as op

    launches: list = []
    for name in ("_lora_shrink_kernel", "_lora_shrink_reduce_kernel"):
        setattr(op, name, _LaunchRecorder(name, getattr(op, name), launches))

    inputs, weights, mapping = _lora_data(16, 2049, 32, 1, 4, torch.bfloat16, 1)
    out = _shrink(inputs, weights, _meta(mapping, 4), _new_out(weights, 16), 0.5)
    _assert_matches_fp64(out, inputs, weights, mapping, 0.5)

    names = [name for name, _ in launches]
    if not split_k:
        assert names == ["_lora_shrink_kernel"], names
        assert not launches[0][1].get("STORE_PARTIALS", False)
        return
    assert names == ["_lora_shrink_kernel", "_lora_shrink_reduce_kernel"], names
    (_, first), (_, reduce) = launches
    assert first["STORE_PARTIALS"] and first["out_ptr"].dtype == torch.float32
    launch = dict(BLOCK_M=32, BLOCK_N=16, SPLIT_K=8, num_warps=4, num_stages=2)
    assert {key: first[key] for key in launch} == launch
    assert (first["BLOCK_K"], first["num_ctas"]) == (256, 1)
    assert {key: reduce[key] for key in launch} == launch


def _case_batch_invariance(nslices):
    from vllm.lora.ops.triton_ops import lora_expand
    from vllm.lora.ops.triton_ops.utils import _LORA_B_PTR_DICT

    # K=2560 is ten BK=256 blocks, so every one of the eight splits is used.
    hidden, rank, num_loras, scaling = 2560, 16, 3, 0.7
    g = torch.Generator(device=DEVICE).manual_seed(1234 + nslices)

    def randn(*shape):
        return torch.randn(shape, dtype=torch.bfloat16, device=DEVICE, generator=g)

    lora_a = [randn(num_loras, rank, hidden) for _ in range(nslices)]
    lora_b = [randn(num_loras, hidden, rank) for _ in range(nslices)]
    target, target_residual = randn(hidden), randn(hidden * nslices)

    def run(m, pos, others):
        ids = torch.tensor(others[:pos] + [0] + others[pos:], device=DEVICE)
        inputs, residual = randn(m, hidden), randn(m, hidden * nslices)
        inputs[pos], residual[pos] = target, target_residual
        meta = _meta(ids.to(torch.int32), num_loras)
        shrunk = _shrink(inputs, lora_a, meta, _new_out(lora_a, m), scaling)
        _LORA_B_PTR_DICT.clear()
        args = meta.meta_args(token_nums=m, specialize_active_lora=False)
        lora_expand(shrunk, lora_b, residual, *args, add_inputs=True)
        return shrunk[:, pos].clone(), residual[pos].clone()

    ref_shrunk, ref_expanded = run(1, 0, [])
    for m, pos, others in [  # (M, target row, adapters of the other rows)
        (127, 0, [0] * 126),
        (127, 126, [1, 2] * 63),
        (128, 0, [1] * 127),
        (129, 128, [2] * 128),
        (31, 15, [0, 1, 2] * 10),
        (32, 16, [0, 1, 2] * 10 + [1]),
        (33, 16, [0, 1, 2] * 10 + [1, 2]),
        (64, 40, ([0, 1, 2] * 21)[:63]),
    ]:
        for _ in range(20):
            shrunk, expanded = run(m, pos, others)
            assert torch.equal(shrunk, ref_shrunk), (m, pos)
            assert torch.equal(expanded, ref_expanded), (m, pos)


def _case_empty_and_metadata_swap():
    rank, scaling = 8, 1.0
    inputs, weights, _ = _lora_data(16, 128, rank, 1, 2, torch.bfloat16, seed=3)

    no_rows = torch.empty((0,), dtype=torch.int32, device=DEVICE)
    _shrink(inputs[:0], weights, _meta(no_rows, 2), _new_out(weights, 0), scaling)

    untouched = _new_out(weights, 16, fill=-1.0)
    no_lora = torch.full((16,), -1, dtype=torch.int32, device=DEVICE)
    _shrink(inputs, weights, _meta(no_lora, 2), untouched, scaling)
    assert torch.all(untouched == -1.0)

    mappings = [
        torch.tensor(ids * 2, dtype=torch.int32, device=DEVICE)
        for ids in ([0, -1, 1, -1, 0, 1, -1, 0], [0, 0, -1, -1] * 2, [-1, 1] * 4)
    ]
    out, meta, first = _new_out(weights, 16, fill=-1.0), None, None
    for mapping in mappings + mappings[:1]:
        meta = _meta(mapping, 2, meta)
        _shrink(inputs, weights, meta, out, scaling)
        _assert_matches_fp64(out, inputs, weights, mapping, scaling)
        first = out.clone() if first is None else first
    assert torch.equal(out, first)


def _case_cuda_graph_replay():
    hidden, m, scaling, bf16 = 2560, 96, 0.6, torch.bfloat16
    inputs, weights, mapping = _lora_data(m, hidden, 16, 1, 3, bf16, 5)

    def eager(x, ids):
        return _shrink(x, weights, _meta(ids, 3), _new_out(weights, x.size(0)), scaling)

    reference = eager(inputs, mapping)
    static_inputs, static_out = inputs.clone(), _new_out(weights, m)
    meta = _meta(mapping, 3)
    stream = torch.cuda.Stream(device=DEVICE)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _shrink(static_inputs, weights, meta, static_out, scaling)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _shrink(static_inputs, weights, meta, static_out, scaling)
    graph.replay()
    assert torch.equal(static_out, reference)

    # A larger eager call between replays must not disturb the graph's scratch.
    with torch.cuda.stream(stream):
        large_inputs, _, large_mapping = _lora_data(256, hidden, 16, 1, 3, bf16, 6)
        eager(large_inputs, large_mapping)
    torch.cuda.current_stream().wait_stream(stream)

    new_inputs, _, new_mapping = _lora_data(m, hidden, 16, 1, 3, bf16, 7)
    for x, ids, expected in (
        (new_inputs, new_mapping, eager(new_inputs, new_mapping)),
        (inputs, mapping, reference),
    ):
        static_inputs.copy_(x)
        _meta(ids, 3, meta)
        graph.replay()
        assert torch.equal(static_out, expected)


def _case_concurrent_streams():
    # Same lane on two streams, as MoE shared-experts overlap schedules it.
    aux_stream = torch.cuda.Stream(device=DEVICE)
    for i in range(50):
        calls = [
            _lora_data(32, 2560, 16, 1, 2, torch.bfloat16, seed=2 * i + j)
            for j in range(2)
        ]
        refs = [
            _shrink(x, w, _meta(ids, 2), _new_out(w, 32), 0.9) for x, w, ids in calls
        ]
        metas = [_meta(ids, 2) for _, _, ids in calls]
        outs = [_new_out(w, 32) for _, w, _ in calls]
        inputs_ready = torch.cuda.Event()
        inputs_ready.record()
        (x0, w0, _), (x1, w1, _) = calls
        _shrink(x0, w0, metas[0], outs[0], 0.9)
        with torch.cuda.stream(aux_stream):
            inputs_ready.wait(aux_stream)
            _shrink(x1, w1, metas[1], outs[1], 0.9)
            aux_done = torch.cuda.Event()
            aux_done.record(aux_stream)
        aux_done.wait(torch.cuda.current_stream())
        assert torch.equal(outs[0], refs[0]) and torch.equal(outs[1], refs[1]), i


@pytest.mark.parametrize(
    "k,rank,nslices,dtype,scaling",
    [
        (2560, 8, 3, "bfloat16", 0.73),
        (9728, 16, 1, "float16", 1.0),
        (4097, 64, 1, "bfloat16", 0.5),
        (1024, 8, 1, "float16", 1.0),
    ],
    ids=["K2560_r8_3slices", "K9728_r16", "K4097_r64_tail", "K1024_empty_split"],
)
def test_splitk_matches_fp64_reference(k, rank, nslices, dtype, scaling):
    _run_case(S8, "_case_numerics", k, rank, nslices, dtype, scaling)


@pytest.mark.parametrize(
    "env,split_k", [(S8, 8), (S1, 0), (BI_OFF, 0)], ids=["S8", "BI_S1", "BI_off"]
)
def test_shrink_dispatch(env, split_k):
    _run_case(env, "_case_dispatch", split_k)


@pytest.mark.parametrize("nslices", [1, 3])
def test_splitk_batch_invariance(nslices):
    _run_case(S8, "_case_batch_invariance", nslices)


@pytest.mark.parametrize(
    "case",
    [
        "_case_empty_and_metadata_swap",
        "_case_cuda_graph_replay",
        "_case_concurrent_streams",
    ],
)
def test_splitk(case):
    _run_case(S8, case)


@pytest.mark.parametrize(
    "env,message",
    [
        ({**S8, "VLLM_BATCH_INVARIANT": "0"}, "VLLM_BATCH_INVARIANT"),
        ({**S8, "VLLM_LORA_ENABLE_DUAL_STREAM": "1"}, "VLLM_LORA_ENABLE_DUAL_STREAM"),
    ],
    ids=["requires_batch_invariant", "rejects_dual_stream"],
)
def test_splitk_rejects_unsupported_env(env, message):
    proc = _subprocess(env, "import vllm.lora.ops.triton_ops.lora_shrink_op")
    assert proc.returncode != 0 and message in proc.stderr, proc.stderr
