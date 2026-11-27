# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact prefill selection, workspace reuse, and capacity recovery on SM100."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100)
    or current_platform.get_cuda_runtime_major() < 13,
    reason="DoSA kernels require SM100 and CUDA 13",
)
TOPK, CANARY = 2048, -77


def _owner(heads, rows, keys, dtype=torch.bfloat16, alternate=False):
    from vllm.model_executor.layers.dosa import DoSAWorkspace

    device = torch.accelerator.current_device_index()
    owner = DoSAWorkspace(
        torch.device("cuda", device),
        max_queries=rows,
        max_keys=keys,
        heads=heads,
        logits_dtype=dtype,
    )
    prepare_device = (device + int(alternate)) % torch.accelerator.device_count()
    with torch.accelerator.device_index(prepare_device):
        owner.prepare()
    return owner


def _inputs(heads, rows, keys, seed, dtype, *, hot=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    opts = dict(device="cuda", generator=g)
    weights = torch.full((rows, heads), -0.125, dtype=dtype, device="cuda")
    idx = torch.arange(keys, device="cuda", dtype=torch.int32)
    if hot:
        q = torch.zeros((rows, heads, 128), device="cuda")
        q[:, 0] = 2 * torch.randint(0, 2, (128,), **opts) - 1
        k = 2 * torch.randint(0, 2, (keys, 128), dtype=torch.int8, **opts) - 1
        weights[:, 0] = 0.125
        exponent = 126 + (idx >= keys // 2).to(torch.int32)
    else:
        q = torch.randint(-2, 3, (rows, heads, 128), dtype=torch.float32, **opts)
        k = torch.randint(-2, 3, (keys, 128), dtype=torch.float32, **opts)
        weights[:, :14] = 0.125
        exponent = 125 + idx % 5
    packed = exponent * 0x01010101
    ends = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
    q, k = q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn)
    return q, k, packed, weights, torch.zeros_like(ends), ends


def _output(rows, padding=17):
    storage = torch.full(
        (rows + 2, TOPK + padding), CANARY, device="cuda", dtype=torch.int32
    )
    return storage, storage[1:-1, :TOPK]


def _check(data, storage, output, dtype, *, probes=None):
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, mqa_logits_mx_inputs

    q, k, packed, weights, starts, ends = data
    if probes is not None:
        index = torch.tensor(probes, device=q.device)
        q = q.view(torch.uint8)[index].view(torch.float8_e4m3fn)
        weights, starts, ends, output = (
            x[index] for x in (weights, starts, ends, output)
        )
    q_scale, weights = mqa_logits_mx_inputs(q, None, weights, dtype)
    scores = fp8_fp4_mqa_logits(
        (q, q_scale), (k, packed), weights, starts, ends, len(k), logits_dtype=dtype
    ).float()
    assert torch.all((output >= 0) & (output < ends[:, None]))
    assert torch.all(output.sort(dim=1).values.diff(dim=1) > 0)
    valid = torch.arange(len(k), device=k.device) < ends[:, None]
    selected = scores.gather(1, output.long()).sort(descending=True).values
    expected = scores.masked_fill(~valid, -torch.inf).topk(TOPK).values
    assert torch.equal(selected, expected)
    assert torch.all(storage[0] == CANARY) and torch.all(storage[-1] == CANARY)
    assert torch.all(storage[1:-1, TOPK:] == CANARY)


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dosa_reuses_scratch_across_streams(heads, dtype):
    owner = _owner(heads, 128, 196608, dtype, alternate=True)
    cases = [
        _inputs(heads, q, k, seed, dtype)
        for q, k, seed in ((16, 131073, 41), (11, 196601, 92))
    ]
    # The second call introduces both an inadmissible row and a mixed MX scale.
    cases[1][3][0].zero_()
    if dtype == torch.float32:
        cases[1][2][-1] = 0x817F7E80 - (1 << 32)
    caller = torch.cuda.current_stream()
    streams = [torch.cuda.Stream() for _ in cases]
    outputs = [_output(len(case[0])) for case in cases]
    for case, stream, (_, ids) in zip(cases, streams, outputs):
        with torch.cuda.stream(stream):
            stream.wait_stream(caller)
            assert owner.run(*case, ids, common_end=len(case[1]) - len(case[0]) + 1)
    for case, stream, (storage, ids) in zip(cases, streams, outputs):
        stream.synchronize()
        _check(case, storage, ids, dtype)
    assert owner.error_original[0].item() & 1024


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dosa_hot_keeps_native_scores_across_slabs(dtype):
    owner = _owner(32, 2048, 530432, dtype)
    rows, keys = owner.rows + 17, 530432
    data = _inputs(32, rows, keys, 947, dtype, hot=True)
    mixed_key = 12288 + 1024
    if dtype == torch.float32:
        data[2][mixed_key] = 0x77767574
    storage, ids = _output(rows)
    assert owner.run(*data, ids, common_end=keys - rows + 1)
    assert owner._layout_active.item() == 1
    assert 12288 < owner._hot_end.item() < owner._second_end.item()
    if dtype == torch.float32:
        physical = owner.pair_rank[mixed_key].item()
        assert owner.fp32.packed_scales[physical].item() == 0x77767574
        assert owner.rho[physical].item() == 255
    _check(data, storage, ids, dtype, probes=[0, owner.rows - 1, owner.rows, rows - 1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dosa_tight_compaction_fits_partial_pages(monkeypatch, dtype):
    from vllm.model_executor.layers import dosa

    budget = (
        "CANDIDATE_SCRATCH_BYTES"
        if dtype == torch.float32
        else "BF16_PACKED_CANDIDATE_SCRATCH_BYTES"
    )
    page_bytes = 512 if dtype == torch.float32 else dosa.PACKED_ID_PAGE_BYTES
    monkeypatch.setattr(dosa, budget, 32 * page_bytes)
    owner = _owner(32, 128, 1048576, dtype)
    data = _inputs(32, 16, dosa.SEED + 1, 117, dtype)
    data[0].zero_()
    data[3].fill_(-0.125)
    data[3][:, 0] = 0.125
    data[-1].fill_(dosa.SEED + 1)
    storage, ids = _output(16)
    assert owner.run(*data, ids, common_end=dosa.SEED, causal=False)
    assert not owner.error_original[:16].any()
    assert owner.pool_head.item() == owner.rows // 4
    _check(data, storage, ids, dtype)


@pytest.mark.parametrize(
    "heads,dtype", [(32, torch.bfloat16), (64, torch.bfloat16), (32, torch.float32)]
)
def test_short_prefill_keeps_native_scores(heads, dtype):
    from vllm.model_executor.layers.sparse_attn_indexer import _short_native_prefill

    if dtype == torch.bfloat16 and not hasattr(torch.ops.deep_select, "topk"):
        pytest.skip("DeepSelect is unavailable")
    q, k, packed, weights, starts, ends = data = _inputs(heads, 16, 131072, 93, dtype)
    storage, ids = _output(len(q), 8 if dtype == torch.bfloat16 else 0)
    _short_native_prefill(q, None, k, packed, weights, starts, ends, len(k), ids, dtype)
    _check(data, storage, ids, dtype)
