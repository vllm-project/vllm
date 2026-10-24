# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native-score, recovery and lifetime properties of the real DoSA owner."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability(100)
    or current_platform.get_cuda_runtime_major() < 13,
    reason="DoSA kernels require SM100 and CUDA 13",
)
TOPK = 2048
CANARY = -77


@pytest.fixture(scope="module", params=(32, 64))
def workspace(request):
    from vllm.model_executor.layers.dosa import DoSAWorkspace

    owner = DoSAWorkspace(
        torch.device("cuda", torch.accelerator.current_device_index()),
        max_queries=128,
        max_keys=196608,
        heads=request.param,
    )
    owner.prepare()
    return owner


def _inputs(heads, rows, keys, seed, positive_heads=16):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(
        -2,
        3,
        (rows, heads, 128),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    k = torch.randint(
        -2, 3, (keys, 128), device="cuda", dtype=torch.float32, generator=generator
    ).to(torch.float8_e4m3fn)
    weights = (
        torch.randint(1, 3, (rows, heads), device="cuda", generator=generator) / 8
    ).to(torch.bfloat16)
    positive = torch.where(
        torch.arange(rows, device="cuda") < 8, positive_heads, heads // 2 + 4
    )
    sign = torch.arange(heads, device="cuda")[None, :] < positive[:, None]
    weights = torch.where(sign, weights, -weights)
    exponent = torch.arange(keys, device="cuda", dtype=torch.int32) % 5 + 125
    packed = exponent.to(torch.uint8).repeat_interleave(4).view(torch.int32)
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
    return q, k, packed, weights, starts, ends


def _scores(q, k, weights, packed):
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, mqa_logits_mx_inputs

    starts = torch.zeros(len(q), device=q.device, dtype=torch.int32)
    ends = torch.full_like(starts, len(k))
    q_scale, native_weights = mqa_logits_mx_inputs(q, None, weights)
    return fp8_fp4_mqa_logits(
        (q, q_scale), (k, packed), native_weights, starts, ends, len(k)
    ).float()


def _output(rows):
    storage = torch.full(
        (rows + 2, TOPK + 17), CANARY, device="cuda", dtype=torch.int32
    )
    return storage, storage[1:-1, :TOPK]


def _check(scores, ends, storage, output):
    assert torch.all((output >= 0) & (output < ends[:, None]))
    assert torch.all(output.sort(dim=1).values.diff(dim=1) > 0)
    scores = scores.masked_fill(
        torch.arange(scores.shape[1], device="cuda")[None, :] >= ends[:, None],
        -torch.inf,
    )
    selected = scores.gather(1, output.long()).sort(dim=1, descending=True).values
    assert torch.equal(selected, scores.topk(TOPK, dim=1).values)
    assert torch.all(storage[0] == CANARY) and torch.all(storage[-1] == CANARY)
    assert torch.all(storage[1:-1, TOPK:] == CANARY)


def test_dosa_dirty_reuse_orders_streams_and_preserves_strided_output(workspace):
    """Independent requests may change Q/K while sharing only serialized scratch."""
    first = _inputs(workspace.heads, 16, 131072, 41)
    second = _inputs(workspace.heads, 11, 196601, 92, positive_heads=14)
    expected = [_scores(data[0], data[1], data[3], data[2]) for data in (first, second)]
    outputs = [_output(len(data[0])) for data in (first, second)]
    ready = torch.cuda.Event()
    ready.record()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    completed = [torch.cuda.Event(), torch.cuda.Event()]
    for data, (storage, output), stream, event in zip(
        (first, second), outputs, streams, completed
    ):
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            assert workspace.run(
                *data[:6], output, common_end=len(data[1]) - len(data[0]) + 1
            )
            event.record()
    for event in completed:
        event.synchronize()
    for scores, data, (storage, output) in zip(expected, (first, second), outputs):
        _check(scores, data[5], storage, output)
    storage, output = _output(16)
    assert workspace.run(*first[:6], output, common_end=131057)
    _check(expected[0], first[5], storage, output)


@pytest.mark.parametrize(
    "zero_weights", [False, True], ids=["record_overflow", "guard"]
)
def test_dosa_exact_recovery_of_sixteen_equal_score_rows(workspace, zero_weights):
    """Overflow and inadmissible certificates must both recover complete rows."""
    data = _inputs(workspace.heads, 16, 131072, 117)
    if zero_weights:
        data[3].zero_()
    else:
        data[0].zero_()
        data[3].fill_(-0.125)
        data[3][:, 0] = 0.125
    storage, output = _output(16)
    assert workspace.run(*data[:6], output, common_end=131057)
    error_bit = 1024 if zero_weights else 16
    assert torch.all((workspace.error_original[:16] & error_bit) != 0)
    scores = torch.zeros((16, 131072), device="cuda", dtype=torch.float32)
    _check(scores, data[5], storage, output)


def test_dosa_dense_scores_match_native_bf16_rounding(workspace):
    q = torch.zeros((4, workspace.heads, 128), device="cuda")
    q[:, 0, 0], q[:, 0, 1] = 1, 2.0**-4
    k = torch.zeros((128, 128), device="cuda")
    k[:, 0], k[:, 1] = 1, 2.0**-4
    q, k = q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn)
    weights = torch.zeros((4, workspace.heads), device="cuda", dtype=torch.bfloat16)
    weights[:, 0] = 1
    scales = torch.ones(128, device="cuda")
    packed = torch.full((128,), 0x7F7F7F7F, device="cuda", dtype=torch.int32)
    native = _scores(q, k, weights, packed)
    actual = torch.empty_like(native)
    score = getattr(torch.ops._C, f"dosa_h{workspace.heads}_sample_dense_score")
    score(q, k, scales, weights.float(), actual, workspace.sms)
    # The true dot is 1+2^-8; native rounds the head's BF16 midpoint to 1.
    assert torch.all(native == 1)
    assert torch.equal(actual.view(torch.int32), native.view(torch.int32))


def test_dosa_batch_offsets_and_fallback_before_writes(workspace):
    from vllm.model_executor.layers.sparse_attn_indexer import _dosa_prefill
    from vllm.v1.attention.backends.mla.indexer import build_prefill_chunk_metadata

    full = workspace.max_q
    first = _inputs(workspace.heads, full, 131072, 41)
    second = _inputs(workspace.heads, full, 131072, 92, positive_heads=14)
    q = torch.cat((first[0], second[0]))
    keys = torch.cat((first[1], second[1]))
    packed = torch.cat((first[2], second[2])).view(torch.uint8).reshape(-1, 4)
    weights = torch.cat((first[3], second[3])).float()
    queries_cpu = torch.tensor([0, full, 2 * full], dtype=torch.int32)
    lengths_cpu = torch.tensor([131072, 131072], dtype=torch.int32)
    lengths = lengths_cpu.to("cuda")
    chunk = build_prefill_chunk_metadata(
        start_idx=0,
        end_idx=2,
        query_start_loc=queries_cpu.to("cuda"),
        query_start_loc_cpu=queries_cpu,
        uncompressed_seq_lens=lengths,
        compressed_seq_lens=lengths,
        compressed_seq_lens_cpu=lengths_cpu,
        block_table=torch.arange(4096, device="cuda", dtype=torch.int32).reshape(
            2, 2048
        ),
        compress_ratio=1,
        collect_request_ranges=True,
    )
    assert chunk is not None
    storage, output = _output(2 * full)
    assert not _dosa_prefill(
        workspace, chunk, 0, 2 * full - 1, q, keys, packed, weights, output
    )
    torch.accelerator.synchronize()
    assert torch.all(storage == CANARY)
    assert _dosa_prefill(
        workspace, chunk, 0, 2 * full, q, keys, packed, weights, output
    )
    for index, data in enumerate((first, second)):
        expected = _scores(data[0], data[1], data[3], data[2])
        _check(expected, data[5], storage, output[index * full : (index + 1) * full])
