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


@pytest.fixture(autouse=True)
def ieee_matmul():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


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
    scales = torch.ldexp(
        torch.ones(keys, device="cuda", dtype=torch.float32), exponent - 127
    )
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.arange(keys - rows + 1, keys + 1, device="cuda", dtype=torch.int32)
    return q, k, packed, weights, starts, ends, scales


def _scores(q, k, weights, scales):
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, mqa_logits_mx_inputs

    bits = scales.view(torch.int32)
    assert torch.all(
        (scales > 0)
        & torch.isfinite(scales)
        & (((bits & 0x7FFFFF) == 0) | (bits == 0x00400000))
    )
    exponent = ((bits >> 23) & 255).to(torch.uint8)
    packed = exponent.repeat_interleave(4).view(torch.int32)
    starts = torch.zeros(len(q), device=q.device, dtype=torch.int32)
    ends = torch.full_like(starts, len(k))
    q_scale, native_weights = mqa_logits_mx_inputs(q, None, weights)
    logits = fp8_fp4_mqa_logits(
        (q, q_scale), (k, packed), native_weights, starts, ends, len(k)
    )
    assert logits.dtype == torch.bfloat16
    return logits.float()


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
    expected = [_scores(data[0], data[1], data[3], data[6]) for data in (first, second)]
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


def test_dosa_ue8m0_extremes_decode_without_losing_subnormal():
    from vllm.model_executor.layers import dosa_ops

    exponent = torch.tensor(
        [0, 1, 127, 128, 254, 255], device="cuda", dtype=torch.uint8
    )
    packed = exponent.repeat_interleave(4).view(torch.int32)
    decoded = torch.empty(6, device="cuda", dtype=torch.float32)
    dosa_ops.decode_ue8m0[(1,)](packed.view(torch.uint8), decoded, 6, 256)
    expected = torch.tensor(
        [2.0**-127, 2.0**-126, 1, 2, 2.0**127], device="cuda", dtype=torch.float32
    )
    assert torch.equal(decoded[:5].view(torch.int32), expected.view(torch.int32))
    assert torch.isnan(decoded[5])


def test_profiled_dispatch_survives_model_config_scope():
    from vllm.config import (
        AttentionConfig,
        SchedulerConfig,
        VllmConfig,
        get_current_vllm_config_or_none,
        set_current_vllm_config,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.dosa import DoSAWorkspace
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
    )

    config = VllmConfig(
        attention_config=AttentionConfig(indexer_dosa=True),
        scheduler_config=SchedulerConfig.default_factory(
            max_model_len=196608, max_num_batched_tokens=2048, max_num_seqs=1
        ),
    )
    output = torch.full((1, TOPK), CANARY, device="cuda", dtype=torch.int32)
    with set_current_vllm_config(config):
        cache = DeepseekV32IndexerCache(
            128, torch.uint8, "dispatch.indexer", config.cache_config
        )
        layer = SparseAttnIndexer(
            cache, 128, "ue8m0", TOPK, 128, 196608, 196608, output
        )
    q = torch.zeros((1, 32, 128), device="cuda", dtype=torch.float8_e4m3fn)
    weights = torch.zeros((1, 32), device="cuda")
    hidden = torch.zeros((1, 1), device="cuda")
    init_workspace_manager(q.device)
    manager = current_workspace_manager()
    with set_forward_context(None, config):
        assert get_current_vllm_config_or_none() is None
        layer.forward_cuda(hidden, q, None, weights)
        manager.lock()
        try:
            owner = manager.get_persistent_resource(
                (DoSAWorkspace, 32, 4096, 196608), lambda: pytest.fail("not profiled")
            )
            assert owner.max_queries == layer.dosa_max_queries == 4096
            assert (
                layer.forward_cuda(hidden, q, None, weights).data_ptr()
                == output.data_ptr()
            )
            assert torch.all(output == CANARY)
            torch.accelerator.synchronize()
        finally:
            manager.unlock()


def test_dosa_short_intersection_falls_back_before_any_request_writes(workspace):
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
    table = torch.arange(4096, device="cuda", dtype=torch.int32).reshape(2, 2048)
    chunk = build_prefill_chunk_metadata(
        start_idx=0,
        end_idx=2,
        query_start_loc=queries_cpu.to("cuda"),
        query_start_loc_cpu=queries_cpu,
        uncompressed_seq_lens=lengths,
        compressed_seq_lens=lengths,
        compressed_seq_lens_cpu=lengths_cpu,
        block_table=table,
        compress_ratio=1,
        collect_request_ranges=True,
    )
    assert chunk is not None
    storage = torch.full(
        (2 * full + 2, TOPK + 17), CANARY, device="cuda", dtype=torch.int32
    )
    output = storage[1:-1]
    epoch = workspace.plan_epoch
    # Check full-then-short and short-then-full after real row intersection.
    for begin, end in ((0, 2 * full - 1), (1, 2 * full)):
        assert not _dosa_prefill(
            workspace, chunk, begin, end, q, keys, packed, weights, output
        )
        torch.accelerator.synchronize()
        assert bool((storage == CANARY).all())
        assert workspace.plan_epoch == epoch

    # The same input is checked against the actual native BF16 score.
    expected = _scores(first[0], first[1], first[3], first[6])
    assert _dosa_prefill(workspace, chunk, 0, full, q, keys, packed, weights, output)
    torch.accelerator.synchronize()
    _check(expected, first[5], storage[: full + 2], output[:full, :TOPK])
    assert bool((storage[full + 1 :] == CANARY).all())

    assert _dosa_prefill(
        workspace, chunk, 0, 2 * full, q, keys, packed, weights, output
    )
    torch.accelerator.synchronize()
    _check(expected, first[5], storage, output[:full, :TOPK])
    second_scores = _scores(second[0], second[1], second[3], second[6])
    _check(second_scores, second[5], storage, output[full : 2 * full, :TOPK])


@pytest.mark.parametrize("heads", (64, 32))
def test_dosa_bootstrap_query_offsets_above_signed_32bit(heads):
    """Bootstrap must copy distant Q rows without signed source-offset wrap."""
    from vllm.model_executor.layers.dosa_ops import _spread_queries

    width = heads * 128
    prefix = 1 << 31
    rows = prefix // width + 1
    # Both the intended final anchor and the old wrapped offset are allocated.
    storage = torch.empty(prefix + rows * width, device="cuda", dtype=torch.uint8)
    storage[:8192].fill_(3)
    q = storage[prefix:].view(rows, heads, 128)
    weights = torch.empty((rows, heads), device="cuda", dtype=torch.bfloat16)
    expected_q = torch.arange(4 * width, device="cuda", dtype=torch.int32)
    expected_q = (expected_q.remainder(97) + 11).to(torch.uint8).view(4, heads, 128)
    expected_w = torch.arange(1, 4 * heads + 1, device="cuda", dtype=torch.float32)
    expected_w = expected_w.view(4, heads).div(8).to(torch.bfloat16)
    for anchor in range(4):
        row = anchor * (rows - 1) // 3
        q[row].copy_(expected_q[anchor])
        weights[row].copy_(expected_w[anchor])
    output_q = torch.empty_like(expected_q)
    output_w = torch.empty((4, heads), device="cuda", dtype=torch.float32)
    _spread_queries[(4,)](q, weights, output_q, output_w, rows, heads, num_warps=4)
    torch.accelerator.synchronize()
    assert torch.equal(output_q[:3], expected_q[:3])
    assert torch.equal(output_w, expected_w.float())
    assert torch.equal(output_q[3], expected_q[3])


def _op(heads, name, *args):
    import vllm._dosa_C  # noqa: F401

    return getattr(torch.ops._C, f"dosa_h{heads}_{name}")(*args)


def _record_codes(scores):
    bits = scores.view(torch.int32).long() & 0xFFFFFFFF
    bits = torch.where((bits & 0x7FFFFFFF) == 0, 0, bits)
    ordered = torch.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    return ((~ordered) & 0xFFFFFFFF) >> 12


@pytest.mark.parametrize("heads", (32, 64))
def test_dosa_dense_scores_match_native_bf16_rounding(heads):
    q = torch.zeros((4, heads, 128), device="cuda")
    q[:, 0, 0], q[:, 0, 1] = 1, 2.0**-4
    k = torch.zeros((128, 128), device="cuda")
    k[:, 0], k[:, 1] = 1, 2.0**-4
    q, k = q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn)
    weights = torch.zeros((4, heads), device="cuda", dtype=torch.bfloat16)
    weights[:, 0] = 1
    scales = torch.ones(128, device="cuda")
    native = _scores(q, k, weights, scales)
    actual = torch.empty_like(native)
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    _op(heads, "sample_dense_score", q, k, scales, weights.float(), actual, sms)
    # The true dot is 1+2^-8; native rounds the head's BF16 midpoint to 1.
    assert torch.all(native == 1)
    assert torch.equal(actual.view(torch.int32), native.view(torch.int32))


@pytest.mark.parametrize("heads", (32, 64))
def test_dosa_native_seed_selects_exact_witnesses_and_routes_bad_scores(heads):
    seed, rows = 12288, 7
    index = torch.arange(seed, device="cuda", dtype=torch.float32)
    scores = torch.zeros((rows, seed), device="cuda")
    scores[0], scores[1] = index.remainder(9) - 4, -index.remainder(5000) / 8
    scores = scores.to(torch.bfloat16).float()
    scores[2, ::2] = -0.0
    scores[3, 17], scores[4, 33], scores[5, 65] = torch.nan, torch.inf, -torch.inf
    scores[6, 19] = 1 + 2.0**-8
    low = torch.full((rows, seed + 3), 0xA5, device="cuda", dtype=torch.uint8)
    words = torch.full((rows, seed + 3), CANARY, device="cuda", dtype=torch.int32)
    counts = torch.empty(rows, device="cuda", dtype=torch.int32)
    tau = torch.empty(rows, device="cuda")
    error = torch.zeros_like(counts)
    lengths = torch.empty_like(counts)
    diag = torch.empty((rows, 4), device="cuda", dtype=torch.int32)
    _op(heads, "seed_prepare5", scores, low, words, counts, tau, error, lengths, diag)
    ids = words[:3, :TOPK] & 0xFFFFF
    assert torch.all((ids >= 0) & (ids < seed))
    assert torch.all(ids.sort(dim=1).values.diff(dim=1) > 0)
    selected = scores[:3].gather(1, ids.long()).sort(dim=1, descending=True).values
    expected = scores[:3].topk(TOPK, dim=1).values
    # +/-0 are equal native scores; no unstable native tie ID is prescribed.
    assert torch.equal(selected, expected)
    assert torch.equal(tau[:3], expected[:, -1])
    cells = ((words[:3, :TOPK].long() & 0xFFFFFFFF) >> 20) << 8
    cells |= low[:3, :TOPK].long()
    assert torch.equal(cells, _record_codes(scores[:3].gather(1, ids.long())))
    assert torch.all(counts[:3] == TOPK) and torch.all(error[:3] == 0)
    assert torch.all(counts[3:] == 0) and torch.all(torch.isneginf(tau[3:]))
    assert torch.all((error[3:] & 1) != 0) and torch.all(diag[3:, 3] == 3)
    assert torch.all(words[:, TOPK:] == CANARY) and torch.all(low[:, TOPK:] == 0xA5)
    assert torch.all(words[3:] == CANARY) and torch.all(low[3:] == 0xA5)


@pytest.mark.parametrize("heads", (32, 64))
def test_dosa_record5_native_score_ties_use_frontier_ids_and_strides(heads):
    rows, keys = 4, 8192
    equal = (1, 129, 513, 4096)
    q = torch.zeros((rows, heads, 128), device="cuda")
    k = torch.full((keys, 128), 0.5, device="cuda")
    before = [TOPK - min(n, 64) for n in equal]
    for row, (b, e) in enumerate(zip(before, equal)):
        q[row, 0, row] = 1
        if row % 2:
            k[:, row], k[:b, row] = 2, 0.5
        else:
            k[:b, row] = 2
        k[b : b + e, row] = 1
    q, k = q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn)
    weights = torch.zeros((rows, heads), device="cuda", dtype=torch.bfloat16)
    weights[:, 0] = 1
    weights[1::2, 0] = -1
    scores = _scores(q, k, weights, torch.ones(keys, device="cuda"))
    code = _record_codes(scores)
    low = code.to(torch.uint8)
    ids = torch.arange(keys, device="cuda", dtype=torch.int64)
    words = (((code >> 8) << 20) | ids).to(torch.int32)
    counts = torch.full((rows,), keys, device="cuda", dtype=torch.int32)
    starts = torch.zeros_like(counts)
    error = torch.zeros_like(counts)
    pool = torch.zeros(1, device="cuda", dtype=torch.int32)
    diag = torch.empty((rows, 2), device="cuda", dtype=torch.int32)
    boundary = torch.full((rows, 4096), CANARY, device="cuda", dtype=torch.int32)
    boundary_counts = torch.empty_like(counts)
    storage = torch.full((rows + 2, TOPK), CANARY, device="cuda", dtype=torch.int32)
    output = storage[1:-1]
    _op(
        heads,
        "select_select_dense5",
        low,
        words,
        counts,
        starts,
        counts,
        pool,
        error,
        output,
        diag,
        boundary,
        boundary_counts,
        keys,
    )
    assert torch.all(error == 0)
    assert boundary_counts.cpu().tolist() == list(equal)
    guarded = torch.full(
        (rows + 2, 4096 + 17), CANARY, device="cuda", dtype=torch.int32
    )
    strided = guarded[1:-1, :4096]
    strided.copy_(boundary)
    permutation = torch.arange(keys - 1, -1, -1, device="cuda", dtype=torch.int32)
    _op(
        heads,
        "post_complete_ties",
        strided,
        boundary_counts,
        diag,
        starts,
        counts,
        error,
        output,
        permutation,
    )
    assert torch.all(error == 0)
    _check(scores, counts, storage, output)
    for row, b in enumerate(before):
        # Only the published frontier E is ordered, not all global score ties.
        logical = permutation[output[row, b:].long()]
        assert torch.all(logical.diff() > 0)
    assert torch.all(guarded[0] == CANARY) and torch.all(guarded[-1] == CANARY)
    assert torch.all(guarded[1:-1, 4096:] == CANARY)
