# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Flint split-KV decode attention kernel must match an fp64 reference at
bf16 output-rounding level, mirror FlashInfer trtllm-gen's numerics, and stay
correct under CUDA graph replay with new sequence lengths."""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda() or not current_platform.is_device_capability_family(
    100
):
    pytest.skip("Flint decode attention targets SM100", allow_module_level=True)

from flashinfer.decode import trtllm_batch_decode_with_kv_cache  # noqa: E402

from vllm.models.flint.ops.decode_attention import flint_decode_attention  # noqa: E402

HQ, HKV, D, PAGE = 32, 4, 128, 16
SCALE = D**-0.5
WINDOW = 513


def _make(
    seq_lens: list[int], seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random q/K/V/sinks; blocks are shuffled so block tables are non-trivial."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = [(s + PAGE - 1) // PAGE for s in seq_lens]
    num_blocks = sum(pages) + 5
    kv = torch.randn(num_blocks, HKV, PAGE, 2 * D, device="cuda", generator=g)
    perm = torch.randperm(num_blocks, device="cuda", generator=g).int()
    bt = torch.zeros(len(seq_lens), max(pages + [1]) + 2, dtype=torch.int32)
    bt = bt.cuda()
    o = 0
    for i, n in enumerate(pages):
        bt[i, :n] = perm[o : o + n]
        o += n
    q = torch.randn(len(seq_lens), HQ, D, device="cuda", generator=g) * 2
    sinks = torch.randn(HQ, device="cuda", generator=g) * 2
    sl = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    return q.bfloat16(), kv.bfloat16(), bt, sl, sinks


def _reference(q, kv, bt, seq_lens, sinks, window):
    out = torch.zeros(q.shape, dtype=torch.float64, device=q.device)
    for i, s in enumerate(seq_lens.tolist()):
        if s == 0:
            continue
        n = (s + PAGE - 1) // PAGE
        x = kv[bt[i, :n].long()].permute(1, 0, 2, 3).reshape(HKV, n * PAGE, 2 * D)
        x = x[:, max(s - window, 0) if window else 0 : s].double()
        k, v = x[..., :D], x[..., D:]
        scores = torch.einsum("hgd,htd->hgt", q[i].double().view(HKV, -1, D), k)
        scores = scores * SCALE
        sink = sinks.double().view(HKV, -1, 1)
        m = torch.maximum(scores.amax(-1, keepdim=True), sink)
        p = torch.exp(scores - m)
        denom = p.sum(-1, keepdim=True) + torch.exp(sink - m)
        out[i] = (torch.einsum("hgt,htd->hgd", p, v) / denom).reshape(HQ, D)
    return out


def _flashinfer(q, kv, bt, seq_lens, sinks, window):
    workspace = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
    k_cache, v_cache = kv.split(D, dim=-1)
    return trtllm_batch_decode_with_kv_cache(
        q,
        (k_cache, v_cache),
        workspace,
        bt,
        seq_lens,
        int(seq_lens.max()),
        bmm1_scale=SCALE,
        window_left=window - 1 if window else -1,
        sinks=sinks,
        kv_layout="HND",
        backend="trtllm-gen",
    )


def _flint(q, kv, bt, seq_lens, sinks, window, out=None):
    out = torch.empty_like(q) if out is None else out
    return flint_decode_attention(q, kv, bt, seq_lens, sinks, window, SCALE, out)


def _rel_err(x, ref):
    """|x - ref| relative to each output row's max magnitude."""
    scale = ref.abs().amax(-1, keepdim=True).clamp_min(1e-6)
    return (x.double() - ref).abs() / scale


@pytest.mark.parametrize(
    "seq_lens",
    [
        [1],
        [17],
        [1000],
        [20011],
        [5, 77, 1000, 4097, 16, 33, 513, 3001],
        [100, 2000, 37] * 10 + [1, 9000],
    ],
    ids=["1", "17", "1000", "20011", "mixed8", "mixed32"],
)
@pytest.mark.parametrize("window", [None, WINDOW], ids=["full", "swa"])
@torch.inference_mode()
def test_matches_reference_and_flashinfer(seq_lens: list[int], window: int | None):
    args = _make(seq_lens)
    ref = _reference(*args, window)
    out = _flint(*args, window)
    fi_err = _rel_err(_flashinfer(*args, window), ref)
    err = _rel_err(out, ref)
    # One bf16 ulp of the row max is 2^-8; trtllm-gen also rounds its split
    # partials to bf16, so its error bounds ours.
    assert err.max() < 8e-3
    assert err.max() <= fi_err.max() + 1e-3
    assert err.mean() <= fi_err.mean() + 1e-4


@pytest.mark.parametrize("seq_len", [513, 514, 529, 530, 545, 20000])
@torch.inference_mode()
def test_window_crosses_block_boundary(seq_len: int):
    """The first attended page is partial; earlier tokens must be masked."""
    args = _make([seq_len, seq_len + 7], seed=seq_len)
    err = _rel_err(_flint(*args, WINDOW), _reference(*args, WINDOW))
    assert err.max() < 8e-3


@torch.inference_mode()
def test_launch_from_another_stream_waits_for_the_previous_one():
    """Launches share the split workspace and arrival counters, so a launch
    from a different stream must not start before the previous one ends."""
    args = _make([20011] * 4)
    _flint(*args, None)
    stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
    torch.accelerator.synchronize()
    with torch.cuda.stream(stream_a):
        torch.cuda._sleep(100_000_000)  # ~50 ms at ~2 GHz
        a_done = torch.cuda.Event()
        _flint(*args, None)
        a_done.record()
    with torch.cuda.stream(stream_b):
        _flint(*args, None)
        b_done = torch.cuda.Event()
        b_done.record()
    b_done.synchronize()
    assert a_done.query()


@torch.inference_mode()
def test_empty_request_writes_zeros():
    q, kv, bt, sl, sinks = _make([0, 40])
    out = torch.full_like(q, float("nan"))
    _flint(q, kv, bt, sl, sinks, None, out)
    assert (out[0] == 0).all()
    assert _rel_err(out[1:], _reference(q, kv, bt, sl, sinks, None)[1:]).max() < 8e-3


@torch.inference_mode()
def test_probabilities_summed_in_fp32():
    """Like trtllm-gen, P is rounded to bf16 for P @ V but the softmax
    denominator sums the fp32 probabilities. With V = 1, one max-logit token
    per page (so every split sees the same running max) and all other
    probabilities just below a bf16 rounding midpoint, the output is visibly
    below 1 (it would be exactly 1 with fp32 P or a bf16-P denominator)."""
    n = 1000
    q, kv, bt, sl, sinks = _make([n])
    kv.zero_()
    q.zero_()
    q[..., 0] = 1.0
    q[..., 1] = 2**-6
    flat = kv.view(-1, HKV, PAGE, 2 * D)[bt[0, : (n + PAGE - 1) // PAGE].long()]
    flat[..., 0] = -7.8125  # p = exp2((s - 0) * scale * log2e) = 0.50195..
    flat[..., 1] = 0.9296875
    flat[:, :, 0, :2] = 0.0  # first token of every page at the max (p = 1)
    flat[..., D:] = 1.0
    kv.view(-1, HKV, PAGE, 2 * D)[bt[0, : (n + PAGE - 1) // PAGE].long()] = flat
    sinks.fill_(-100.0)
    out = _flint(q, kv, bt, sl, sinks, None)
    expected = _flashinfer(q, kv, bt, sl, sinks, None)
    assert (expected == 0.99609375).all()
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_cuda_graph_replay_with_new_lengths():
    """Grid shape depends only on the batch size; the split-combine counters
    reset themselves, so replays with different lengths stay correct."""
    lens_a = [30000, 17, 4100, 515]
    lens_b = [1, 31000, 513, 9000]
    q, kv, bt, sl, sinks = _make(lens_b)
    out = torch.empty_like(q)
    _, _, bt_a, sl_a, _ = _make(lens_a)
    bt_a = torch.nn.functional.pad(bt_a, (0, bt.shape[1] - bt_a.shape[1]))
    static_bt, static_sl = bt_a.clone(), sl_a.clone()
    for window in (None, WINDOW):
        _flint(q, kv, static_bt, static_sl, sinks, window, out)  # warm up / compile
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _flint(q, kv, static_bt, static_sl, sinks, window, out)
        for bt_x, sl_x in ((bt, sl), (bt_a, sl_a), (bt, sl)):
            static_bt.copy_(bt_x)
            static_sl.copy_(sl_x)
            graph.replay()
            ref = _reference(q, kv, static_bt, static_sl, sinks, window)
            assert _rel_err(out, ref).max() < 8e-3


def _early_trigger_writer():
    """A PDL kernel that releases its dependents immediately, idles ~2 us,
    then writes one token's K/V into the cache, like
    ``qk_norm_rope_cache(launch_dependents=True)``."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64

    class Writer:
        @cute.jit
        def __call__(self, src, dst, cycles: Int32, stream):
            self.kernel(src, dst, cycles).launch(
                grid=(1, 1, 1), block=(256, 1, 1), stream=stream, use_pdl=True
            )

        @cute.kernel
        def kernel(self, src, dst, cycles: Int32):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
            t0 = cute.arch.clock64()
            while cute.arch.clock64() - t0 < Int64(cycles):
                pass
            tid, _, _ = cute.arch.thread_idx()
            dst[tid] = src[tid]

    from cutlass.cute.runtime import make_fake_stream
    from quack.compile_utils import make_fake_tensor

    bf16 = cutlass.BFloat16
    return cute.compile(
        Writer(),
        make_fake_tensor(bf16, (256,), divisibility=8),
        make_fake_tensor(bf16, (256,), divisibility=8),
        Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@pytest.mark.parametrize("window", [None, WINDOW], ids=["full", "swa"])
@torch.inference_mode()
def test_current_token_written_by_pdl_predecessor(window: int | None):
    """K/V pages are prefetched before the grid-dependency wait, except the
    page holding the current token: K/V written there by an early-triggering
    predecessor must be seen."""
    writer = _early_trigger_writer()
    # The last write (request 3, head 3) races the attention kernel; with one
    # token, a stale read would change that output entirely.
    lens = [20011, 4096, 17, 1]
    q, kv, bt, sl, sinks = _make(lens)
    new_kv = torch.randn(len(lens), HKV, 2 * D, device="cuda").bfloat16() * 4
    out = torch.empty_like(q)
    blks = [int(bt[i, (s - 1) // PAGE]) for i, s in enumerate(lens)]
    _flint(q, kv, bt, sl, sinks, window, out)  # compile / workspace
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i, s in enumerate(lens):
            for h in range(HKV):
                writer(new_kv[i, h], kv[blks[i], h, (s - 1) % PAGE], 4000)
        _flint(q, kv, bt, sl, sinks, window, out)
    kv_before = kv.clone()
    graph.replay()
    torch.accelerator.synchronize()
    for i, s in enumerate(lens):
        kv_before[blks[i], :, (s - 1) % PAGE] = new_kv[i]
    assert torch.equal(kv, kv_before)
    ref = _reference(q, kv, bt, sl, sinks, window)
    assert _rel_err(out, ref).max() < 8e-3
