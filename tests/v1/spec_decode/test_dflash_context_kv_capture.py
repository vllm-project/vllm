# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gates for capturing the DFlash context-KV precompute in the FULL graph.

Replay reads persistent buffers sliced to the graph shape's context row
count, so correctness rests on three properties, each tested here:
  * the opt-in is fail-closed (``context_kv_capture_supported``),
  * PAD_SLOT_ID rows never write KV through ``do_kv_cache_update``,
  * a captured replay over PAD-padded persistent buffers produces byte-exact
    drafter KV vs the eager call, across batch shapes with
    ``num_reqs < num_reqs_padded`` (100-step poisoned-cache A/B).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    context_kv_capture_supported,
)


def test_context_kv_capture_gate_fails_closed():
    opt_in = SimpleNamespace(context_kv_capture_safe=True)
    full = CUDAGraphMode.FULL_DECODE_ONLY

    def supported(model=opt_in, mode=full, cp_size=1, query=4, spec=3):
        return context_kv_capture_supported(model, mode, cp_size, query, spec)

    assert supported()
    # Models that do not declare the audited precompute (e.g. wrappers that
    # do not inherit DFlashQwen3ForCausalLM) block the capture.
    assert not supported(model=SimpleNamespace())
    assert not supported(model=SimpleNamespace(context_kv_capture_safe=False))
    # Eager drafter keeps the eager precompute.
    assert not supported(mode=CUDAGraphMode.NONE)
    # DCP stays on the eager path.
    assert not supported(cp_size=2)
    # Anchor-style DSpark (N queries, 1+N target tokens): the context row
    # count is still a pure function of the graph's request count (one
    # extra row per request), so the capture is supported.
    assert supported(query=3, spec=3)
    # Any other query layout is unknown and stays on the eager path.
    assert not supported(query=2, spec=3)
    assert not supported(query=5, spec=3)


def test_dflash_qwen3_declares_capture_safe():
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM

    assert DFlashQwen3ForCausalLM.context_kv_capture_safe is True


NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 64
HIDDEN = 128
NUM_BLOCKS = 8
BLOCK_SIZE = 16
NUM_SLOTS = NUM_BLOCKS * BLOCK_SIZE
MAX_POS = 512
SENTINEL = 512.0


class _TritonCacheWriter:
    """The real TritonAttentionImpl.do_kv_cache_update over the attrs it reads."""

    attn_type = AttentionType.DECODER
    _is_per_token_head_quant = False
    head_size = HEAD_DIM
    kv_cache_dtype = "auto"

    do_kv_cache_update = TritonAttentionImpl.do_kv_cache_update


def _make_fake_drafter(device: torch.device, dtype: torch.dtype):
    """A stand-in self for the real DFlashQwen3Model precompute method.

    Carries exactly the fused buffers _build_fused_kv_buffers would create,
    with real TritonAttentionImpl cache writes, so the unbound
    precompute_and_store_context_kv runs its production code path.
    """
    from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

    torch.manual_seed(0)
    kv_size = NUM_KV_HEADS * HEAD_DIM
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    freqs = torch.outer(torch.arange(MAX_POS, dtype=torch.float32), inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(device, dtype)
    layers = [
        SimpleNamespace(
            kv_cache=torch.full(
                (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_DIM),
                SENTINEL,
                dtype=dtype,
                device=device,
            ),
            impl=_TritonCacheWriter(),
            _k_scale=torch.ones(1, dtype=torch.float32, device=device),
            _v_scale=torch.ones(1, dtype=torch.float32, device=device),
        )
        for _ in range(NUM_LAYERS)
    ]
    fake = SimpleNamespace(
        _num_attn_layers=NUM_LAYERS,
        _kv_size=kv_size,
        _head_dim=HEAD_DIM,
        _num_kv_heads=NUM_KV_HEADS,
        _rms_norm_eps=1e-6,
        _hidden_norm_weight=torch.rand(HIDDEN, dtype=dtype, device=device) + 0.5,
        _fused_kv_weight=0.05
        * torch.randn(NUM_LAYERS * 2 * kv_size, HIDDEN, dtype=dtype, device=device),
        _fused_kv_bias=None,
        _k_norm_weights=(
            torch.rand(NUM_LAYERS, HEAD_DIM, dtype=dtype, device=device) + 0.5
        ).contiguous(),
        _rope_head_size=HEAD_DIM,
        _rope_cos_sin_cache=cos_sin,
        _rope_is_neox=True,
        _attn_layers=layers,
    )
    # HARNESS FIX (phase-C2, era-port rule): the production
    # precompute_and_store_context_kv dispatches through two ordinary class
    # helpers (self._project_context_kv / self._normalize_context_k) that a
    # SimpleNamespace stand-in cannot resolve. Bind the REAL branch class
    # methods onto the stub so the unbound production path (and only it)
    # still runs; they consume exactly the buffer attributes carried above.
    from types import MethodType

    fake._project_context_kv = MethodType(DFlashQwen3Model._project_context_kv, fake)
    fake._normalize_context_k = MethodType(DFlashQwen3Model._normalize_context_k, fake)

    def precompute(hidden, positions, slots):
        DFlashQwen3Model.precompute_and_store_context_kv(fake, hidden, positions, slots)

    return fake, precompute


def _slot_rows(cache: torch.Tensor) -> torch.Tensor:
    """[num_blocks, nkv, block_size, 2*hd] -> [slot, nkv, 2*hd]."""
    return cache.permute(0, 2, 1, 3).reshape(NUM_SLOTS, NUM_KV_HEADS, 2 * HEAD_DIM)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_precompute_skips_pad_slot_rows():
    device = torch.device("cuda")
    fake, precompute = _make_fake_drafter(device, torch.bfloat16)
    hidden = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device=device)
    positions = torch.tensor([5, 9, 100, 200], dtype=torch.int64, device=device)
    slots = torch.tensor(
        [17, PAD_SLOT_ID, 33, PAD_SLOT_ID], dtype=torch.int64, device=device
    )

    precompute(hidden, positions, slots)
    torch.accelerator.synchronize()

    written = {17, 33}
    for layer in fake._attn_layers:
        rows = _slot_rows(layer.kv_cache)
        for slot in range(NUM_SLOTS):
            if slot in written:
                assert not torch.all(rows[slot] == SENTINEL)
            else:
                assert torch.all(rows[slot] == SENTINEL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_captured_precompute_matches_eager_over_100_steps():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    fake, precompute = _make_fake_drafter(device, dtype)
    caches = [layer.kv_cache for layer in fake._attn_layers]
    padded = 16  # num_reqs_padded=4 x num_query_per_req=4

    # Persistent buffers, exactly the speculator's capture contract: slots
    # PAD-filled and positions zeroed before capture.
    hidden = torch.zeros(padded, HIDDEN, dtype=dtype, device=device)
    positions = torch.zeros(padded, dtype=torch.int64, device=device)
    slots = torch.full((padded,), PAD_SLOT_ID, dtype=torch.int64, device=device)

    def reset_caches():
        for cache in caches:
            cache.fill_(SENTINEL)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            precompute(hidden, positions, slots)
    torch.cuda.current_stream().wait_stream(stream)

    reset_caches()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        precompute(hidden, positions, slots)
    torch.accelerator.synchronize()
    for cache in caches:
        assert torch.all(cache == SENTINEL), "capture pass must write no KV"

    rng = torch.Generator()
    rng.manual_seed(1234)
    live_counts = [16, 4, 8, 12]  # n == padded and three n < padded shapes
    for step in range(100):
        n = live_counts[step % len(live_counts)]
        hidden[:n].copy_(torch.randn(n, HIDDEN, generator=rng))
        hidden[n:].fill_(1024.0)  # poison the padded tail
        positions[:n].copy_(torch.randint(0, MAX_POS, (n,), generator=rng))
        positions[n:].zero_()
        slot_cpu = torch.randperm(NUM_SLOTS, generator=rng)[:n]
        if step % 3 == 0 and n > 2:
            slot_cpu[1] = PAD_SLOT_ID  # a rejected row inside the live span
        slots[:n].copy_(slot_cpu)
        slots[n:].fill_(PAD_SLOT_ID)

        reset_caches()
        graph.replay()
        torch.accelerator.synchronize()
        replayed = [cache.clone() for cache in caches]

        reset_caches()
        precompute(hidden, positions, slots)
        for got, want in zip(replayed, (cache.clone() for cache in caches)):
            assert torch.equal(got, want), f"replay != eager at step {step}"

        # Today's eager call runs at the unpadded shape; live rows must agree
        # (bitwise when the GEMM shapes coincide).
        reset_caches()
        precompute(hidden[:n], positions[:n], slots[:n])
        live_cpu = slot_cpu[slot_cpu >= 0]
        live = live_cpu.to(device)
        blocks, offsets = live // BLOCK_SIZE, live % BLOCK_SIZE
        for got_cache, want_cache in zip(replayed, caches):
            got = got_cache[blocks, :, offsets, :]
            want = want_cache[blocks, :, offsets, :]
            if n == padded:
                assert torch.equal(got, want), f"live rows differ at step {step}"
            else:
                torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)

        # Everything the live slots did not claim keeps the poison: padded
        # tail rows and in-span PAD rows never write.
        untouched = torch.ones(NUM_SLOTS, dtype=torch.bool)
        untouched[live_cpu] = False
        untouched = untouched.to(device)
        for got_cache in replayed:
            rows = _slot_rows(got_cache)
            assert torch.all(rows[untouched] == SENTINEL), (
                f"a PAD row wrote KV at step {step}"
            )
