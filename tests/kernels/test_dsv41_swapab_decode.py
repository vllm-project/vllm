# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 swap-AB decode attention vs FlashMLA's mega kernel.

The swap-AB kernel keeps the mega kernel's contract: it reads Q in the
chunk-interleaved, head-padded layout and writes MXFP8 in the o-fused layout
``wo_a`` reads, for the live heads' groups only. The caches are written by
V4.1's own insert kernels, and both kernels are held to an fp64 reference over
the dequantized records.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.models.deepseek_v41.common.ops import quantize_and_insert_k_cache
from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import (
    rope_quant_insert,
)
from vllm.models.deepseek_v41.common.ops.fused_layout import (
    WV_GROUP_SIZE,
    permute_q_to_fused,
)
from vllm.models.deepseek_v41.nvidia import flash_mla_mega_attn as mega_attn
from vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn import (
    _ROPE_ARGS,
    _SF_ARGS,
    DeepseekV4MegaAttnAttention,
    alloc_mega_attn_output,
    is_flashmla_mega_attn_supported,
)
from vllm.platforms import current_platform

HEAD_DIM, ROPE_DIM = 512, 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
FP8_RECORD, NVFP4_RECORD = 528, 288
PADDED_HEADS = 64
SCALE = HEAD_DIM**-0.5
MAX_POS = 65536
# V4.1-Flash's decode lists: the 128-slot window and 512 compressed slots.
SWA_WIDTH, EXTRA_WIDTH = 128, 512
# Slots per page: SWA, and compressed on ratio-2 layers (128 on ratio-1 ones).
SWA_BLOCK, EXTRA_BLOCK = 32, 64
SWA_SLOTS, EXTRA_SLOTS = 4096, 16384
# A block also holds the other pages of its KV cache group, so a layer's pages
# sit further apart than its records span. As in a packed block, the stride
# stays a multiple of 512 B and of the record, which FlashMLA requires: it
# reads the records as rows of the whole cache.
PAD_RECORDS = 32
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Output bytes no kernel wrote: 448 in e4m3, 2**-1 as a ue8m0 scale.
SENTINEL = 0x7E
SENTINEL_WORD = 0x7E7E7E7E

_MEGA_ATTN_SUPPORTED, _MEGA_ATTN_REASON = is_flashmla_mega_attn_supported()
requires_mega_attn = pytest.mark.skipif(
    not _MEGA_ATTN_SUPPORTED, reason=f"{_MEGA_ATTN_REASON}"
)


@dataclass
class Case:
    heads: int
    q: torch.Tensor  # [s_q, heads, 512] bf16, head-major
    q_fused: torch.Tensor  # [s_q, 64, 512] bf16, zero-padded, chunk-interleaved
    positions: torch.Tensor  # [s_q] int32
    sink: torch.Tensor  # [64] fp32, -inf past the live heads
    cos_sin: torch.Tensor  # [MAX_POS, 64] fp32
    swa: torch.Tensor  # [num_blocks, SWA_BLOCK, 528] uint8 view of padded blocks
    swa_idx: torch.Tensor  # [s_q, SWA_WIDTH] int32 slot ids, -1 = none
    swa_len: torch.Tensor  # [s_q] int32
    extra: torch.Tensor  # [num_blocks, extra_block, 528 or 288], likewise
    extra_idx: torch.Tensor  # [s_q, EXTRA_WIDTH] int32
    extra_len: torch.Tensor  # [s_q] int32


@pytest.fixture(scope="module")
def cos_sin_cache() -> torch.Tensor:
    """A cos | sin table in the layout of V4.1's rotary embedding."""
    inv_freq = 1.0 / 10000 ** (torch.arange(0, ROPE_DIM, 2, device="cuda") / ROPE_DIM)
    freqs = torch.outer(torch.arange(MAX_POS, device="cuda").float(), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


def _swapab_module() -> Any:
    """The swap-AB kernel's module; skips where the CuTe DSL is not installed."""
    pytest.importorskip("cutlass")
    from vllm.models.deepseek_v41.nvidia.ops import swapab_decode_attn

    return swapab_decode_attn


@pytest.fixture(scope="module")
def swapab_kernel() -> Any:
    """The swap-AB kernel, where the CuTe DSL build can compile it."""
    swapab = _swapab_module()
    if not swapab.swapab_decode_supported(16, SWA_WIDTH, EXTRA_WIDTH):
        pytest.skip("The CuTe DSL build predates PTX ISA 9.2 (CUDA 13.2).")
    return swapab._SWAPAB_DECODE_KERNEL


def _paged_cache(
    num_slots: int, block_size: int, record_bytes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zeroed ``[num_blocks, block_bytes]`` blocks and the view of their pages.

    The view is ``[num_blocks, block_size, record_bytes]``; every block ends in
    ``PAD_RECORDS`` records' worth of bytes that no page covers.
    """
    num_blocks = num_slots // block_size + 1
    blocks = torch.zeros(
        num_blocks,
        block_size + PAD_RECORDS,
        record_bytes,
        dtype=torch.uint8,
        device="cuda",
    )
    return blocks.flatten(1), blocks[:, :block_size]


def _make_case(
    s_q: int,
    heads: int,
    extra: str,
    cos_sin: torch.Tensor,
    seed: int,
    full: bool,
    extra_block: int = EXTRA_BLOCK,
) -> Case:
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device="cuda")

    def randint(low: int, high: int, *shape: int) -> torch.Tensor:
        return torch.randint(
            low, high, shape, generator=gen, device="cuda", dtype=torch.int32
        )

    swa_blocks, swa = _paged_cache(SWA_SLOTS, SWA_BLOCK, FP8_RECORD)
    slots = torch.arange(swa.shape[0] * SWA_BLOCK, device="cuda")
    quantize_and_insert_k_cache(
        randn(slots.numel(), HEAD_DIM).bfloat16(),
        swa_blocks,
        slots,
        block_size=SWA_BLOCK,
        bytes_per_token=FP8_RECORD,
    )
    record = NVFP4_RECORD if extra == "fp4" else FP8_RECORD
    _, extra_view = _paged_cache(EXTRA_SLOTS, extra_block, record)
    slots = torch.arange(extra_view.shape[0] * extra_block, device="cuda")
    # The compressed cache's own writer; it RoPEs each row at its position.
    rope_quant_insert(
        randn(slots.numel(), HEAD_DIM).bfloat16(),
        slots % MAX_POS,
        cos_sin,
        extra_view,
        slots,
        1,
    )

    swa_idx = randint(0, swa.shape[0] * SWA_BLOCK, s_q, SWA_WIDTH)
    extra_idx = randint(0, extra_view.shape[0] * extra_block, s_q, EXTRA_WIDTH)
    for idx in (swa_idx, extra_idx):
        idx[torch.rand(idx.shape, generator=gen, device="cuda") < 0.05] = -1
    if full:
        swa_len = torch.full((s_q,), SWA_WIDTH, dtype=torch.int32, device="cuda")
        extra_len = torch.full((s_q,), EXTRA_WIDTH, dtype=torch.int32, device="cuda")
    else:
        swa_len = randint(1, SWA_WIDTH + 1, s_q)
        extra_len = randint(0, EXTRA_WIDTH + 1, s_q)
        if s_q > 1:
            extra_len[-1] = 0  # nothing compressed yet

    q = (randn(s_q, heads, HEAD_DIM) * 2).bfloat16()
    q_padded = q.new_zeros(s_q, PADDED_HEADS, HEAD_DIM)
    q_padded[:, :heads] = q
    sink = torch.full((PADDED_HEADS,), -math.inf, device="cuda")
    sink[:heads] = randn(heads)
    return Case(
        heads=heads,
        q=q,
        q_fused=permute_q_to_fused(q_padded),
        positions=randint(0, MAX_POS, s_q),
        sink=sink,
        cos_sin=cos_sin,
        swa=swa,
        swa_idx=swa_idx,
        swa_len=swa_len,
        extra=extra_view,
        extra_idx=extra_idx,
        extra_len=extra_len,
    )


def _dequant_records(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """The fp32 ``[n, 512]`` rows of V4.1 records ``slots`` of a paged cache.

    Records are interleaved: each holds a token's data, then its own scales.
    """
    block_size, record_bytes = cache.shape[1:]
    records = cache[slots // block_size, slots % block_size]
    if record_bytes == FP8_RECORD:
        # 512 e4m3, then one ue8m0 exponent per 32 dims.
        values = records[:, :512].view(torch.float8_e4m3fn).float().view(-1, 16, 32)
        scales = torch.exp2(records[:, 512:].float() - 127)
    else:
        # e2m1 pairs, even element in the low nibble, then one e4m3 scale per
        # 16 dims.
        packed = records[:, :256].long()
        codes = torch.stack([packed & 15, packed >> 4], dim=-1).view(-1, 32, 16)
        magnitudes = torch.tensor(E2M1_MAGNITUDES, device=slots.device)
        values = magnitudes[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
        scales = records[:, 256:].view(torch.float8_e4m3fn).float()
    return (values * scales[..., None]).view(-1, HEAD_DIM)


def _rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """GPT-J RoPE of the last ``ROPE_DIM`` dims of ``x``."""
    if inverse:
        sin = -sin
    x = x.clone()
    even, odd = x[..., NOPE_DIM::2], x[..., NOPE_DIM + 1 :: 2]
    x[..., NOPE_DIM::2], x[..., NOPE_DIM + 1 :: 2] = (
        even * cos - odd * sin,
        odd * cos + even * sin,
    )
    return x


def _reference(case: Case) -> torch.Tensor:
    """fp64 attention over the dequantized records, ``[s_q, heads, 512]``.

    As both kernels do it: Q is RoPEd in fp32 and rounded to bf16, the sink
    only enters the softmax denominator, and the output gets the inverse RoPE.
    """
    cos_sin = case.cos_sin[case.positions.long()]
    cos, sin = cos_sin[:, None, : ROPE_DIM // 2], cos_sin[:, None, ROPE_DIM // 2 :]
    q = _rope(case.q.float(), cos, sin).bfloat16().double()
    sink = case.sink[: case.heads, None].double()
    out = []
    for t in range(q.shape[0]):
        keys = []
        for cache, idx, length in (
            (case.swa, case.swa_idx, case.swa_len),
            (case.extra, case.extra_idx, case.extra_len),
        ):
            slots = idx[t, : int(length[t])].long()
            slots = slots[slots >= 0]
            keys.append(_dequant_records(cache, slots))
        k = torch.cat(keys).double()
        scores = q[t] @ k.T * SCALE
        m = torch.cat([scores, sink], dim=1).amax(dim=1, keepdim=True)
        p = torch.exp(scores - m)
        out.append(p @ k / (p.sum(dim=1, keepdim=True) + torch.exp(sink - m)))
    return _rope(torch.stack(out).float(), cos, sin, inverse=True)


def _alloc_output(num_tokens: int) -> QuantizedActivation:
    """A mega-attention output pair with every byte set to ``SENTINEL``."""
    out = alloc_mega_attn_output(
        num_tokens, PADDED_HEADS // WV_GROUP_SIZE, torch.device("cuda")
    )
    out.data.view(torch.uint8).fill_(SENTINEL)
    out.scale.fill_(SENTINEL_WORD)
    return out


def _dequant_output(out: QuantizedActivation, heads: int) -> torch.Tensor:
    """The live heads of a mega-attention output, ``[s_q, heads, 512]`` fp32.

    Group ``g`` holds heads ``8g + h``, their 32-dim chunk ``c`` at chunk
    ``8c + h``; chunk ``k``'s ue8m0 exponent is byte ``k`` of the group's
    scale words.
    """
    num_tokens, groups = out.data.shape[0], heads // WV_GROUP_SIZE
    values = out.data[:, :groups].float().view(num_tokens, groups, -1, 32)
    exponents = out.scale[:, :groups].contiguous().view(torch.uint8).float()
    values = values * torch.exp2(exponents[..., None] - 127)
    values = values.view(num_tokens, groups, HEAD_DIM // 32, WV_GROUP_SIZE, 32)
    return values.transpose(2, 3).reshape(num_tokens, heads, HEAD_DIM)


def _run_mega(case: Case, out: QuantizedActivation) -> None:
    """The mega kernel, called as ``_forward_decode_mega`` calls it."""
    torch.ops._flashmla_C.fused_norm_rope_attn_rope_cast_decode(
        case.q_fused,
        case.swa.unsqueeze(-2),
        case.swa_idx,
        SCALE,
        HEAD_DIM,
        case.sink,
        case.swa_len,
        case.extra.unsqueeze(-2),
        case.extra_idx,
        case.extra_len,
        False,  # enable_q_norm
        0.0,  # rms_norm_eps
        case.positions,
        *_ROPE_ARGS,
        case.cos_sin,
        PADDED_HEADS // WV_GROUP_SIZE,
        *_SF_ARGS,
        out.data,
        out.scale,
    )


def _run_swapab(kernel: Any, case: Case, out: QuantizedActivation) -> None:
    kernel(
        q=case.q_fused,
        swa_cache=case.swa,
        swa_idx=case.swa_idx,
        swa_len=case.swa_len,
        extra_cache=case.extra,
        extra_idx=case.extra_idx,
        extra_len=case.extra_len,
        positions=case.positions,
        cos_sin_cache=case.cos_sin,
        sink=case.sink,
        heads=case.heads,
        softmax_scale=SCALE,
        out_data=out.data,
        out_scale=out.scale,
    )


def _assert_same_output(
    actual: QuantizedActivation, expected: QuantizedActivation, heads: int
) -> None:
    groups = heads // WV_GROUP_SIZE
    assert torch.equal(
        actual.data[:, :groups].view(torch.uint8),
        expected.data[:, :groups].view(torch.uint8),
    )
    assert torch.equal(actual.scale[:, :groups], expected.scale[:, :groups])


@requires_mega_attn
@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize("s_q", [1, 2, 5, 16])
@pytest.mark.parametrize("extra", ["fp4", "fp8"])
@pytest.mark.parametrize("extra_block", [64, 128])
@pytest.mark.parametrize("full", [False, True])
def test_matches_reference(
    heads: int,
    s_q: int,
    extra: str,
    extra_block: int,
    full: bool,
    cos_sin_cache: torch.Tensor,
    swapab_kernel: Any,
):
    """Swap-AB lands no further from exact attention than the mega kernel."""
    case = _make_case(
        s_q,
        heads,
        extra,
        cos_sin_cache,
        seed=7 * s_q + heads + full + extra_block,
        full=full,
        extra_block=extra_block,
    )
    mega_out, swapab_out = _alloc_output(s_q), _alloc_output(s_q)
    _run_mega(case, mega_out)
    _run_swapab(swapab_kernel, case, swapab_out)

    ref = _reference(case)
    mega = _dequant_output(mega_out, heads)
    swapab = _dequant_output(swapab_out, heads)
    assert torch.isfinite(swapab).all()
    # Both outputs are MXFP8, so neither matches exactly; the swap-AB kernel's
    # own rounding may not add to the mega kernel's.
    assert (swapab - ref).norm() <= 1.1 * (mega - ref).norm()
    assert (swapab - ref).abs().amax() <= 1.1 * (mega - ref).abs().amax()
    assert (swapab - mega).norm() < 2e-2 * mega.norm()


@requires_mega_attn
@pytest.mark.parametrize("heads", [16, 32])
def test_untouched_padding(heads: int, cos_sin_cache: torch.Tensor, swapab_kernel: Any):
    """Only the live heads' groups are written: ``wo_a`` reads no others."""
    case = _make_case(3, heads, "fp4", cos_sin_cache, seed=heads, full=False)
    out = _alloc_output(3)
    _run_swapab(swapab_kernel, case, out)
    groups = heads // WV_GROUP_SIZE
    assert (out.data[:, groups:].view(torch.uint8) == SENTINEL).all()
    assert (out.scale[:, groups:] == SENTINEL_WORD).all()


@requires_mega_attn
@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize("extra", ["fp4", "fp8"])
@pytest.mark.parametrize("s_q", [1, 7, 16])
def test_cuda_graph_and_determinism(
    heads: int,
    extra: str,
    s_q: int,
    cos_sin_cache: torch.Tensor,
    swapab_kernel: Any,
):
    """Repeated calls and a graph replay agree bit for bit.

    Each CTA sums its cluster peers' partials in a fixed order, so the output
    does not depend on which CTA finishes first.
    """
    case = _make_case(s_q, heads, extra, cos_sin_cache, seed=77 + s_q, full=False)
    first, second, replayed = (_alloc_output(s_q) for _ in range(3))
    _run_swapab(swapab_kernel, case, first)  # compiles outside the capture
    _run_swapab(swapab_kernel, case, second)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_swapab(swapab_kernel, case, replayed)
    graph.replay()
    torch.accelerator.synchronize()
    _assert_same_output(second, first, heads)
    _assert_same_output(replayed, first, heads)

    # A replay reads what the inputs hold now, as every decode step rewrites them.
    case.q_fused.neg_()
    case.swa_len.copy_((case.swa_len + 1) // 2)
    graph.replay()
    _run_swapab(swapab_kernel, case, first)
    torch.accelerator.synchronize()
    _assert_same_output(replayed, first, heads)


def _cpu_layer(
    heads: int, compress_ratio: int, extra_width: int = EXTRA_WIDTH
) -> DeepseekV4MegaAttnAttention:
    """A mega-attention layer with just what its decode dispatch reads."""
    layer = object.__new__(DeepseekV4MegaAttnAttention)
    layer.n_local_heads = heads
    layer.compress_ratio = compress_ratio
    layer.window_size = SWA_WIDTH
    layer.topk_indices_buffer = torch.zeros(1, extra_width, dtype=torch.int32)
    layer.n_wv_group = PADDED_HEADS // WV_GROUP_SIZE
    layer.scale = SCALE
    layer.attn_sink = torch.zeros(PADDED_HEADS)
    layer.rotary_emb = SimpleNamespace(cos_sin_cache=torch.zeros(1, ROPE_DIM))
    layer.swa_cache_layer = SimpleNamespace(
        block_size=SWA_BLOCK,
        kv_cache=torch.zeros(1, SWA_BLOCK, FP8_RECORD, dtype=torch.uint8),
    )
    return layer


def _decode_step(
    layer: DeepseekV4MegaAttnAttention, num_tokens: int, swa_width: int = SWA_WIDTH
) -> None:
    """``_forward_decode_mega`` on a step of ``num_tokens`` zero tokens."""
    swa_metadata = SimpleNamespace(
        decode_swa_indices=torch.zeros(num_tokens, 1, swa_width, dtype=torch.int32),
        decode_swa_lens=torch.zeros(num_tokens, dtype=torch.int32),
    )
    layer._forward_decode_mega(
        torch.zeros(num_tokens, PADDED_HEADS, HEAD_DIM, dtype=torch.bfloat16),
        torch.zeros(num_tokens, dtype=torch.int32),
        None,
        swa_metadata,
        alloc_mega_attn_output(num_tokens, layer.n_wv_group, torch.device("cpu")),
    )


@pytest.mark.parametrize(
    (
        "heads",
        "compress_ratio",
        "num_tokens",
        "swa_width",
        "extra_width",
        "batch_invariant",
        "cutedsl",
        "expected",
    ),
    [
        (16, 1, 1, 128, 512, False, True, "swapab"),
        (32, 2, 16, 128, 512, False, True, "swapab"),
        (64, 1, 1, 128, 512, False, True, "mega"),  # TP1: no padded heads to save
        (16, 0, 1, 128, 512, False, True, "mega"),  # sliding-window-only layer
        (16, 1, 17, 128, 512, False, True, "mega"),  # past the small-step regime
        (16, 1, 1, 128, 1024, False, True, "mega"),  # KV tiles > a portable cluster
        (16, 1, 1, 192, 512, False, True, "mega"),  # DSpark's non-causal draft step
        (16, 1, 1, 128, 512, True, True, "mega"),  # the kernels round differently
        (16, 1, 1, 128, 512, False, False, "mega"),  # no CuTe DSL to import
    ],
)
def test_dispatch_rule(
    monkeypatch: pytest.MonkeyPatch,
    heads: int,
    compress_ratio: int,
    num_tokens: int,
    swa_width: int,
    extra_width: int,
    batch_invariant: bool,
    cutedsl: bool,
    expected: str,
):
    """Which kernel a decode step takes. Both kernels are mocked: no GPU."""
    swapab = _swapab_module()
    calls: list[str] = []

    def mock_swapab(**kwargs: Any) -> None:
        assert kwargs["heads"] == heads
        assert kwargs["swa_idx"].shape == (num_tokens, SWA_WIDTH)
        assert kwargs["extra_cache"].shape == (1, EXTRA_BLOCK, NVFP4_RECORD)
        calls.append("swapab")

    monkeypatch.setattr(swapab, "_SWAPAB_DECODE_KERNEL", mock_swapab)
    # Whatever CUDA the CuTe DSL here was built with.
    monkeypatch.setattr(swapab, "target_version", lambda **_: True)
    monkeypatch.setattr(
        torch.ops,
        "_flashmla_C",
        SimpleNamespace(
            fused_norm_rope_attn_rope_cast_decode=lambda *_: calls.append("mega")
        ),
    )
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability == 100,
    )
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    monkeypatch.setattr(mega_attn, "has_cutedsl", lambda: cutedsl)

    layer = _cpu_layer(heads, compress_ratio, extra_width)
    layer._use_swapab = mega_attn._swapab_decode_supported(layer)
    compressed = (
        (
            torch.zeros(1, EXTRA_BLOCK, 1, NVFP4_RECORD, dtype=torch.uint8),
            torch.zeros(num_tokens, extra_width, dtype=torch.int32),
            torch.zeros(num_tokens, dtype=torch.int32),
        )
        if compress_ratio
        else (None, None, None)
    )
    monkeypatch.setattr(layer, "_decode_compressed_kv_and_topk", lambda *_: compressed)

    _decode_step(layer, num_tokens, swa_width)
    assert calls == [expected]


@pytest.mark.parametrize("extra_record", [NVFP4_RECORD, FP8_RECORD])
@pytest.mark.parametrize("extra_block", [64, 128])
def test_warmup_key_matches_launch(
    monkeypatch: pytest.MonkeyPatch, extra_record: int, extra_block: int
):
    """``bind_kv_cache`` registers the compile key a decode step launches.

    Any other key would compile at its first use, after the JIT warmup.
    """
    swapab = _swapab_module()
    registered: list[dict[str, int]] = []
    launched: list[Any] = []
    monkeypatch.setattr(
        swapab.SwapABDecodeKernel,
        "register_warmup",
        lambda self, **kwargs: registered.append(kwargs),
    )
    monkeypatch.setattr(
        swapab.SwapABDecodeKernel,
        "launch",
        lambda self, launch_spec, inputs: launched.append(launch_spec[0]),
    )

    layer = _cpu_layer(16, compress_ratio=1)
    layer.is_kv_source = True
    layer._use_swapab = layer._swapab_jit_warmup = True
    # [num_blocks, 1, page, record], as the model runner binds it.
    layer.bind_kv_cache(torch.zeros(2, 1, extra_block, extra_record, dtype=torch.uint8))
    num_tokens = 3
    compressed = (
        layer._compressed_kv_cache().unsqueeze(-2),
        torch.zeros(num_tokens, EXTRA_WIDTH, dtype=torch.int32),
        torch.zeros(num_tokens, dtype=torch.int32),
    )
    monkeypatch.setattr(layer, "_decode_compressed_kv_and_topk", lambda *_: compressed)

    _decode_step(layer, num_tokens)
    assert len(registered) == 1
    assert swapab._SWAPAB_DECODE_KERNEL.get_warmup_keys(**registered[0]) == launched
