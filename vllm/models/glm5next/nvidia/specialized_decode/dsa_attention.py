# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA attention stage (``dsa_attn``): the MLA half of the DSA layer around the
indexer and the TRTLLM-Gen sparse attention kernel, in CuTe kernels.

Stock decode runs the q/kv a-norm (Triton), then after the indexer's top-k the
MLA KV cache write, the cuBLAS absorb bmm (query x W_UK), the query FP8 quant
and a zero fill of the attention kernel's multi-CTA counters, all between the
top-k and the attention, then the cuBLAS W_UV bmm. Here:

- ``norm`` is ``fused_q_kv_rmsnorm`` in CuTe;
- ``head_gemv`` with ``cache`` is the absorb bmm with the static FP8 query quant
  as its epilogue, and M more CTAs write the normed latent into the FP8 KV cache
  (``concat_and_cache_mla``). It runs right after ``q_b_proj``, before the
  indexer, so only the index remap separates the top-k from the attention;
- the attention kernel gets one caller-owned counter buffer (it resets the
  counters itself), so FlashInfer zero-fills none per call;
- ``head_gemv`` is the W_UV bmm (and, for the KDA stage, the output gate's
  g_b_proj).

The bmms are per-head ``x @ W`` products with W row-major [I, J]: a thread
holds RPT rows of eight consecutive columns, loaded before the PDL wait, and
the row groups are summed across lanes and warps in a fixed order. Their FP32
sums differ from cuBLAS's only in summation order; the BF16 rounding and the
FP8 conversions follow the stock kernels.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
    stream,
    to_cute,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    alloc,
    asm,
    bf16x2,
    div_rn,
    fp8_bits,
    load_vector,
    load_words,
    pack_fp8,
    store_words,
    unpack_bf16,
    warp_sum,
)

from .mega_moe import _store_bf16
from .shared_expert import _round_bf16

FP8_MAX = 448.0
MAX_TOKENS = 8  # FP32 accumulators per thread: 8 columns x tokens

_compiled: dict = {}


@asm(Float32, pure=True)
def _div_full(a, b):
    """Triton's FP32 ``/``."""
    return "div.full.f32 $0,$1,$2;", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def _rsqrt(value):
    """Triton's ``tl.rsqrt``."""
    return "rsqrt.approx.f32 $0,$1;", "=f,f", (value,)


@cute.jit
def _clamp_fp8(value: Float32):
    """``scaled_fp8_conversion``'s clamp before the saturating conversion."""
    return cute.arch.fmax(cute.arch.fmin(value, Float32(FP8_MAX)), Float32(-FP8_MAX))


class _NormKernel:
    """``fused_q_kv_rmsnorm`` of the a-projection's q and kv parts (CTA (t, 0)
    and (t, 1)). A thread owns eight values; the sum of squares adds the
    thread's, then the lanes' and then the warps' partials."""

    def __init__(self, q_size: int, kv_size: int, pitch: int):
        assert q_size % 8 == 0 and kv_size % 8 == 0 and q_size >= kv_size
        self.q_size, self.kv_size, self.pitch = q_size, kv_size, pitch
        self.threads = (q_size // 8 + 31) // 32 * 32

    @cute.jit
    def __call__(
        self, qkv: cute.Tensor, q_w: cute.Tensor, kv_w: cute.Tensor,
        q_out: cute.Tensor, kv_out: cute.Tensor, eps: Float32, tokens: Int32,
        st: cuda.CUstream,
    ):  # fmt: skip
        self.kernel(qkv, q_w, kv_w, q_out, kv_out, eps).launch(
            grid=(tokens, 2, 1), block=(self.threads, 1, 1), stream=st, use_pdl=True
        )

    @cute.kernel
    def kernel(
        self, qkv: cute.Tensor, q_w: cute.Tensor, kv_w: cute.Tensor,
        q_out: cute.Tensor, kv_out: cute.Tensor, eps: Float32,
    ):  # fmt: skip
        tid, _, _ = cute.arch.thread_idx()
        t, part, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        partial = alloc(smem, Float32, (self.threads // 32,))
        if part == 0:
            self.row(qkv, q_w, q_out, partial, t, tid, 0, self.q_size, eps)
        else:
            self.row(qkv, kv_w, kv_out, partial, t, tid, self.q_size, self.kv_size, eps)

    @cute.jit
    def row(
        self, qkv, weight, out, partial, t, tid, start, size: cutlass.Constexpr, eps
    ):
        """The thread's eight columns of the ``size``-wide part at ``start``."""
        lane = cute.arch.lane_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        active = tid < size // 8
        gamma = cute.make_rmem_tensor(cute.make_layout((8,)), Float32)
        gamma.fill(0.0)
        if active:
            got = load_words(weight.iterator + tid * 8, count=4, word_type="u32")
            for i in cutlass.range_constexpr(4):
                gamma[2 * i], gamma[2 * i + 1] = unpack_bf16(got[i])
        cute.arch.griddepcontrol_wait()
        # Dependents read the outputs only after their own wait.
        cute.arch.griddepcontrol_launch_dependents()
        x = cute.make_rmem_tensor(cute.make_layout((8,)), Float32)
        x.fill(0.0)
        if active:
            got = load_words(
                qkv.iterator + t * self.pitch + start + tid * 8,
                count=4, word_type="u32", mutable=True,
            )  # fmt: skip
            for i in cutlass.range_constexpr(4):
                x[2 * i], x[2 * i + 1] = unpack_bf16(got[i])
        squares = Float32(0.0)
        for i in cutlass.range_constexpr(8):
            squares = squares + x[i] * x[i]
        squares = warp_sum(squares)
        if lane == 0:
            partial[warp] = squares
        cute.arch.sync_threads()
        total = Float32(0.0)
        for w in cutlass.range_constexpr(self.threads // 32):
            total = total + partial[w]
        rrms = _rsqrt(_div_full(total, Float32(size)) + eps)
        if active:
            y = [x[i] * rrms * gamma[i] for i in range(8)]
            words = [bf16x2(y[2 * i], y[2 * i + 1]) for i in range(4)]
            store_words(out.iterator + t * size + tid * 8, words)


def norm(qkv, q_weight, kv_weight, eps, q_size):
    """``fused_q_kv_rmsnorm(qkv[:, :q_size], qkv[:, q_size:], ...)``: the normed
    (q [M, q_size], kv [M, kv_size]) BF16 parts of ``qkv``."""
    m, pitch = qkv.shape
    kv_size = kv_weight.shape[0]
    assert qkv.stride(1) == 1 and qkv.stride(0) == pitch
    assert pitch == q_size + kv_size
    q_out = torch.empty((m, q_size), dtype=qkv.dtype, device=qkv.device)
    kv_out = torch.empty((m, kv_size), dtype=qkv.dtype, device=qkv.device)
    args = (qkv, q_weight, kv_weight, q_out, kv_out)
    st = stream(qkv.device)
    key = ("norm", m, q_size, kv_size, qkv.device.index)
    if key not in _compiled:
        _compiled[key] = cute.compile(
            _NormKernel(q_size, kv_size, pitch),
            *(to_cute(t, 16) for t in args), Float32(eps), Int32(m), st,
        )  # fmt: skip
    _compiled[key](*(to_cute(t, 16) for t in args), Float32(eps), Int32(m), st)
    return q_out, kv_out


class _HeadGemvKernel:
    """``out[t, h] = x[t, h] @ W[h]`` for ``heads`` row-major [I, J] weights.

    CTA (h, tile) owns J columns ``tile * cols ...``; thread (g, c) holds rows
    ``g * rpt ...`` of columns ``8c ...``, loaded before the PDL wait. Lanes of
    one column group sum their row groups with butterflies, then the warps' sums
    add in warp order. ``fp8``: the BF16 product times the inverse of
    ``scale[0]``, clamped and converted as the stock static FP8 quant.

    ``cache`` (latent width): CTA ``gemv CTAs + t`` writes token t's latent row
    into the FP8 KV cache as ``concat_and_cache_mla`` (``kv / k_scale``, padded
    slots skipped).
    """

    def __init__(
        self, heads, rows, cols, tile, rpt, tokens, w_strides, fp8, cache=0,
        x_stride=None,
    ):  # fmt: skip
        assert tile % 8 == 0 and cols % tile == 0 and rows % rpt == 0 and rpt % 8 == 0
        self.heads, self.rows, self.cols, self.tile, self.rpt = (
            heads, rows, cols, tile, rpt,
        )  # fmt: skip
        self.tokens, self.fp8, self.cache = tokens, fp8, cache
        self.head_stride, self.row_stride = w_strides
        self.x_stride = heads * rows if x_stride is None else x_stride
        self.col_threads = tile // 8
        self.threads = self.col_threads * (rows // rpt)
        assert 32 % self.col_threads == 0 and self.threads % 32 == 0
        assert self.threads <= 1024 and cache // 8 <= self.threads
        self.warps = self.threads // 32
        self.tiles = cols // tile
        self.ctas = heads * self.tiles + (tokens if cache else 0)

    @cute.jit
    def __call__(
        self, w: cute.Tensor, x: cute.Tensor, out: cute.Tensor,
        scale: cute.Tensor, kv: cute.Tensor, cache: cute.Tensor,
        slots: cute.Tensor, k_scale: cute.Tensor, block_size: Int32,
        block_stride: Int64, entry_stride: Int64, n_slots: Int32,
        st: cuda.CUstream,
    ):  # fmt: skip
        self.kernel(
            w, x, out, scale, kv, cache, slots, k_scale, block_size, block_stride,
            entry_stride, n_slots,
        ).launch(
            grid=(self.ctas, 1, 1),
            block=(self.threads, 1, 1),
            stream=st,
            use_pdl=True,
        )  # fmt: skip

    @cute.kernel
    def kernel(
        self, w: cute.Tensor, x: cute.Tensor, out: cute.Tensor,
        scale: cute.Tensor, kv: cute.Tensor, cache: cute.Tensor,
        slots: cute.Tensor, k_scale: cute.Tensor, block_size: Int32,
        block_stride: Int64, entry_stride: Int64, n_slots: Int32,
    ):  # fmt: skip
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        partial = alloc(smem, Float32, (self.warps, self.tokens, self.tile))
        if block < self.heads * self.tiles:
            self.gemv(w, x, out, scale, partial, tid, block)
        else:
            self.write_cache(
                kv, cache, slots, k_scale, block_size, block_stride, entry_stride,
                n_slots, tid, block - self.heads * self.tiles,
            )  # fmt: skip

    @cute.jit
    def write_cache(
        self, kv, cache, slots, k_scale, block_size, block_stride, entry_stride,
        n_slots, tid, t,
    ):  # fmt: skip
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        # Rows past the slot mapping (CUDA graph padding) are not written.
        slot = Int64(-1)
        if t < n_slots:
            slot = Int64(slots[t])
        if tid < self.cache // 8 and slot >= 0:
            got = load_words(
                kv.iterator + t * self.cache + tid * 8,
                count=4, word_type="u32", mutable=True,
            )  # fmt: skip
            scale = k_scale[0]
            v = []
            for i in cutlass.range_constexpr(4):
                lo, hi = unpack_bf16(got[i])
                v += [div_rn(lo, scale), div_rn(hi, scale)]
            block = slot // Int64(block_size)
            offset = slot - block * Int64(block_size)
            address = cache.iterator + block * block_stride
            address = address + offset * entry_stride + tid * 8
            store_words(address, (pack_fp8(tuple(v[:4])), pack_fp8(tuple(v[4:]))))

    @cute.jit
    def gemv(self, w, x, out, scale, partial, tid, block):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        h = block // self.tiles
        col = block % self.tiles * self.tile + tid % self.col_threads * 8
        row = tid // self.col_threads * self.rpt
        b = cute.make_rmem_tensor(cute.make_layout((self.rpt, 4)), Uint32)
        base = Int64(h) * self.head_stride + col
        for r in cutlass.range_constexpr(self.rpt):
            got = load_vector(
                w.iterator + base + Int64(row + r) * self.row_stride, cache="weight"
            )
            for i in cutlass.range_constexpr(4):
                b[r, i] = got[i]
        inverse = Float32(1.0)
        cute.arch.griddepcontrol_wait()
        # Dependents read the outputs only after their own wait.
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.fp8):
            inverse = div_rn(Float32(1.0), scale[0])

        weights = cute.make_rmem_tensor(cute.make_layout((self.rpt, 8)), Float32)
        for r in cutlass.range_constexpr(self.rpt):
            for i in cutlass.range_constexpr(4):
                weights[r, 2 * i], weights[r, 2 * i + 1] = unpack_bf16(b[r, i])
        acc = cute.make_rmem_tensor(cute.make_layout((self.tokens, 8)), Float32)
        acc.fill(0.0)
        for t in cutlass.range_constexpr(self.tokens):
            a = cute.make_rmem_tensor(cute.make_layout((self.rpt,)), Float32)
            for p in cutlass.range_constexpr(self.rpt // 8):
                got = load_words(
                    x.iterator + t * self.x_stride + h * self.rows + row + p * 8,
                    count=4, word_type="u32", mutable=True,
                )  # fmt: skip
                for i in cutlass.range_constexpr(4):
                    a[p * 8 + 2 * i], a[p * 8 + 2 * i + 1] = unpack_bf16(got[i])
            for r in cutlass.range_constexpr(self.rpt):
                for k in cutlass.range_constexpr(8):
                    acc[t, k] = acc[t, k] + a[r] * weights[r, k]
        # Row groups of a warp: lanes c, c + col_threads, ... of one column group.
        offset = self.col_threads
        while offset < 32:
            for t in cutlass.range_constexpr(self.tokens):
                for k in cutlass.range_constexpr(8):
                    acc[t, k] = acc[t, k] + cute.arch.shuffle_sync_bfly(
                        acc[t, k], offset=offset
                    )
            offset *= 2
        if lane < self.col_threads:
            for t in cutlass.range_constexpr(self.tokens):
                for k in cutlass.range_constexpr(8):
                    partial[warp, t, lane * 8 + k] = acc[t, k]
        cute.arch.sync_threads()
        for e in cutlass.range(tid, self.tokens * self.tile, self.threads, unroll=1):
            t = e // self.tile
            c = e % self.tile
            total = Float32(0.0)
            for v in cutlass.range_constexpr(self.warps):
                total = total + partial[v, t, c]
            j = block % self.tiles * self.tile + c
            index = (t * self.heads + h) * self.cols + j
            if cutlass.const_expr(self.fp8):
                out[index] = fp8_bits(_clamp_fp8(_round_bf16(total) * inverse))
            else:
                _store_bf16(out.iterator + index, total)


# (cols per CTA, rows per thread) by (rows, cols) and tokens, from a GB300
# sweep with the weights rotated out of L2.
TACTICS = {
    (256, 512): {1: (64, 8), 2: (64, 8), 4: (64, 16), 8: (64, 16)},  # absorb
    (512, 256): {1: (64, 16), 2: (32, 16), 4: (32, 16)},  # value up
    (128, 2048): {1: (64, 8), 2: (64, 8), 4: (64, 8), 8: (64, 8)},  # KDA g_b
}


def _flat(tensor):
    """``tensor``'s storage span as a 1-D view."""
    span = sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride())) + 1
    return tensor.as_strided((span,), (1,))


def head_gemv(x, weight, out, scale=None, cache=None):
    """``out[t, h] = x[t, h] @ weight[h]`` with ``x`` [M, H, I] BF16 (any token
    stride) and ``weight`` [H, I, J] BF16 (J contiguous); ``out`` [M, H, J] is
    BF16, or FP8 (statically quantized by ``scale``, a one-element FP32 tensor)
    when ``scale`` is given.

    Args:
        x: [M, H, I] BF16 activations.
        weight: [H, I, J] BF16 weights.
        out: [M, H, J] BF16 or FP8 output.
        scale: the static FP8 scale of an FP8 ``out``.
        cache: optionally (kv [M, L] BF16, kv_cache [blocks, block size, L] FP8,
            slots int64, k_scale), written as ``concat_and_cache_mla``.

    """
    m, heads, rows = x.shape
    _, _, cols = weight.shape
    assert weight.shape == (heads, rows, cols) and weight.stride(2) == 1
    assert x.stride(2) == 1 and x.stride(1) == rows and x.stride(0) % 8 == 0
    assert out.is_contiguous() and m <= MAX_TOKENS
    dev = x.device
    fp8 = scale is not None
    if fp8:
        assert out.dtype == torch.float8_e4m3fn
        out = out.view(torch.uint8)
    dummy = torch.empty(64, dtype=torch.bfloat16, device=dev)
    scale = scale if fp8 else dummy.view(torch.float32)[:1]
    width, ints = 0, (Int32(1), Int64(0), Int64(0), Int32(0))
    kv_args = (dummy, dummy.view(torch.uint8), dummy.view(torch.int64)[:1], scale)
    if cache is not None:
        kv, kv_cache, slots, k_scale = cache
        width = kv.shape[1]
        assert kv.shape[0] == m and kv.is_contiguous() and kv_cache.shape[-1] == width
        assert kv_cache.dim() == 3 and kv_cache.stride(2) == 1
        assert slots.dim() == 1 and slots.dtype == torch.int64
        raw = kv_cache.view(torch.uint8)
        kv_args = (kv.view(-1), _flat(raw), slots, k_scale)
        ints = (
            Int32(raw.shape[1]), Int64(raw.stride(0)), Int64(raw.stride(1)),
            Int32(min(slots.shape[0], m)),
        )  # fmt: skip
    tile, rpt = TACTICS.get((rows, cols), {}).get(m, (64, 8))
    strides = (weight.stride(0), weight.stride(1))
    args = (_flat(weight), _flat(x), out.view(-1), scale, *kv_args)
    st = stream(dev)
    key: tuple = ("gemv", heads, rows, cols, tile, rpt, m, strides, fp8, width)
    key += (x.stride(0), dev.index) + tuple(t.shape for t in args)
    if key not in _compiled:
        kernel = _HeadGemvKernel(
            heads, rows, cols, tile, rpt, m, strides, fp8, width, x.stride(0)
        )
        _compiled[key] = cute.compile(
            kernel, *(to_cute(t, 16) for t in args), *ints,
            st,
        )  # fmt: skip
    _compiled[key](*(to_cute(t, 16) for t in args), *ints, st)
    return out


class DsaAttentionStage:
    """``MultiHeadLatentAttentionWrapper.forward`` of a DSA layer inside planned
    steps of up to ``MAX_TOKENS`` tokens. Installed as the wrapper's ``forward``;
    every other call runs the stock forward."""

    def __init__(self, attention, value_up_tokens: set[int]):
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
            FlashInferMLASparseImpl,
        )

        w = attention.mla_attn
        mla = w.mla_attn
        impl = mla.impl
        checks = (
            (w.q_lora_rank is not None and w.fuse_qkv_rmsnorm, "a fused q/kv a-norm"),
            (w.rotary_emb is None and w.qk_rope_head_dim == 0, "NoPE"),
            (w.indexer is not None and not w.skip_topk, "its own indexer"),
            (w.g_proj is None and not w.dcp_q_replicate, "no gate or DCP query"),
            (isinstance(impl, FlashInferMLASparseImpl), "the TRTLLM-Gen sparse MLA"),
            (str(mla.kv_cache_dtype) in ("fp8", "fp8_e4m3"), "an FP8 KV cache"),
            (mla.q_pad_num_heads is None and not mla.use_pcp, "no padding or PCP"),
            (mla.hisparse_cache is None and impl.dcp_world_size == 1, "no HiSparse"),
            (mla.W_UK_T.stride(-1) == 1, "a column-major W_UK"),
        )
        missing = [what for ok, what in checks if not ok]
        if missing:
            raise ValueError("DSA attention stage needs " + ", ".join(missing))
        self.wrapper, self.mla, self.impl = w, mla, impl
        self.heads = w.num_heads
        self.q_size = w.q_lora_rank
        self.w_uk = mla.W_UK_T  # [N, P, L]
        self.w_uv = mla.W_UV.contiguous()  # [N, L, V]
        # Token counts whose W_UV runs as the CuTe GEMV; cuBLAS's bmm otherwise.
        self.value_up_tokens = value_up_tokens
        self.stock = w.forward
        w.forward = self.forward

    def supports(self, md, m: int) -> bool:
        return (
            m <= MAX_TOKENS
            and md is not None
            and md.num_actual_tokens == m
            and md.num_decode_tokens == m
        )

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        from vllm.forward_context import get_forward_context
        from vllm.platforms import current_platform

        from .linear import planned

        m = hidden_states.shape[0]
        mla = self.mla
        if not planned() or llama_4_scaling is not None:
            return self.stock(positions, hidden_states, llama_4_scaling)
        context = get_forward_context()
        md = context.attn_metadata[mla.layer_name]
        if not self.supports(md, m):
            return self.stock(positions, hidden_states, llama_4_scaling)
        w = self.wrapper
        qkv = w.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c = norm(
            qkv,
            w.q_a_layernorm.weight.data,
            w.kv_a_layernorm.weight.data,
            w.q_a_layernorm.variance_epsilon,
            self.q_size,
        )
        q = w.q_b_proj(q_c)[0].view(m, self.heads, -1)
        q_fp8 = torch.empty(
            (m, self.heads, self.w_uk.shape[2]),
            dtype=torch.float8_e4m3fn,
            device=q.device,
        )
        cache = (kv_c, mla.kv_cache, context.slot_mapping[mla.layer_name], mla._k_scale)
        head_gemv(q, self.w_uk, q_fp8, mla._q_scale, cache)
        if not w.skip_topk:  # MTP draft steps 1+ reuse step 0's top-k
            w.indexer(hidden_states, q_c, positions, w.indexer_rope_emb)
        self.impl.record_logical_topk_ready()
        kv_cache = mla.kv_cache.view(current_platform.fp8_dtype())
        latent, _ = self.impl.forward_mqa(q_fp8, kv_cache, md, mla)
        out = torch.empty(
            (m, self.heads, self.w_uv.shape[2]), dtype=q.dtype, device=q.device
        )
        latent = latent.view(m, self.heads, -1)
        if m in self.value_up_tokens:
            head_gemv(latent, self.w_uv, out)
        else:
            torch.bmm(latent.transpose(0, 1), mla.W_UV, out=out.transpose(0, 1))
        return w.o_proj(out.view(m, -1))[0]
