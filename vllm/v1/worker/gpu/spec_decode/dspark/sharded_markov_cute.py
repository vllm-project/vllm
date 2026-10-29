# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark's sequential Markov sampling for a small batch, in CuTe DSL.

The kernel samples up to 8 request rows, with the Markov head sharded by
vocabulary across the tensor-parallel ranks (``sharded_markov.py`` explains the
design). It is compiled for a number of draft positions ``steps`` (1..8) and
the element types of the base logits and of the draft-logits cache (BF16 or
FP32).

For request ``r``, step ``i`` needs r's previous token ``p`` (its anchor for
``i = 0``)::

    logits_ri = cache_round(base_ri + bf16(markov_head . markov_embed[p]))
    d_r,i+1   = argmax(logits_ri)                     (temperature 0)
              = argmax(logits_ri / T_r + gumbel_ri)   (otherwise)

Each rank owns rows ``[rank * V/world, (rank + 1) * V/world)`` of the head.
Each warp keeps one 16-row tile of that shard in registers for all steps, so
the head is read once per launch; the ``m16n8k16`` MMA's eight columns carry the
eight request rows' embeddings, so every row's bias comes from the same MMAs.
A step is those MMAs, a CTA reduction and a two-level exchange of 64-bit
candidate keys per request row: the CTAs of a rank fold their best key into a
rank-local maximum, the last CTA to arrive stores that key, tagged with the
launch epoch, into every rank's mailbox, and every CTA waits for the ``world``
records of each row in its own mailbox and takes their maximum. The exchange is
the grid barrier and the cross-rank argmax at once, so the grid must be
co-resident: the kernel is launched cooperatively, and its 512-thread
requirement keeps one CTA per SM within the register file.

The base logits may hold the full vocabulary or only this rank's shard (the
draft LM head's output before its all-gather): ``base_column`` is the column of
this rank's first vocabulary entry in them, ``rank * V/world`` or 0.

Rows whose request-state index is negative (CUDA-graph padding) are inert, as
in the fused path: their tokens are drawn greedily and their logits are not
cached.

The draft-logits cache must hold the full vocabulary on every rank. Each rank
writes its shard into its own cache and into a staging buffer of every peer;
after the last exchange each rank copies the peers' shards from its staging
buffer into its cache. Staging and mailbox records alternate between two
parities so that a launch never overwrites what a peer may still be reading
from the previous launch.

Peer buffer layout (bytes): mailbox records u64
``[parity 2][step 8][row 8][source rank 8]`` (8 KiB), then the staging logits
``[parity 2][max_rows][steps][vocab]`` of the cache type at ``STAGING_OFFSET``.
Local control words (uint32): combine maxima u64 ``[step 8][row 8]`` at words
0..127, arrivals ``[step 8]`` at 128..135, launches at 136, end arrivals at 137.

Global memory is addressed with 64-bit integers and accessed with explicit PTX,
so that each access has the state space, cache operator and memory ordering
the protocol needs (relaxed system-scope mailbox records, L2-only staging
reads, non-coherent reads of the weights).
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

RANK_DIM = 256
CHUNKS = RANK_DIM // 32  # 16-byte head vectors per lane and row
WARPS = 16
THREADS = WARPS * 32
MAX_STEPS = 8
MAX_RANKS = 8
MAX_ROWS = 8  # MMA columns
RECORD_WORDS = 2 * MAX_STEPS * MAX_ROWS * MAX_RANKS
STAGING_OFFSET = 16384
ARRIVAL_WORD = 128
LAUNCH_WORD = 136
END_WORD = 137
TAG_BITS = 15
TAG_MASK = (1 << TAG_BITS) - 1
INDEX_BITS = 17
INDEX_MASK = (1 << INDEX_BITS) - 1
ROW_BYTES = RANK_DIM * 2  # one BF16 embedding or head row

assert RECORD_WORDS * 8 <= STAGING_OFFSET


def _asm(result, operands: list, text: str, constraints: str, effects: bool, loc, ip):
    """A PTX inline assembly operation; ``effects`` keeps it in program order
    with the other side-effecting operations."""
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _values(values: list, loc, ip) -> list:
    """The IR values of DSL scalars."""
    return [value.ir_value(loc=loc, ip=ip) for value in values]


def _fields(result, count: int, kind, loc, ip) -> list:
    """The ``count`` fields of an inline assembly struct result as ``kind``."""
    ir_type = T.f32() if kind is Float32 else T.i32()
    return [
        kind(llvm.extractvalue(ir_type, result, [index], loc=loc, ip=ip))
        for index in range(count)
    ]


@dsl_user_op
def load_v4(address: Int64, opcode: str, effects: bool, *, loc=None, ip=None) -> list:
    """Four 32-bit words at a 16-byte aligned global address."""
    result = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        _values([Int64(address)], loc, ip),
        f"{opcode} {{$0, $1, $2, $3}}, [$4];",
        "=r,=r,=r,=r,l",
        effects,
        loc,
        ip,
    )
    return _fields(result, 4, Uint32, loc, ip)


@dsl_user_op
def store_global_v4(address: Int64, words: list, *, loc=None, ip=None) -> None:
    """``st.global.v4.u32``."""
    _asm(
        None,
        _values([Int64(address), *(Uint32(w) for w in words)], loc, ip),
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def shared_address(pointer: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """The shared-window address of a shared-memory pointer."""
    return Int32(pointer.toint(loc=loc, ip=ip))


@dsl_user_op
def load_shared_v4(address: Int32, *, loc=None, ip=None) -> list:
    """Four 32-bit words at a 16-byte aligned shared address."""
    result = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        _values([Int32(address)], loc, ip),
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        True,
        loc,
        ip,
    )
    return _fields(result, 4, Uint32, loc, ip)


@dsl_user_op
def store_shared_v4(address: Int32, words: list, *, loc=None, ip=None) -> None:
    """``st.shared.v4.u32``."""
    _asm(
        None,
        _values([Int32(address), *(Uint32(w) for w in words)], loc, ip),
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def load_scalar(
    address: Int64, opcode: str, kind, effects: bool = False, *, loc=None, ip=None
):
    """One 32- or 64-bit global value of DSL type ``kind`` (``opcode`` names
    the load, e.g. ``ld.global.nc.u64``)."""
    if kind is Float32:
        ir_type, constraint = T.f32(), "f"
    elif kind in (Int64, Uint64):
        ir_type, constraint = T.i64(), "l"
    else:
        ir_type, constraint = T.i32(), "r"
    return kind(
        _asm(
            ir_type,
            _values([Int64(address)], loc, ip),
            f"{opcode} $0, [$1];",
            f"={constraint},l",
            effects,
            loc,
            ip,
        )
    )


@dsl_user_op
def load_bf16_as_f32(address: Int64, *, loc=None, ip=None) -> Float32:
    """A global BF16 value, widened."""
    return Float32(
        _asm(
            T.f32(),
            _values([Int64(address)], loc, ip),
            "{\n.reg .b16 h;\nld.global.b16 h, [$1];\ncvt.f32.bf16 $0, h;\n}",
            "=f,l",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def store_scalar(address: Int64, value, opcode: str, *, loc=None, ip=None) -> None:
    """One 32- or 64-bit global store (``opcode``, e.g. ``st.global.u32``)."""
    constraint = "l" if isinstance(value, (Int64, Uint64)) else "r"
    _asm(
        None,
        _values([Int64(address), value], loc, ip),
        f"{opcode} [$0], $1;",
        f"l,{constraint}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def atomic(address: Int64, value, opcode: str, *, loc=None, ip=None):
    """A global atomic returning the old value (``opcode`` e.g.
    ``atom.global.add.u32``); ``value`` is Uint32 or Uint64."""
    wide = isinstance(value, Uint64)
    return type(value)(
        _asm(
            T.i64() if wide else T.i32(),
            _values([Int64(address), value], loc, ip),
            f"{opcode} $0, [$1], $2;",
            "=l,l,l" if wide else "=r,l,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def fence(scope: str, *, loc=None, ip=None) -> None:
    """``membar.gl`` (``__threadfence``) or ``membar.sys``
    (``__threadfence_system``)."""
    _asm(None, [], f"membar.{scope};", "", True, loc, ip)


@dsl_user_op
def prefetch_row_l2(address: Int64, *, loc=None, ip=None) -> None:
    """Pull one 512-byte row into L2."""
    text = "\n".join(
        f"prefetch.global.L2 [$0+{line * 128}];" for line in range(ROW_BYTES // 128)
    )
    _asm(None, _values([Int64(address)], loc, ip), text, "l", True, loc, ip)


@dsl_user_op
def trap(*, loc=None, ip=None) -> None:
    """Abort the kernel."""
    _asm(None, [], "trap;", "", True, loc, ip)


@dsl_user_op
def mma_16816(acc: list, a: list, b: list, *, loc=None, ip=None) -> list:
    """``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32``: ``acc`` four
    FP32, ``a`` four and ``b`` two packed BF16 pairs."""
    result = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        _values(
            [
                *(Uint32(v) for v in a),
                *(Uint32(v) for v in b),
                *(Float32(v) for v in acc),
            ],
            loc,
            ip,
        ),
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        False,
        loc,
        ip,
    )
    return _fields(result, 4, Float32, loc, ip)


@dsl_user_op
def div_full(a, b, *, loc=None, ip=None) -> Float32:
    """``div.full.f32``: Triton's FP32 ``/``, as the stock sampler divides."""
    return Float32(
        _asm(
            T.f32(),
            _values([Float32(a), Float32(b)], loc, ip),
            "div.full.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def add_rn(a, b, *, loc=None, ip=None) -> Float32:
    """FP32 add, never contracted into a neighbouring multiply."""
    return Float32(
        _asm(
            T.f32(),
            _values([Float32(a), Float32(b)], loc, ip),
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def round_bf16(value, *, loc=None, ip=None) -> Float32:
    """``value`` rounded to BF16 (nearest even) and widened back."""
    return Float32(
        _asm(
            T.f32(),
            _values([Float32(value)], loc, ip),
            "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\ncvt.f32.bf16 $0, h;\n}",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def float_bits(value, *, loc=None, ip=None) -> Uint32:
    """The bits of an FP32 value."""
    return Uint32(
        _asm(
            T.i32(),
            _values([Float32(value)], loc, ip),
            "mov.b32 $0, $1;",
            "=r,f",
            False,
            loc,
            ip,
        )
    )


@cute.jit
def candidate_key(score: Float32, index: Int32) -> Uint64:
    """[48:17] order-preserving score bits, [16:0] inverted index (ties prefer
    the smaller index, as ``torch.argmax`` does). Larger is better; 0 means "no
    key"."""
    bits = float_bits(score)
    ordered = bits | Uint32(0x80000000)
    if bits & Uint32(0x80000000) != Uint32(0):
        ordered = bits ^ Uint32(0xFFFFFFFF)
    return (Uint64(ordered) << INDEX_BITS) | Uint64(Uint32(INDEX_MASK) - Uint32(index))


@cute.jit
def key_index(key: Uint64) -> Int32:
    """The vocabulary index of a candidate key."""
    return Int32(Uint32(INDEX_MASK) - (Uint32(key) & Uint32(INDEX_MASK)))


@cute.jit
def shuffle_xor_u64(value: Uint64, offset: Int32) -> Uint64:
    """``__shfl_xor_sync`` of a 64-bit value over the full warp."""
    lo = cute.arch.shuffle_sync_bfly(Uint32(value), offset)
    hi = cute.arch.shuffle_sync_bfly(Uint32(value >> 32), offset)
    return (Uint64(Uint32(hi)) << 32) | Uint64(Uint32(lo))


@cute.jit
def shuffle_idx_u64(value: Uint64, source: Int32) -> Uint64:
    """``__shfl_sync`` of a 64-bit value from lane ``source``."""
    lo = cute.arch.shuffle_sync(Uint32(value), source)
    hi = cute.arch.shuffle_sync(Uint32(value >> 32), source)
    return (Uint64(Uint32(hi)) << 32) | Uint64(Uint32(lo))


@cute.jit
def max_u64(a: Uint64, b: Uint64) -> Uint64:
    """The larger of two keys (unsigned; tensor loads come back signed)."""
    result = Uint64(a)
    other = Uint64(b)
    if other > result:
        result = other
    return result


@cute.jit
def await_record(
    address: Int64, tag: Uint32, timeout_ns: Int64, step: Int32, source: Int32
) -> Uint64:
    """Spin until the mailbox record at ``address`` carries ``tag``; return
    its key. With a positive ``timeout_ns`` a wait that exceeds it reports and
    traps instead of hanging (debug aid for ranks that disagree on the launch
    sequence)."""
    value = load_scalar(address, "ld.relaxed.sys.global.u64", Uint64, True)
    if Uint32(value) & Uint32(TAG_MASK) != tag:
        start = Int64(0)
        if timeout_ns > 0:
            start = cute.arch.globaltimer()
        spins = Uint32(0)
        while Uint32(value) & Uint32(TAG_MASK) != tag:
            value = load_scalar(address, "ld.relaxed.sys.global.u64", Uint64, True)
            spins = spins + Uint32(1)
            if timeout_ns > 0 and spins & Uint32(1023) == Uint32(0):
                waited = cute.arch.globaltimer() - start
                if waited > timeout_ns:
                    cute.printf(
                        "sharded Markov sampler: step {} waited {} ns for rank {} "
                        "(tag {}, saw {})",
                        step,
                        timeout_ns,
                        source,
                        tag,
                        Uint32(value) & Uint32(TAG_MASK),
                    )
                    trap()
    return value >> TAG_BITS


class ShardedMarkovSample:
    """The sharded Markov chain kernel for ``steps`` draft positions, base
    logits of ``base_dtype`` and a draft-logits cache of ``cache_dtype``
    (``cutlass.BFloat16`` or ``cutlass.Float32``)."""

    def __init__(self, steps: int, base_dtype, cache_dtype):
        assert 1 <= steps <= MAX_STEPS
        self.steps = steps
        self.base_bf16 = base_dtype is BFloat16
        self.cache_dtype = cache_dtype
        self.cache_bf16 = cache_dtype is BFloat16
        self.cache_bytes = 2 if self.cache_bf16 else 4
        # A 16-row tile of logits is this many 16-byte vectors of this many
        # elements.
        self.vector_elements = 16 // self.cache_bytes
        self.tile_vectors = 16 // self.vector_elements

    @cute.jit
    def __call__(
        self,
        base_logits: Int64,  # [rows, steps, vocab or vocab_local]
        base_row_stride: Int64,  # elements
        base_step_stride: Int64,
        base_column: Int64,  # column of this rank's shard
        embed_table: Int64,  # BF16 [vocab, 256]
        head: Int64,  # this rank's BF16 [vocab_local, 256]
        input_ids: Int64,  # int32; row r's anchor at [r * query_per_row]
        query_per_row: Int32,
        idx_mapping: Int64,  # int64 [rows * steps] request-state rows
        temperature: Int64,  # FP32 [max_num_reqs]
        noise: Int64,  # FP32 [rows * steps, vocab_local]
        logits_cache: Int64,  # [max_num_reqs, steps, vocab]
        cache_stride0: Int64,
        cache_stride1: Int64,
        draft_tokens: Int64,  # int64 [rows, steps]
        draft_stride: Int64,  # row stride
        peers: Int64,  # u64 [world] peer buffer bases
        control: Int64,  # local control words
        rank: Int32,
        world: Int32,
        vocab_local: Int32,
        rows: Int32,
        max_rows: Int32,
        timeout_ns: Int64,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """Launch the chain as a cooperative grid of ``ctas`` CTAs."""
        self.kernel(
            base_logits,
            base_row_stride,
            base_step_stride,
            base_column,
            embed_table,
            head,
            input_ids,
            query_per_row,
            idx_mapping,
            temperature,
            noise,
            logits_cache,
            cache_stride0,
            cache_stride1,
            draft_tokens,
            draft_stride,
            peers,
            control,
            rank,
            world,
            vocab_local,
            rows,
            max_rows,
            timeout_ns,
        ).launch(
            grid=(ctas, 1, 1), block=[THREADS, 1, 1], stream=stream, cooperative=True
        )

    @cute.jit
    def peer_base(self, peers: Int64, peer: Int32) -> Int64:
        """The base address of ``peer``'s buffer."""
        return load_scalar(peers + Int64(peer) * 8, "ld.global.nc.u64", Int64)

    @cute.jit
    def load_base(
        self,
        base_logits: Int64,
        row_stride: Int64,
        step_stride: Int64,
        column: Int64,
        r: Int32,
        step: Int32,
        row: Int32,
    ) -> Float32:
        """Base logit of request row ``r``, ``step`` and shard row ``row``."""
        element = (
            Int64(r) * row_stride + Int64(step) * step_stride + column + Int64(row)
        )
        value = Float32(0.0)
        if cutlass.const_expr(self.base_bf16):
            value = load_bf16_as_f32(base_logits + element * 2)
        else:
            value = load_scalar(base_logits + element * 4, "ld.global.f32", Float32)
        return value

    @cute.jit
    def load_noise(
        self, noise: Int64, r: Int32, step: Int32, row: Int32, vocab_local: Int32
    ) -> Float32:
        """Gumbel noise of request row ``r``, ``step`` and shard row ``row``."""
        element = Int64(r * self.steps + step) * Int64(vocab_local) + Int64(row)
        return load_scalar(noise + element * 4, "ld.global.f32", Float32)

    @cute.jit
    def load_inputs(
        self,
        base: cute.Tensor,
        gumbel: cute.Tensor,
        step: Int32,
        tile: Int32,
        g: Int32,
        t: Int32,
        rows: Int32,
        base_logits: Int64,
        row_stride: Int64,
        step_stride: Int64,
        column: Int64,
        noise: Int64,
        vocab_local: Int32,
    ):
        """Base logits and noise of this lane's resident tile rows and request
        rows for ``step``: ``base``/``gumbel`` [col * 2 + half]."""
        for col in cutlass.range_constexpr(2):
            r = 2 * t + col
            if r < rows:
                for half in cutlass.range_constexpr(2):
                    row = tile * 16 + g + 8 * half
                    base[col * 2 + half] = self.load_base(
                        base_logits, row_stride, step_stride, column, r, step, row
                    )
                    gumbel[col * 2 + half] = self.load_noise(
                        noise, r, step, row, vocab_local
                    )

    @cute.kernel
    def kernel(
        self,
        base_logits: Int64,
        base_row_stride: Int64,
        base_step_stride: Int64,
        base_column: Int64,
        embed_table: Int64,
        head: Int64,
        input_ids: Int64,
        query_per_row: Int32,
        idx_mapping: Int64,
        temperature: Int64,
        noise: Int64,
        logits_cache: Int64,
        cache_stride0: Int64,
        cache_stride1: Int64,
        draft_tokens: Int64,
        draft_stride: Int64,
        peers: Int64,
        control: Int64,
        rank: Int32,
        world: Int32,
        vocab_local: Int32,
        rows: Int32,
        max_rows: Int32,
        timeout_ns: Int64,
    ):
        """One CTA of 16 warps; see the module docstring."""
        steps = self.steps
        cache_bytes = self.cache_bytes
        vector_elements = self.vector_elements
        tile_vectors = self.tile_vectors

        @cute.struct
        class Shared:
            embedding: cute.struct.Align[
                cute.struct.MemRange[Uint32, MAX_ROWS * RANK_DIM // 2], 16
            ]
            tile_logits: cute.struct.Align[
                cute.struct.MemRange[Uint32, WARPS * MAX_ROWS * 16 * cache_bytes // 4],
                16,
            ]
            warp_best: cute.struct.MemRange[Uint64, WARPS * MAX_ROWS]
            winner: cute.struct.MemRange[Uint64, MAX_ROWS]
            count: cute.struct.MemRange[Uint32, 1]

        shared = cutlass.utils.SmemAllocator().allocate(Shared)
        embedding = shared_address(shared.embedding.data_ptr())
        tile_words = shared.tile_logits.data_ptr()
        tile_base = shared_address(tile_words)
        tile_logits = cute.make_tensor(
            cute.recast_ptr(tile_words, dtype=self.cache_dtype),
            cute.make_layout((WARPS, MAX_ROWS, 16), stride=(MAX_ROWS * 16, 16, 1)),
        )
        warp_best = cute.make_tensor(
            shared.warp_best.data_ptr(),
            cute.make_layout((WARPS, MAX_ROWS), stride=(MAX_ROWS, 1)),
        )
        winner = cute.make_tensor(shared.winner.data_ptr(), cute.make_layout(MAX_ROWS))
        count = cute.make_tensor(shared.count.data_ptr(), cute.make_layout(1))

        thread, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        warp = thread // 32
        lane = thread % 32
        g = lane // 4
        t = lane % 4
        tiles = vocab_local // 16
        stride = ctas * WARPS
        first_tile = cta * WARPS + warp
        vocab_offset = rank * vocab_local
        vocab = Int64(vocab_local) * Int64(world)
        resident = first_tile < tiles

        # The resident tile: rows g and g + 8, 16 bytes at k = 32 chunk + 8 t.
        lo = cute.make_rmem_tensor((CHUNKS * 4,), Uint32)
        hi = cute.make_rmem_tensor((CHUNKS * 4,), Uint32)
        if resident:
            tile_row = head + Int64(first_tile * 16 + g) * ROW_BYTES + Int64(t * 16)
            for chunk in cutlass.range_constexpr(CHUNKS):
                words_lo = load_v4(tile_row + chunk * 64, "ld.global.nc.v4.u32", False)
                words_hi = load_v4(
                    tile_row + 8 * ROW_BYTES + chunk * 64, "ld.global.nc.v4.u32", False
                )
                for j in cutlass.range_constexpr(4):
                    lo[chunk * 4 + j] = words_lo[j]
                    hi[chunk * 4 + j] = words_hi[j]

        # Base logits and noise of the resident tile for the current step; the
        # next step's are loaded while the exchange is in flight.
        base = cute.make_rmem_tensor((4,), Float32)
        gumbel = cute.make_rmem_tensor((4,), Float32)
        base.fill(0.0)
        gumbel.fill(0.0)
        if resident:
            self.load_inputs(
                base,
                gumbel,
                0,
                first_tile,
                g,
                t,
                rows,
                base_logits,
                base_row_stride,
                base_step_stride,
                base_column,
                noise,
                vocab_local,
            )

        if thread == 0:
            count[0] = load_scalar(
                control + LAUNCH_WORD * 4, "ld.volatile.global.u32", Uint32, True
            )
        anchor_words = cute.make_rmem_tensor((4,), Uint32)
        anchor_words.fill(0)
        if warp < MAX_ROWS:
            if warp < rows:
                anchor = load_scalar(
                    input_ids + Int64(warp * query_per_row) * 4, "ld.global.u32", Int32
                )
                loaded = load_v4(
                    embed_table + Int64(anchor) * ROW_BYTES + Int64(lane * 16),
                    "ld.global.nc.v4.u32",
                    False,
                )
                for j in cutlass.range_constexpr(4):
                    anchor_words[j] = loaded[j]
            store_shared_v4(
                embedding + warp * ROW_BYTES + lane * 16,
                [anchor_words[j] for j in range(4)],
            )
        cute.arch.sync_threads()
        launches = count[0]
        tag = Uint32(1) + launches % Uint32(32767)
        parity = Int32(launches & Uint32(1))
        records = self.peer_base(peers, rank)
        arrivals = control + ARRIVAL_WORD * 4
        # Staging elements of this launch's parity: [max_rows][steps][vocab].
        staging_parity = (
            Int64(STAGING_OFFSET // cache_bytes)
            + Int64(parity * max_rows * steps) * vocab
        )

        best = cute.make_rmem_tensor((2,), Uint64)
        temp = cute.make_rmem_tensor((2,), Float32)
        acc = cute.make_rmem_tensor((4,), Float32)
        for step in cutlass.range(steps, unroll=1):
            last_step = step + 1 == steps
            for col in cutlass.range_constexpr(2):
                temp[col] = Float32(0.0)
                if 2 * t + col < rows:
                    state = load_scalar(
                        idx_mapping + Int64((2 * t + col) * steps + step) * 8,
                        "ld.global.u64",
                        Int64,
                    )
                    if state >= 0:
                        temp[col] = load_scalar(
                            temperature + state * 4, "ld.global.f32", Float32
                        )
            best[0] = Uint64(0)
            best[1] = Uint64(0)
            for tile in cutlass.range(first_tile, tiles, stride, unroll=1):
                acc.fill(0.0)
                emb_row = embedding + g * ROW_BYTES + t * 16
                if tile == first_tile:
                    for chunk in cutlass.range_constexpr(CHUNKS):
                        x = load_shared_v4(emb_row + chunk * 64)
                        k = chunk * 4
                        c = [acc[0], acc[1], acc[2], acc[3]]
                        c = mma_16816(
                            c, [lo[k], hi[k], lo[k + 1], hi[k + 1]], [x[0], x[1]]
                        )
                        c = mma_16816(
                            c,
                            [lo[k + 2], hi[k + 2], lo[k + 3], hi[k + 3]],
                            [x[2], x[3]],
                        )
                        for j in cutlass.range_constexpr(4):
                            acc[j] = c[j]
                else:
                    # Tiles beyond one per warp (small grids) stream from L2.
                    stream_row = head + Int64(tile * 16 + g) * ROW_BYTES + Int64(t * 16)
                    for chunk in cutlass.range_constexpr(CHUNKS):
                        y = load_shared_v4(emb_row + chunk * 64)
                        wa = load_v4(
                            stream_row + chunk * 64, "ld.global.nc.v4.u32", False
                        )
                        wb = load_v4(
                            stream_row + 8 * ROW_BYTES + chunk * 64,
                            "ld.global.nc.v4.u32",
                            False,
                        )
                        d = [acc[0], acc[1], acc[2], acc[3]]
                        d = mma_16816(d, [wa[0], wb[0], wa[1], wb[1]], [y[0], y[1]])
                        d = mma_16816(d, [wa[2], wb[2], wa[3], wb[3]], [y[2], y[3]])
                        for j in cutlass.range_constexpr(4):
                            acc[j] = d[j]
                for col in cutlass.range_constexpr(2):
                    r = 2 * t + col
                    if r < rows:
                        for half in cutlass.range_constexpr(2):
                            row = tile * 16 + g + 8 * half
                            base_value = base[col * 2 + half]
                            noise_value = gumbel[col * 2 + half]
                            if tile != first_tile:
                                base_value = self.load_base(
                                    base_logits,
                                    base_row_stride,
                                    base_step_stride,
                                    base_column,
                                    r,
                                    step,
                                    row,
                                )
                                noise_value = self.load_noise(
                                    noise, r, step, row, vocab_local
                                )
                            # The bias GEMM's BF16 output, then the eager add's
                            # FP32 sum rounded once to the cache type.
                            bias = round_bf16(acc[2 * half + col])
                            value = add_rn(base_value, bias)
                            if cutlass.const_expr(self.cache_bf16):
                                value = round_bf16(value)
                            tile_logits[warp, r, g + 8 * half] = value.to(
                                self.cache_dtype
                            )
                            score = value
                            if temp[col] != Float32(0.0):
                                score = add_rn(div_full(value, temp[col]), noise_value)
                            best[col] = max_u64(
                                best[col], candidate_key(score, vocab_offset + row)
                            )
                # The tile's logits leave as 16-byte vectors: one per lane and
                # request row for the local cache and for each peer's staging
                # buffer.
                cute.arch.sync_warp()
                if lane < rows * tile_vectors:
                    vec_row = lane // tile_vectors
                    vec = lane - vec_row * tile_vectors
                    vec_state = load_scalar(
                        idx_mapping + Int64(vec_row * steps + step) * 8,
                        "ld.global.u64",
                        Int64,
                    )
                    if vec_state >= 0:
                        packed = load_shared_v4(
                            tile_base
                            + (warp * MAX_ROWS + vec_row) * 16 * cache_bytes
                            + vec * 16
                        )
                        element = Int64(
                            vocab_offset + tile * 16 + vec * vector_elements
                        )
                        store_global_v4(
                            logits_cache
                            + (
                                vec_state * cache_stride0
                                + Int64(step) * cache_stride1
                                + element
                            )
                            * cache_bytes,
                            packed,
                        )
                        staged = (
                            staging_parity
                            + Int64(vec_row * steps + step) * vocab
                            + element
                        ) * cache_bytes
                        for peer in cutlass.range(world, unroll=1):
                            if peer != rank:
                                store_global_v4(
                                    self.peer_base(peers, peer) + staged, packed
                                )
                cute.arch.sync_warp()
            if step + 1 < steps and resident:
                self.load_inputs(
                    base,
                    gumbel,
                    step + 1,
                    first_tile,
                    g,
                    t,
                    rows,
                    base_logits,
                    base_row_stride,
                    base_step_stride,
                    base_column,
                    noise,
                    vocab_local,
                )

            # Each request row's best key over the warp (lanes sharing t), then
            # the CTA.
            for col in cutlass.range_constexpr(2):
                for level in cutlass.range_constexpr(3):
                    best[col] = max_u64(
                        best[col], shuffle_xor_u64(best[col], 4 << level)
                    )
            if g == 0:
                warp_best[warp, 2 * t] = best[0]
                warp_best[warp, 2 * t + 1] = best[1]
            cute.arch.sync_threads()
            if warp == 0:
                if lane < rows:
                    key = Uint64(0)
                    for w in cutlass.range_constexpr(WARPS):
                        key = max_u64(key, warp_best[w, lane])
                    atomic(
                        control + Int64(step * MAX_ROWS + lane) * 8,
                        key,
                        "atom.global.max.u64",
                    )
                cute.arch.sync_warp()
                last = Int32(0)
                if lane == 0:
                    # The last step's fence also publishes this CTA's peer
                    # stores of every step before its arrival is counted.
                    if last_step:
                        fence("sys")
                    else:
                        fence("gl")
                    arrived = atomic(
                        arrivals + Int64(step) * 4, Uint32(1), "atom.global.add.u32"
                    )
                    if arrived == Uint32(ctas - 1):
                        last = Int32(1)
                last = cute.arch.shuffle_sync(last, 0)
                # Lane L handles request row L / world and rank L % world.
                ex_row = lane // world
                other_rank = lane - ex_row * world
                exchanging = lane < rows * world
                # Post every row slot up to max_rows (rows past `rows` carry
                # the zero key) so each slot is rewritten on every launch of
                # its parity. Graphs of different row counts share the launch
                # counter; a slot left from a larger launch could otherwise
                # carry a tag that matches a later one.
                posting = lane < max_rows * world
                record = (
                    (Int64(parity) * MAX_STEPS + Int64(step)) * MAX_ROWS + Int64(ex_row)
                ) * MAX_RANKS
                if last != 0:
                    if lane == 0:
                        fence("gl")
                    cute.arch.sync_warp()
                    rank_key = Uint64(0)
                    if lane < rows:
                        rank_key = atomic(
                            control + Int64(step * MAX_ROWS + lane) * 8,
                            Uint64(0),
                            "atom.global.exch.b64",
                        )
                    if lane == 0:
                        atomic(
                            arrivals + Int64(step) * 4,
                            Uint32(0),
                            "atom.global.exch.b32",
                        )
                    mine = shuffle_idx_u64(rank_key, ex_row)
                    if posting:
                        # Only the last step's record must publish (system
                        # scope) the logits this rank pushed to its peers; a
                        # system fence on the other steps would wait for those
                        # stores to drain.
                        if last_step:
                            fence("sys")
                        store_scalar(
                            self.peer_base(peers, other_rank) + (record + rank) * 8,
                            (mine << TAG_BITS) | Uint64(tag),
                            "st.relaxed.sys.global.u64",
                        )
                found = Uint64(0)
                if exchanging:
                    found = await_record(
                        records + (record + other_rank) * 8,
                        tag,
                        timeout_ns,
                        step,
                        other_rank,
                    )
                    if last_step:
                        fence("sys")
                    elif found != Uint64(0):
                        # Each rank's candidate may be the winner: start its
                        # embedding row's HBM read while the other ranks'
                        # records are still in flight.
                        prefetch_row_l2(
                            embed_table + Int64(key_index(found)) * ROW_BYTES
                        )
                for level in cutlass.range_constexpr(3):
                    if (1 << level) < world:
                        found = max_u64(found, shuffle_xor_u64(found, 1 << level))
                if exchanging and other_rank == 0:
                    winner[ex_row] = found
            cute.arch.sync_threads()
            if cta == 0 and thread < rows:
                store_scalar(
                    draft_tokens + (Int64(thread) * draft_stride + Int64(step)) * 8,
                    Int64(key_index(winner[thread])),
                    "st.global.u64",
                )
            if step + 1 < steps and warp < rows:
                next_words = load_v4(
                    embed_table
                    + Int64(key_index(winner[warp])) * ROW_BYTES
                    + Int64(lane * 16),
                    "ld.global.nc.v4.u32",
                    False,
                )
                store_shared_v4(embedding + warp * ROW_BYTES + lane * 16, next_words)
            cute.arch.sync_threads()

        # The peers' shards are in this rank's staging buffer: copy them into
        # the cache.
        vectors = vocab_local * cache_bytes // 16
        per_row_step = (world - 1) * vectors
        total = rows * steps * per_row_step
        for task in cutlass.range(
            cta * THREADS + thread, total, ctas * THREADS, unroll=1
        ):
            row_step = task // per_row_step
            within = task - row_step * per_row_step
            source = within // vectors
            source_vector = within - source * vectors
            if source >= rank:
                source = source + 1
            tail_state = load_scalar(
                idx_mapping + Int64(row_step) * 8, "ld.global.u64", Int64
            )
            if tail_state >= 0:
                tail_step = row_step % steps
                tail_element = Int64(source) * Int64(vocab_local) + Int64(
                    source_vector * vector_elements
                )
                shard = load_v4(
                    records
                    + (staging_parity + Int64(row_step) * vocab + tail_element)
                    * cache_bytes,
                    "ld.global.cg.v4.u32",
                    True,
                )
                store_global_v4(
                    logits_cache
                    + (
                        tail_state * cache_stride0
                        + Int64(tail_step) * cache_stride1
                        + tail_element
                    )
                    * cache_bytes,
                    shard,
                )

        if thread == 0:
            fence("gl")
            previous = atomic(control + END_WORD * 4, Uint32(1), "atom.global.add.u32")
            if previous == Uint32(ctas - 1):
                store_scalar(control + END_WORD * 4, Uint32(0), "st.global.u32")
                store_scalar(
                    control + LAUNCH_WORD * 4, launches + Uint32(1), "st.global.u32"
                )
