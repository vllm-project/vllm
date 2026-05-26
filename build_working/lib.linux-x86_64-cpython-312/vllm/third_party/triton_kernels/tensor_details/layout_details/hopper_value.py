import torch
import triton
import triton.language as tl
from .base import Layout


def right_shift_unsigned(x, shift):
    return (x >> shift) & ((1 << (32 - shift)) - 1)


# -----------------------------------------------------------------------
# Interleave the bits of four consecutive fp4 values (i.e. 16-bits) as:
#     1000000111000000         (first fp4)
#        1000000111000000      (second fp4)
#           1000000111000000   (third fp4)
#     0110110000000000         (fourth fp4)
# This is done so that dequantization can be done in 14 SASS instructions
# -----------------------------------------------------------------------


def _compress_fp4(x):
    x = x.to(torch.int32)
    return ((x & 0x8) << 12) | ((x & 0x7) << 6)


def _compress_fourth(x):
    x = x.to(torch.int32)
    return ((x & 0x8) << 11) | ((x & 0x6) << 9) | ((x & 0x1) << 13)


def _pack_bits(x: torch.Tensor, mx_axis: int):
    x = x.contiguous()
    assert x.shape[-1] % 4 == 0, "Input tensor must have a last dimension divisible by 4"
    x = x.reshape(x.shape[:-1] + (x.shape[-1] // 4, 4))
    first = _compress_fp4(x[..., 0]) | (_compress_fp4(x[..., 0] >> 4) << 16)
    second = _compress_fp4(x[..., 1]) | (_compress_fp4(x[..., 1] >> 4) << 16)
    third = _compress_fp4(x[..., 2]) | (_compress_fp4(x[..., 2] >> 4) << 16)
    fourth = _compress_fourth(x[..., 3]) | (_compress_fourth(x[..., 3] >> 4) << 16)
    x = first | right_shift_unsigned(second, 3) | right_shift_unsigned(third, 6) | fourth
    assert x.is_contiguous()
    x = x.view(torch.uint8)
    return x


# -----------------------------------------------------------------------
# inverse operation of _pack_bits
# -----------------------------------------------------------------------


def _bf16_to_fp4e2m1(x):
    # 0bAxxxxxxBCDxxxxxx (int16) -> 0b0000ABCD (uint8)
    assert x.dtype == torch.int16
    s = (right_shift_unsigned(x, 15) & 0x1) << 3
    em = right_shift_unsigned(x, 6) & 0x7
    return (s | em).to(torch.uint8)


def _bf16x2_to_fp4e2m1x2(x):
    # 0bAxxxxxxBCDxxxxxx_0bExxxxxxFGHxxxxxx  (int32) -> 0bABCD_EFGH (uint8)
    assert x.dtype == torch.int32
    lo = (x & 0xFFFF).to(torch.int16)
    hi = (right_shift_unsigned(x, 16) & 0xFFFF).to(torch.int16)
    ret_lo = _bf16_to_fp4e2m1(lo)
    ret_hi = _bf16_to_fp4e2m1(hi)
    return ret_lo | (ret_hi << 4)


def _unpack_bits(x, mx_axis: int):
    x = x.view(torch.int32)
    m = 0b10000001110000001000000111000000
    a = (x << 1) & 0b10000000000000001000000000000000
    b = right_shift_unsigned(x, 3) & 0b00000001100000000000000110000000
    c = right_shift_unsigned(x, 7) & 0b00000000010000000000000001000000
    unpacked = [x & m, (x << 3) & m, (x << 6) & m, (a | b) | c]
    x = torch.stack(unpacked, dim=-1)
    x = x.flatten(-2, -1)
    x = _bf16x2_to_fp4e2m1x2(x)
    return x


# -----------------------------------------------------------------------


class HopperMXValueLayout(Layout):
    name: str = "HOPPER_VALUE"

    def __init__(self, shape, mx_axis, mma_version=3):
        super().__init__(shape)
        assert mx_axis in range(len(shape))
        self.mx_axis = mx_axis
        self.mma_version = mma_version
        *self.leading_shape, self.K, self.N, = shape

    def _maybe_mT(self, data):
        if self.mx_axis == len(self.leading_shape):
            return data.mT
        return data

    def swizzle_data(self, data):
        """
        Given a uint8 tensor of shape (*, M, K), returns a tensor of shape
        (*, M // 4, K * 4) such that:

        1) Groups contiguously all the elements owned by the same thread of 4
        mma tiles along the K axis. The following animation shows a similar
        grouping for 2 tiles along M and 2 tiles along K rather than 4 along K
        as done here:
        https://neuralmagic.com/wp-content/uploads/2024/10/animation_4.gif

        2) Moves the elements belonging to thread 4-7 to be contiguous with those
        from thread 0-3. This is done to get a full cache line when loading them
        from HBM.

        mx_axis selects the lhs or rhs of the matmul.

        WARNING: Assumes that the matmul will be done in bf16 or fp16!
        Implementing it for fp8 is as easy as making the tile size (8, 8)
        """
        batch = data.ndim - 2
        assert batch >= 0
        assert self.mma_version in (2, 3)
        data = self._maybe_mT(data)
        init_shape = data.shape

        # We are loading 8 bf16 elements per thread to use ld.global.v4
        # Every u8 represents 2 mxfp4 elements
        u8_kwidth = 8 // 2 if self.mma_version == 2 else 1

        # Pack the 4 // u8_kwidth subtiles of an mma into a u4x8
        contig = (1, u8_kwidth)
        scott_trick = (2, 1)
        threads = (4, 4)
        warp_tile = (2, 2)
        k_tile = (1, 4 // u8_kwidth)

        sizes = list(data.shape[:-2])
        pads = []
        # [rest, K, tile, threads] per dimension
        for i, (a, b, c, s, d) in enumerate(zip(k_tile, warp_tile, threads, scott_trick, contig)):
            pack = a * b * c * s * d
            size = data.shape[batch + i]
            pad = (pack - size % pack) % pack
            pads += [(0, pad)]
            sizes.append((size + pad) // pack)
            sizes += [a, b, c, s, d]

        pads = tuple(x for t in pads[::-1] for x in t)
        data = torch.nn.functional.pad(data, pads)
        init_shape = data.shape
        # 0: rest[0]
        # 1: k_tile[0]
        # 2: warp_tile[0]
        # 3: threads[0]
        # 4: scott_trick[0]
        # 5: contig[0]
        # 6: rest[1]
        # 7: k_tile[1]
        # 8: warp_tile[1]
        # 9: threads[1]
        # 10: scott_trick[1]
        # 11: contig[1]
        data = data.view(*sizes)
        # Want [rest[0], threads[0], rest[1], scott_trick[0], scott_trick[0], threads[1], contig[1], contig[0], k_tile[1], k_tile[0], warp_tile[1], warp_tile[0]]
        perm = [0, 3, 6, 10, 4, 9, 7, 1, 8, 2, 5, 11]
        perm = list(range(batch)) + [batch + p for p in perm]
        data = data.permute(*perm).contiguous()
        # These are views
        data = data.flatten(-10, -1)
        data = data.flatten(-3, -2)
        assert data.is_contiguous()
        assert data.shape[-2] == init_shape[-2] // 4
        assert data.shape[-1] == init_shape[-1] * 4
        # twiddle the bits
        data = _pack_bits(data, self.mx_axis)
        data = self._maybe_mT(data)
        return data

    def unswizzle_data(self, data):
        data = self._maybe_mT(data)
        data = _unpack_bits(data, self.mx_axis)
        *batch, M, K = data.shape
        # We have two times the elements if we already upcasted to bfloat16
        mult = 2 if data.dtype == torch.bfloat16 else 1
        assert M % 4 == 0, "M must be divisible by 4"
        assert K % (4 * 8 * 2 * 2 * mult) == 0, f"K must be divisible by {4 * 8 * 2 * 2 * mult}"
        # We are loading 8 bf16 elements per thread to use ld.global.v4
        # Every u8 represents 2 mxfp4 elements
        u8_kwidth = 8 // 2 if self.mma_version == 2 else 1
        data = data.reshape(*batch, M // 4, 4, K // (4 * 8 * 2 * 2 * mult), 2, 4, 8 // u8_kwidth, 2, u8_kwidth * mult)
        b = len(batch)
        perm = [0, 6, 1, 3, 2, 5, 4, 7]
        perm = list(range(b)) + [b + p for p in perm]
        data = data.permute(*perm)
        data = data.reshape(*batch, M * 4, K // 4)
        data = self._maybe_mT(data)
        return data[..., :self.K, :self.N]

    def swizzle_block_shape(self, block_shape):
        return block_shape


@triton.jit
def _unshuffle_triton(x, mma_version: tl.constexpr):
    """
    Triton inverse of swizzle_mxfp4_value_hopper
    """
    tl.static_assert(mma_version == 2 or mma_version == 3, "mma_version must be 2 or 3")
    # if mx_axis == 0:
    #     x = x.trans()

    # We have two times the elements if we already upcasted to bfloat16
    mult: tl.constexpr = 2 if x.dtype == tl.bfloat16 else 1
    M: tl.constexpr = x.shape[0]
    K: tl.constexpr = x.shape[1]
    tl.static_assert(M % 4 == 0, "M must be divisible by 4")
    tl.static_assert(K % (4 * 8 * 2 * 2 * mult) == 0, f"K must be divisible by {4 * 8 * 2 * 2 * mult}")

    # We are loading 8 bf16 elements per thread to use ld.global.v4
    # Every u8 represents 2 mxfp4 elements
    u8_kwidth: tl.constexpr = 8 // 2 if mma_version == 2 else 1
    x = x.reshape(M // 4, 4, K // (4 * 8 * 2 * 2 * mult), 2, 4, 8 // u8_kwidth, 2, u8_kwidth * mult)
    x = x.trans(0, 6, 1, 3, 2, 5, 4, 7)
    x = x.reshape(M * 4, K // 4)
    # if mx_axis == 0:
    #     x = x.trans()
    return x


@triton.jit
def _unpack_fp4_to_bf16_triton(x):
    # For now we implement just H100 support (mul.bf16x2)
    # A100 support is possible via fma
    r0, r1 = tl.inline_asm_elementwise(
        r"""
        {
            .reg .b32 b, c, d<7>, scale;
            .reg .b32 bias;
            mov.b32 bias, 0x7e807e80; // 2 ** 126 == 2 ** (bias_bf16 - bias_fp2)
            // We add the missing bias to the scale directly
            and.b32 $0, $4, 0b10000001110000001000000111000000;
            mul.bf16x2 $0, $0, bias;
            shl.b32 b, $4, 3;
            and.b32 $1, b,  0b10000001110000001000000111000000;
            mul.bf16x2 $1, $1, bias;
            shl.b32 c, $4, 6;
            and.b32 $2, c,  0b10000001110000001000000111000000;
            mul.bf16x2 $2, $2, bias;
            // Unpack last two elements
            shl.b32 d0, $4, 1;
            and.b32 d1, d0, 0b10000000000000001000000000000000;
            shr.b32 d2, $4, 3;
            and.b32 d3, d2, 0b00000001100000000000000110000000;
            or.b32 d4, d1, d3;
            shr.b32 d5, $4, 7;
            and.b32 d6, d5, 0b00000000010000000000000001000000;
            or.b32 $3, d4, d6;
            mul.bf16x2 $3, $3, bias;
        }
        """,
        constraints="=r,=r,=r,=r,r",
        args=[x],
        dtype=(tl.bfloat16, tl.bfloat16),
        is_pure=True,
        pack=4,
    )
    # Concat each pack of 4
    x = tl.join(r0, r1)
    x = x.reshape(x.shape[0], x.shape[1] // 4, 4, x.shape[2])
    x = x.trans(0, 1, 3, 2)
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
    return x


@triton.jit
def mxfp4_to_bf16_triton(x, scale, mx_axis: tl.constexpr):
    """
    Implements the bit-untwiddling of a 32-bit integer (8 mxfp4 elements):
    (x << 0) & 0b1000000111000000
    (x << 3) & 0b1000000111000000
    (x << 6) & 0b1000000111000000
    ((x << 1) & 0b1000000000000000) | ((x >> 3) & 0b0000000110000000) | ((x >> 7) & 0b0000000001000000)
    """
    # upcast values to bfloat16
    tl.static_assert(len(x.shape) == 2)
    tl.static_assert(mx_axis == 0 or mx_axis == 1, "mx_axis must be 0 or 1")
    tl.static_assert(x.shape[1] % 4 == 0)
    tl.static_assert(x.dtype == tl.uint8)
    if mx_axis == 0:
        x = x.trans()
    x = _unpack_fp4_to_bf16_triton(x)
    x = _unshuffle_triton(x, mma_version=3)
    if mx_axis == 0:
        x = x.trans()

    # upcast scale to bfloat16
    # Add bias missing from the bf16 upcasting sequence
    # triton / LLVM generates terrible code for this sequence
    # scale = scale.to(tl.uint16)
    # scale = scale << 7
    # scale = scale.to(tl.bfloat16, bitcast=True)
    scale = tl.inline_asm_elementwise(
        r"""
        {
            prmt.b32 $0, $2, 0, 0x5140;
            shl.b32 $0, $0, 7;
            prmt.b32 $1, $2, 0, 0x7362;
            shl.b32 $1, $1, 7;
        }
        """,
        constraints="=r,=r,r",
        args=[scale],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=4,
    )
    # Broadcast scale
    scale = scale.expand_dims(mx_axis + 1)
    scale = scale.broadcast_to(scale.shape[:mx_axis + 1] + [32] + scale.shape[mx_axis + 2:])
    scale = scale.reshape(x.shape)

    # Combine scale and x
    x = x * scale
    return x
