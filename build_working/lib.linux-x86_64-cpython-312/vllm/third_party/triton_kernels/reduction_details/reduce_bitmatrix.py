import torch
import triton
import triton.language as tl


@triton.jit
def vpopc(x):
    """
    Vertical popcount
    Input  x : uint32[..., N]
    Output y : uint32[..., 32]
    semantics : y[..., i] = sum_j((x[..., j] >> i) & 1)
    credits: @apgoucher
    """

    tl.static_assert(x.dtype == tl.uint32, "x should consist of 32-bit unsigned integers")

    BLOCK_N: tl.constexpr = x.shape[-1]  # summation axis
    BATCHES: tl.constexpr = x.numel // BLOCK_N  # number of batches
    if BLOCK_N >= 8:
        sa1: tl.constexpr = 8
    else:
        sa1: tl.constexpr = BLOCK_N
    # create 8-way sums in 4-bit fields:
    y = tl.reshape(x, [BATCHES, BLOCK_N // sa1, sa1, 1])
    y = (y >> tl.arange(0, 4)[None, None, None, :]) & 0x11111111
    y = tl.sum(y, 2)  # [BATCHES, BLOCK_N // sa1, 4]
    if BLOCK_N >= 128:
        sa2: tl.constexpr = 16
    else:
        sa2: tl.constexpr = BLOCK_N // sa1
    # create 128-way sums in 8-bit fields:
    y = tl.reshape(y, [BATCHES, BLOCK_N // (sa1 * sa2), sa2, 1, 4])
    y = (y >> (4 * tl.arange(0, 2))[None, None, None, :, None]) & 0x0f0f0f0f
    y = tl.sum(y, 2)  # [BATCHES, BLOCK_N // (sa1 * sa2), 2, 4]
    sa3: tl.constexpr = BLOCK_N // (sa1 * sa2)
    # create N-way sums in 32-bit fields:
    y = tl.reshape(y, [BATCHES, 1, sa3, 8])
    y = (y >> (8 * tl.arange(0, 4))[None, :, None, None]) & 0x000000ff
    y = tl.sum(y, 2)  # [BATCHES, 4, 8]
    y = tl.reshape(y, x.shape[:-1] + [32])
    return y


@triton.jit
def _sum_bitmatrix_memset(Ret, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(Ret + offs, 0)


@triton.jit
def _sum_bitmatrix_rows(B, shape_bm, stride_bm: tl.constexpr, stride_bn: tl.constexpr,  # input bitmatrix
                        Ret, Partials, stride_pm: tl.constexpr, stride_pn, shape_pn,  # outputs
                        BLOCK_MM: tl.constexpr, BLOCK_M: tl.constexpr):

    tl.static_assert(BLOCK_MM % BLOCK_M == 0)
    TILE_SIZE: tl.constexpr = BLOCK_MM // BLOCK_M
    if isinstance(shape_bm, tl.tensor) and shape_bm.dtype.is_ptr():
        shape_bm = tl.load(shape_bm)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_MM + tl.arange(0, BLOCK_MM)
    offs_n = pid_n * 32 + tl.arange(0, 32)
    n_rows = shape_bm
    bits = tl.load(B + pid_n * stride_bn + offs_m * stride_bm, mask=offs_m < n_rows, other=0)
    bits = tl.reshape(bits, [TILE_SIZE, BLOCK_M])
    ret = vpopc(bits)  # [TILE_SIZE, 32]

    offs_t = pid_m * TILE_SIZE + tl.arange(0, TILE_SIZE)

    tl.atomic_add(Ret + offs_n, tl.sum(ret, 0), sem="relaxed")
    tl.store(Partials + offs_t[:, None] * stride_pm + offs_n[None, :] * stride_pn, ret)


def clear_sums(n_cols, device, MEMSET_BLOCK=512):
    cdiv = triton.cdiv
    blocks = cdiv(n_cols, MEMSET_BLOCK)
    out_ret = torch.empty((blocks * MEMSET_BLOCK, ), device=device, dtype=torch.int32)
    _sum_bitmatrix_memset[(blocks, )](out_ret, MEMSET_BLOCK)
    return out_ret


def sum_bitmatrix_rows(x, out_ret, partials_block_size=None):
    assert partials_block_size is not None
    cdiv = triton.cdiv
    PARTIALS_BLOCK_M = partials_block_size
    n_rows, n_cols = x.shape
    n_rows_max = x.shape_max[0]
    assert out_ret.shape == (n_cols, )

    TILE_SIZE = max(1, 128 // PARTIALS_BLOCK_M)
    BLOCK_MM = PARTIALS_BLOCK_M * TILE_SIZE

    pids_x = cdiv(n_rows_max, BLOCK_MM)
    pids_y = cdiv(n_cols, 32)
    out_partials = torch.empty((pids_y * 32, pids_x * TILE_SIZE), device=out_ret.device, dtype=torch.int32)
    out_partials = torch.transpose(out_partials, 0, 1)

    # output tensors
    _sum_bitmatrix_rows[(pids_x, pids_y)](
        x.storage.data, n_rows, x.stride(0), x.stride(1),  # input
        out_ret,  # output [final reduction]
        out_partials, out_partials.stride(0), out_partials.stride(1),
        out_partials.shape[1],  # output [partial reductions]
        BLOCK_M=PARTIALS_BLOCK_M, BLOCK_MM=BLOCK_MM,  # constants
        num_warps=8)

    out_partials = out_partials[:cdiv(n_rows_max, PARTIALS_BLOCK_M), :]

    return out_ret, out_partials
