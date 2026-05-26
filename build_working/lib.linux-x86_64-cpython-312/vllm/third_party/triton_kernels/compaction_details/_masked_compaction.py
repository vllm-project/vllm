import triton
import triton.language as tl


@triton.jit
def _masked_compaction(Yv, Yi, BitMask, stride_bm, stride_bn, RetYv, RetYi, sentinel, K: tl.constexpr):
    pid_m = tl.program_id(0)
    yv = tl.load(Yv + pid_m * K + tl.arange(0, K))
    yi = tl.load(Yi + pid_m * K + tl.arange(0, K))
    div = yi // 32
    rem = yi % 32
    active_bits = (tl.load(BitMask + pid_m * stride_bm + div * stride_bn) >> rem) & 1
    exc_cumsum = tl.cumsum(active_bits, 0) - active_bits
    active_flags = active_bits.to(tl.int1)
    rev_arange = tl.where(active_flags, 0, K - 1 - tl.arange(0, K))
    write_indx = exc_cumsum + rev_arange
    yv = tl.where(active_flags, yv, sentinel)
    yi = tl.where(active_flags, yi, sentinel)
    tl.store(RetYv + pid_m * K + write_indx, yv)
    tl.store(RetYi + pid_m * K + write_indx, yi)
