import torch
import triton
import triton.language as tl
from .base import Layout


class HopperMXScaleLayout(Layout):
    name: str = "HOPPER_SCALE"

    def __init__(self, shape, mx_axis, num_warps=8) -> None:
        assert num_warps & (num_warps - 1) == 0, "warps_n must be a power of 2"
        super().__init__(shape)
        self.mx_axis = mx_axis
        self.num_warps = num_warps
        *self.leading_shape, _, _ = shape

    def _maybe_mT(self, data):
        if self.mx_axis == len(self.leading_shape):
            return data.contiguous().mT
        return data

    def swizzle_data(self, data):
        data = self._maybe_mT(data).contiguous()
        *batch, M, K = data.shape
        SWIZZLE_ALIGN_M = 2 * self.num_warps * 2 * 8
        SWIZZLE_ALIGN_K = 2
        pad_m = (SWIZZLE_ALIGN_M - (M % SWIZZLE_ALIGN_M)) % SWIZZLE_ALIGN_M
        pad_k = (SWIZZLE_ALIGN_K - (K % SWIZZLE_ALIGN_K)) % SWIZZLE_ALIGN_K
        data = torch.nn.functional.pad(data, (0, pad_k, 0, pad_m))
        *batch, M, K = data.shape
        assert data.is_contiguous()
        assert M % (
            2 * self.num_warps * 2 *
            8) == 0 and K % 2 == 0, f"Input tensor must have a subtile of shape (..., {2 * self.num_warps * 2 * 8}, 2)"
        b = len(batch)
        data = data.reshape(*batch, M // (2 * self.num_warps * 2 * 8), 2, self.num_warps, 2, 8, K // 2, 2)
        perm = [0, 2, 5, 1, 4, 6, 3]
        perm = list(range(b)) + [b + p for p in perm]
        data = data.permute(*perm)
        data = data.flatten(-5, -1)
        data = data.flatten(-3, -2)
        assert data.shape[-2] == M // 32
        assert data.shape[-1] == K * 32
        data = self._maybe_mT(data)
        return data

    def unswizzle_data(self, data):
        data = self._maybe_mT(data)
        *batch, M, K = data.shape
        b = len(batch)
        data = data.reshape(*batch, M // self.num_warps, self.num_warps, K // 64, 2, 8, 2, 2)
        perm = [0, 3, 1, 6, 4, 2, 5]
        perm = list(range(b)) + [b + p for p in perm]
        data = data.permute(*perm)
        data = data.reshape(*batch, M * 32, K // 32)
        data = self._maybe_mT(data)
        return data

    def swizzle_block_shape(self, block_shape):
        return block_shape


@triton.jit
def unswizzle_mxfp4_scale_hopper(x, mx_axis: tl.constexpr, num_warps: tl.constexpr):
    """
    Triton inverse of swizzle_mxfp4_scale_hopper
    """
    tl.static_assert(len(x.shape) == 2, "NYI")
    # implementation assumes mxfp data is packed along the last dimension
    x = x.trans() if mx_axis == 0 else x
    M: tl.constexpr = x.shape[0]
    K: tl.constexpr = x.shape[1]
    tl.static_assert(M % num_warps == 0, f"M must be divisible by {num_warps}. Got {M}")
    tl.static_assert(K % 64 == 0, f"K must be divisible by 64. Got {K}")
    x = x.reshape(M // num_warps, num_warps, K // 64, 2, 8, 2, 2)
    x = x.trans(0, 3, 1, 6, 4, 2, 5)
    x = x.reshape(M * 32, K // 32)
    # implementation assumed mxfp data is packed along the last dimension
    x = x.trans() if mx_axis == 0 else x
    return x
