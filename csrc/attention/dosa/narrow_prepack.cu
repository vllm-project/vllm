#include "common.cuh"

namespace DOSA_NAMESPACE::h18_narrow {
struct Source {
  const uint8_t* fine;
  const float *scale, *meta, *raw, *out;
  const int* hint;
  int fine_stride;
};
struct Target {
  uint8_t* fine;
  float *scale, *meta, *raw, *out;
  int* hint;
  int fine_stride;
};
__device__ float decode(uint8_t bits) {
  const __half_raw value = __nv_cvt_fp8_to_halfraw(bits, __NV_E4M3);
  return __half2float(static_cast<__half>(value));
}
__device__ double sum_up(double value, int lane) {
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    const double other = __shfl_down_sync(0xffffffffu, value, offset);
    if (lane + offset < 32) value = __dadd_ru(value, other);
  }
  return value;
}

// One Q8 per CTA, one warp per row. Source is the actual unchanged P16 plan;
// destination is the existing P14 storage. Tags commit only after all writes.
__global__ void prepack(const __grid_constant__ Source src,
                        const __grid_constant__ Target dst, int* budget,
                        const int* error) {
  const int group = blockIdx.x, t = threadIdx.x, warp = t / 32, lane = t & 31;
  if (budget[2 * group] != 16 || budget[2 * group + 1] != 16) return;
  const int row = group * 8 + warp;
  const uint8_t* fine = src.fine + int64_t(row) * src.fine_stride;
  const float* meta = src.meta + int64_t(row) * 64;
  __shared__ int good, omitted[8][2];
  __shared__ float guards[8];
  if (t == 0) good = 1;
  __syncthreads();
  bool valid = src.hint[row] == 2 && error[row] == 0;
  double best[2] = {INFINITY, INFINITY};
  int selected[2] = {-1, -1};
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const float m = meta[2 + j], c = __fmul_rn(m, 0.5f);
    const uint32_t bits = __float_as_uint(c) & 0x7fffffffu;
    valid &= isfinite(m) && (bits == 0 || bits >= 0x00800000u) &&
             __fmul_rn(c, 2.f) == m;
    float norm2 = 0.f;
#pragma unroll
    for (int d = lane; d < 128; d += 32) {
      const float q = decode(fine[(2 + j) * 128 + d]);
      norm2 +=
          q * q;  // Selection may be approximate; the bound below is outward.
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
      norm2 += __shfl_down_sync(0xffffffffu, norm2, offset);
    const double cost = double(c) * c * __shfl_sync(0xffffffffu, norm2, 0);
    if (c > 0.f && cost < best[0]) {
      best[1] = best[0];
      selected[1] = selected[0];
      best[0] = cost;
      selected[0] = j;
    } else if (c > 0.f && cost < best[1]) {
      best[1] = cost;
      selected[1] = j;
    }
  }
  valid &= selected[0] >= 0 && selected[1] >= 0;
  if (lane == 0 && !valid) atomicExch(&good, 0);
  __syncthreads();
  if (!good) return;
  const int first = selected[0], second = selected[1];
  const double c0 = double(__fmul_rn(meta[2 + first], 0.5f));
  const double c1 = double(__fmul_rn(meta[2 + second], 0.5f));
  double plus2 = 0., minus2 = 0.;
#pragma unroll
  for (int d = lane; d < 128; d += 32) {
    // FP32 half-coefficient times FP8 has at most28 significant bits, so the
    // binary64 products are exact. Interval addition covers both joint signs.
    const double a = __dmul_rn(c0, double(decode(fine[(2 + first) * 128 + d])));
    const double b =
        __dmul_rn(c1, double(decode(fine[(2 + second) * 128 + d])));
    const double plus = fmax(fabs(__dadd_rd(a, b)), fabs(__dadd_ru(a, b)));
    const double minus = fmax(fabs(__dadd_rd(a, -b)), fabs(__dadd_ru(a, -b)));
    plus2 = __dadd_ru(plus2, __dmul_ru(plus, plus));
    minus2 = __dadd_ru(minus2, __dmul_ru(minus, minus));
  }
  plus2 = sum_up(plus2, lane);
  minus2 = sum_up(minus2, lane);
  if (lane == 0) {
    constexpr double Sqrt128Up = 0x1.6a09e667f3bcep+3;
    const double joint = __dsqrt_ru(fmax(plus2, minus2));
    const float guard = __double2float_ru(
        __dadd_ru(double(meta[18]), __dmul_ru(Sqrt128Up, joint)));
    valid &= isfinite(guard) && guard >= 0.f;
    guards[warp] = guard;
    omitted[warp][0] = min(first, second);
    omitted[warp][1] = max(first, second);
    if (!valid) atomicExch(&good, 0);
  }
  __syncthreads();
  if (!good) return;
  const int skip0 = omitted[warp][0], skip1 = omitted[warp][1];
#pragma unroll
  for (int h = 0; h < 16; ++h) {
    int old = h;
    if (h >= 2) {
      int paid = h - 2;
      if (paid >= skip0) ++paid;
      if (paid >= skip1) ++paid;
      old = paid + 2;
    }
#pragma unroll
    for (int d = lane; d < 128; d += 32)
      dst.fine[int64_t(row) * dst.fine_stride + h * 128 + d] = fine[old * 128 + d];
    if (lane == h) {
      dst.scale[int64_t(row) * 16 + h] = src.scale[int64_t(row) * 18 + old];
      dst.meta[int64_t(row) * 64 + h] = meta[old];
    }
  }
  for (int i = lane + 16; i < 64; i += 32)
    dst.meta[int64_t(row) * 64 + i] = i == 16 ? guards[warp] : 0.f;
  if (lane == 0) {
    dst.raw[row] = src.raw[row];
    dst.out[row] = src.out[row];
    dst.hint[row] = src.hint[row];
  }
  __syncthreads();
  if (t == 0) {
    budget[2 * group] = 14;
    budget[2 * group + 1] = 14;
  }
}

void run(std::vector<Tensor> source, std::vector<Tensor> target, Tensor budget,
         Tensor error) {
  DeviceGuard guard(source[0].get_device_index());
  Source src{
      source[0].const_data_ptr<uint8_t>(), source[1].const_data_ptr<float>(),
      source[2].const_data_ptr<float>(),   source[3].const_data_ptr<float>(),
      source[4].const_data_ptr<float>(),   source[5].const_data_ptr<int>(),
      int(source[0].stride(0))};
  Target dst{target[0].mutable_data_ptr<uint8_t>(),
             target[1].mutable_data_ptr<float>(),
             target[2].mutable_data_ptr<float>(),
             target[3].mutable_data_ptr<float>(),
             target[4].mutable_data_ptr<float>(),
             target[5].mutable_data_ptr<int>(), int(target[0].stride(0))};
  launch<prepack, 256>(source[0].size(0) / 8, src, dst,
                       budget.mutable_data_ptr<int>(),
                       error.const_data_ptr<int>());
}
}  // namespace DOSA_NAMESPACE::h18_narrow

DOSA_REGISTER("h18_to_h16",
              "(Tensor[] source, Tensor(a!)[] target, Tensor! budget, Tensor "
              "error) -> ()",
              CUDA, h18_narrow::run)
