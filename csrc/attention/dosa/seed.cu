#include "common.cuh"
#include "record.cuh"

namespace DOSA_NAMESPACE {
namespace native_seed {
constexpr int Threads = 256, Seed = 12288, Top = 2048;
struct Output {
  uint8_t* low;
  int *words, *counts;
  float* tau;
  int *error, *lengths, *diag;
  int64_t low_stride, word_stride;
};

__global__ void prepare(const float* scores, int64_t stride, Output a) {
  const int row = blockIdx.x, t = threadIdx.x;
  const float* input = scores + int64_t(row) * stride;
  __shared__ unsigned hist[4 * 256], pivot, before, equal, written;
  if (t == 0) {
    a.counts[row] = a.lengths[row] = 0;
    a.tau[row] = -INFINITY;
    for (int i = 0; i < 4; ++i) a.diag[row * 4 + i] = 0;
    written = 0;
  }
  bool bad = false;
  for (int j = t; j < Seed; j += Threads) {
    float value = input[j];
    bad |= !isfinite(value) || (__float_as_uint(value) & 0xFFFFu) != 0;
  }
  if (__syncthreads_or(bad)) {
    if (t == 0) {
      atomicOr(a.error + row, 1);
      a.diag[row * 4 + 3] = 3;
    }
    return;
  }
  for (int pass = 0; pass < 2; ++pass) {
    for (int j = t; j < 4 * 256; j += Threads) hist[j] = 0;
    __syncthreads();
    const int shift = pass == 0 ? 12 : 4;
    for (int j = t; j < Seed; j += Threads) {
      const unsigned code = h32_record5::encode(input[j]);
      if (pass == 0 || (code >> 12) == (pivot >> 12))
        atomicAdd(hist + (t / 64) * 256 + ((code >> shift) & 255u), 1u);
    }
    __syncthreads();
    const unsigned count =
        hist[t] + hist[256 + t] + hist[512 + t] + hist[768 + t];
    __syncthreads();
    hist[t] = count;
    __syncthreads();
    if (t == 0) {
      unsigned total = pass == 0 ? 0u : before;
      for (unsigned bin = 0; bin < 256; ++bin) {
        if (total < Top && Top <= total + hist[bin]) {
          pivot = (pass == 0 ? 0u : pivot & 0xFF000u) | (bin << shift);
          before = total;
          equal = hist[bin];
          break;
        }
        total += hist[bin];
      }
    }
    __syncthreads();
  }
  if (t == 0) {
    // Positive ordered cells end in 0xF; negative cells end in zero.
    pivot |= (pivot & 0x80000u) ? 0u : 15u;
    hist[0] = 0;
  }
  __syncthreads();
  const unsigned needed = Top - before;
  // Seed ties affect only the witness subset: every chosen score is >=tau.
  for (int j = t; j < Seed; j += Threads) {
    const float value = input[j];
    const unsigned code = h32_record5::encode(value);
    unsigned slot = Top;
    if (code < pivot) slot = atomicAdd(&written, 1u);
    if (code == pivot) {
      const unsigned rank = atomicAdd(hist, 1u);
      if (rank < needed) slot = atomicAdd(&written, 1u);
    }
    if (slot < Top) {
      a.low[int64_t(row) * a.low_stride + slot] = uint8_t(code);
      a.words[int64_t(row) * a.word_stride + slot] =
          int32_t(h32_record5::pack_word(code, unsigned(j)));
    }
  }
  __syncthreads();
  if (t == 0) {
    unsigned ordered = ~(pivot << 12);
    unsigned bits = ordered & 0x80000000u ? ordered ^ 0x80000000u : ~ordered;
    a.tau[row] = __uint_as_float(bits & 0xFFFF0000u);
    a.counts[row] = int(written);
    a.diag[row * 4] = int(before);
    a.diag[row * 4 + 1] = int(equal);
    a.diag[row * 4 + 2] = int(needed);
    if (written != Top) atomicOr(a.error + row, 1);
  }
}
}  // namespace native_seed

void prepare5(Tensor scores, Tensor low, Tensor words, Tensor counts,
              Tensor tau, Tensor error, Tensor lengths, Tensor diag) {
  DeviceGuard guard(scores.get_device_index());
  native_seed::Output output{low.mutable_data_ptr<uint8_t>(),
                             words.mutable_data_ptr<int>(),
                             counts.mutable_data_ptr<int>(),
                             tau.mutable_data_ptr<float>(),
                             error.mutable_data_ptr<int>(),
                             lengths.mutable_data_ptr<int>(),
                             diag.mutable_data_ptr<int>(),
                             low.stride(0),
                             words.stride(0)};
  native_seed::prepare<<<scores.size(0), 256, 0, get_current_cuda_stream()>>>(
      scores.const_data_ptr<float>(), scores.stride(0), output);
  DOSA_CUDA_CHECK(cudaGetLastError());
}
}  // namespace DOSA_NAMESPACE

STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(DOSA_OP("prepare5") "(Tensor scores, Tensor! low, Tensor! words, Tensor! counts, Tensor! tau, Tensor! error, Tensor! lengths, Tensor! diag) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl(DOSA_OP("prepare5"), TORCH_BOX(&DOSA_NAMESPACE::prepare5));
}
