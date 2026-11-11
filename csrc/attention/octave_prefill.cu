// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "octave_prefill.cuh"

#if defined(USE_ROCM)
  #include <algorithm>
  #include <mutex>
  #include <string>
  #include <unordered_map>
  #include <vector>

  #include <c10/cuda/CUDAGuard.h>
  #include <ATen/cuda/CUDAContext.h>

  #include "octave_format.cuh"

namespace {

// Return per-device storage for prefix partials. Retain older allocations so
// pointers captured by GPU graphs remain valid after the workspace grows.
float* workspace(int64_t need, const at::TensorOptions& opts) {
  static std::mutex mtx;
  static std::unordered_map<int, std::vector<at::Tensor>> kept;
  std::lock_guard<std::mutex> lock(mtx);
  std::vector<at::Tensor>& bufs = kept[opts.device().index()];
  if (bufs.empty() || bufs.back().numel() < need) {
    int64_t n = 8 << 20;
    while (n < need) n *= 2;
    bufs.push_back(at::empty({n}, opts));
  }
  return (float*)bufs.back().data_ptr();
}

}  // namespace

// Validate tensors and choose the target, split count, and prefix workspace
// before invoking the architecture-specific two-phase prefill launcher.
void octave_prefill(torch::Tensor out, torch::Tensor q, torch::Tensor k,
                    torch::Tensor v, torch::Tensor cache,
                    torch::Tensor block_table, torch::Tensor cu_seqlens_q,
                    torch::Tensor seq_lens, int64_t max_query_len,
                    double sm_scale, int64_t fmt) {
  using octave::D;
  TORCH_CHECK(q.dtype() == at::kHalf && k.dtype() == at::kHalf &&
                  v.dtype() == at::kHalf && out.dtype() == at::kHalf,
              "octave_prefill: fp16 only");
  TORCH_CHECK(q.size(2) == D && q.stride(2) == 1 && k.stride(2) == 1 &&
              v.stride(2) == 1 && out.stride(2) == 1);
  TORCH_CHECK(cache.dtype() == at::kByte && cache.dim() == 4 &&
              cache.stride(3) == 1);
  TORCH_CHECK(cache.size(3) == octave_slot_bytes(fmt),
              "octave_prefill: slot size mismatch");
  const int align = fmt & octave::kCompactFlag ? 4 : 8;
  TORCH_CHECK(cache.stride(2) % align == 0 && cache.stride(1) % align == 0 &&
                  cache.stride(0) % align == 0,
              "octave_prefill: misaligned slots");

  // 0: gfx1100 (WMMA), 1: gfx942 (MFMA, octave_prefill_cdna3.cu).
  static const int arch = [] {
    const std::string name =
        at::cuda::getCurrentDeviceProperties()->gcnArchName;
    if (name.rfind("gfx1100", 0) == 0) return 0;
    if (name.rfind("gfx942", 0) == 0) return 1;
    return -1;
  }();
  TORCH_CHECK(arch >= 0, "octave_prefill: gfx1100 or gfx942 only");
  const int num_seqs = seq_lens.size(0);
  if (num_seqs == 0 || q.size(0) == 0) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(q));
  const int hq = q.size(1), hkv = cache.size(1);
  TORCH_CHECK(hq % hkv == 0);
  const int hpk = hq / hkv;
  const int max_blocks = block_table.size(1);
  TORCH_CHECK(block_table.stride(1) == 1 && block_table.stride(0) == max_blocks,
              "octave_prefill: block_table rows must be contiguous");
  const int total_q = q.size(0);

  // TODO(tuning): When adding RDNA3.5/4, use its prefix row tile for this
  // block count estimate; tune the 16-split cap and 1024-block target.
  const int rowtiles = (int)((max_query_len * hpk + 127) / 128);
  const int base_blocks = std::max(1, num_seqs * hkv * rowtiles);
  // Blocks whose split has no keys exit at once, so ask for plenty.
  const int num_splits =
      std::min(16, std::max(1, (1024 + base_blocks - 1) / base_blocks));
  float* ws = workspace((int64_t)num_splits * total_q * hq * (D + 2),
                        q.options().dtype(at::kFloat));
  const octave::PrefillArgs args{out,
                                 q,
                                 k,
                                 v,
                                 cache,
                                 block_table,
                                 cu_seqlens_q,
                                 seq_lens,
                                 max_query_len,
                                 sm_scale,
                                 fmt,
                                 ws,
                                 num_splits};
  if (arch == 1)
    octave_cdna::octave_prefill(args);
  else
    octave_rdna::octave_prefill(args);
}

#else
void octave_prefill(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    int64_t, double, int64_t) {
  TORCH_CHECK(false, "octave requires ROCm");
}
#endif
