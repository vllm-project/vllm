// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Host-side launcher of the wvSplitKrc skiny GEMM. The kernels (and the device code
// they instantiate) live in skinny_gemms.cuh; each family is kept in its own
// translation unit so ninja compiles the families in parallel (skinny_gemms.cu
// used to hold all of them and took ~58 s of device code generation on one
// core).
#include "skinny_gemms.cuh"

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount) {
  static constexpr bool _DTRMNSTC = true;

  auto M_in = in_b.size(0);
  auto N_in = in_a.size(0);
  auto K_in = in_b.size(1);
  auto Kap_in = in_a.stride(0);

  auto Bx_in =
      (in_bias.has_value() && in_bias->numel() > 0)
          ? (in_bias->sizes().size() == 2) ? in_bias->size(1) : in_bias->size(0)
          : 1;
  auto By_in = (in_bias.has_value() && in_bias->numel() > 0 &&
                in_bias->sizes().size() == 2)
                   ? in_bias->size(0)
                   : 1;

  TORCH_CHECK(in_a.dtype() == in_b.dtype());
  TORCH_CHECK(K_in % 8 == 0, "k % 8 == 0");
  TORCH_CHECK(in_a.dtype() == torch::kFloat16 ||
              in_a.dtype() == torch::kBFloat16);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(in_a));

  auto out_c = torch::empty(
      {N_in, M_in},
      torch::TensorOptions().dtype(in_a.dtype()).device(in_a.device()));

  auto N_p2 = 1U << (32 - __builtin_clz(N_in - 1));

  dim3 grid(CuCount);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // const int max_lds_len = get_lds_size() / 2;

  // With 64 Ms per CU (each of 4 SIMDs working on a 16x16 tile),
  // and each working on a 512-shard of K, how many CUs would we need?
  int rndup_cus = ((M_in + 64 - 1) / 64) * ((K_in + 512 - 1) / 512);

  // How many of 4 waves in a group can work on same 16 Ms at same time? First
  // try to maximize this. This reduces the Ms each group works on, i.e.
  // increasing the number of CUs needed.
  int GrpsShrB = min(N_p2 / 16, 4);

  // Given the above, how many CUs would we need?
  int CuNeeded = rndup_cus * GrpsShrB;

  if (CuNeeded > CuCount) throw std::runtime_error("Invalid wvSplitKrc size");

  // Can we increase SplitK by shrinking the K-shard to 256? That doubles both
  // the CUs needed and the k-shards to read back, and the readback is serial in
  // the shard count, so it pays only up to a crossover. The crossover is not a
  // constant -- it falls as M grows -- so one cap cannot sit on it for every
  // shape; this value minimizes total time over the admitted shapes. Retune it
  // if the readback cost changes, and note the directions are not symmetric:
  // capping too high costs about twice what capping too low does.
  // Mirrored by rocm_unquantized_gemm_impl() in
  // vllm/model_executor/layers/utils.py; both must pick the same chunkk or the
  // host-side fit check bounds the wrong k_rnd.
  constexpr int64_t CHUNKK2_MAX_SHARDS = 11;
  const int64_t shards_chunkk2 = (K_in + 256 - 1) / 256;  // 256 == 512 / 2
  const int chunkk = (N_p2 != 16 && CuNeeded * 2 <= CuCount &&
                      shards_chunkk2 <= CHUNKK2_MAX_SHARDS)
                         ? 2
                         : 1;

  // One fp32 partial per (M, N, k-shard) must fit the split-K workspace.
  const int64_t k_rnd = (K_in + 512 / chunkk - 1) / (512 / chunkk);
  constexpr int64_t glbl_cap = 128 * 1024 * (_DTRMNSTC ? 12 : 1);
  const int64_t glbl_needed =
      _DTRMNSTC ? (int64_t)N_p2 * M_in * k_rnd : (int64_t)N_p2 * M_in;
  TORCH_CHECK(glbl_needed <= glbl_cap,
              "wvSplitKrc: split-K workspace too small", " (need ", glbl_needed,
              " > ", glbl_cap, " floats) for N=", N_in, " K=", K_in,
              " M=", M_in);

  // Split-K workspaces: one pre-partitioned pool per device.
  // These must be zeroed before first use, and the kernel re-zeros them after
  // each use.
  // To avoid unnecessary overhead, we do not allocate them zeroed
  // per-invocation. However, a simple static allocation does not work with
  // multi-streams, because concurrent streams can alias into the same
  // workspace. Solution: pre-allocate kWvSlots = 8 slots per device, plus
  // kWvCaptureSlots reserved for capturing streams, and zero the whole pool
  // once at creation. On first use, a stream takes one of the slots if any are
  // still available. On subsequent uses, it re-uses the slot. If no slots are
  // available, fall back to allocating a zeroed one per call, which is safe but
  // adds overhead.
  // A capturing stream must neither allocate nor enqueue anything -- either
  // would be captured into the graph -- so it is only handed a reserved slot.
  struct WvSplitKrcPool {
    torch::Tensor glbl, cntr;
    int64_t glbl_stride = 0, cntr_stride = 0;
    int next_slot = 0;
    int next_capture_slot = 0;
    std::map<cudaStream_t, int> slot_of;
    std::map<cudaStream_t, int> capture_slot_of;
  };
  // mutex not strictly necessary if caller is always single-threaded (Python)
  // but this preserves Torch thread_local semantics for the C++ API
  static std::mutex wv_pool_mu;
  static std::map<int, WvSplitKrcPool> wv_pools;

  float* glbl;
  int* cntr;
  // Fallback space
  torch::Tensor spill_glbl, spill_cntr;
  {
    std::lock_guard<std::mutex> wv_lk(wv_pool_mu);
    const int wv_dev = static_cast<int>(in_a.device().index());
    WvSplitKrcPool& P = wv_pools[wv_dev];
    constexpr int64_t wv_gf = 128 * 1024 * (_DTRMNSTC ? 12 : 1);
    constexpr int64_t wv_ci = wv_gf / 4;
    constexpr int64_t wv_all_slots = kWvSlots + kWvCaptureSlots;

    // A stream whose capture was already invalidated reports an error; treat
    // that as not capturing, since the capture is lost either way.
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    unsigned long long capture_id = 0;
    if (cudaStreamGetCaptureInfo(stream, &capture_status, &capture_id) !=
        cudaSuccess) {
      capture_status = cudaStreamCaptureStatusNone;
    }
    const bool wv_capturing = capture_status != cudaStreamCaptureStatusNone;

    if (!P.glbl.defined()) {
      TORCH_CHECK(!wv_capturing,
                  "wvSplitKrc: the split-K workspace pool cannot be allocated "
                  "during graph capture. Call "
                  "warmup_rocm_skinny_gemm_workspaces() "
                  "(vllm/model_executor/layers/utils.py) eagerly first.");
      P.glbl_stride = wv_gf;
      P.cntr_stride = wv_ci;
      P.glbl = torch::zeros(wv_gf * wv_all_slots, torch::TensorOptions()
                                                      .dtype(torch::kFloat32)
                                                      .device(in_a.device()))
                   .detach();
      P.cntr =
          torch::zeros(
              wv_ci * wv_all_slots,
              torch::TensorOptions().dtype(torch::kInt).device(in_a.device()))
              .detach();
      // The mutex orders host access to the map, not device execution, so it
      // alone would let another stream be handed a slot the fill above is
      // still clearing. Synchronizing here, inside the lock, is what makes
      // every slot safe to use without any further per-slot zeroing.
      const cudaError_t wv_err = cudaStreamSynchronize(stream);
      TORCH_CHECK(wv_err == cudaSuccess,
                  "wvSplitKrc: failed to zero the split-K workspace pool: ",
                  cudaGetErrorString(wv_err));
    }

    int wv_slot = -1;
    if (wv_capturing) {
      // One workspace per capturing stream, so graphs captured on the same
      // stream share it -- safe because at most one of them replays at a time
      // (vLLM captures one graph per padded batch size, and runs one per step).
      auto cap_it = P.capture_slot_of.find(stream);
      if (cap_it == P.capture_slot_of.end()) {
        TORCH_CHECK(P.next_capture_slot < kWvCaptureSlots,
                    "wvSplitKrc: no free capture workspace slot; more than ",
                    kWvCaptureSlots,
                    " distinct streams have captured a graph containing "
                    "wvSplitKrc. Raise kWvCaptureSlots in "
                    "csrc/rocm/skinny_gemms.cu.");
        cap_it = P.capture_slot_of
                     .emplace(stream, static_cast<int>(kWvSlots) +
                                          P.next_capture_slot++)
                     .first;
      }
      wv_slot = cap_it->second;
    } else {
      auto slot_it = P.slot_of.find(stream);
      if (slot_it != P.slot_of.end()) {
        wv_slot = slot_it->second;
      } else if (P.next_slot < kWvSlots) {
        wv_slot = P.next_slot++;
        P.slot_of.emplace(stream, wv_slot);
      }
    }

    if (wv_slot >= 0) {
      glbl = P.glbl.data_ptr<float>() + wv_slot * P.glbl_stride;
      cntr = P.cntr.data_ptr<int>() + wv_slot * P.cntr_stride;
    } else {
      spill_glbl = torch::zeros(
          wv_gf,
          torch::TensorOptions().dtype(torch::kFloat32).device(in_a.device()));
      spill_cntr = torch::zeros(
          wv_ci,
          torch::TensorOptions().dtype(torch::kInt).device(in_a.device()));
      glbl = spill_glbl.data_ptr<float>();
      cntr = spill_cntr.data_ptr<int>();
    }
  }

#define WVSPLITKrc(_N, _GrpsShrB, _CHUNKK)                                     \
  {                                                                            \
    dim3 block(64, 4);                                                         \
    if constexpr (_DTRMNSTC)                                                   \
      wvSplitKrc_<fptype, 64, 16, 4, 8, 1, _N, _GrpsShrB, _CHUNKK, 1>          \
          <<<grid, block, 0, stream>>>(N_in, K_in, Kap_in, M_in, Bx_in, By_in, \
                                       af4, bf4, biasf4, glbl, cntr, c,        \
                                       CuCount);                               \
    else                                                                       \
      wvSplitKrc_<fptype, 64, 16, 4, 8, 1, _N, _GrpsShrB, _CHUNKK, 0>          \
          <<<grid, block, 0, stream>>>(N_in, K_in, Kap_in, M_in, Bx_in, By_in, \
                                       af4, bf4, biasf4, glbl, cntr, c,        \
                                       CuCount);                               \
  }

  AT_DISPATCH_REDUCED_FLOATING_TYPES(in_a.scalar_type(), "wvSplitKrc", [&] {
    using fptype = typename scalar<scalar_t>::type;
    const fptype* af4 = reinterpret_cast<const fptype*>(in_a.data_ptr());
    const fptype* bf4 = reinterpret_cast<const fptype*>(in_b.data_ptr());
    const fptype* biasf4 =
        (in_bias.has_value() && in_bias->numel() > 0)
            ? reinterpret_cast<const fptype*>(in_bias->data_ptr())
            : nullptr;
    fptype* c = reinterpret_cast<fptype*>(out_c.data_ptr());

    switch (N_p2) {
      case 16:
        WVSPLITKrc(16, 1, 1) break;
      case 32:
        if (chunkk == 2) WVSPLITKrc(32, 2, 2) else WVSPLITKrc(32, 2, 1) break;
      case 64:
        if (chunkk == 2) WVSPLITKrc(64, 4, 2) else WVSPLITKrc(64, 4, 1) break;
      case 128:
        if (chunkk == 2) WVSPLITKrc(128, 4, 2) else WVSPLITKrc(128, 4, 1) break;
      default:
        throw std::runtime_error(
            "Unsupported N value: " + std::to_string(M_in) + "," +
            std::to_string(K_in) + "," + std::to_string(N_in));
    }
  });
  return out_c;
}
