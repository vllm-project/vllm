#pragma once

#ifndef _WIN32
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // #ifndef _WIN32

#include <torch/all.h>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <vector>
#include "gemm_configs.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/arch/arch.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/gemm.h"
#include "gemm_configs.h"

#ifndef _WIN32
  #pragma GCC diagnostic pop
#endif  // #ifndef _WIN32

#include "fp4_gemm_template_sm100.h"
#include "cudaUtils.h"

using namespace cute;

enum Out_Dtype { BFLOAT16, HALF, FLOAT32, UNDEFINED };

template <typename Gemm>
typename Gemm::Arguments prepareGemmArgs(void* D, void const* A, void const* B,
                                         void const* input_sf,
                                         void const* weight_sf,
                                         float const* global_sf, int m, int n,
                                         int k) {
  using Sm100BlkScaledConfig =
      typename Gemm::CollectiveMainloop::Sm100BlkScaledConfig;
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementC = void;
  using ElementD = typename Gemm::ElementD;
  using ElementCompute = float;

  typename Gemm::Arguments operator_args;
  operator_args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  auto& fusion_args = operator_args.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(global_sf);

  operator_args.problem_shape = cute::make_shape(m, n, k, /* batch_count */ 1);

  operator_args.mainloop.ptr_A = static_cast<ElementA const*>(A);
  operator_args.mainloop.ptr_B = static_cast<ElementB const*>(B);
  operator_args.mainloop.ptr_SFA = static_cast<ElementSFA const*>(input_sf);
  operator_args.mainloop.ptr_SFB = static_cast<ElementSFB const*>(weight_sf);
  operator_args.epilogue.ptr_C = static_cast<ElementC const*>(D);
  operator_args.epilogue.ptr_D = static_cast<ElementD*>(D);

  operator_args.mainloop.dA =
      cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideA>(k, 0);
  operator_args.mainloop.dB =
      cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideB>(k, 0);
  operator_args.epilogue.dC =
      cute::make_int_tuple_from<typename Gemm::GemmKernel::StrideC>(n, 0);
  operator_args.epilogue.dD = operator_args.epilogue.dC;

  operator_args.mainloop.layout_SFA =
      Sm100BlkScaledConfig::tile_atom_to_shape_SFA(operator_args.problem_shape);
  operator_args.mainloop.layout_SFB =
      Sm100BlkScaledConfig::tile_atom_to_shape_SFB(operator_args.problem_shape);

  if constexpr (!std::is_const_v<
                    decltype(operator_args.scheduler.max_swizzle_size)>) {
    operator_args.scheduler.max_swizzle_size = 1;
  }
  if constexpr (!std::is_const_v<
                    decltype(operator_args.scheduler.raster_order)>) {
    using Enum_t = decltype(operator_args.scheduler.raster_order);
    operator_args.scheduler.raster_order = Enum_t::Heuristic;
  }
  if constexpr (Gemm::ArchTag::kMinComputeCapability >= 100) {
    operator_args.hw_info.cluster_shape = dim3(1, 1, 1);
    operator_args.hw_info.cluster_shape_fallback = dim3(0, 0, 0);
  }
  return operator_args;
}

size_t Float_Fp4GemmKernelLauncher(void* D, void const* A, void const* B,
                                   void const* input_sf, void const* weight_sf,
                                   float const* global_sf, int m, int n, int k,
                                   CutlassGemmConfig gemmConfig,
                                   char* workspace, const size_t workspaceBytes,
                                   cudaStream_t stream) {
  using Fp4GemmOperator = DeviceGemmFp4GemmSm100_Float::Gemm;
  Fp4GemmOperator gemm;

  auto args = prepareGemmArgs<Fp4GemmOperator>(D, A, B, input_sf, weight_sf,
                                               global_sf, m, n, k);

  // Check shared memory size; throw when SMEM exceeds
  int smem_size =
      int(sizeof(typename Fp4GemmOperator::GemmKernel::SharedStorage));
  static int mMaxSmemSize = getMaxSharedMemoryPerBlockOptin();
  if (smem_size > mMaxSmemSize) {
    std::string errMsg = "SMEM size exceeds maximum allowed. Required " +
                         std::to_string(smem_size) + ", got " +
                         std::to_string(mMaxSmemSize);
    throw std::runtime_error("[FP4 gemm Runner] " + errMsg);
  }
  // Return workspace size
  if (!A && !B && !D) {
    return gemm.get_workspace_size(args);
  }
  if (gemm.get_workspace_size(args) > workspaceBytes) {
    std::string errMsg("Requested workspace size insufficient. Required " +
                       std::to_string(gemm.get_workspace_size(args)) +
                       ", got " + std::to_string(workspaceBytes));
    throw std::runtime_error("[FP4 gemm Runner] " + errMsg);
  }
  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string errMsg =
        "FP4 Gemm cutlass kernel will fail for params. Error: " +
        std::string(cutlassGetStatusString(can_implement));
    throw std::runtime_error("[FP4 gemm Runner] " + errMsg);
  }
  auto initStatus = gemm.initialize(args, workspace, stream);
  if (initStatus != cutlass::Status::kSuccess) {
    std::string errMsg = "Failed to initialize cutlass FP4 gemm. Error: " +
                         std::string(cutlassGetStatusString(initStatus));
    throw std::runtime_error("[FP4 gemm Runner] " + errMsg);
  }
  auto runStatus = gemm.run(args, workspace, stream);
  if (runStatus != cutlass::Status::kSuccess) {
    std::string errMsg = "Failed to run cutlass FP4 gemm. Error: " +
                         std::string(cutlassGetStatusString(runStatus));
    throw std::runtime_error("[FP4 gemm Runner] " + errMsg);
  }
  return gemm.get_workspace_size(args);
}

size_t Half_Fp4GemmKernelLauncher(void* D, void const* A, void const* B,
                                  void const* input_sf, void const* weight_sf,
                                  float const* global_sf, int m, int n, int k,
                                  CutlassGemmConfig gemmConfig, char* workspace,
                                  const size_t workspaceBytes,
                                  cudaStream_t stream) {
  using Fp4GemmOperator = DeviceGemmFp4GemmSm100_Half::Gemm;
  Fp4GemmOperator gemm;
  auto args = prepareGemmArgs<Fp4GemmOperator>(D, A, B, input_sf, weight_sf,
                                               global_sf, m, n, k);
  // Check shared memory size; throw when SMEM exceeds
  int smem_size =
      int(sizeof(typename Fp4GemmOperator::GemmKernel::SharedStorage));
  static int mMaxSmemSize = getMaxSharedMemoryPerBlockOptin();
  if (smem_size > mMaxSmemSize) {
    std::string errMsg = "SMEM size exceeds maximum allowed. Required " +
                         std::to_string(smem_size) + ", got " +
                         std::to_string(mMaxSmemSize);
    throw std::runtime_error("[FP4 gemm Runner] " + errMsg);
  }
  // Return workspace size
  if (!A && !B && !D) {
    return gemm.get_workspace_size(args);
  }
  if (gemm.get_workspace_size(args) > workspaceBytes) {
    std::string errMsg("Requested workspace size insufficient. Required " +
                       std::to_string(gemm.get_workspace_size(args)) +
                       ", got " + std::to_string(workspaceBytes));
    throw std::runtime_error("[FP4 gemm Runner] " + errMsg);
  }
  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string errMsg =
        "FP4 Gemm cutlass kernel will fail for params. Error: " +
        std::string(cutlassGetStatusString(can_implement));
    throw std::runtime_error("[FP4 gemm Runner] " + errMsg);
  }
  auto initStatus = gemm.initialize(args, workspace, stream);
  if (initStatus != cutlass::Status::kSuccess) {
    std::string errMsg = "Failed to initialize cutlass FP4 gemm. Error: " +
                         std::string(cutlassGetStatusString(initStatus));
    throw std::runtime_error("[FP4 gemm Runner] " + errMsg);
  }
  auto runStatus = gemm.run(args, workspace, stream);
  if (runStatus != cutlass::Status::kSuccess) {
    std::string errMsg = "Failed to run cutlass FP4 gemm. Error: " +
                         std::string(cutlassGetStatusString(runStatus));
    throw std::runtime_error("[FP4 gemm Runner] " + errMsg);
  }
  return gemm.get_workspace_size(args);
}

size_t BFloat16_Fp4GemmKernelLauncher(
    void* D, void const* A, void const* B, void const* input_sf,
    void const* weight_sf, float const* global_sf, int m, int n, int k,
    CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes,
    cudaStream_t stream) {
  using Fp4GemmOperator = DeviceGemmFp4GemmSm100_BFloat16::Gemm;
  Fp4GemmOperator gemm;
  auto args = prepareGemmArgs<Fp4GemmOperator>(D, A, B, input_sf, weight_sf,
                                               global_sf, m, n, k);
  // Check shared memory size; throw when SMEM exceeds
  int smem_size =
      int(sizeof(typename Fp4GemmOperator::GemmKernel::SharedStorage));
  static int mMaxSmemSize = getMaxSharedMemoryPerBlockOptin();
  if (smem_size > mMaxSmemSize) {
    std::string errMsg = "SMEM size exceeds maximum allowed. Required " +
                         std::to_string(smem_size) + ", got " +
                         std::to_string(mMaxSmemSize);
    throw std::runtime_error("[FP4 gemm Runner] " + errMsg);
  }
  // Return workspace size
  if (!A && !B && !D) {
    return gemm.get_workspace_size(args);
  }
  if (gemm.get_workspace_size(args) > workspaceBytes) {
    std::string errMsg("Requested workspace size insufficient. Required " +
                       std::to_string(gemm.get_workspace_size(args)) +
                       ", got " + std::to_string(workspaceBytes));
    throw std::runtime_error("[FP4 gemm Runner] " + errMsg);
  }
  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string errMsg =
        "FP4 Gemm cutlass kernel will fail for params. Error: " +
        std::string(cutlassGetStatusString(can_implement));
    throw std::runtime_error("[FP4 gemm Runner] " + errMsg);
  }
  auto initStatus = gemm.initialize(args, workspace, stream);
  if (initStatus != cutlass::Status::kSuccess) {
    std::string errMsg = "Failed to initialize cutlass FP4 gemm. Error: " +
                         std::string(cutlassGetStatusString(initStatus));
    throw std::runtime_error("[FP4 gemm Runner] " + errMsg);
  }
  auto runStatus = gemm.run(args, workspace, stream);
  if (runStatus != cutlass::Status::kSuccess) {
    std::string errMsg = "Failed to run cutlass FP4 gemm. Error: " +
                         std::string(cutlassGetStatusString(runStatus));
    throw std::runtime_error("[FP4 gemm Runner] " + errMsg);
  }
  return gemm.get_workspace_size(args);
}

template <typename T, typename Arch, typename TileShape>
size_t genericFp4GemmKernelLauncher(
    void* D, void const* A, void const* B, void const* input_sf,
    void const* weight_sf, float const* global_sf, int m, int n, int k,
    CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes,
    cudaStream_t stream, Out_Dtype out_dtype, int* occupancy = nullptr) {
  if (out_dtype == Out_Dtype::BFLOAT16) {
    return BFloat16_Fp4GemmKernelLauncher(D, A, B, input_sf, weight_sf,
                                          global_sf, m, n, k, gemmConfig,
                                          workspace, workspaceBytes, stream);
  } else if (out_dtype == Out_Dtype::HALF) {
    return Half_Fp4GemmKernelLauncher(D, A, B, input_sf, weight_sf, global_sf,
                                      m, n, k, gemmConfig, workspace,
                                      workspaceBytes, stream);

  } else if (out_dtype == Out_Dtype::FLOAT32) {
    return Float_Fp4GemmKernelLauncher(D, A, B, input_sf, weight_sf, global_sf,
                                       m, n, k, gemmConfig, workspace,
                                       workspaceBytes, stream);
  }

  throw std::runtime_error(
      "[FP4 gemm runner] Undefined FP4 Datatype during "
      "Gemm Operator initialization.");
}

template <typename T, typename Arch>
size_t dispatchGemmToCutlassSm100(T* D, void const* A, void const* B,
                                  void const* input_sf, void const* weight_sf,
                                  float const* global_sf, int m, int n, int k,
                                  CutlassGemmConfig gemmConfig, char* workspace,
                                  const size_t workspaceBytes,
                                  cudaStream_t stream, Out_Dtype Out_dtype,
                                  int* occupancy = nullptr) {
  switch (gemmConfig.tile_config_sm100) {
    case CutlassTileConfigSM100::CtaShape128x128x64B:
      return genericFp4GemmKernelLauncher<
          T, Arch, cute::Shape<cute::_128, cute::_128, cute::_64>>(
          D, A, B, input_sf, weight_sf, global_sf, m, n, k, gemmConfig,
          workspace, workspaceBytes, stream, Out_dtype, occupancy);
      break;
    case CutlassTileConfigSM100::CtaShape128x256x64B:
      return genericFp4GemmKernelLauncher<
          T, Arch, cute::Shape<cute::_128, cute::_256, cute::_64>>(
          D, A, B, input_sf, weight_sf, global_sf, m, n, k, gemmConfig,
          workspace, workspaceBytes, stream, Out_dtype, occupancy);
      break;
    // case CutlassTileConfigSM100::CtaShape256x256x64B:
    //     return genericFp4GemmKernelLauncher<T, Arch, cute::Shape<cute::_256,
    //     cute::_256, cute::_64>>(
    //         D, A, B, input_sf, weight_sf, global_sf, m, n, k, gemmConfig,
    //         workspace, workspaceBytes, stream, occupancy);
    //     break;
    case CutlassTileConfigSM100::Undefined:
      throw std::runtime_error(
          "[FP4][dispatch_gemm_to_cutlass] gemm config undefined.");
      break;
    case CutlassTileConfigSM100::ChooseWithHeuristic:
      throw std::runtime_error(
          "[FP4][dispatch_gemm_to_cutlass] gemm config should have already "
          "been set by "
          "heuristic.");
      break;
    default:
      throw std::runtime_error(
          "[FP4][dispatch_gemm_to_cutlass] Config is invalid for FP4 GEMM.");
      break;
  }
}

template <typename T>
size_t dispatchToArch(T* D, void const* A, void const* B, void const* input_sf,
                      void const* weight_sf, float const* global_sf, int m,
                      int n, int k, CutlassGemmConfig gemmConfig,
                      char* workspace, const size_t workspaceBytes,
                      cudaStream_t stream,
                      Out_Dtype out_type_enum = Out_Dtype::UNDEFINED,
                      int* occupancy = nullptr) {
  if (getSMVersion() == 100) {
    return dispatchGemmToCutlassSm100<T, cutlass::arch::Sm100>(
        D, A, B, input_sf, weight_sf, global_sf, m, n, k, gemmConfig, workspace,
        workspaceBytes, stream, out_type_enum, occupancy);
  } else {
    throw std::runtime_error(
        "[GEMM Dispatch] Arch unsupported for CUTLASS "
        "FP4 GEMM");
  }
}

template size_t dispatchToArch<__nv_bfloat16>(
    __nv_bfloat16* D, void const* A, void const* B, void const* input_sf,
    void const* weight_sf, float const* global_sf, int m, int n, int k,
    CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes,
    cudaStream_t stream, Out_Dtype out_type_enum = Out_Dtype::BFLOAT16,
    int* occupancy = nullptr);

template size_t dispatchToArch<at::Half>(
    at::Half* D, void const* A, void const* B, void const* input_sf,
    void const* weight_sf, float const* global_sf, int m, int n, int k,
    CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes,
    cudaStream_t stream, Out_Dtype out_type_enum = Out_Dtype::HALF,
    int* occupancy = nullptr);

template size_t dispatchToArch<float>(
    float* D, void const* A, void const* B, void const* input_sf,
    void const* weight_sf, float const* global_sf, int m, int n, int k,
    CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes,
    cudaStream_t stream, Out_Dtype out_type_enum = Out_Dtype::FLOAT32,
    int* occupancy = nullptr);

std::vector<CutlassGemmConfig> getConfigs() {
  std::vector<CutlassGemmConfig> candidateConfigs;
  std::vector<CutlassTileConfigSM100> tilesSm100 = {
      CutlassTileConfigSM100::CtaShape128x128x64B,
      CutlassTileConfigSM100::CtaShape128x256x64B,
      // Error for M=256.
      // CutlassTileConfigSM100::CtaShape256x256x64B,
  };
  for (auto const& tile_config : tilesSm100) {
    CutlassGemmConfig config(tile_config, MainloopScheduleType::AUTO,
                             EpilogueScheduleType::AUTO,
                             ClusterShape::ClusterShape_1x1x1);
    candidateConfigs.push_back(config);
  }

  return candidateConfigs;
}

template <typename T>
size_t getWorkspaceSizeImpl(int const m, int const n, int const k) {
  size_t workspace_size = 0;
  auto gemmConfigs = getConfigs();
  for (auto const& gemmConfig : gemmConfigs) {
    try {
      size_t curr_workspace_size =
          dispatchToArch<T>(nullptr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, m, n, k, gemmConfig, nullptr, 0, 0);
      workspace_size = std::max(workspace_size, curr_workspace_size);
    } catch (std::runtime_error& e) {
      // Swallow errors when SMEM exceeds maximum allowed
      continue;
    }
  }
  return workspace_size;
}

template <typename T>
size_t getWorkspaceSize(int const m, int const n, int const k) {
  // Custom hash function for the MNK type
  using MNK = std::tuple<int, int, int>;

  struct MNKHash {
    size_t operator()(const MNK& mnk) const {
      auto h1 = std::hash<int>{}(std::get<0>(mnk));
      auto h2 = std::hash<int>{}(std::get<1>(mnk));
      auto h3 = std::hash<int>{}(std::get<2>(mnk));
      return h1 ^ h2 ^ h3;
    }
  };

  static std::unordered_map<MNK, size_t, MNKHash> workspace_hashmap;

  size_t workspace_size = 0;
  if (workspace_hashmap.find(std::make_tuple(m, n, k)) ==
      workspace_hashmap.end()) {
    workspace_size = getWorkspaceSizeImpl<T>(m, n, k);
    workspace_hashmap[std::make_tuple(m, n, k)] = workspace_size;
  } else {
    workspace_size = workspace_hashmap[std::make_tuple(m, n, k)];
  }
  return workspace_size;
}

#define CHECK_TYPE(x, st, m) \
  TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type:", m)
#define CHECK_TH_CUDA(x, m) TORCH_CHECK(x.is_cuda(), m, "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) \
  TORCH_CHECK(x.is_contiguous(), m, " must be contiguous")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)

// TODO(kaixih@nvidia): switch to use native fp4 dtype when available.
constexpr auto FLOAT4_E2M1X2 = at::ScalarType::Byte;
constexpr auto SF_DTYPE = at::ScalarType::Float8_e4m3fn;

void check_input(torch::Tensor& A, c10::ScalarType type, const char* str) {
  TORCH_CHECK(A.is_contiguous(), str, "must be contiguous");
  TORCH_CHECK(A.is_cuda(), str, "must be a CUDA tensor");
  TORCH_CHECK(A.scalar_type() == type, str, "got Tensor Type unexpected.");
  TORCH_CHECK(A.dim() == 2, str, " must be a matrix of rank 2.");
}

/// Expectations: A: [m, k] Contiguous dim: k
///               B: [n, k] Contiguous dim: k
void cutlass_fp4_gemm(torch::Tensor& D, torch::Tensor& A, torch::Tensor& B,
                      torch::Tensor& input_sf, torch::Tensor& weight_sf,
                      torch::Tensor& global_sf, torch::Tensor& workspace,
                      const int64_t workspaceBytes) {
  check_input(A, at::ScalarType::Byte, "Matrix A ");
  check_input(B, at::ScalarType::Byte, "Matrix B ");
  check_input(input_sf, at::ScalarType::Float8_e4m3fn,
              "Block Scale of Matrix A ");
  check_input(weight_sf, at::ScalarType::Float8_e4m3fn,
              "Block Scale of Matrix B");
  TORCH_CHECK(global_sf.scalar_type() == at::ScalarType::Float,
              "Alpha, i.e. rec(input_scale * weight_scale_2) ",
              " got Tensor Type unexpected.");

  int32_t m = A.size(0);
  int32_t n = B.size(0);
  int32_t k = A.size(1) * 2;  // since k is packed

  constexpr int alignment = 32;
  TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment,
              ", but got a shape: (", A.sizes()[0], "x", A.sizes()[1],
              "), k: ", k, ".");
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment,
              ", but got b shape: (", B.sizes()[0], "x", B.sizes()[1], ").");

  auto global_sf_ptr = static_cast<float const*>(global_sf.data_ptr());
  auto workspace_ptr = static_cast<char*>(workspace.data_ptr());
  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());
  auto gemmConfig = getConfigs()[0];  // TODO: fix this based on the size

  switch (D.scalar_type()) {
    case torch::kFloat: {
      dispatchToArch<float>(reinterpret_cast<float*>(D.data_ptr()),
                            A.data_ptr(), B.data_ptr(), input_sf.data_ptr(),
                            weight_sf.data_ptr(), global_sf_ptr, m, n, k,
                            gemmConfig, workspace_ptr, workspaceBytes, stream,
                            Out_Dtype::FLOAT32);
      break;
    }
    case torch::kHalf: {
      dispatchToArch<at::Half>(
          reinterpret_cast<at::Half*>(D.data_ptr()), A.data_ptr(), B.data_ptr(),
          input_sf.data_ptr(), weight_sf.data_ptr(), global_sf_ptr, m, n, k,
          gemmConfig, workspace_ptr, workspaceBytes, stream, Out_Dtype::HALF);
      break;
    }
    case torch::kBFloat16: {
      dispatchToArch<__nv_bfloat16>(
          reinterpret_cast<__nv_bfloat16*>(D.data_ptr()), A.data_ptr(),
          B.data_ptr(), input_sf.data_ptr(), weight_sf.data_ptr(),
          global_sf_ptr, m, n, k, gemmConfig, workspace_ptr, workspaceBytes,
          stream, Out_Dtype::BFLOAT16);
      break;
    }
    default:
      throw std::runtime_error("Unsupported data type for Fp4GemmRunner.");
  }
}

__device__ int computeSFIndex(int rowIdx, int colIdx, int totalRow,
                              int totalColumn) {
  constexpr int kColumnGroup0Size = 4;
  constexpr int kRowGroup0Size = 32;
  constexpr int kRowGroup1Size = 128;

  // Padding logic
  int paddedColumn =
      ((totalColumn + kColumnGroup0Size - 1) / kColumnGroup0Size) *
      kColumnGroup0Size;

  // Compute indices
  int columnIdxInGroup0 = colIdx % kColumnGroup0Size;
  int columnGroupIdx = colIdx / kColumnGroup0Size;
  int columnGroupStride = 512;

  int rowIdxInGroup0 = rowIdx % kRowGroup0Size;
  int rowGroup0Stride = 16;
  int rowIdxInGroup1 = (rowIdx % kRowGroup1Size) / kRowGroup0Size;
  int rowGroup1Stride = 4;
  int rowGroupIdx = rowIdx / kRowGroup1Size;
  int rowGroupStride = kRowGroup1Size * paddedColumn;

  return columnIdxInGroup0 + columnGroupIdx * columnGroupStride +
         rowIdxInGroup0 * rowGroup0Stride + rowIdxInGroup1 * rowGroup1Stride +
         rowGroupIdx * rowGroupStride;
}

__global__ void blockscale_interleave_fp4_kernel(int8_t* output_ptr,
                                                 const int8_t* input_ptr,
                                                 int rows, int cols,
                                                 int num_experts,
                                                 int expert_out_size) {
  int eIdx = blockIdx.z;  // Expert index (z-dimension of grid)
  int rIdx = blockIdx.y * blockDim.y + threadIdx.y;  // Row index
  int cIdx = blockIdx.x * blockDim.x + threadIdx.x;  // Column index

  if (eIdx < num_experts && rIdx < rows && cIdx < cols) {
    auto globalRowIdx = eIdx * rows + rIdx;
    const int8_t* blockScalePtr = input_ptr + globalRowIdx * cols;
    int8_t* interleavedBlockScalePtr = output_ptr + eIdx * expert_out_size;

    int sf_index = computeSFIndex(rIdx, cIdx, rows, cols);
    interleavedBlockScalePtr[sf_index] = blockScalePtr[cIdx];
  }
}

void blockscale_interleave_fp4(torch::Tensor& output, torch::Tensor& input,
                               int64_t rows, int64_t cols, int64_t num_experts,
                               int64_t expert_out_size) {
  auto input_ptr = static_cast<int8_t*>(input.data_ptr());
  auto output_ptr = static_cast<int8_t*>(output.data_ptr());

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (rows + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 num_experts);

  blockscale_interleave_fp4_kernel<<<numBlocks, threadsPerBlock>>>(
      output_ptr, input_ptr, rows, cols, num_experts, expert_out_size);

  cudaDeviceSynchronize();
}
