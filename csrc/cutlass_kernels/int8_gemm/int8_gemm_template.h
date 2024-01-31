/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // #ifndef _WIN32

// clang-format off
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass_extensions/gemm/device/gemm_universal_base_compat.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
// clang-format on

#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/epilogue/threadblock/epilogue_per_row_per_col_scale.h"
#include "cutlass_extensions/epilogue/threadblock/epilogue_tensor_op_int32.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm_configs.h"

#include "cutlass_extensions/gemm/kernel/default_int8_traits.h"
#include "cutlass_extensions/gemm/kernel/gemm_with_epilogue_visitor.h"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif // #ifndef _WIN32

#include "tensorrt_llm/common/allocator.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/kernels/cutlass_kernels/int8_gemm/int8_gemm.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"

#include <chrono>
#include <sstream>

namespace tk = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

template <typename T, typename arch, typename ThreadblockShape, typename WarpShape, int Stages>
void genericInt8GemmKernelLauncher(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
    const float* alphaRow, T* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspace,
    size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    using ElementInput = int8_t;

    using ElementOutput_ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
#ifdef ENABLE_BF16
    using ElementOutput =
        typename cutlass::platform::conditional<cutlass::platform::is_same<ElementOutput_, __nv_bfloat16>::value,
            cutlass::bfloat16_t, ElementOutput_>::type;
#else
    using ElementOutput = ElementOutput_;
#endif

    using ElementAccumulator = int32_t;
    using ElementCompute = float;

    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    using OperatorClass = typename cutlass::gemm::kernel::Int8GemmArchTraits<arch>::OperatorClass;
    using InstructionShape = typename cutlass::gemm::kernel::Int8GemmArchTraits<arch>::InstructionShape;

    using DefaultGemmConf = typename cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass, arch, ElementInput,
        ElementInput, ElementOutput, ElementCompute>;
    using GemmOp = typename DefaultGemmConf::Operator;
    using EpilogueOp = typename DefaultGemmConf::EpilogueOutputOp;

    // only TN is supported (s8 * s8 + s32)
    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<ElementInput, cutlass::layout::RowMajor,
        DefaultGemmConf::kAlignmentA, ElementInput, cutlass::layout::ColumnMajor, DefaultGemmConf::kAlignmentB,
        ElementOutput, cutlass::layout::RowMajor, ElementAccumulator, OperatorClass, arch, ThreadblockShape, WarpShape,
        InstructionShape, EpilogueOp, ThreadblockSwizzle, Stages, true, GemmOp>::GemmKernel;

    using AlphaColTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
        cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
            typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Shape,
            typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Count,
            GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::kThreads,
            GemmKernel_::Epilogue::OutputTileIterator::kElementsPerAccess, cutlass::sizeof_bits<ElementOutput>::value>,
        ElementCompute>;

    // Epilogue visitor
    using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<ThreadblockShape,
        GemmKernel_::kThreadCount, AlphaColTileIterator, typename GemmKernel_::Epilogue::OutputTileIterator,
        ElementAccumulator, ElementCompute, EpilogueOp>;

    /// Epilogue
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<EpilogueVisitor,
        typename GemmKernel_::Epilogue>::Epilogue;

    // GEMM
    using GemmKernel
        = cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmKernel_::Mma, Epilogue, ThreadblockSwizzle>;

    if (occupancy != nullptr)
    {
        *occupancy = tensorrt_llm::cutlass_extensions::compute_occupancy_for_kernel<GemmKernel>();
        return;
    }

    using Gemm = cutlass::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

    typename EpilogueOp::Params linearScalingParams; // TODO: right now it's unused (scaling is done in
                                                     // visitor, no activation needed)
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kBatched, {m, n, k}, 1,
        {reinterpret_cast<ElementInput*>(const_cast<ElementInput*>(A)), k},
        {reinterpret_cast<ElementInput*>(const_cast<ElementInput*>(B)), k}, quantOption,
        {reinterpret_cast<ElementCompute*>(const_cast<float*>(alphaCol)), 0},
        {reinterpret_cast<ElementCompute*>(const_cast<float*>(alphaRow)), 0}, {nullptr, 0},
        {reinterpret_cast<ElementOutput*>(C), n}, 0, 0,
        typename EpilogueVisitor::Arguments(linearScalingParams, 0, 0, 0)};

    Gemm gemm;
    // TODO: handle that
    if (gemm.get_workspace_size(args) > workspaceBytes)
    {
        TLLM_LOG_WARNING(
            "Requested split-k but workspace size insufficient. Falling back to non-split-k implementation.");
        // If requested split-k factor will require more workspace bytes, revert to standard gemm.
        args.batch_count = 1;
    }

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess)
    {
        std::string errMsg = "int8gemm cutlass kernel will fail for params. Error: "
            + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[TensorRT-LLM Error][int8gemm Runner] " + errMsg);
    }

    auto initStatus = gemm.initialize(args, workspace, stream);
    if (initStatus != cutlass::Status::kSuccess)
    {
        std::string errMsg
            = "Failed to initialize cutlass int8 gemm. Error: " + std::string(cutlassGetStatusString(initStatus));
        throw std::runtime_error("[TensorRT-LLM Error][int8gemm Runner] " + errMsg);
    }

    auto runStatus = gemm.run(stream);
    if (runStatus != cutlass::Status::kSuccess)
    {
        std::string errMsg
            = "Failed to run cutlass int8 gemm. Error: " + std::string(cutlassGetStatusString(runStatus));
        throw std::runtime_error("[TensorRT-LLM Error][int8gemm Runner] " + errMsg);
    }
}

template <typename T, typename arch, typename ThreadblockShape, typename WarpShape, int Stages, typename Enable = void>
struct dispatchStages
{
    static void dispatch(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
        const float* alphaRow, T* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspace,
        size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
    {
        TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
        std::string errMsg = "Cutlass int8 gemm. Not instantiates for arch "
            + std::to_string(arch::kMinComputeCapability) + " with stages set to " + std::to_string(Stages);
        throw std::runtime_error("[TensorRT-LLM Error][dispatchStages::dispatch] " + errMsg);
    }
};

template <typename T, typename arch, typename ThreadblockShape, typename WarpShape>
struct dispatchStages<T, arch, ThreadblockShape, WarpShape, 2>
{
    static void dispatch(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
        const float* alphaRow, T* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspace,
        size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
    {
        TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
        genericInt8GemmKernelLauncher<T, arch, ThreadblockShape, WarpShape, 2>(A, B, quantOption, alphaCol, alphaRow, C,
            m, n, k, gemmConfig, workspace, workspaceBytes, stream, occupancy);
    }
};

template <typename T, typename ThreadblockShape, typename WarpShape, int Stages>
struct dispatchStages<T, cutlass::arch::Sm80, ThreadblockShape, WarpShape, Stages,
    typename std::enable_if<(Stages > 2)>::type>
{
    static void dispatch(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
        const float* alphaRow, T* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspace,
        size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
    {

        TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
        genericInt8GemmKernelLauncher<T, cutlass::arch::Sm80, ThreadblockShape, WarpShape, Stages>(A, B, quantOption,
            alphaCol, alphaRow, C, m, n, k, gemmConfig, workspace, workspaceBytes, stream, occupancy);
    }
};

template <typename T, typename arch, typename ThreadblockShape, typename WarpShape>
void dispatchGemmConfig(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
    const float* alphaRow, T* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspace,
    size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr)
{

    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    switch (gemmConfig.stages)
    {
    case 2:
        using DispatcherStages2 = dispatchStages<T, arch, ThreadblockShape, WarpShape, 2>;
        DispatcherStages2::dispatch(A, B, quantOption, alphaCol, alphaRow, C, m, n, k, gemmConfig, workspace,
            workspaceBytes, stream, occupancy);
        break;
    case 3:
        using DispatcherStages3 = dispatchStages<T, arch, ThreadblockShape, WarpShape, 3>;
        DispatcherStages3::dispatch(A, B, quantOption, alphaCol, alphaRow, C, m, n, k, gemmConfig, workspace,
            workspaceBytes, stream, occupancy);
        break;
    case 4:
        using DispatcherStages4 = dispatchStages<T, arch, ThreadblockShape, WarpShape, 4>;
        DispatcherStages4::dispatch(A, B, quantOption, alphaCol, alphaRow, C, m, n, k, gemmConfig, workspace,
            workspaceBytes, stream, occupancy);
        break;
    case 5:
        using DispatcherStages5 = dispatchStages<T, arch, ThreadblockShape, WarpShape, 5>;
        DispatcherStages5::dispatch(A, B, quantOption, alphaCol, alphaRow, C, m, n, k, gemmConfig, workspace,
            workspaceBytes, stream, occupancy);
        break;
    case 6:
        using DispatcherStages6 = dispatchStages<T, arch, ThreadblockShape, WarpShape, 6>;
        DispatcherStages6::dispatch(A, B, quantOption, alphaCol, alphaRow, C, m, n, k, gemmConfig, workspace,
            workspaceBytes, stream, occupancy);
        break;
    default:
        std::string errMsg = "dispatchGemmConfig does not support stages " + std::to_string(gemmConfig.stages);
        throw std::runtime_error("[TensorRT-LLM Error][dispatch_gemm_config] " + errMsg);
        break;
    }
}

template <typename T, typename arch>
void dispatchGemmToCutlass(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
    const float* alphaRow, T* C, int m, int n, int k, char* workspace, size_t workspaceBytes,
    tkc::CutlassGemmConfig gemmConfig, cudaStream_t stream, int* occupancy = nullptr)
{

    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    switch (gemmConfig.tile_config)
    {
    case tkc::CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64:
        dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 32, 64>>(A, B,
            quantOption, alphaCol, alphaRow, C, m, n, k, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64:
        dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<256, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>>(A, B,
            quantOption, alphaCol, alphaRow, C, m, n, k, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
        dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<32, 128, 64>, cutlass::gemm::GemmShape<32, 32, 64>>(A, B,
            quantOption, alphaCol, alphaRow, C, m, n, k, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
        dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<64, 128, 64>, cutlass::gemm::GemmShape<64, 32, 64>>(A, B,
            quantOption, alphaCol, alphaRow, C, m, n, k, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64:
        dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<64, 64, 128>, cutlass::gemm::GemmShape<32, 64, 64>>(A, B,
            quantOption, alphaCol, alphaRow, C, m, n, k, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64:
        dispatchGemmConfig<T, arch, cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>>(A, B,
            quantOption, alphaCol, alphaRow, C, m, n, k, gemmConfig, workspace, workspaceBytes, stream, occupancy);
        break;
    case tkc::CutlassTileConfig::Undefined:
        throw std::runtime_error("[TensorRT-LLM Error][int8][dispatch_gemm_to_cutlass] gemm config undefined.");
        break;
    case tkc::CutlassTileConfig::ChooseWithHeuristic:
        throw std::runtime_error(
            "[TensorRT-LLM Error][int8][dispatch_gemm_to_cutlass] gemm config should have already been set by "
            "heuristic.");
        break;
    default:
        throw std::runtime_error(
            "[TensorRT-LLM Error][int8][dispatch_gemm_to_cutlass] Config is invalid for int8 GEMM.");
        break;
    }
}

template <typename T>
CutlassInt8GemmRunner<T>::CutlassInt8GemmRunner()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    int device{-1};
    tk::check_cuda_error(cudaGetDevice(&device));
    mSm = tk::getSMVersion();
    tk::check_cuda_error(cudaDeviceGetAttribute(&mMultiProcessorCount, cudaDevAttrMultiProcessorCount, device));
}

template <typename T>
CutlassInt8GemmRunner<T>::~CutlassInt8GemmRunner()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template <typename T>
void CutlassInt8GemmRunner<T>::dispatchToArch(const int8_t* A, const int8_t* B, tk::QuantMode quantOption,
    const float* alphaCol, const float* alphaRow, T* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig,
    char* workspacePtr, const size_t workspaceBytes, cudaStream_t stream, int* occupancy)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (mSm >= 70 && mSm < 72)
    {
        dispatchGemmToCutlass<T, cutlass::arch::Sm70>(A, B, quantOption, alphaCol, alphaRow, C, m, n, k, workspacePtr,
            workspaceBytes, gemmConfig, stream, occupancy);
    }
    else if (mSm >= 72 && mSm < 75)
    {
        dispatchGemmToCutlass<T, cutlass::arch::Sm72>(A, B, quantOption, alphaCol, alphaRow, C, m, n, k, workspacePtr,
            workspaceBytes, gemmConfig, stream, occupancy);
    }
    else if (mSm >= 75 && mSm < 80)
    {
        dispatchGemmToCutlass<T, cutlass::arch::Sm75>(A, B, quantOption, alphaCol, alphaRow, C, m, n, k, workspacePtr,
            workspaceBytes, gemmConfig, stream, occupancy);
    }
    else if (mSm >= 80 && mSm <= 90)
    {
        dispatchGemmToCutlass<T, cutlass::arch::Sm80>(A, B, quantOption, alphaCol, alphaRow, C, m, n, k, workspacePtr,
            workspaceBytes, gemmConfig, stream, occupancy);
    }
    else
    {
        throw std::runtime_error(
            "[TensorRT-LLM Error][CutlassInt8GemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS int8 GEMM");
    }
}

template <typename T>
void CutlassInt8GemmRunner<T>::gemm(const int8_t* A, const int8_t* B, tk::QuantMode quantOption, const float* alphaCol,
    const float* alphaRow, void* C, int m, int n, int k, tkc::CutlassGemmConfig gemmConfig, char* workspacePtr,
    const size_t workspaceBytes, cudaStream_t stream)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    dispatchToArch(A, B, quantOption, alphaCol, alphaRow, reinterpret_cast<T*>(C), m, n, k, gemmConfig, workspacePtr,
        workspaceBytes, stream);
}

template <typename T>
std::vector<tkc::CutlassGemmConfig> CutlassInt8GemmRunner<T>::getConfigs() const
{
    static constexpr bool isWeightOnly = false;
    std::vector<tkc::CutlassGemmConfig> candidateConfigs
        = get_candidate_configs(mSm, isWeightOnly, mSm <= 70, /* SIMT configs */
            true, SPLIT_K_LIMIT);                             /* INT8 configs */
    return candidateConfigs;
}

template <typename T>
size_t CutlassInt8GemmRunner<T>::getWorkspaceSize(const int m, const int n, const int k)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    // These are the min tile sizes for each config, which would launch the maximum number of blocks
    const int maxGridM = cutlass::ceil_div(m, MIN_M_TILE);
    const int maxGridN = cutlass::ceil_div(m, MIN_N_TILE);
    // We need 4 bytes per block in the worst case. We launch SPLIT_K_LIMIT in z dim.
    return static_cast<size_t>(maxGridM * maxGridN * SPLIT_K_LIMIT * 4);
}

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
