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

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"
#include "cutlass_extensions/gemm/threadblock/default_mma.h"

#pragma GCC diagnostic pop

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>

namespace tensorrt_llm
{

// ============================= Variable batched Gemm things ===========================
template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
    typename WarpShape, int Stages>
void genericMoeGemmKernelLauncher(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
    int64_t* total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    cutlass_extensions::CutlassGemmConfig gemm_config, const int multi_processor_count, cudaStream_t stream,
    int* kernel_occupancy = nullptr)
{
#ifdef ENABLE_BF16
    static_assert(cutlass::platform::is_same<T, __nv_bfloat16>::value || cutlass::platform::is_same<T, half>::value
            || cutlass::platform::is_same<T, float>::value,
        "Specialized for bfloat16, half, float");
#else
    static_assert(cutlass::platform::is_same<T, half>::value || cutlass::platform::is_same<T, float>::value,
        "Specialized for half, float");
#endif

    static_assert(cutlass::platform::is_same<T, WeightType>::value
            || cutlass::platform::is_same<WeightType, uint8_t>::value
            || cutlass::platform::is_same<WeightType, cutlass::uint4b_t>::value,
        "");

    // The cutlass type for the input elements. This is needed to convert to cutlass::half_t if necessary.
    using ElementType_ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value, cutlass::half_t, T>::type;
#ifdef ENABLE_BF16
    using ElementType =
        typename cutlass::platform::conditional<cutlass::platform::is_same<ElementType_, __nv_bfloat16>::value,
            cutlass::bfloat16_t, ElementType_>::type;
#else
    using ElementType = ElementType_;
#endif

    using CutlassWeightType_ =
        typename cutlass::platform::conditional<cutlass::platform::is_same<WeightType, half>::value, cutlass::half_t,
            WeightType>::type;
#ifdef ENABLE_BF16
    using CutlassWeightType =
        typename cutlass::platform::conditional<cutlass::platform::is_same<CutlassWeightType_, __nv_bfloat16>::value,
            cutlass::bfloat16_t, CutlassWeightType_>::type;
#else
    using CutlassWeightType = CutlassWeightType_;
#endif

    // We need separate config for each architecture since we will target different tensorcore instructions. For float,
    // we do not target TCs.
    using MixedGemmArchTraits = cutlass::gemm::kernel::MixedGemmArchTraits<ElementType, CutlassWeightType, arch>;
    using ElementAccumulator = typename MixedGemmArchTraits::AccType;

    using EpilogueOp = typename tensorrt_llm::cutlass_extensions::Epilogue<ElementType,
        MixedGemmArchTraits::ElementsPerAccessC, ElementAccumulator, EpilogueTag>::Op;

    // Finally, set up the kernel.
    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemmGrouped<ElementType, cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone, MixedGemmArchTraits::ElementsPerAccessA, CutlassWeightType,
        typename MixedGemmArchTraits::LayoutB, cutlass::ComplexTransform::kNone,
        MixedGemmArchTraits::ElementsPerAccessB, ElementType, cutlass::layout::RowMajor, ElementAccumulator,
        typename MixedGemmArchTraits::OperatorClass, arch, ThreadblockShape, WarpShape,
        typename MixedGemmArchTraits::InstructionShape, EpilogueOp,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, Stages,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly, typename MixedGemmArchTraits::Operator>::GemmKernel;

    using GemmKernel = cutlass::gemm::kernel::MoeFCGemm<typename GemmKernel_::Mma, typename GemmKernel_::Epilogue,
        typename GemmKernel_::ThreadblockSwizzle,
        arch, // Ensure top level arch is used for dispatch
        GemmKernel_::kGroupScheduleMode>;

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    if (kernel_occupancy != nullptr)
    {
        *kernel_occupancy = tensorrt_llm::cutlass_extensions::compute_occupancy_for_kernel<GemmKernel>();
        return;
    }
    int occupancy = std::min(2, GemmGrouped::maximum_active_blocks());
    TLLM_CHECK_WITH_INFO(occupancy != 0, "GPU lacks the shared memory resources to run GroupedGEMM kernel");
    const int threadblock_count = multi_processor_count * occupancy;

    typename EpilogueOp::Params epilogue_op(
        ElementAccumulator(1.f), biases ? ElementAccumulator(1.f) : ElementAccumulator(0.f));

    const int group_size = gemm_k;
    typename GemmGrouped::Arguments args(num_experts, threadblock_count, group_size, epilogue_op,
        reinterpret_cast<const ElementType*>(A), reinterpret_cast<const CutlassWeightType*>(B),
        reinterpret_cast<const ElementType*>(weight_scales), reinterpret_cast<const ElementType*>(biases),
        reinterpret_cast<ElementType*>(C), total_rows_before_expert, gemm_n, gemm_k);

    GemmGrouped gemm;

    auto can_implement = gemm.can_implement(args);
    TLLM_CHECK_WITH_INFO(can_implement == cutlass::Status::kSuccess,
        "MoE FC kernel will fail for params. Error: " + std::string(cutlassGetStatusString(can_implement)));

    auto init_status = gemm.initialize(args);
    TLLM_CHECK_WITH_INFO(init_status == cutlass::Status::kSuccess,
        "Failed to initialize cutlass variable batched gemm. Error: "
            + std::string(cutlassGetStatusString(init_status)));

    auto run_status = gemm.run(stream);
    TLLM_CHECK_WITH_INFO(run_status == cutlass::Status::kSuccess,
        "Failed to run cutlass variable batched gemm. Error: " + std::string(cutlassGetStatusString(run_status)));
}

template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
    typename WarpShape, int Stages, typename Enable = void>
struct dispatch_stages
{
    static void dispatch(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
        int64_t* total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
        cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, cudaStream_t stream,
        int* occupancy = nullptr)
    {
        TLLM_THROW("Cutlass fpA_intB gemm. Not instantiated for arch %d with stages set to %d",
            arch::kMinComputeCapability, Stages);
    }
};

template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
    typename WarpShape>
struct dispatch_stages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2>
{
    static void dispatch(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
        int64_t* total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
        cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, cudaStream_t stream,
        int* occupancy = nullptr)
    {
        genericMoeGemmKernelLauncher<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2>(A, B,
            weight_scales, biases, C, total_rows_before_expert, num_rows, gemm_n, gemm_k, num_experts, gemm_config,
            multi_processor_count, stream, occupancy);
    }
};

template <typename T, typename WeightType, typename EpilogueTag, typename ThreadblockShape, typename WarpShape,
    int Stages>
struct dispatch_stages<T, WeightType, cutlass::arch::Sm80, EpilogueTag, ThreadblockShape, WarpShape, Stages,
    typename std::enable_if<(Stages > 2)>::type>
{
    static void dispatch(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
        int64_t* total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
        cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, cudaStream_t stream,
        int* occupancy = nullptr)
    {
        genericMoeGemmKernelLauncher<T, WeightType, cutlass::arch::Sm80, EpilogueTag, ThreadblockShape, WarpShape,
            Stages>(A, B, weight_scales, biases, C, total_rows_before_expert, num_rows, gemm_n, gemm_k, num_experts,
            gemm_config, multi_processor_count, stream, occupancy);
    }
};

template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
    typename WarpShape>
void dispatchGemmConfig(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
    int64_t* total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, cudaStream_t stream,
    int* occupancy = nullptr)
{
    switch (gemm_config.stages)
    {
    case 2:
        using DispatcherStages2 = dispatch_stages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2>;
        DispatcherStages2::dispatch(A, B, weight_scales, biases, C, total_rows_before_expert, num_rows, gemm_n, gemm_k,
            num_experts, gemm_config, multi_processor_count, stream, occupancy);
        break;
    case 3:
        using DispatcherStages3 = dispatch_stages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 3>;
        DispatcherStages3::dispatch(A, B, weight_scales, biases, C, total_rows_before_expert, num_rows, gemm_n, gemm_k,
            num_experts, gemm_config, multi_processor_count, stream, occupancy);
        break;
    case 4:
        using DispatcherStages4 = dispatch_stages<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 4>;
        DispatcherStages4::dispatch(A, B, weight_scales, biases, C, total_rows_before_expert, num_rows, gemm_n, gemm_k,
            num_experts, gemm_config, multi_processor_count, stream, occupancy);
        break;
    default: TLLM_THROW("dispatchGemmConfig does not support stages %d", gemm_config.stages); break;
    }
}

// This overload will handle tensorop gemms. It is disabled via SFINAE for fp32.
// This overload is only enabled when T == WeightType.
template <typename T, typename WeightType, typename arch, typename EpilogueTag,
    typename std::enable_if<!std::is_same<T, float>::value && std::is_same<T, WeightType>::value>::type* = nullptr>
void dispatchMoeGemmToCutlass(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
    int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    cutlass_extensions::CutlassGemmConfig gemm_config, int sm_version, int multi_processor_count, cudaStream_t stream,
    int* occupancy = nullptr)
{
    switch (gemm_config.tile_config)
    {
    case cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
        dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<32, 128, 64>,
            cutlass::gemm::GemmShape<32, 32, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows,
            gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
        dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<64, 128, 64>,
            cutlass::gemm::GemmShape<32, 64, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows,
            gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
        dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 64>,
            cutlass::gemm::GemmShape<64, 32, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows,
            gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::Undefined: TLLM_THROW("GEMM config undefined."); break;
    case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
        TLLM_THROW("GEMM config should have already been set by heuristic.");
        break;
    default: TLLM_THROW("Config is invalid for same type tensorop GEMM."); break;
    }
}

// Tensorop GEMM overload
// Overload for quantize MoE GEMMs. We disable some warp configs here since they will not be used and we can improve
// compile time
template <typename T, typename WeightType, typename arch, typename EpilogueTag,
    typename std::enable_if<!std::is_same<T, float>::value && !std::is_same<T, WeightType>::value>::type* = nullptr>
void dispatchMoeGemmToCutlass(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
    int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    cutlass_extensions::CutlassGemmConfig gemm_config, int sm_version, int multi_processor_count, cudaStream_t stream,
    int* occupancy = nullptr)
{
    switch (gemm_config.tile_config)
    {
    case cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
        dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<32, 128, 64>,
            cutlass::gemm::GemmShape<32, 32, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows,
            gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
        dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<64, 128, 64>,
            cutlass::gemm::GemmShape<64, 32, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows,
            gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
        dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 64>,
            cutlass::gemm::GemmShape<128, 32, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows,
            gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::Undefined: TLLM_THROW("GEMM config undefined."); break;
    case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
        TLLM_THROW("GEMM config should have already been set by heuristic.");
        break;
    default: TLLM_THROW("Config is invalid for mixed type tensorop GEMM."); break;
    }
}

// This overload will handle simt gemms. It is disabled via SFINAE for tensorop.
template <typename T, typename WeightType, typename arch, typename EpilogueTag,
    typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr>
void dispatchMoeGemmToCutlass(const T* A, const WeightType* B, const T* weight_scales, const T* biases, T* C,
    int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    cutlass_extensions::CutlassGemmConfig gemm_config, int sm_version, int multi_processor_count, cudaStream_t stream,
    int* occupancy = nullptr)
{
    switch (gemm_config.tile_config)
    {
    case cutlass_extensions::CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8:
        dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 8>,
            cutlass::gemm::GemmShape<64, 64, 8>>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows,
            gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, stream, occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::Undefined: TLLM_THROW("GEMM config undefined."); break;
    case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
        TLLM_THROW("GEMM config should have already been set by heuristic.");
        break;
    default: TLLM_THROW("Unsupported config for float MoE gemm."); break;
    }
}

template <typename T, typename WeightType>
std::vector<cutlass_extensions::CutlassGemmConfig> MoeGemmRunner<T, WeightType>::getConfigs()
{
    static constexpr bool is_weight_only = !std::is_same<T, WeightType>::value;
    static constexpr bool only_simt_configs = std::is_same<T, float>::value;
    std::vector<cutlass_extensions::CutlassGemmConfig> candidate_configs
        = kernels::cutlass_kernels::get_candidate_configs(sm_, is_weight_only, only_simt_configs);
    return candidate_configs;
}

template <typename T, typename WeightType>
MoeGemmRunner<T, WeightType>::MoeGemmRunner()
{
    int device{-1};
    tensorrt_llm::common::check_cuda_error(cudaGetDevice(&device));
    sm_ = tensorrt_llm::common::getSMVersion();
    tensorrt_llm::common::check_cuda_error(
        cudaDeviceGetAttribute(&multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
}

template <typename T, typename WeightType>
template <typename EpilogueTag>
void MoeGemmRunner<T, WeightType>::dispatchToArch<EpilogueTag>(const T* A, const WeightType* B, const T* weight_scales,
    const T* biases, T* C, int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
    int num_experts, cutlass_extensions::CutlassGemmConfig gemm_config, cudaStream_t stream, int* occupancy)
{
    if (sm_ >= 70 && sm_ < 75)
    {
        dispatchMoeGemmToCutlass<T, WeightType, cutlass::arch::Sm70, EpilogueTag>(A, B, weight_scales, biases, C,
            total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, sm_, multi_processor_count_,
            stream, occupancy);
    }
    else if (sm_ >= 75 && sm_ < 80)
    {
        dispatchMoeGemmToCutlass<T, WeightType, cutlass::arch::Sm75, EpilogueTag>(A, B, weight_scales, biases, C,
            total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, sm_, multi_processor_count_,
            stream, occupancy);
    }
    else if (sm_ >= 80 && sm_ < 90)
    {
        dispatchMoeGemmToCutlass<T, WeightType, cutlass::arch::Sm80, EpilogueTag>(A, B, weight_scales, biases, C,
            total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, sm_, multi_processor_count_,
            stream, occupancy);
    }
    else if (sm_ >= 90)
    {
        // TODO Update the arch to Sm90 once CUTLASS hopper specialisations are available
        dispatchMoeGemmToCutlass<T, WeightType, cutlass::arch::Sm80, EpilogueTag>(A, B, weight_scales, biases, C,
            total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, sm_, multi_processor_count_,
            stream, occupancy);
    }
    else
    {
        TLLM_THROW("Arch unsupported for MoE GEMM");
    }
}

template <typename T, typename WeightType>
template <typename EpilogueTag>
void MoeGemmRunner<T, WeightType>::runGemm<EpilogueTag>(const T* A, const WeightType* B, const T* weight_scales,
    const T* biases, T* C, int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
    int num_experts, cudaStream_t stream)
{
    auto chosen_conf = this->best_config_;
    if (!chosen_conf)
    {
        auto candidate_configs = getConfigs();
        std::vector<int> occupancies(candidate_configs.size());

        for (size_t ii = 0; ii < candidate_configs.size(); ++ii)
        {
            dispatchToArch<EpilogueTag>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n,
                gemm_k, num_experts, candidate_configs[ii], stream, &occupancies[ii]);
        }

        static constexpr int workspace_bytes = 0; // No workspace for MoE GEMMs.
        static constexpr int split_k_limit = 1;   // MoE GEMM does not support split-k.

        static constexpr bool is_weight_only = !std::is_same<T, WeightType>::value;
        chosen_conf = kernels::cutlass_kernels::estimate_best_config_from_occupancies(candidate_configs, occupancies,
            total_rows, gemm_n, gemm_k, num_experts, split_k_limit, workspace_bytes, multi_processor_count_,
            is_weight_only);
    }
    assert(chosen_conf);
    dispatchToArch<EpilogueTag>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k,
        num_experts, *chosen_conf, stream);
}

template <typename T, typename WeightType>
void MoeGemmRunner<T, WeightType>::moeGemmBiasAct(const T* A, const WeightType* B, const T* weight_scales,
    const T* biases, T* C, int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k,
    int num_experts, ActivationType activation_type, cudaStream_t stream)
{
    switch (activation_type)
    {
    case ActivationType::Relu:
        runGemm<cutlass_extensions::EpilogueOpDefaultReLU>(
            A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, stream);
        break;
    case ActivationType::Gelu:
        runGemm<cutlass_extensions::EpilogueOpDefaultFtGelu>(
            A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, stream);
        break;
    case ActivationType::Silu:
        runGemm<cutlass_extensions::EpilogueOpDefaultSilu>(
            A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, stream);
        break;
    case ActivationType::Identity:
        runGemm<cutlass_extensions::EpilogueOpDefault>(
            A, B, weight_scales, biases, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, stream);
        break;
    case ActivationType::InvalidType: TLLM_THROW("Activation type for fpA_intB must be valid."); break;
    default: TLLM_THROW("Invalid activation type."); break;
    }
}

template <typename T, typename WeightType>
void MoeGemmRunner<T, WeightType>::moeGemm(const T* A, const WeightType* B, const T* weight_scales, T* C,
    int64_t* total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
    cudaStream_t stream)
{
    runGemm<cutlass_extensions::EpilogueOpDefault>(
        A, B, weight_scales, nullptr, C, total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, stream);
}

} // namespace tensorrt_llm
