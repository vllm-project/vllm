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

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"

#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"
#include "cutlass_extensions/gemm/threadblock/default_mma.h"

#pragma GCC diagnostic pop

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "cutlass_heuristic.h"
#include "cutlass_type_conversion.h"

//#include "moe_gemm_kernels_template_sm90.h"
//#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_launcher_sm90.h"
#include "moe_gemm_kernels.h"
//#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_sm90_traits.h"
#include "fused_moe_gemm_launcher_sm80.h"
//#include <tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>
#include <sstream>

namespace tensorrt_llm
{
    namespace kernels::cutlass_kernels
    {

        // ============================= Variable batched Gemm things ===========================
        template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
                  typename WarpShape, int Stages>
        void genericMoeGemmKernelLauncher(T const *A, WeightType const *B, T const *weight_scales, T const *biases, T *C,
                                          int64_t *total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
                                          cutlass_extensions::CutlassGemmConfig gemm_config, int const multi_processor_count, bool use_fused_moe,
                                          cudaStream_t stream, int *kernel_occupancy = nullptr)
        {
#ifdef ENABLE_BF16
            static_assert(cutlass::platform::is_same<T, __nv_bfloat16>::value || cutlass::platform::is_same<T, half>::value || cutlass::platform::is_same<T, float>::value,
                          "Specialized for bfloat16, half, float");
#else
            static_assert(cutlass::platform::is_same<T, half>::value || cutlass::platform::is_same<T, float>::value,
                          "Specialized for half, float");
#endif

            static_assert(cutlass::platform::is_same<T, WeightType>::value || cutlass::platform::is_same<WeightType, uint8_t>::value || cutlass::platform::is_same<WeightType, cutlass::uint4b_t>::value,
                          "");

            static_assert(!cutlass::platform::is_same<arch, cutlass::arch::Sm90>::value,
                          "Sm90 architecture should use specialised kernels");

            // The cutlass type for the input elements. This is needed to convert to cutlass::half_t if necessary.
            using ElementType = typename TllmToCutlassTypeAdapter<T>::type;
            using CutlassWeightType = typename TllmToCutlassTypeAdapter<WeightType>::type;
            if (!use_fused_moe)
            {
                // We need separate config for each architecture since we will target different tensorcore instructions. For
                // float, we do not target TCs.
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
                TLLM_CHECK_WITH_INFO(occupancy > 0, "GPU lacks the shared memory resources to run GroupedGEMM kernel");
                int const threadblock_count = multi_processor_count * occupancy;

                typename EpilogueOp::Params epilogue_op(
                    ElementAccumulator(1.f), biases ? ElementAccumulator(1.f) : ElementAccumulator(0.f));

                int const group_size = gemm_k;
                typename GemmGrouped::Arguments args(num_experts, threadblock_count, group_size, epilogue_op,
                                                     reinterpret_cast<ElementType const *>(A), reinterpret_cast<CutlassWeightType const *>(B),
                                                     reinterpret_cast<ElementType const *>(weight_scales), reinterpret_cast<ElementType const *>(biases),
                                                     reinterpret_cast<ElementType *>(C), total_rows_before_expert, gemm_n, gemm_k);

                GemmGrouped gemm;

                auto can_implement = gemm.can_implement(args);
                TLLM_CHECK_WITH_INFO(can_implement == cutlass::Status::kSuccess,
                                     "MoE FC kernel will fail for params. Error: " + std::string(cutlassGetStatusString(can_implement)));

                auto init_status = gemm.initialize(args);
                TLLM_CHECK_WITH_INFO(init_status == cutlass::Status::kSuccess,
                                     "Failed to initialize cutlass variable batched gemm. Error: " + std::string(cutlassGetStatusString(init_status)));

                auto run_status = gemm.run(stream);
                TLLM_CHECK_WITH_INFO(run_status == cutlass::Status::kSuccess,
                                     "Failed to run cutlass variable batched gemm. Error: " + std::string(cutlassGetStatusString(run_status)));
            }
            else if constexpr (sizeof(ElementType) == 2 && sizeof(CutlassWeightType) == 2 && (std::is_same_v<EpilogueTag, cutlass_extensions::EpilogueOpDefaultSilu> || std::is_same_v<EpilogueTag, cutlass_extensions::EpilogueOpDefaultFtGelu>)) // use fused moe gemm
                                                                                                                                                                                                                                                   // kernel.. (only support
                                                                                                                                                                                                                                                   // fp16 or bf16)
            {
                sm80_generic_fused_moe_gemm_kernelLauncher<ElementType, CutlassWeightType, ThreadblockShape::kM,
                                                           ThreadblockShape::kN, ThreadblockShape::kK, Stages, EpilogueTag>(reinterpret_cast<ElementType const *>(A),
                                                                                                                            reinterpret_cast<CutlassWeightType const *>(B), reinterpret_cast<ElementType const *>(biases),
                                                                                                                            reinterpret_cast<ElementType *>(C), total_rows_before_expert, num_rows, gemm_n, gemm_k, num_experts,
                                                                                                                            multi_processor_count, stream, kernel_occupancy);
            }
        }

    } // namespace kernels::cutlass_kernels

    template <typename T, typename WeightType, typename Arch, typename EpilogueTag, typename ThreadblockShape,
              typename WarpShape, int Stages>
    static void dispatch(T const *A, WeightType const *B, T const *weight_scales, T const *biases, T *C,
                         int64_t *total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
                         cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, bool use_fused_moe,
                         cudaStream_t stream, int *occupancy = nullptr)
    {
        static_assert(!std::is_same_v<Arch, cutlass::arch::Sm90>, "Use TMA specialised functions for arch SM90");
        constexpr bool isFp8 = std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>;
        if constexpr ((Stages == 2 || Arch::kMinComputeCapability >= 80) && !isFp8)
        {
            kernels::cutlass_kernels::genericMoeGemmKernelLauncher<T, WeightType, Arch, EpilogueTag, ThreadblockShape,
                                                                   WarpShape, Stages>(A, B, weight_scales, biases, C, total_rows_before_expert, num_rows, gemm_n, gemm_k,
                                                                                      num_experts, gemm_config, multi_processor_count, use_fused_moe, stream, occupancy);
        }
        else
        {
            TLLM_THROW(
                "Cutlass gemm. Not instantiated for arch %d with stages set to %d", Arch::kMinComputeCapability, Stages);
        }
    }

    template <typename T, typename WeightType, typename arch, typename EpilogueTag, typename ThreadblockShape,
              typename WarpShape>
    void dispatchGemmConfig(T const *A, WeightType const *B, T const *weight_scales, T const *biases, T *C,
                            int64_t *total_rows_before_expert, int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
                            cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, bool use_fused_moe,
                            cudaStream_t stream, int *occupancy = nullptr)
    {
        switch (gemm_config.stages)
        {
        case 2:
            dispatch<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 2>(A, B, weight_scales, biases, C,
                                                                                       total_rows_before_expert, num_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
                                                                                       use_fused_moe, stream, occupancy);
            break;
        case 3:
            dispatch<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 3>(A, B, weight_scales, biases, C,
                                                                                       total_rows_before_expert, num_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
                                                                                       use_fused_moe, stream, occupancy);
            break;
        case 4:
            dispatch<T, WeightType, arch, EpilogueTag, ThreadblockShape, WarpShape, 4>(A, B, weight_scales, biases, C,
                                                                                       total_rows_before_expert, num_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count,
                                                                                       use_fused_moe, stream, occupancy);
            break;
        default:
            TLLM_THROW("dispatchGemmConfig does not support stages %d", gemm_config.stages);
            break;
        }
    }

    // This overload will handle tensorop gemms. It is disabled via SFINAE for fp32.
    // This overload is only enabled when T == WeightType.
    template <typename T, typename WeightType, typename arch, typename EpilogueTag,
              typename std::enable_if<!std::is_same<T, float>::value && std::is_same<T, WeightType>::value>::type * = nullptr>
    void dispatchMoeGemmToCutlass(T const *A, WeightType const *B, T const *weight_scales, T const *biases, T *C,
                                  int64_t *total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
                                  cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, bool use_fused_moe,
                                  cudaStream_t stream, int *occupancy = nullptr)
    {
        switch (gemm_config.tile_config)
        {
        case cutlass_extensions::CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64:
            TLLM_CHECK_WITH_INFO(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
            if constexpr (arch::kMinComputeCapability >= 75)
            {
                dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<16, 128, 64>,
                                   cutlass::gemm::GemmShape<16, 32, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                                         total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, use_fused_moe, stream,
                                                                         occupancy);
            }
            break;
        case cutlass_extensions::CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64:
            TLLM_CHECK_WITH_INFO(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
            if constexpr (arch::kMinComputeCapability >= 75)
            {
                dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<16, 256, 64>,
                                   cutlass::gemm::GemmShape<16, 64, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                                         total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, use_fused_moe, stream,
                                                                         occupancy);
            }
            break;
        case cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
            dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<32, 128, 64>,
                               cutlass::gemm::GemmShape<32, 32, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows,
                                                                     gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, use_fused_moe, stream, occupancy);
            break;
        case cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
            dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<64, 128, 64>,
                               cutlass::gemm::GemmShape<32, 64, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows,
                                                                     gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, use_fused_moe, stream, occupancy);
            break;
        case cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
            dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 64>,
                               cutlass::gemm::GemmShape<64, 32, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows,
                                                                     gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, use_fused_moe, stream, occupancy);
            break;
        case cutlass_extensions::CutlassTileConfig::Undefined:
            TLLM_THROW("GEMM config undefined.");
            break;
        case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
            TLLM_THROW("GEMM config should have already been set by heuristic.");
            break;
        default:
            TLLM_THROW("Config is invalid for same type tensorop GEMM.");
            break;
        }
    }

    // Tensorop GEMM overload
    // Overload for quantize MoE GEMMs. We disable some warp configs here since they will not be used and we can improve
    // compile time
    template <typename T, typename WeightType, typename arch, typename EpilogueTag,
              typename std::enable_if<!std::is_same<T, float>::value && !std::is_same<T, WeightType>::value>::type * = nullptr>
    void dispatchMoeGemmToCutlass(T const *A, WeightType const *B, T const *weight_scales, T const *biases, T *C,
                                  int64_t *total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
                                  cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, bool use_fused_moe,
                                  cudaStream_t stream, int *occupancy = nullptr)
    {
        switch (gemm_config.tile_config)
        {
        case cutlass_extensions::CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64:
            TLLM_CHECK_WITH_INFO(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
            if constexpr (arch::kMinComputeCapability >= 75)
            {
                dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<16, 128, 64>,
                                   cutlass::gemm::GemmShape<16, 32, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                                         total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, use_fused_moe, stream,
                                                                         occupancy);
            }
            break;
        case cutlass_extensions::CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64:
            TLLM_CHECK_WITH_INFO(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
            if constexpr (arch::kMinComputeCapability >= 75)
            {
                dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<16, 256, 64>,
                                   cutlass::gemm::GemmShape<16, 64, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                                         total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, use_fused_moe, stream,
                                                                         occupancy);
            }
            break;
        case cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
            dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<32, 128, 64>,
                               cutlass::gemm::GemmShape<32, 32, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows,
                                                                     gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, use_fused_moe, stream, occupancy);
            break;
        case cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
            dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<64, 128, 64>,
                               cutlass::gemm::GemmShape<64, 32, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows,
                                                                     gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, use_fused_moe, stream, occupancy);
            break;
        case cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
            dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 64>,
                               cutlass::gemm::GemmShape<128, 32, 64>>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows,
                                                                      gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, use_fused_moe, stream, occupancy);
            break;
        case cutlass_extensions::CutlassTileConfig::Undefined:
            TLLM_THROW("GEMM config undefined.");
            break;
        case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
            TLLM_THROW("GEMM config should have already been set by heuristic.");
            break;
        default:
            TLLM_THROW("Config is invalid for mixed type tensorop GEMM.");
            break;
        }
    }

    // This overload will handle simt gemms. It is disabled via SFINAE for tensorop.
    template <typename T, typename WeightType, typename arch, typename EpilogueTag,
              typename std::enable_if<std::is_same<T, float>::value>::type * = nullptr>
    void dispatchMoeGemmToCutlass(T const *A, WeightType const *B, T const *weight_scales, T const *biases, T *C,
                                  int64_t *total_rows_before_expert, int64_t total_rows, int64_t gemm_n, int64_t gemm_k, int num_experts,
                                  cutlass_extensions::CutlassGemmConfig gemm_config, int multi_processor_count, bool use_fused_moe,
                                  cudaStream_t stream, int *occupancy = nullptr)
    {
        switch (gemm_config.tile_config)
        {
        case cutlass_extensions::CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8:
            dispatchGemmConfig<T, WeightType, arch, EpilogueTag, cutlass::gemm::GemmShape<128, 128, 8>,
                               cutlass::gemm::GemmShape<64, 64, 8>>(A, B, weight_scales, biases, C, total_rows_before_expert, total_rows,
                                                                    gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count, use_fused_moe, stream, occupancy);
            break;
        case cutlass_extensions::CutlassTileConfig::Undefined:
            TLLM_THROW("GEMM config undefined.");
            break;
        case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
            TLLM_THROW("GEMM config should have already been set by heuristic.");
            break;
        default:
            TLLM_THROW("Unsupported config for float MoE gemm.");
            break;
        }
    }

    template <typename T, typename WeightType>
    std::vector<cutlass_extensions::CutlassGemmConfig> MoeGemmRunner<T, WeightType>::getConfigs() const
    {
        return getConfigs(sm_);
    }

    template <typename T, typename WeightType>
    std::vector<cutlass_extensions::CutlassGemmConfig> MoeGemmRunner<T, WeightType>::getConfigs(int sm)
    {
        std::vector<cutlass_extensions::CutlassGemmConfig> candidate_configs = getAmpereConfigs(sm);
        return candidate_configs;
    }

    template <typename T, typename WeightType>
    std::vector<cutlass_extensions::CutlassGemmConfig> MoeGemmRunner<T, WeightType>::getAmpereConfigs(int sm)
    {
        using tensorrt_llm::cutlass_extensions::CutlassGemmConfig;
        static constexpr auto weight_only_flag = std::is_same<T, WeightType>::value ? CutlassGemmConfig::NONE : CutlassGemmConfig::WEIGHT_ONLY;
        static constexpr auto simt_only_flag = std::is_same<T, float>::value ? CutlassGemmConfig::SIMT_ONLY : CutlassGemmConfig::NONE;
        int const max_split_k = 2;
        int const grouped_gemm_flag = CutlassGemmConfig::GROUPED_GEMM;
        int const enable_hopper = CutlassGemmConfig::NONE;

        auto config_type_param = static_cast<CutlassGemmConfig::CandidateConfigTypeParam>(
            weight_only_flag | simt_only_flag | grouped_gemm_flag | enable_hopper);

        /*
        if (!kernels::cutlass_kernels::isValidAmpereMOESpecialisation<T, WeightType>())
        {
            return {};
        }
        */

        std::vector<cutlass_extensions::CutlassGemmConfig> ampere_configs = kernels::cutlass_kernels::get_candidate_configs(sm, max_split_k, config_type_param);
        return ampere_configs;
    }

    template <typename T, typename WeightType>
    bool MoeGemmRunner<T, WeightType>::isHopperSpecialised() const
    {
        bool config_is_sm90 = best_config_ && best_config_->is_sm90;
        return supportsHopperSpecialisation() && config_is_sm90;
    }

    template <typename T, typename WeightType>
    bool MoeGemmRunner<T, WeightType>::supportsHopperSpecialisation() const
    {
        // return sm_ == 90 && kernels::cutlass_kernels::isValidHopperMOESpecialisation<T, WeightType>();
        return false;
    }

    template <typename T, typename WeightType>
    int MoeGemmRunner<T, WeightType>::getSM() const
    {
        return this->sm_;
    }

    // currently support sm80 bf16/fp16 gate ativation, only set predication tensor for m direction
    template <typename T, typename WeightType>
    bool MoeGemmRunner<T, WeightType>::isFusedGatedActivation(bool is_gated_activation, int gemm_n, int gemm_k) const
    {
        return is_gated_activation && std::is_same_v<T, WeightType> && (!std::is_same_v<T, float>) && (!this->isHopperSpecialised()) && this->getSM() >= 80 && (gemm_k % 32 == 0) && (gemm_n % 32 == 0);
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
    void MoeGemmRunner<T, WeightType>::dispatchToArch<EpilogueTag>(T const *A, WeightType const *B, T const *weight_scales,
                                                                   T const *biases, T *C, int64_t *total_rows_before_expert, HopperGroupedGemmInput hopper_input, int64_t total_rows,
                                                                   int64_t gemm_n, int64_t gemm_k, int num_experts, cutlass_extensions::CutlassGemmConfig gemm_config,
                                                                   bool use_fused_moe, cudaStream_t stream, int *occupancy)
    {
        if (sm_ >= 80 && sm_ < 90)
        {
            dispatchMoeGemmToCutlass<T, WeightType, cutlass::arch::Sm80, EpilogueTag>(A, B, weight_scales, biases, C,
                                                                                      total_rows_before_expert, total_rows, gemm_n, gemm_k, num_experts, gemm_config, multi_processor_count_,
                                                                                      use_fused_moe, stream, occupancy);
        }
        else
        {
            TLLM_THROW("Arch unsupported for MoE GEMM");
        }
    }

    template <typename T, typename WeightType>
    size_t MoeGemmRunner<T, WeightType>::calcMaxWorkspaceSize(int num_experts) const
    {
        return 0;
    }

    template <typename T, typename WeightType>
    template <typename EpilogueTag>
    void MoeGemmRunner<T, WeightType>::runGemm<EpilogueTag>(T const *A, WeightType const *B, T const *weight_scales,
                                                            T const *biases, T *C, int64_t *total_rows_before_expert, HopperGroupedGemmInput hopper_input, int64_t total_rows,
                                                            int64_t gemm_n, int64_t gemm_k, int num_experts, bool use_fused_moe, cudaStream_t stream)
    {
        TLLM_CHECK_WITH_INFO(this->best_config_, "No MOE GEMM config set at runtime");
        auto chosen_conf = *this->best_config_;
        dispatchToArch<EpilogueTag>(A, B, weight_scales, biases, C, total_rows_before_expert, hopper_input, total_rows,
                                    gemm_n, gemm_k, num_experts, chosen_conf, use_fused_moe, stream);
    }

    template <typename T, typename WeightType>
    void MoeGemmRunner<T, WeightType>::moeGemmBiasAct(T const *A, WeightType const *B, T const *weight_scales,
                                                      T const *biases, T *C, int64_t *total_rows_before_expert, HopperGroupedGemmInput hopper_input, int64_t total_rows,
                                                      int64_t gemm_n, int64_t gemm_k, int num_experts, ActivationType activation_type, bool use_fused_moe,
                                                      cudaStream_t stream)
    {
        switch (activation_type)
        {
        case ActivationType::Relu:
            runGemm<cutlass_extensions::EpilogueOpDefaultReLU>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                               hopper_input, total_rows, gemm_n, gemm_k, num_experts, use_fused_moe, stream);
            break;
        case ActivationType::Gelu:
            runGemm<cutlass_extensions::EpilogueOpDefaultFtGelu>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                                 hopper_input, total_rows, gemm_n, gemm_k, num_experts, use_fused_moe, stream);
            break;
        case ActivationType::Silu:
            runGemm<cutlass_extensions::EpilogueOpDefaultSilu>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                               hopper_input, total_rows, gemm_n, gemm_k, num_experts, use_fused_moe, stream);
            break;
        case ActivationType::Identity:
            runGemm<cutlass_extensions::EpilogueOpDefault>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                           hopper_input, total_rows, gemm_n, gemm_k, num_experts, use_fused_moe, stream);
            break;
        case ActivationType::Swiglu:
            runGemm<cutlass_extensions::EpilogueOpDefaultSilu>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                               hopper_input, total_rows, gemm_n, gemm_k, num_experts, use_fused_moe, stream);
            break;
        case ActivationType::Geglu:
            runGemm<cutlass_extensions::EpilogueOpDefaultFtGelu>(A, B, weight_scales, biases, C, total_rows_before_expert,
                                                                 hopper_input, total_rows, gemm_n, gemm_k, num_experts, use_fused_moe, stream);
            break;
        case ActivationType::InvalidType:
            TLLM_THROW("Activation type for fpA_intB must be valid.");
            break;
        default:
            TLLM_THROW("Invalid activation type.");
            break;
        }
    }

    template <typename T, typename WeightType>
    void MoeGemmRunner<T, WeightType>::moeGemm(T const *A, WeightType const *B, T const *weight_scales, T *C,
                                               int64_t *total_rows_before_expert, HopperGroupedGemmInput hopper_input, int64_t total_rows, int64_t gemm_n,
                                               int64_t gemm_k, int num_experts, bool use_fused_moe, cudaStream_t stream)
    {
        runGemm<cutlass_extensions::EpilogueOpDefault>(A, B, weight_scales, nullptr, C, total_rows_before_expert,
                                                       hopper_input, total_rows, gemm_n, gemm_k, num_experts, use_fused_moe, stream);
    }

} // namespace tensorrt_llm
