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

#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/gemm/device/gemm_universal_base_compat.h"

#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/fpA_intB_gemm.h"
#include "cutlass_extensions/gemm/threadblock/default_mma.h"
#include "cutlass_extensions/gemm_configs.h"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif // #ifndef _WIN32

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

#include "cutlass_heuristic.h"
#include "cutlass_type_conversion.h"
#include "fpA_intB_gemm.h"
//#include "fpA_intB_gemm_template_sm90.h"

namespace tk = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{
    namespace kernels
    {
        namespace cutlass_kernels
        {

            template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
                      typename arch, cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag, typename ThreadblockShape,
                      typename WarpShape, int Stages>
            void generic_mixed_gemm_kernelLauncher(ActivationType const *A, WeightType const *B, ScaleZeroType const *weight_scales,
                                                   ScaleZeroType const *weight_zero_points, BiasType const *biases, float const alpha, OutputType *C, int m, int n,
                                                   int k, int const group_size, tkc::CutlassGemmConfig gemm_config, char *workspace, size_t workspace_bytes,
                                                   cudaStream_t stream, int *occupancy = nullptr)
            {
                TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

#ifdef ENABLE_BF16
                static_assert(
#ifdef ENABLE_FP8
                    cutlass::platform::is_same<ActivationType, __nv_fp8_e4m3>::value ||
#endif
                        cutlass::platform::is_same<ActivationType, __nv_bfloat16>::value || cutlass::platform::is_same<ActivationType, half>::value || cutlass::platform::is_same<ActivationType, float>::value,
                    "Specialized for bfloat16, half, float");
#else
                static_assert(cutlass::platform::is_same<ActivationType, half>::value || cutlass::platform::is_same<ActivationType, float>::value,
                              "Specialized for half, float");
#endif

                static_assert(cutlass::platform::is_same<ActivationType, WeightType>::value || cutlass::platform::is_same<WeightType, uint8_t>::value || cutlass::platform::is_same<WeightType, cutlass::uint4b_t>::value,
                              "");

                // The cutlass type for the input elements. This is needed to convert to cutlass::half_t if necessary.
                using CutlassActivationType = typename TllmToCutlassTypeAdapter<ActivationType>::type;
                using CutlassWeightType = typename TllmToCutlassTypeAdapter<WeightType>::type;
                using CutlassScaleZeroType = typename TllmToCutlassTypeAdapter<ScaleZeroType>::type;
                using CutlassBiasType = typename TllmToCutlassTypeAdapter<BiasType>::type;
                using CutlassOutputType = typename TllmToCutlassTypeAdapter<OutputType>::type;

                // We need separate config for each architecture since we will target different tensorcore instructions. For float,
                // we do not target TCs.
                using MixedGemmArchTraits = cutlass::gemm::kernel::MixedGemmArchTraits<CutlassActivationType, CutlassWeightType, arch>;
                using ElementAccumulator = typename MixedGemmArchTraits::AccType;

                constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<CutlassOutputType>::value;
                using EpilogueOp =
                    typename tkc::Epilogue<CutlassOutputType, ElementsPerAccessC, ElementAccumulator, EpilogueTag>::Op;

                using Operator = typename MixedGemmArchTraits::Operator;
                using TaggedOperator = typename cutlass::arch::TagOperator<Operator, QuantOp>::TaggedOperator;

                using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<CutlassActivationType, cutlass::layout::RowMajor,
                                                                                MixedGemmArchTraits::ElementsPerAccessA, CutlassWeightType, typename MixedGemmArchTraits::LayoutB,
                                                                                MixedGemmArchTraits::ElementsPerAccessB, CutlassOutputType, cutlass::layout::RowMajor, ElementAccumulator,
                                                                                cutlass::arch::OpClassTensorOp, arch, ThreadblockShape, WarpShape,
                                                                                typename MixedGemmArchTraits::InstructionShape, EpilogueOp,
                                                                                typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages, true,
                                                                                TaggedOperator>::GemmKernel;

                using GemmKernel = cutlass::gemm::kernel::GemmFpAIntB<typename GemmKernel_::Mma, typename GemmKernel_::Epilogue,
                                                                      typename GemmKernel_::ThreadblockSwizzle,
                                                                      arch, // Ensure top level arch is used for dispatch
                                                                      GemmKernel_::kSplitKSerial>;

                if (occupancy != nullptr)
                {
                    *occupancy = tensorrt_llm::cutlass_extensions::compute_occupancy_for_kernel<GemmKernel>();
                    return;
                }

                using Gemm = cutlass::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

                int const ldb = cutlass::platform::is_same<cutlass::layout::RowMajor, typename MixedGemmArchTraits::LayoutB>::value
                                    ? n
                                    : k * GemmKernel::kInterleave;

                if (weight_scales == nullptr)
                {
                    throw std::runtime_error("Weight scales must always be set to a non-null value.");
                }

                if constexpr (cutlass::isFinegrained(QuantOp))
                {
                    /*
                    if constexpr (cutlass::platform::is_same<CutlassActivationType, float_e4m3_t>::value)
                    {
                        if (group_size != 128)
                        {
                            throw std::runtime_error("Only group size 128 supported for fine grained W4A(fp)8 kernels.");
                        }
                    }
                    */
                    if (group_size != 64 && group_size != 128)
                    {
                        throw std::runtime_error("Only group size 64 and 128 supported for fine grained kernels.");
                    }

                    if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY)
                    {
                        if (weight_zero_points != nullptr)
                        {
                            throw std::runtime_error("Weight zero pointer must be a nullptr for scale only fine grained");
                        }
                    }
                    else if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS)
                    {
                        if (weight_zero_points == nullptr)
                        {
                            throw std::runtime_error("Weight zero pointer must be valid for scale and bias fine grained");
                        }
                    }
                }
                else
                {
                    if (group_size != k)
                    {
                        throw std::runtime_error("Invalid group size for per column scaling kernels.");
                    }

                    if (weight_zero_points != nullptr)
                    {
                        throw std::runtime_error("Weight zero-points must be null when running per column scaling");
                    }
                }

                int const ld_scale_zero = cutlass::isFinegrained(QuantOp) ? n : 0;
                ElementAccumulator output_op_beta = (biases == nullptr) ? ElementAccumulator(0.f) : ElementAccumulator(1.f);
                typename Gemm::Arguments args({m, n, k}, group_size,
                                              {reinterpret_cast<CutlassActivationType *>(const_cast<ActivationType *>(A)), k},
                                              {reinterpret_cast<CutlassWeightType *>(const_cast<WeightType *>(B)), ldb},
                                              {reinterpret_cast<CutlassScaleZeroType *>(const_cast<ScaleZeroType *>(weight_scales)), ld_scale_zero},
                                              {reinterpret_cast<CutlassScaleZeroType *>(const_cast<ScaleZeroType *>(weight_zero_points)), ld_scale_zero},
                                              {reinterpret_cast<CutlassBiasType *>(const_cast<BiasType *>(biases)), 0},
                                              {reinterpret_cast<CutlassOutputType *>(C), n}, gemm_config.split_k_factor,
                                              {ElementAccumulator(alpha), output_op_beta});

                // This assertion is enabled because because for the column interleaved layout, K MUST be a multiple of
                // threadblockK. The reason for this is that the default pitchlinear iterators are used to handle walking over the
                // interleaved matrix. The way masking in handled in these do not map to the interleaved layout. We need to write
                // our own predicated iterator in order to relax this limitation.
                if (GemmKernel::kInterleave > 1 && ((k % MixedGemmArchTraits::ThreadblockK) || ((k / gemm_config.split_k_factor) % MixedGemmArchTraits::ThreadblockK)))
                {
                    throw std::runtime_error("Temp assertion: k must be multiple of threadblockK");
                }

                Gemm gemm;
                if (gemm.get_workspace_size(args) > workspace_bytes)
                {
                    TLLM_LOG_WARNING(
                        "Requested split-k but workspace size insufficient. Falling back to non-split-k implementation.");
                    // If requested split-k factor will require more workspace bytes, revert to standard gemm.
                    args.batch_count = 1;
                }

                auto can_implement = gemm.can_implement(args);
                if (can_implement != cutlass::Status::kSuccess)
                {
                    std::string err_msg = "fpA_intB cutlass kernel will fail for params. Error: " + std::string(cutlassGetStatusString(can_implement));
                    throw std::runtime_error("[TensorRT-LLm Error][fpA_intB Runner] " + err_msg);
                }

                auto init_status = gemm.initialize(args, workspace, stream);
                if (init_status != cutlass::Status::kSuccess)
                {
                    std::string err_msg = "Failed to initialize cutlass fpA_intB gemm. Error: " + std::string(cutlassGetStatusString(init_status));
                    throw std::runtime_error("[TensorRT-LLm Error][fpA_intB Runner] " + err_msg);
                }

                auto run_status = gemm.run(stream);
                if (run_status != cutlass::Status::kSuccess)
                {
                    std::string err_msg = "Failed to run cutlass fpA_intB gemm. Error: " + std::string(cutlassGetStatusString(run_status));
                    throw std::runtime_error("[TensorRT-LLm Error][fpA_intB Runner] " + err_msg);
                }
            }

            // This filters out invalid template combinations that we DON'T want instantiated in CUTLASS. For example,
            // instantiating SM=75, Stages=3 is invalid so we would need to filter that out. Fine grained
            // quanitzation is only supported on Ampere+ GPUs. FP8 GEMM is only supported on Ada+ GPUs.
            template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
                      typename arch, cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag, typename ThreadblockShape,
                      typename WarpShape, int Stages>
            void filter_and_run_mixed_gemm(ActivationType const *A, WeightType const *B, ScaleZeroType const *weight_scales,
                                           ScaleZeroType const *weight_zero_points, BiasType const *biases, float const alpha, OutputType *C, int m, int n,
                                           int k, int const group_size, tkc::CutlassGemmConfig gemm_config, char *workspace, size_t workspace_bytes,
                                           cudaStream_t stream, int *occupancy = nullptr)
            {

                TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
                if constexpr (Stages > 2 && arch::kMinComputeCapability < 80)
                {
                    // Multistage only supported on Ampere
                    std::string err_msg = "Cutlass fpA_intB gemm not supported for arch " + std::to_string(arch::kMinComputeCapability) + " with stages set to " + std::to_string(Stages);
                    throw std::runtime_error("[TensorRT-LLm Error][filter_and_run_mixed_gemm] " + err_msg);
                }
                else if constexpr (Stages == 2 && arch::kMinComputeCapability >= 89)
                {
                    // Multistage only supported on Ampere
                    std::string err_msg = "Cutlass fpA_intB gemm not supported for arch " + std::to_string(arch::kMinComputeCapability) + " with stages set to " + std::to_string(Stages);
                    throw std::runtime_error("[TensorRT-LLm Error][filter_and_run_mixed_gemm] " + err_msg);
                }
                else if constexpr (cutlass::platform::is_same<ActivationType, __nv_fp8_e4m3>::value && arch::kMinComputeCapability < 89)
                {
                    // FP8 activation type only supported on Ada+ GPUs
                    std::string err_msg = "Cutlass fpA_intB gemm not supported for arch " + std::to_string(arch::kMinComputeCapability) + " with activation type set to FP8";
                    throw std::runtime_error("[TensorRT-LLm Error][filter_and_run_mixed_gemm] " + err_msg);
                }
                else
                {
                    generic_mixed_gemm_kernelLauncher<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, arch,
                                                      QuantOp, EpilogueTag, ThreadblockShape, WarpShape, Stages>(A, B, weight_scales, weight_zero_points, biases,
                                                                                                                 alpha, C, m, n, k, group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
                }
            }

            template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
                      typename arch, cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag, typename ThreadblockShape,
                      typename WarpShape>
            void dispatch_gemm_config(ActivationType const *A, WeightType const *B, ScaleZeroType const *weight_scales,
                                      ScaleZeroType const *weight_zero_points, BiasType const *biases, float const alpha, OutputType *C, int m, int n,
                                      int k, int const group_size, tkc::CutlassGemmConfig gemm_config, char *workspace, size_t workspace_bytes,
                                      cudaStream_t stream, int *occupancy = nullptr)
            {

                TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
                switch (gemm_config.stages)
                {
                case 2:
                    filter_and_run_mixed_gemm<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, arch, QuantOp,
                                              EpilogueTag, ThreadblockShape, WarpShape, 2>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m,
                                                                                           n, k, group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
                    break;
                case 3:
                    filter_and_run_mixed_gemm<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, arch, QuantOp,
                                              EpilogueTag, ThreadblockShape, WarpShape, 3>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m,
                                                                                           n, k, group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
                    break;
                case 4:
                    filter_and_run_mixed_gemm<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, arch, QuantOp,
                                              EpilogueTag, ThreadblockShape, WarpShape, 4>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m,
                                                                                           n, k, group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
                    break;
                default:
                    std::string err_msg = "dispatch_gemm_config does not support stages " + std::to_string(gemm_config.stages);
                    throw std::runtime_error("[TensorRT-LLm Error][dispatch_gemm_config] " + err_msg);
                    break;
                }
            }

            template <typename T>
            constexpr bool is_fp8()
            {
                return std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>;
            }

            template <typename ActivationType, typename WeightType, typename ScaleZeroType, typename BiasType, typename OutputType,
                      typename arch, cutlass::WeightOnlyQuantOp QuantOp, typename EpilogueTag>
            void dispatch_gemm_to_cutlass(ActivationType const *A, WeightType const *B, ScaleZeroType const *weight_scales,
                                          ScaleZeroType const *weight_zero_points, BiasType const *biases, float const alpha, OutputType *C, int m, int n,
                                          int k, int const group_size, char *workspace, size_t workspace_bytes, tkc::CutlassGemmConfig gemm_config,
                                          cudaStream_t stream, int *occupancy = nullptr)
            {

                TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

                // Don't instantiate configs that are not supported pre-hopper. Produce a sensible error instead.
                constexpr bool any_is_fp8 = is_fp8<ActivationType>() || is_fp8<WeightType>() || is_fp8<ScaleZeroType>() || is_fp8<BiasType>() || is_fp8<OutputType>();

                constexpr bool all_types_are_the_same = std::is_same_v<ActivationType, ScaleZeroType> && std::is_same_v<ActivationType, BiasType> && std::is_same_v<ActivationType, OutputType>;

                constexpr bool is_valid_pre_hopper = (all_types_are_the_same && !any_is_fp8) || (arch::kMinComputeCapability >= 89);

                if constexpr (is_valid_pre_hopper)
                {
                    // Note that SIMT configs are omitted here since they are not supported for fpA_intB.
                    // We also only instantiate configs here where threadblockShapeM == warpShapeM since those usually perform the
                    // best for mixed type gemms.
                    constexpr int tile_shape_k = 128 * 8 / cutlass::sizeof_bits<ActivationType>::value;
                    switch (gemm_config.tile_config)
                    {
                    case tkc::CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64:
                        TLLM_CHECK_WITH_INFO(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
                        if constexpr (arch::kMinComputeCapability >= 75)
                        {
                            dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, arch, QuantOp,
                                                 EpilogueTag, cutlass::gemm::GemmShape<16, 128, tile_shape_k>,
                                                 cutlass::gemm::GemmShape<16, 32, tile_shape_k>>(A, B, weight_scales, weight_zero_points, biases,
                                                                                                 alpha, C, m, n, k, group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
                        }
                        break;
                    case tkc::CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64:
                        TLLM_CHECK_WITH_INFO(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
                        if constexpr (arch::kMinComputeCapability >= 75)
                        {
                            dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, arch, QuantOp,
                                                 EpilogueTag, cutlass::gemm::GemmShape<16, 256, tile_shape_k>,
                                                 cutlass::gemm::GemmShape<16, 64, tile_shape_k>>(A, B, weight_scales, weight_zero_points, biases,
                                                                                                 alpha, C, m, n, k, group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
                        }
                        break;
                    case tkc::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
                        dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, arch, QuantOp,
                                             EpilogueTag, cutlass::gemm::GemmShape<32, 128, tile_shape_k>,
                                             cutlass::gemm::GemmShape<32, 32, tile_shape_k>>(A, B, weight_scales, weight_zero_points, biases, alpha,
                                                                                             C, m, n, k, group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
                        break;
                    case tkc::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
                        dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, arch, QuantOp,
                                             EpilogueTag, cutlass::gemm::GemmShape<64, 128, tile_shape_k>,
                                             cutlass::gemm::GemmShape<64, 32, tile_shape_k>>(A, B, weight_scales, weight_zero_points, biases, alpha,
                                                                                             C, m, n, k, group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
                        break;
                    case tkc::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
                        TLLM_CHECK_WITH_INFO(arch::kMinComputeCapability >= 75, "Invalid config on Volta");
                        if constexpr (arch::kMinComputeCapability >= 75)
                        {
                            dispatch_gemm_config<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, arch, QuantOp,
                                                 EpilogueTag, cutlass::gemm::GemmShape<128, 128, tile_shape_k>,
                                                 cutlass::gemm::GemmShape<128, 32, tile_shape_k>>(A, B, weight_scales, weight_zero_points, biases,
                                                                                                  alpha, C, m, n, k, group_size, gemm_config, workspace, workspace_bytes, stream, occupancy);
                        }
                        break;
                    case tkc::CutlassTileConfig::Undefined:
                        throw std::runtime_error("[TensorRT-LLm Error][fpA_intB][dispatch_gemm_to_cutlass] gemm config undefined.");
                        break;
                    case tkc::CutlassTileConfig::ChooseWithHeuristic:
                        throw std::runtime_error(
                            "[TensorRT-LLm Error][fpA_intB][dispatch_gemm_to_cutlass] gemm config should have already been set by "
                            "heuristic.");
                        break;
                    default:
                        throw std::runtime_error(
                            "[TensorRT-LLm Error][fpA_intB][dispatch_gemm_to_cutlass] Config is invalid for mixed type GEMM.");
                        break;
                    }
                }
                else
                {
                    // This is not a limitation in CUTLASS. We just do not need to support this case.
                    std::string err_msg = "The activation type must equal the scale, bias and output types on Ampere and earlier.";
                    throw std::runtime_error("[TensorRT-LLm Error][dispatch_gemm_to_cutlass] " + err_msg);
                }
            }

            template <typename ActivationType, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp, typename ScaleZeroType,
                      typename BiasType, typename OutputType>
            CutlassFpAIntBGemmRunner<ActivationType, WeightType, QuantOp, ScaleZeroType, BiasType,
                                     OutputType>::CutlassFpAIntBGemmRunner()
            {
                TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
                int device{-1};
                tk::check_cuda_error(cudaGetDevice(&device));
                sm_ = tk::getSMVersion();
                tk::check_cuda_error(cudaDeviceGetAttribute(&multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
            }

            template <typename ActivationType, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp, typename ScaleZeroType,
                      typename BiasType, typename OutputType>
            CutlassFpAIntBGemmRunner<ActivationType, WeightType, QuantOp, ScaleZeroType, BiasType,
                                     OutputType>::~CutlassFpAIntBGemmRunner()
            {
                TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
            }

            template <typename ActivationType, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp, typename ScaleZeroType,
                      typename BiasType, typename OutputType>
            template <typename EpilogueTag>
            void CutlassFpAIntBGemmRunner<ActivationType, WeightType, QuantOp, ScaleZeroType, BiasType,
                                          OutputType>::dispatch_to_arch<EpilogueTag>(ActivationType const *A, WeightType const *B,
                                                                                     ScaleZeroType const *weight_scales, ScaleZeroType const *weight_zero_points, BiasType const *biases,
                                                                                     float const alpha, OutputType *C, int m, int n, int k, int const group_size, tkc::CutlassGemmConfig gemm_config,
                                                                                     char *workspace_ptr, const size_t workspace_bytes, cudaStream_t stream, int *occupancy)
            {
                TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
                if (sm_ >= 70 && sm_ < 75)
                {
                    dispatch_gemm_to_cutlass<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, cutlass::arch::Sm70,
                                             QuantOp, EpilogueTag>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size,
                                                                   workspace_ptr, workspace_bytes, gemm_config, stream, occupancy);
                }
                else if (sm_ >= 75 && sm_ < 80)
                {
                    dispatch_gemm_to_cutlass<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, cutlass::arch::Sm75,
                                             QuantOp, EpilogueTag>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size,
                                                                   workspace_ptr, workspace_bytes, gemm_config, stream, occupancy);
                }
                else if (sm_ >= 80 && sm_ < 89)
                {
                    dispatch_gemm_to_cutlass<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, cutlass::arch::Sm80,
                                             QuantOp, EpilogueTag>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size,
                                                                   workspace_ptr, workspace_bytes, gemm_config, stream, occupancy);
                }
                else if (sm_ == 89)
                {
#if ENABLE_FP8 && ((__CUDACC_VER_MAJOR__ < 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 4))
                    if constexpr (cutlass::platform::is_same<ActivationType, __nv_fp8_e4m3>::value)
                    {
                        throw std::runtime_error(
                            "[TensorRT-LLM Error][CutlassFpAIntBGemmRunner][dispatch_to_arch] INT4xFP8 GEMM for Ada needs "
                            "CUDA>=12.4");
                    }
#endif
                    dispatch_gemm_to_cutlass<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, cutlass::arch::Sm89,
                                             QuantOp, EpilogueTag>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size,
                                                                   workspace_ptr, workspace_bytes, gemm_config, stream, occupancy);
                }
                /*
                else if (sm_ == 90)
                {
                    sm90_dispatch_gemm_to_cutlass<ActivationType, WeightType, ScaleZeroType, BiasType, OutputType, QuantOp,
                                                  EpilogueTag>(A, B, weight_scales, weight_zero_points, biases, alpha, C, m, n, k, group_size, workspace_ptr,
                                                               workspace_bytes, gemm_config, stream, occupancy);
                }
                */
                else
                {
                    throw std::runtime_error(
                        "[TensorRT-LLM Error][CutlassFpAIntBGemmRunner][dispatch_to_arch] Arch unsupported for CUTLASS mixed type "
                        "GEMM");
                }
            }

            template <typename ActivationType, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp, typename ScaleZeroType,
                      typename BiasType, typename OutputType>
            void CutlassFpAIntBGemmRunner<ActivationType, WeightType, QuantOp, ScaleZeroType, BiasType, OutputType>::gemm(
                void const *A, void const *B, void const *weight_scales, void const *weight_zero_points, void const *biases,
                float const alpha, void *C, int m, int n, int k, int const group_size, tkc::CutlassGemmConfig gemmConfig,
                char *workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
            {
                TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
                if constexpr ((QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS) || (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY))
                {
                    dispatch_to_arch<tkc::EpilogueOpBias>((ActivationType const *)A, (WeightType const *)B,
                                                          (ScaleZeroType const *)weight_scales, (ScaleZeroType const *)weight_zero_points, (BiasType const *)biases,
                                                          alpha, (OutputType *)C, m, n, k, group_size, gemmConfig, workspace_ptr, workspace_bytes, stream, nullptr);
                }
                else
                {
                    throw std::runtime_error(
                        "Overload with scale, zero and group size only supported for fine grained bias template.");
                }
            }

            template <typename ActivationType, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp, typename ScaleZeroType,
                      typename BiasType, typename OutputType>
            void CutlassFpAIntBGemmRunner<ActivationType, WeightType, QuantOp, ScaleZeroType, BiasType, OutputType>::gemm(
                void const *A, void const *B, void const *weight_scales, void const *weight_zero_points, void const *biases,
                void *C, int m, int n, int k, int const group_size, tkc::CutlassGemmConfig gemmConfig, char *workspace_ptr,
                const size_t workspace_bytes, cudaStream_t stream)
            {
                TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
                gemm(A, B, weight_scales, weight_zero_points, biases, 1.f, C, m, n, k, group_size, gemmConfig, workspace_ptr,
                     workspace_bytes, stream);
            }

            template <typename ActivationType, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp, typename ScaleZeroType,
                      typename BiasType, typename OutputType>
            void CutlassFpAIntBGemmRunner<ActivationType, WeightType, QuantOp, ScaleZeroType, BiasType, OutputType>::gemm(
                void const *A, void const *B, void const *weight_scales, float const alpha, void *C, int m, int n, int k,
                tkc::CutlassGemmConfig gemmConfig, char *workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
            {
                TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

                if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY)
                {
                    dispatch_to_arch<tkc::EpilogueOpBias>((ActivationType const *)A, (WeightType const *)B,
                                                          (ScaleZeroType const *)weight_scales, nullptr, nullptr, alpha, (OutputType *)C, m, n, k, k, gemmConfig,
                                                          workspace_ptr, workspace_bytes, stream, nullptr);
                }
                else
                {
                    throw std::runtime_error("Overload with scale only (and no group size) only supported for per column scaling.");
                }
            }

            template <typename ActivationType, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp, typename ScaleZeroType,
                      typename BiasType, typename OutputType>
            void CutlassFpAIntBGemmRunner<ActivationType, WeightType, QuantOp, ScaleZeroType, BiasType, OutputType>::gemm(
                void const *A, void const *B, void const *weight_scales, void *C, int m, int n, int k,
                tkc::CutlassGemmConfig gemmConfig, char *workspace_ptr, const size_t workspace_bytes, cudaStream_t stream)
            {
                TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
                gemm(A, B, weight_scales, 1.f, C, m, n, k, gemmConfig, workspace_ptr, workspace_bytes, stream);
            }

            template <typename ActivationType, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp, typename ScaleZeroType,
                      typename BiasType, typename OutputType>
            std::vector<tkc::CutlassGemmConfig>
            CutlassFpAIntBGemmRunner<ActivationType, WeightType, QuantOp, ScaleZeroType, BiasType, OutputType>::getConfigs() const
            {
                static constexpr bool is_weight_only = !std::is_same<ActivationType, WeightType>::value;
                tkc::CutlassGemmConfig::CandidateConfigTypeParam config_type_param = tkc::CutlassGemmConfig::CandidateConfigTypeParam::HOPPER;
                if (is_weight_only)
                {
                    config_type_param = static_cast<tkc::CutlassGemmConfig::CandidateConfigTypeParam>(
                        config_type_param | tkc::CutlassGemmConfig::CandidateConfigTypeParam::WEIGHT_ONLY);
                }
                std::vector<tkc::CutlassGemmConfig> candidateConfigs = get_candidate_configs(sm_, SPLIT_K_LIMIT, config_type_param);
                return candidateConfigs;
            }

            template <typename ActivationType, typename WeightType, cutlass::WeightOnlyQuantOp QuantOp, typename ScaleZeroType,
                      typename BiasType, typename OutputType>
            size_t
            CutlassFpAIntBGemmRunner<ActivationType, WeightType, QuantOp, ScaleZeroType, BiasType, OutputType>::getWorkspaceSize(
                int const m, int const n, int const k)
            {
                TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
                // These are the min tile sizes for each config, which would launch the maximum number of blocks
                int const max_grid_m = cutlass::ceil_div(m, MIN_M_TILE);
                int const max_grid_n = cutlass::ceil_div(n, MIN_N_TILE);
                // We need 4 bytes per block in the worst case. We launch split_k_limit in z dim.
                return static_cast<size_t>(max_grid_m * max_grid_n * SPLIT_K_LIMIT * 4);
            }

        } // namespace cutlass_kernels
    } // namespace kernels
} // namespace tensorrt_llm
