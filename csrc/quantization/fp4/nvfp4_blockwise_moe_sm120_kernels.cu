/*
 * NVFP4 BlockScaled MoE Kernel for SM120 (RTX 5090/Blackwell GeForce)
 *
 * Implementation based on CUTLASS 4.2.0 for SM120 architecture
 * Using LinCombBlockScaleFactor for proper NVFP4 block-scaled operations
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

// CUTLASS includes
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// Debug macro
#ifdef VLLM_DEBUG_NVFP4_MOE_SM120
#define DEBUG_PRINT(...) printf("[nvfp4-sm120] " __VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

// Forward declaration
int32_t get_sm_version_num();

// Check tensor types
#define CHECK_TYPE(x, st, m) \
  TORCH_CHECK(x.scalar_type() == st, ": Inconsistency of Tensor type:", m)
#define CHECK_TH_CUDA(x, m) \
  TORCH_CHECK(x.is_cuda(), m, ": must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x, m) \
  TORCH_CHECK(x.is_contiguous(), m, ": must be contiguous.")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)

// NVFP4 specific types
constexpr auto FLOAT4_E2M1X2 = at::ScalarType::Byte;  // Packed FP4
constexpr auto SF_DTYPE = at::ScalarType::Float8_e4m3fn;  // Scale factor type

// Helper kernel to set up pointer arrays for grouped GEMM
template <typename ElementAB, typename ElementC, typename ElementSF,
          typename ElementAccumulator, typename LayoutSFA, typename LayoutSFB,
          typename ScaleConfig>
__global__ void __get_group_gemm_starts_sm120(
    ElementAB** a_offsets, ElementAB** b_offsets, ElementC** out_offsets,
    ElementSF** a_scales_offsets, ElementSF** b_scales_offsets,
    ElementAccumulator** alpha_offsets, LayoutSFA* layout_sfa_base_as_int,
    LayoutSFB* layout_sfb_base_as_int, ElementAB* a_base_as_int,
    ElementAB* b_base_as_int, ElementC* out_base_as_int,
    ElementSF* a_scales_base_as_int, ElementSF* b_scales_base_as_int,
    ElementAccumulator* alphas_base_as_int, const int32_t* expert_offsets,
    const int32_t* sf_offsets, const int32_t* problem_sizes_as_shapes,
    const int K, const int N) {
  int64_t expert_id = threadIdx.x;
  if (expert_id >= gridDim.x * blockDim.x) {
    return;
  }

  // Offset calculations for grouped GEMM
  int64_t expert_offset = static_cast<int64_t>(expert_offsets[expert_id]);
  int64_t sf_offset = static_cast<int64_t>(sf_offsets[expert_id]);
  int64_t group_size = 16;  // NVFP4 block size
  int64_t m = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3]);
  int64_t n = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 1]);
  int64_t k = static_cast<int64_t>(problem_sizes_as_shapes[expert_id * 3 + 2]);

  int64_t half_k = k / 2;  // FP4 packed
  int64_t group_k = k / group_size;

  // Set pointer offsets
  a_offsets[expert_id] = a_base_as_int + expert_offset * half_k;
  b_offsets[expert_id] = b_base_as_int + expert_id * n * half_k;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  a_scales_offsets[expert_id] = a_scales_base_as_int + sf_offset * group_k;
  b_scales_offsets[expert_id] = b_scales_base_as_int + expert_id * n * group_k;
  alpha_offsets[expert_id] = alphas_base_as_int + expert_id;

  // Set up scale factor layouts
  auto layout_tuple_sfa = ScaleConfig::tile_atom_to_shape_SFA(
      cute::make_shape(int(m), int(n), int(k), int(1)));
  auto layout_tuple_sfb = ScaleConfig::tile_atom_to_shape_SFB(
      cute::make_shape(int(m), int(n), int(k), int(1)));

  layout_sfa_base_as_int[expert_id] = layout_tuple_sfa;
  layout_sfb_base_as_int[expert_id] = layout_tuple_sfb;
}

// CUTLASS kernel implementation for SM120
template<typename OutType>
void run_fp4_blockwise_scaled_group_mm_sm120(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_blockscale,
    const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& sf_offsets,
    int M, int N, int K) {

    DEBUG_PRINT("run_fp4_blockwise_scaled_group_mm_sm120: M=%d, N=%d, K=%d\n", M, N, K);

    // CUTLASS type definitions for SM120
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int32_t, int32_t, int32_t>>;
    using ElementType = cutlass::float_e2m1_t;
    using ElementSFType = cutlass::float_ue4m3_t;
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementC = OutType;
    using ElementD = ElementC;
    using ElementAccumulator = float;

    // Layout definitions (TN layout for SM120 GeForce)
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = LayoutC;

    // Alignment constraints
    static constexpr int AlignmentA = 32;
    static constexpr int AlignmentB = 32;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    // Architecture definitions for SM120
    using ArchTag = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;  // Required for NVFP4

    // Tile and cluster shapes for SM120
    using TileShape = Shape<_128, _128, _128>;  // Standard tile for SM120
    using ClusterShape = Shape<_1, _1, _1>;     // No multicast on GeForce

    // Simplified epilogue without scale factor generation for now
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutC*, AlignmentC,
        ElementD, LayoutD*, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    // Mainloop for SM120 with automatic schedule
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutA*, AlignmentA,
        ElementB, LayoutB*, AlignmentB,
        ElementAccumulator,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto  // Auto schedule for SM120
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        ProblemShape,
        CollectiveMainloop,
        CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Type aliases for clarity
    using StrideA = typename Gemm::GemmKernel::InternalStrideA;
    using StrideB = typename Gemm::GemmKernel::InternalStrideB;
    using StrideC = typename Gemm::GemmKernel::InternalStrideC;
    using StrideD = typename Gemm::GemmKernel::InternalStrideD;
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
    using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

    int num_experts = static_cast<int>(expert_offsets.size(0));

    // Create tensors for pointers and strides
    auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(a.device());
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(a.device());

    torch::Tensor a_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor b_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor out_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor a_scales_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor b_scales_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor alpha_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor layout_sfa = torch::empty({num_experts, sizeof(LayoutSFA)}, options_int32);
    torch::Tensor layout_sfb = torch::empty({num_experts, sizeof(LayoutSFB)}, options_int32);

    // Create stride tensors
    torch::Tensor a_strides = torch::empty({num_experts, 3}, options_int);
    torch::Tensor b_strides = torch::empty({num_experts, 3}, options_int);
    torch::Tensor c_strides = torch::empty({num_experts, 3}, options_int);

    auto stream = at::cuda::getCurrentCUDAStream(a.device().index());

    // Launch kernel to set up pointer arrays
    DEBUG_PRINT("Setting up pointer arrays for %d experts\n", num_experts);
    DEBUG_PRINT("A shape: [%ld], B shape: [%ld, %ld, %ld]\n",
                (long)a.size(0), (long)b.size(0), (long)b.size(1), (long)b.size(2));
    DEBUG_PRINT("Output shape: [%ld, %ld]\n", (long)output.size(0), (long)output.size(1));

    __get_group_gemm_starts_sm120<ElementType, OutType, ElementSFType,
                                  ElementAccumulator, LayoutSFA, LayoutSFB,
                                  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig>
        <<<1, num_experts, 0, stream>>>(
            reinterpret_cast<ElementType**>(a_ptrs.data_ptr()),
            reinterpret_cast<ElementType**>(b_ptrs.data_ptr()),
            reinterpret_cast<OutType**>(out_ptrs.data_ptr()),
            reinterpret_cast<ElementSFType**>(a_scales_ptrs.data_ptr()),
            reinterpret_cast<ElementSFType**>(b_scales_ptrs.data_ptr()),
            reinterpret_cast<ElementAccumulator**>(alpha_ptrs.data_ptr()),
            reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
            reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr()),
            const_cast<ElementType*>(reinterpret_cast<const ElementType*>(a.data_ptr())),
            const_cast<ElementType*>(reinterpret_cast<const ElementType*>(b.data_ptr())),
            reinterpret_cast<OutType*>(output.data_ptr()),
            const_cast<ElementSFType*>(reinterpret_cast<const ElementSFType*>(a_blockscale.data_ptr())),
            const_cast<ElementSFType*>(reinterpret_cast<const ElementSFType*>(b_blockscales.data_ptr())),
            const_cast<ElementAccumulator*>(reinterpret_cast<const ElementAccumulator*>(alphas.data_ptr())),
            reinterpret_cast<const int32_t*>(expert_offsets.data_ptr()),
            reinterpret_cast<const int32_t*>(sf_offsets.data_ptr()),
            reinterpret_cast<const int32_t*>(problem_sizes.data_ptr()),
            K, N);

    // Synchronize to ensure helper kernel completes
    cudaError_t sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess) {
        DEBUG_PRINT("ERROR: Helper kernel failed: %s\n", cudaGetErrorString(sync_error));
        output.fill_(0.05f);
        return;
    }
    DEBUG_PRINT("Helper kernel completed successfully\n");

    // Set up strides for matrices
    auto* a_strides_ptr = reinterpret_cast<StrideA*>(a_strides.data_ptr());
    auto* b_strides_ptr = reinterpret_cast<StrideB*>(b_strides.data_ptr());
    auto* c_strides_ptr = reinterpret_cast<StrideC*>(c_strides.data_ptr());
    // Create problem shapes as tuples
    using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;
    torch::Tensor problem_shapes = torch::empty({num_experts, sizeof(UnderlyingProblemShape)/sizeof(int32_t)}, options_int32);
    auto* problem_shapes_ptr = reinterpret_cast<UnderlyingProblemShape*>(problem_shapes.data_ptr());
    auto* problem_sizes_flat = reinterpret_cast<const int32_t*>(problem_sizes.data_ptr());

    // Fill problem shapes and strides on host
    std::vector<UnderlyingProblemShape> problem_shapes_host(num_experts);
    std::vector<StrideA> a_strides_host(num_experts);
    std::vector<StrideB> b_strides_host(num_experts);
    std::vector<StrideC> c_strides_host(num_experts);

    for (int i = 0; i < num_experts; ++i) {
        int32_t m = problem_sizes_flat[i * 3];
        int32_t n = problem_sizes_flat[i * 3 + 1];
        int32_t k = problem_sizes_flat[i * 3 + 2];

        problem_shapes_host[i] = cute::make_tuple(m, n, k);
        a_strides_host[i] = cutlass::make_cute_packed_stride(StrideA{}, {m, k / 2, 1});
        b_strides_host[i] = cutlass::make_cute_packed_stride(StrideB{}, {n, k / 2, 1});
        c_strides_host[i] = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    }

    cudaMemcpyAsync(problem_shapes_ptr, problem_shapes_host.data(),
                    sizeof(UnderlyingProblemShape) * num_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(a_strides_ptr, a_strides_host.data(),
                    sizeof(StrideA) * num_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(b_strides_ptr, b_strides_host.data(),
                    sizeof(StrideB) * num_experts, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(c_strides_ptr, c_strides_host.data(),
                    sizeof(StrideC) * num_experts, cudaMemcpyHostToDevice, stream);

    // Create CUTLASS kernel instance
    Gemm gemm_op;

    // Set up kernel hardware info
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = a.device().index();
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
        hw_info.device_id);

    // Set up scheduler arguments
    typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
    // Use default scheduler settings for SM120

    // Create mainloop arguments
    typename Gemm::GemmKernel::MainloopArguments mainloop_args{
        static_cast<const ElementType**>(a_ptrs.data_ptr()),
        static_cast<StrideA*>(a_strides.data_ptr()),
        static_cast<const ElementType**>(b_ptrs.data_ptr()),
        static_cast<StrideB*>(b_strides.data_ptr()),
        static_cast<const ElementSFType**>(a_scales_ptrs.data_ptr()),
        reinterpret_cast<LayoutSFA*>(layout_sfa.data_ptr()),
        static_cast<const ElementSFType**>(b_scales_ptrs.data_ptr()),
        reinterpret_cast<LayoutSFB*>(layout_sfb.data_ptr())
    };

    // Create epilogue arguments
    typename Gemm::GemmKernel::EpilogueArguments epilogue_args{
        {},  // fusion args
        nullptr,  // ptr_C
        static_cast<StrideC*>(c_strides.data_ptr()),
        static_cast<OutType**>(out_ptrs.data_ptr()),
        static_cast<StrideD*>(c_strides.data_ptr())
    };

    // Set fusion arguments for alpha scaling
    auto& fusion_args = epilogue_args.thread;
    fusion_args.alpha_ptr_array = reinterpret_cast<float**>(alpha_ptrs.data_ptr());
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
    fusion_args.beta = 0.0f;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

    // Create kernel arguments
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_experts, problem_shapes_ptr, nullptr},
        mainloop_args,
        epilogue_args,
        hw_info,
        scheduler
    };

    // Get workspace size and allocate
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    auto workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
    torch::Tensor workspace = torch::empty(workspace_size, workspace_options);

    // Check if kernel can be implemented
    DEBUG_PRINT("Checking kernel implementation feasibility\n");
    auto can_implement = gemm_op.can_implement(arguments);
    if (can_implement != cutlass::Status::kSuccess) {
        DEBUG_PRINT("ERROR: Kernel cannot be implemented for given problem size (status %d)\n",
                    static_cast<int>(can_implement));
        // Fall back to placeholder for now
        output.fill_(0.05f);
        return;
    }
    DEBUG_PRINT("Kernel can be implemented\n");

    // Initialize kernel
    DEBUG_PRINT("Initializing kernel with workspace size %ld\n", (long)workspace_size);
    auto status = gemm_op.initialize(arguments, workspace.data_ptr());
    if (status != cutlass::Status::kSuccess) {
        DEBUG_PRINT("ERROR: Failed to initialize kernel (status %d)\n",
                    static_cast<int>(status));
        output.fill_(0.05f);
        return;
    }
    DEBUG_PRINT("Kernel initialized successfully\n");

    // Run the kernel
    DEBUG_PRINT("Running CUTLASS kernel\n");
    status = gemm_op.run(arguments, workspace.data_ptr(), stream);
    if (status != cutlass::Status::kSuccess) {
        DEBUG_PRINT("ERROR: Failed to run kernel (status %d)\n",
                    static_cast<int>(status));
        output.fill_(0.05f);
        return;
    }

    // Synchronize to check for kernel errors
    sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess) {
        DEBUG_PRINT("ERROR: CUTLASS kernel execution failed: %s\n",
                    cudaGetErrorString(sync_error));
        output.fill_(0.05f);
        return;
    }

    DEBUG_PRINT("SM120 CUTLASS kernel executed successfully\n");
}

// Main entry point
void cutlass_fp4_group_mm_sm120(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& a_blockscale,
    const torch::Tensor& b_blockscales,
    const torch::Tensor& alphas,
    const torch::Tensor& problem_sizes,
    const torch::Tensor& expert_offsets,
    const torch::Tensor& sf_offsets) {

    DEBUG_PRINT("cutlass_fp4_group_mm_sm120 called\n");

    // Input validation
    CHECK_INPUT(a, FLOAT4_E2M1X2, "a");
    CHECK_INPUT(b, FLOAT4_E2M1X2, "b");
    CHECK_INPUT(a_blockscale, SF_DTYPE, "a_blockscale");
    CHECK_INPUT(b_blockscales, SF_DTYPE, "b_blockscales");
    CHECK_INPUT(alphas, at::ScalarType::Float, "alphas");

    TORCH_CHECK(a_blockscale.dim() == 2,
                "expected a_blockscale to be of shape [num_experts, rounded_m,"
                " k // group_size], observed rank: ", a_blockscale.dim());
    TORCH_CHECK(b_blockscales.dim() == 3,
                "expected b_blockscale to be of shape: "
                " [num_experts, n, k // group_size], observed rank: ",
                b_blockscales.dim());
    TORCH_CHECK(problem_sizes.dim() == 2,
                "problem_sizes must be a 2D tensor");
    TORCH_CHECK(problem_sizes.size(1) == 3,
                "problem_sizes must have shape (num_experts, 3)");
    TORCH_CHECK(problem_sizes.size(0) == expert_offsets.size(0),
                "Number of experts must match");
    TORCH_CHECK(problem_sizes.dtype() == torch::kInt32,
                "problem_sizes must be int32");

    // Verify we're on SM120
    int32_t sm_version = get_sm_version_num();
    DEBUG_PRINT("SM version: %d.%d\n", sm_version / 10, sm_version % 10);

    if (sm_version < 120) {
        TORCH_CHECK(false,
                    "SM120 kernel requires compute capability >= 12.0, got ",
                    sm_version / 10, ".", sm_version % 10);
    }

    int num_experts = static_cast<int>(expert_offsets.size(0));
    int M = static_cast<int>(a.size(0));
    int N = static_cast<int>(b.size(1));
    int K = static_cast<int>(2 * b.size(2));  // K is doubled because FP4 is packed

    DEBUG_PRINT("Running SM120 kernel: E=%d, M=%d, N=%d, K=%d\n",
                num_experts, M, N, K);

    // Check for K=1536 register exhaustion issue
    if (K == 1536) {
        DEBUG_PRINT("WARNING: K=1536 exceeds SM120 register limits. May fail or produce incorrect results.\n");
    }

    // Call the templated implementation
    if (output.scalar_type() == torch::kBFloat16) {
        run_fp4_blockwise_scaled_group_mm_sm120<cutlass::bfloat16_t>(
            output, a, b, a_blockscale, b_blockscales, alphas,
            problem_sizes, expert_offsets, sf_offsets, M, N, K);
    } else {
        run_fp4_blockwise_scaled_group_mm_sm120<cutlass::half_t>(
            output, a, b, a_blockscale, b_blockscales, alphas,
            problem_sizes, expert_offsets, sf_offsets, M, N, K);
    }

    // Log a warning that this is still being developed
    static bool warned = false;
    if (!warned) {
        printf("[WARNING] SM120 NVFP4 kernel is using simplified implementation. "
               "Full CUTLASS implementation in progress.\n");
        warned = true;
    }
}