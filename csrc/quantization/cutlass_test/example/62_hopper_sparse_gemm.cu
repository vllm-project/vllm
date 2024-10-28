/***************************************************************************************************
 * Copyright (c) 2024 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Hopper Sparse GEMM example.

  This example demonstrates how to construct and run a structured sparse GEMM kernel
  on NVIDIA Hopper architecture.
    
*/

#include <iostream>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"

#include "util/command_line.h"
#include "util/distribution.h"
#include "util/host_tensor.h"
#include "util/packed_stride.hpp"
#include "util/tensor_view_io.h"
#include "util/reference/device/gemm.h"
#include "util/reference/device/tensor_compare.h"
#include "util/reference/device/tensor_fill.h"

#include "util/helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SPARSE_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::half_t;                                // Element type for A matrix operand
using         LayoutTagA  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::half_t;                                // Element type for B matrix operand
using         LayoutTagB  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = float;                                          // Element type for C and D matrix operands
using         LayoutTagC  = cutlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using TileShape           = Shape<_128,_128,_128>;                          // Threadblock-level tile size for sparse kernel
using TileShapeRef        = Shape<_128,_128, _64>;                          // Threadblock-level tile size for reference (dense) kernel
using ClusterShape        = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster
using KernelSchedule      = cutlass::gemm::KernelTmaWarpSpecialized;        // Kernel schedule policy
using EpilogueSchedule    = cutlass::epilogue::TmaWarpSpecialized;          // Epilogue schedule policy

using ProblemShape = Shape<int,int,int,int>;

// Sparse kernel setup

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutTagC, AlignmentC,
    ElementC, LayoutTagC, AlignmentC,
    EpilogueSchedule
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassSparseTensorOp,
    ElementA, LayoutTagA, AlignmentA,
    ElementB, LayoutTagB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference (dense) kernel setup

using CollectiveEpilogueRef = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShapeRef, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutTagC, AlignmentC,
    ElementC, LayoutTagC, AlignmentC,
    EpilogueSchedule
  >::CollectiveOp;

using CollectiveMainloopRef = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutTagA, AlignmentA,
    ElementB, LayoutTagB, AlignmentB,
    ElementAccumulator,
    TileShapeRef, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule
  >::CollectiveOp;

using GemmKernelRef = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloopRef,
    CollectiveEpilogue
>;

using GemmRef = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelRef>;

// Layouts 
using LayoutA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutA;
using LayoutE = typename Gemm::GemmKernel::CollectiveMainloop::LayoutE;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

// Layouts for reference (non-sparse) tensors
using StrideA = cutlass::gemm::TagToStrideA_t<LayoutTagA>;
using StrideE = StrideA;

using ElementE = typename Gemm::GemmKernel::CollectiveMainloop::ElementE;
using SparseConfig = typename Gemm::GemmKernel::CollectiveMainloop::SparseConfig;

// Offline compressor kernel
using CompressorUtility = cutlass::transform::kernel::StructuredSparseCompressorUtility<
                            ProblemShape,
                            ElementA,
                            LayoutTagA,
                            SparseConfig>;

using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
                            ProblemShape,
                            ElementA,
                            LayoutTagA,
                            SparseConfig,
                            cutlass::arch::Sm90>;

using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

//
// Data members
//

ProblemShape problem_shape;

StrideA stride_A;
StrideA stride_A_compressed;
StrideE stride_E;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;

LayoutA layout_A;
LayoutE layout_E;

uint64_t seed;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementA> block_A_compressed;
cutlass::DeviceAllocation<typename Gemm::CollectiveMainloop::ElementE> block_E;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D_ref;

#endif // defined(CUTLASS_ARCH_MMA_SPARSE_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k, l;

  Options():
    help(false),
    m(5120), n(4096), k(16384), l(1),
    alpha(1.f), beta(0.f),
    iterations(10)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "62_hopper_sparse_gemm\n\n"
      << "  Hopper Sparse GEMM example.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the L extent of the GEMM (batch size)\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "62_hopper_sparse_gemm" << " --m=4096 --n=5120 --k=8192 --l=1 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

#if defined(CUTLASS_ARCH_MMA_SPARSE_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed) {

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = Element(2);
    scope_min = Element(0);
  } else if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(8);
    scope_min = Element(-8);
  }

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

/// Make A structured sparse by replacing elements with 0 and compress it
bool sparsify_and_compress()
{
  auto [M, N, K, L] = problem_shape;
  CompressorUtility compressor_utility(problem_shape, stride_A);

  int ME = compressor_utility.get_metadata_m_physical();
  int KE = compressor_utility.get_metadata_k_physical();
  int KC = compressor_utility.get_tensorA_k_physical();

  block_A_compressed.reset(M * KC * L);
  block_E.reset(ME * KE * L);

  stride_A_compressed = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, KC, L));
  stride_E = cutlass::make_cute_packed_stride(StrideE{}, cute::make_shape(ME, KE, L));

  // Random sparsification is performed on host
  std::vector<ElementA> block_A_host(block_A.size());
  cutlass::device_memory::copy_to_host(block_A_host.data(), block_A.get(), block_A.size());
  compressor_utility.structure_sparse_zero_mask_fill(block_A_host.data(), static_cast<int>(seed + 2024));
  cutlass::device_memory::copy_to_device(block_A.get(), block_A_host.data(), block_A.size());

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  typename Compressor::Arguments arguments {
    problem_shape,
    { block_A.get(),
      stride_A,
      block_A_compressed.get(),
      block_E.get() },
    {hw_info} };

  Compressor compressor_op;
  size_t workspace_size = Compressor::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CUTLASS_CHECK(compressor_op.can_implement(arguments));
  CUTLASS_CHECK(compressor_op.initialize(arguments, workspace.get()));
  CUTLASS_CHECK(compressor_op.run());
  CUDA_CHECK(cudaDeviceSynchronize());

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
bool initialize(Options const& options) {

  problem_shape = make_tuple(options.m, options.n, options.k, options.l);
  auto [M, N, K, L] = problem_shape;

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

  // Allocate memory for tensors
  block_A.reset(M * K * L);
  block_B.reset(N * K * L);
  block_C.reset(M * N * L);
  block_D.reset(M * N * L);
  block_D_ref.reset(M * N * L);

  // Fill input tensors with data
  initialize_block(block_A, seed + 2021);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2023);

  // Replace 0 in A with 1 to avoid metadata changes
  std::vector<ElementA> block_A_host(block_A.size());
  cutlass::device_memory::copy_to_host(block_A_host.data(), block_A.get(), block_A.size());
  for (size_t i = 0; i < block_A.size(); ++i) if (block_A_host[i] == ElementA(0)) block_A_host[i] = ElementA(1.0);
  cutlass::device_memory::copy_to_device(block_A.get(), block_A_host.data(), block_A.size());

  if (!sparsify_and_compress()) {
    return false;
  };

  // Build the compressed/metadata layouts
  layout_A = SparseConfig::fill_layoutA(problem_shape);
  layout_E = SparseConfig::fill_layoutE(problem_shape);

  return true;
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments make_args(Options const& options)
{
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_shape,
    { block_A_compressed.get(), layout_A, block_B.get(), stride_B, block_E.get(), layout_E },
    { { ElementAccumulator(options.alpha), ElementAccumulator(options.beta) },
      block_C.get(), stride_C, block_D.get(), stride_D }
  };

  return arguments;
}

typename GemmRef::Arguments make_args_ref(Options const& options)
{
  typename GemmRef::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_shape,
    { block_A.get(), stride_A, block_B.get(), stride_B },
    { { ElementAccumulator(options.alpha), ElementAccumulator(options.beta) },
      block_C.get(), stride_C, block_D_ref.get(), stride_D }
  };

  return arguments;
}

template<class Engine, class Layout>
void print_device_tensor(cute::Tensor<Engine, Layout> const& t)
{
  // Assumes size = cosize, i.e. compact tensor
  std::vector<typename Engine::value_type> data_host(t.size());
  cutlass::device_memory::copy_to_host(data_host.data(), t.data(), t.size());
  auto t_host = cute::make_tensor(data_host.data(), t.layout());
  cute::print_tensor(t_host);
}

bool verify(Options const& options) {
  CUDA_CHECK(cudaDeviceSynchronize());

  bool passed = cutlass::reference::device::BlockCompareEqual(block_D_ref.get(), block_D.get(), block_D.size());

#if 0
  if (!passed) {
    auto [M, N, K, L] = problem_shape;
    CompressorUtility compressor_utility(problem_shape, stride_A);
    int ME = compressor_utility.get_metadata_m_physical();
    int KE = compressor_utility.get_metadata_k_physical();
    int KC = compressor_utility.get_tensorA_k_physical();

    cute::print("A (original): "); print_device_tensor(make_tensor(block_A.get(), make_shape(M, K, L), stride_A));
    cute::print("A (compressed): "); print_device_tensor(make_tensor(block_A_compressed.get(), make_shape(M, KC, L), stride_A_compressed));
    cute::print("E (physical): "); print_device_tensor(make_tensor(block_E.get(), make_shape(ME, KE, L), stride_E));
    cute::print("E (logical): "); print_device_tensor(make_tensor(block_E.get(), upcast<CollectiveMainloop::ElementEMmaSparsity>(layout_E)));
    cute::print("B: "); print_device_tensor(make_tensor(block_B.get(), make_shape(N, K, L), stride_B));
    cute::print("C: "); print_device_tensor(make_tensor(block_C.get(), make_shape(M, N, L), stride_C));
    cute::print("D reference: "); print_device_tensor(make_tensor(block_D_ref.get(), make_shape(M, N, L), stride_D));
    cute::print("D  computed: "); print_device_tensor(make_tensor(block_D.get(), make_shape(M, N, L), stride_D));
  }
#endif

  return passed;
}

template<typename Gemm>
struct Runner
{
  using Arguments = typename Gemm::Arguments;

  Runner(Arguments args): arguments(args) {
    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    workspace.reset(workspace_size);

    // Check if the problem size is supported or not
    CUTLASS_CHECK(gemm.can_implement(arguments));
  }

  void run() {
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm.run());
  }

  void benchmark(Options const& options) {
    if (options.iterations > 0)
    {
      GpuTimer timer;
      timer.start();
      for (int iter = 0; iter < options.iterations; ++iter) {
        run();
      }
      timer.stop();

      // Compute average runtime and GFLOPs.
      float elapsed_ms = timer.elapsed_millis();
      double avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
      double gflops = options.gflops(avg_runtime_ms / 1000.0);

      std::cout << "  Avg runtime: " << avg_runtime_ms << " ms" << std::endl;
      std::cout << "  GFLOPS: " << gflops << std::endl;
    }
  }

  Gemm gemm;
  Arguments arguments;
  cutlass::device_memory::allocation<uint8_t> workspace;
};

/// Execute the example (verification and timing)
void run(Options &options) {
  bool init = initialize(options);
  if (!init) {
    std::cout << "Initialization failure" << std::endl;
    exit(EXIT_FAILURE);
  }

  Runner<Gemm> gemm(make_args(options));
  Runner<GemmRef> gemm_ref(make_args_ref(options));

  gemm.run();
  gemm_ref.run();

  bool passed = verify(options);

  std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
  std::cout << "  Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

  if (!passed) {
    exit(EXIT_FAILURE);
  }

  std::cout << "Sparse GEMM:" << std::endl;
  gemm.benchmark(options);

  std::cout << "Dense GEMM:" << std::endl;
  gemm_ref.benchmark(options);
}

#endif // defined(CUTLASS_ARCH_MMA_SPARSE_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.2 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 2)) {
    std::cerr << "This example requires CUDA 12.2 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major < 9) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater).\n";
    return 0;
  }
  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //

#if defined(CUTLASS_ARCH_MMA_SPARSE_SM90_SUPPORTED)
  run(options);
#endif

  return EXIT_SUCCESS;
}

/////////////////////////////////////////////////////////////////////////////////////////////////