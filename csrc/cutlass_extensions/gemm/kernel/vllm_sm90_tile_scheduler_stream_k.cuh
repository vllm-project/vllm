// Modified version of:
//  cutlass/gemm/kernel/sm90_tile_scheduler_stream_k.hpp
// with bug fixes, namely
//  1) tile_peer_range was not properly accounting for big units
//  2) get_current_work would prematurely return empty work tiles that should be
//  skipped

// Do not format file to make easier to diff with original
// clang-format off

#pragma once

#include "cutlass/barrier.h"
#include "cutlass/block_striped.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"

#include "cutlass_extensions/gemm/kernel/vllm_tile_scheduler_params.hpp"

namespace cutlass::gemm::kernel::detail {

// Persistent Thread Block (TB) scheduler leveraging stream-K decomposition
template <
  class TileShape,
  class ClusterShape
>
class VLLMPersistentTileSchedulerSm90StreamK {
  //
  // Data members
  //

private:
  using UnderlyingScheduler = PersistentTileSchedulerSm90;

private:
  using UnderlyingArguments = typename UnderlyingScheduler::Arguments;
  using UnderlyingParams = typename UnderlyingScheduler::Params;

  uint64_t current_work_linear_idx_ = 0;

public:

  using RasterOrder = UnderlyingScheduler::RasterOrder;
  using RasterOrderOptions = UnderlyingScheduler::RasterOrderOptions;
  // Use a dummy barrier manager to simply get the type used to store the barrier
  using BarrierType = typename NamedBarrierManager<1>::T;

  using Params = VLLMPersistentTileSchedulerSm90StreamKParams;
  using ReductionMode = Params::ReductionMode;
  using DecompositionMode = Params::DecompositionMode;

  struct WorkTileInfo {
    int32_t M_idx = 0;
    int32_t N_idx = 0;
    int32_t K_idx = 0;
    int32_t L_idx = 0;
    int32_t linear_idx = 0;

    // Number of k tiles to compute for this unit of work. For stream-K, this
    // can indicate the number of K tiles across multiple output tiles.
    uint32_t k_tile_count = 0;

    // Number of k tiles remaining for the work unit as a whole
    uint32_t k_tile_remaining = 0;

    // Whether this unit of work is the final split for the given tile
    bool is_separate_reduction = false;

    CUTLASS_HOST_DEVICE
    bool
    is_valid() const {
      // A work tile that computes no K tiles is invalid unless it is a separate-reduction work tile
      // (which only performs reduction and epilogue)
      return k_tile_count > 0 || is_separate_reduction;
    }

    CUTLASS_HOST_DEVICE
    bool
    is_reduction_unit() const {
      return is_separate_reduction;
    }

    CUTLASS_HOST_DEVICE
    int32_t
    reduction_subtile_idx() const {
      // For separate reduction units, the K_idx of the work tile is unused.
      // Therefore, we override it to contain the subtile of that the reduction
      // unit operates on.
      return is_reduction_unit() ? K_idx : -1;
    }

    CUTLASS_HOST_DEVICE
    void
    setup_separate_reduction(int32_t epilogue_subtile_idx) {
      // Set the epilogue subtile in the K_idx, since this is otherwise unused
      // by separate reduction units.
      K_idx = epilogue_subtile_idx;

      is_separate_reduction = true;
      k_tile_count = 0;
      // Clean up remaining k tiles
      k_tile_remaining = 0;
    }

    CUTLASS_HOST_DEVICE
    static WorkTileInfo
    invalid_work_tile() {
      return {-1, -1, -1, -1, -1, 0};
    }

    CUTLASS_HOST_DEVICE
    bool
    is_final_split(uint32_t k_tiles_per_output_tile) const {
      return (K_idx + k_tile_count) == k_tiles_per_output_tile;
    }
  };

  struct Arguments {

    Arguments() = default;
    Arguments(Arguments const&) = default;
    Arguments(Arguments&&) = default;

    CUTLASS_HOST_DEVICE
    Arguments&
    operator=(Arguments const& args) {
      splits = args.splits;
      max_swizzle_size = args.max_swizzle_size;
      raster_order = args.raster_order;
      reduction_mode = args.reduction_mode;
      decomposition_mode = args.decomposition_mode;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    Arguments&
    operator=(Arguments&& args) noexcept {
      splits = args.splits;
      max_swizzle_size = args.max_swizzle_size;
      raster_order = args.raster_order;
      reduction_mode = args.reduction_mode;
      decomposition_mode = args.decomposition_mode;
      return *this;
    }

    CUTLASS_HOST_DEVICE
    Arguments(int splits_) : splits(splits_) {}

    CUTLASS_HOST_DEVICE
    Arguments(int splits_, int max_swizzle_size_, RasterOrderOptions raster_order_, DecompositionMode decomposition_mode_) :
      splits(splits_),
      max_swizzle_size(max_swizzle_size_),
      raster_order(raster_order_),
      decomposition_mode(decomposition_mode_) {}

    // The splitting factor to be used in a split-K decomposition of the problem.
    // If this is set to a value greater than 1, stream-K decomposition logic
    // is bypassed in favor of a split-K decomposition.
    int splits = 1;
    int max_swizzle_size = 1;
    RasterOrderOptions raster_order = RasterOrderOptions::Heuristic;
    ReductionMode reduction_mode = ReductionMode::Deterministic;
    DecompositionMode decomposition_mode = DecompositionMode::Heuristic;
  };

  // Sink scheduler params as a member
  Params scheduler_params;

  //
  // Methods
  //

  template <class ProblemShape>
  static Params
  to_underlying_arguments(
    ProblemShape problem_shape,
    TileShape tile_shape,
    ClusterShape cluster_shape,
    KernelHardwareInfo const& hw_info,
    Arguments const& args,
    void* workspace,
    const uint32_t epilogue_subtile = 1) {

    static_assert(cute::is_static<TileShape>::value);
    static_assert(cute::is_static<ClusterShape>::value);

    auto problem_shape_mnkl = cute::append<4>(problem_shape, cute::Int<1>{});
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);
    uint32_t k_tile_per_output_tile = cute::size(cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    Params params;
    params.initialize(
      problem_blocks,
      k_tile_per_output_tile,
      to_gemm_coord(cluster_shape),
      hw_info,
      args.splits,
      args.max_swizzle_size,
      args.raster_order,
      args.reduction_mode,
      args.decomposition_mode,
      workspace,
      epilogue_subtile
    );
    return params;
  }

  CUTLASS_HOST_DEVICE
  static bool
  can_implement(Arguments const& args) {
    // Split count > 1 is only valid for heuristic and split-K decomposition modes
    return (args.splits == 1 ||
            args.decomposition_mode == DecompositionMode::Heuristic ||
            args.decomposition_mode == DecompositionMode::SplitK);
  }

  CUTLASS_HOST_DEVICE
  VLLMPersistentTileSchedulerSm90StreamK() {};

  CUTLASS_HOST_DEVICE
  VLLMPersistentTileSchedulerSm90StreamK(Params const& params_) : scheduler_params(params_) {
    if (params_.raster_order_ == RasterOrder::AlongN) {
      current_work_linear_idx_ = uint64_t(blockIdx.x) + uint64_t(blockIdx.y) * uint64_t(gridDim.x);
    }
    else {
      current_work_linear_idx_ = uint64_t(blockIdx.x) * uint64_t(gridDim.y) + uint64_t(blockIdx.y);
    }
  }

  CUTLASS_DEVICE
  WorkTileInfo
  get_current_work() {
    auto work_tile_info = get_current_work_for_linear_idx(current_work_linear_idx_, scheduler_params);
    while (!work_tile_info.is_valid() && work_tile_info.linear_idx >= 0) {
      advance_to_next_work();
      work_tile_info = get_current_work_for_linear_idx(current_work_linear_idx_, scheduler_params);
    }
    return work_tile_info;
  }

  CUTLASS_DEVICE
  static WorkTileInfo
  get_current_work_for_linear_idx(uint64_t linear_idx, Params const& params) {
    // The maximum number of work units is units_per_problem_ * splits_.
    // The multiplication by splits_ is used for handling split-K, in which
    // units_per_problem_ is equal to the total number of output tiles. To account
    // for the fact that we have splits_ peers per output tile, we multiply this
    // value by splits_. For stream-K, this multiplication ends up being a no-op
    // because splits_ is set to 1 for stream-K.
    if(linear_idx >= (params.units_per_problem_ * params.splits_ + params.separate_reduction_units_)) {
      // Invalid work. Return an empty result.
      return WorkTileInfo::invalid_work_tile();
    }

    WorkTileInfo work_tile_info;
    assign_work(params, linear_idx, work_tile_info);
    return work_tile_info;
  }

  // Returns whether the current work_tile_info passed in should continue to be used. This
  // occurs only in the stream-K decomposition with stream-K work units, which encompass
  // work over multiple output tiles. If the current work_tile_info should continue to be
  // used, it is updated to advance to the next output tile it should cover.
  CUTLASS_DEVICE
  bool
  continue_current_work(WorkTileInfo& work_tile_info) const {
    return continue_current_work_for_linear_idx(
      current_work_linear_idx_, work_tile_info, scheduler_params);
  }

  CUTLASS_DEVICE
  static bool
  continue_current_work_for_linear_idx(
    uint64_t linear_idx,
    WorkTileInfo& work_tile_info,
    Params const& params) {

    work_tile_info.k_tile_remaining -= work_tile_info.k_tile_count;

    if (work_tile_info.k_tile_remaining == 0) {
      return false;
    }
    assign_work(params, linear_idx, work_tile_info);
    return work_tile_info.is_valid();
  }

  CUTLASS_DEVICE
  void
  advance_to_next_work(uint32_t advance_count = 1) {
    current_work_linear_idx_ += uint64_t(gridDim.x) * uint64_t(gridDim.y) * uint64_t(gridDim.z) * uint64_t(advance_count);
  }

  // Given the inputs, computes the total number of output blocks this problem will compute over
  // Note that this is only the logical size of our grid, not the physical grid we will actually launch.
  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_tiled_cta_shape_mnl(ProblemShape problem_shape_mnkl, TileShape cta_shape, ClusterShape cluster_shape) {
    return UnderlyingScheduler::get_tiled_cta_shape_mnl(problem_shape_mnkl, cta_shape, cluster_shape);
  }

  // Given the cluster shape, computes the physical grid we should launch.
  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static
  dim3
  get_grid_shape(
    ProblemShape problem_shape,
    TileShape tile_shape,
    ClusterShape cluster_shape,
    KernelHardwareInfo hw_info,
    Arguments arguments) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape, cute::Int<1>{});
    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);

    return Params::get_grid_shape(
      problem_blocks,
      to_gemm_coord(cluster_shape),
      hw_info,
      arguments.max_swizzle_size,
      arguments.raster_order
    );
  }

  // Returns whether fixup is needed for `work_tile_info`.
  CUTLASS_HOST_DEVICE
  static bool
  requires_fixup(Params const& params, WorkTileInfo const& work_tile_info) {
    // Fixup is not needed for invalid or data-parallel tiles
    return work_tile_info.is_valid() && (work_tile_info.k_tile_count != params.divmod_tiles_per_output_tile_.divisor || params.requires_separate_reduction());
  }

  CUTLASS_HOST_DEVICE
  static bool
  requires_separate_reduction(Params const& params) {
    return params.requires_separate_reduction();
  }

  // When the work tile is not special for reduction, it's valid. Otherwise need to skip
  // global loading that producer warpgroup do, also math computation that consumer warpgroup do.
  CUTLASS_DEVICE
  static bool
  valid_warpgroup_in_work_tile(WorkTileInfo const& work_tile_info) {
    return !work_tile_info.is_reduction_unit();
  }

  // Performs the reduction across splits for a given output tile.
  template <class FrgTensorC>
  CUTLASS_DEVICE
  static void
  fixup(
    Params const& params,
    WorkTileInfo const& work_tile_info,
    FrgTensorC& accumulators,
    uint32_t num_barriers,
    uint32_t barrier_idx) {
    static constexpr uint32_t Offset = static_cast<int>(cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);
    static constexpr uint32_t MaxNumNamedBarriers = 2;
    using BarrierManager = NamedBarrierManager<NumThreadsPerWarpGroup, Offset, MaxNumNamedBarriers>;
    return fixup_helper<FrgTensorC, BarrierManager>(
      params, work_tile_info, accumulators, num_barriers, barrier_idx);
  }

  // Helper for performing the reduction across splits for a given output tile.
  template <class FrgTensorC, class BarrierManager>
  CUTLASS_DEVICE
  static void
  fixup_helper(
    Params const& params,
    WorkTileInfo const& work_tile_info,
    FrgTensorC& accumulators,
    uint32_t num_barriers,
    uint32_t barrier_idx,
    uint32_t num_accumulator_mtxs = 1) {

    using ElementAccumulator = typename FrgTensorC::value_type;

    if (!requires_fixup(params, work_tile_info)) {
      return;
    }
    auto tile_idx = output_tile_index(params, work_tile_info);

    // Index of the lock on which to wait
    auto lock_idx = (tile_idx * num_barriers) + barrier_idx;

    auto reduction_tile_idx = tile_idx;
    auto [first_peer_id, my_peer_id, last_peer_id] = tile_peer_range(params, work_tile_info.linear_idx, tile_idx, static_cast<uint32_t>(work_tile_info.K_idx));
    auto reduction_peer_offset = 0;
    if (params.requires_separate_reduction()) {
      // If separate reduction is to be performed, each stream-K unit writes its partials
      // to a separate portion of the workspace. There are as many of these portions as there
      // are peers for a given output tile, so we multiply the tile index by the maximum peer count.
      reduction_tile_idx *= Params::max_peers_per_tile(params.sk_units_, params.sk_tiles_);
      reduction_peer_offset = (my_peer_id - first_peer_id) * cute::size<0>(TileShape{}) * cute::size<1>(TileShape{});
    }

    // Reductions use BlockStripedReduce with a width of BarrierManager::ThreadCount under the hood.
    // Thus, the start of the reduction space is the same across all threads in a warp group.
    int reduction_offset =
      (cute::size<0>(TileShape{}) * cute::size<1>(TileShape{}) * reduction_tile_idx * num_accumulator_mtxs) +
      reduction_peer_offset +
      (size(accumulators) * barrier_idx * BarrierManager::ThreadCount);

    ElementAccumulator* group_reduction_workspace = reinterpret_cast<ElementAccumulator*>(params.reduction_workspace_) + reduction_offset;

    using AccumulatorArrayT = Array<typename FrgTensorC::value_type, size(FrgTensorC{})>;
    using BlockStripedReduceT = BlockStripedReduce<BarrierManager::ThreadCount, AccumulatorArrayT>;

    AccumulatorArrayT* reduction_workspace_array = reinterpret_cast<AccumulatorArrayT*>(group_reduction_workspace);
    AccumulatorArrayT* accumulator_array = reinterpret_cast<AccumulatorArrayT*>(&accumulators);

    int barrier_group_thread_idx = threadIdx.x % BarrierManager::ThreadCount;

    // The number of tiles for which reduction is required is either:
    //   (a) the total number of output tiles (in the case of split-K)
    //   (b) the number of stream-K tiles (potentially multiplied by peer count if using separate reduction)
    // To calculate the total number of output tiles in the split-K case, we
    // note that, in the split-K case, the units_per_problem_ member of Params will be
    // the total number of output tiles.
    uint32_t reduction_tiles = 0;
    if (params.splits_ > 1) {
      reduction_tiles = params.units_per_problem_;
    }
    else if (params.requires_separate_reduction()) {
      reduction_tiles = params.sk_tiles_ * Params::max_peers_per_tile(params.sk_units_, params.sk_tiles_);
    }
    else {
      reduction_tiles = params.sk_tiles_;
    }

    auto reduction_workspace_size = Params::get_reduction_workspace_size(
      reduction_tiles, to_gemm_coord(TileShape{}), sizeof_bits<ElementAccumulator>::value, num_accumulator_mtxs);
    BarrierType* lock_workspace = reinterpret_cast<BarrierType*>(
      reinterpret_cast<uint8_t*>(params.reduction_workspace_) + reduction_workspace_size);

    if (work_tile_info.is_reduction_unit()) {
      plus<AccumulatorArrayT> add_fragments;
      auto peer_offset = size(accumulators) * num_barriers * BarrierManager::ThreadCount;

      // Wait until the peers collaborating on this output tile have all written
      // their accumulators to workspace.
      uint32_t num_peers = last_peer_id - first_peer_id + 1;
      BarrierManager::wait_eq(barrier_idx, lock_workspace, barrier_group_thread_idx, lock_idx, num_peers);

      // Load the first peer's data
      BlockStripedReduceT::load(*accumulator_array, reduction_workspace_array, barrier_group_thread_idx);

      for (int i = 1; i < num_peers; ++i) {
        // Load peer fragment
        AccumulatorArrayT addend_fragment;
        auto peer_reduction_workspace = reinterpret_cast<AccumulatorArrayT*>(group_reduction_workspace + (i * peer_offset));

        BlockStripedReduceT::load(addend_fragment, peer_reduction_workspace, barrier_group_thread_idx);

        // Add peer fragment
        *accumulator_array = add_fragments(*accumulator_array, addend_fragment);
      }
    }
    else if (!compute_epilogue(work_tile_info, params)) {
      if (params.requires_separate_reduction() || work_tile_info.K_idx == 0) {
        // The first peer initializes the workspace partials in the non-separate-reduction case,
        // and all peers write to their own location in workspace when using separate reduction
        BlockStripedReduceT::store(reduction_workspace_array, *accumulator_array, barrier_group_thread_idx);
      }
      else {
        // Wait until the preceding split added its accumulators
        BarrierManager::wait_eq(barrier_idx, lock_workspace, barrier_group_thread_idx, lock_idx, work_tile_info.K_idx);

        // Perform reduction in workspace
        BlockStripedReduceT::reduce(reduction_workspace_array, *accumulator_array, barrier_group_thread_idx);
      }

      // If separate reduction is being performed, each participating stream-K unit increments the barrier
      // by only 1. Otherwise, increment by the K tile count that this unit has processed.
      int32_t increment = params.requires_separate_reduction() ? 1 : work_tile_info.k_tile_count;

      // Signal our arrival
      BarrierManager::arrive_inc(barrier_idx, lock_workspace, barrier_group_thread_idx, lock_idx, increment);
    }
    else {
      if (params.reduction_mode_ == ReductionMode::Deterministic) {
        // Wait until the preceding split added its accumulators
        BarrierManager::wait_eq(barrier_idx, lock_workspace, barrier_group_thread_idx, lock_idx, work_tile_info.K_idx);
      }
      else {
        // Wait unitl the first split has stored its accumulators
        BarrierManager::wait_lt(barrier_idx, lock_workspace, barrier_group_thread_idx, lock_idx, 1);
      }

      // The block computing the final split for the tile adds previously-reduced partials
      // to its accumulators and computes the epilogue.
      BlockStripedReduceT::load_add(*accumulator_array, reduction_workspace_array, barrier_group_thread_idx);
    }
  }

  // Returns whether the block assigned this work should compute the epilogue for the corresponding
  // output tile. For the case of stream-K, this should only occur if the work is marked as the final split.
  CUTLASS_HOST_DEVICE
  static bool
  compute_epilogue(WorkTileInfo const& work_tile_info, Params const& params) {
    // `is_final_split` will be set to `true` for the following scenarios, all of which must compute the epilogue:
    //  1. The tile is computed in data-parallel mode
    //  2. The tile is computed in split-/stream-K mode and this work unit represents the final split of the tile
    //  3. The tile is computed in split-/stream-K mode and separate reduction is used, and this is a separate reduction unit
    return work_tile_info.is_valid() &&
            (work_tile_info.is_final_split(params.divmod_tiles_per_output_tile_.divisor) &&
             !params.requires_separate_reduction()) || work_tile_info.is_separate_reduction;
  }

  // Returns the linearized index of the output tile corresponding to the tile with offset [L, M, K]
  CUTLASS_DEVICE
  static int
  output_tile_index(Params const& params, WorkTileInfo const& work_tile_info) {
    uint64_t linear_idx_in_batch = UnderlyingScheduler::get_linear_idx_from_m_and_n(
      work_tile_info.M_idx, work_tile_info.N_idx,
      params.divmod_cluster_shape_major_,
      params.divmod_cluster_shape_minor_,
      params.divmod_cluster_blk_major_,
      params.log_swizzle_size_,
      params.raster_order_
    );

    uint64_t tiles_mn = params.divmod_batch_.divisor;
    return tiles_mn * work_tile_info.L_idx + linear_idx_in_batch;
  }

  template <class ProblemShape, class ElementAccumulator>
  static size_t
  get_workspace_size(
    Arguments const& args,
    ProblemShape problem_shape,
    KernelHardwareInfo const& hw_info,
    uint32_t mma_warp_groups,
    const uint32_t epilogue_subtile = 1) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape, 1);

    ClusterShape cluster_shape;
    TileShape tile_shape;

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);
    uint32_t k_tile_per_output_tile = cute::size(cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    return Params::get_workspace_size(
      problem_blocks,
      k_tile_per_output_tile,
      to_gemm_coord(tile_shape),
      to_gemm_coord(cluster_shape),
      hw_info,
      args.splits,
      args.max_swizzle_size,
      args.raster_order,
      args.decomposition_mode,
      mma_warp_groups,
      sizeof_bits<BarrierType>::value,
      sizeof_bits<ElementAccumulator>::value,
      epilogue_subtile
    );
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status
  initialize_workspace(
    Arguments const& args,
    void* workspace,
    cudaStream_t stream,
    ProblemShape const& problem_shape,
    KernelHardwareInfo const& hw_info,
    uint32_t mma_warp_groups,
    const uint32_t epilogue_subtile = 1) {

    auto problem_shape_mnkl = cute::append<4>(problem_shape, 1);

    ClusterShape cluster_shape;
    TileShape tile_shape;

    dim3 problem_blocks = get_tiled_cta_shape_mnl(problem_shape_mnkl, tile_shape, cluster_shape);
    uint32_t k_tile_per_output_tile = cute::size(cute::ceil_div(cute::shape<2>(problem_shape_mnkl), cute::shape<2>(TileShape{})));

    return Params::initialize_workspace(
      workspace,
      stream,
      problem_blocks,
      k_tile_per_output_tile,
      to_gemm_coord(tile_shape),
      to_gemm_coord(cluster_shape),
      hw_info,
      args.splits,
      args.max_swizzle_size,
      args.raster_order,
      args.decomposition_mode,
      mma_warp_groups,
      sizeof_bits<BarrierType>::value,
      sizeof_bits<ElementAccumulator>::value,
      epilogue_subtile
    );
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE
  static int
  get_work_k_tile_count(WorkTileInfo const& work_tile_info, ProblemShape, TileShape) {
    return work_tile_info.k_tile_count;
  }

  CUTLASS_HOST_DEVICE
  static uint32_t
  get_work_k_tile_start(WorkTileInfo const& work_tile_info) {
    return work_tile_info.K_idx;
  }

private:

  struct GroupKTiling {
    uint64_t k_tiles;
    uint64_t k_tiles_per_unit;
  };
  
  // For stream-K, stream-k tiles are split into groups, compute how the k-tiles within 
  //  this group should be divided
  CUTLASS_DEVICE
  static GroupKTiling
  compute_group_k_tiling(Params const& params, uint64_t const& group_idx) {
    // Determine whether we are in a "big group" that will process an additional
    // stream-K cluster tile.
    auto sk_cluster_tiles = params.div_cluster_size(params.sk_tiles_);
    auto sk_cluster_tiles_in_group = params.divmod_sk_groups_.divide(sk_cluster_tiles);
    if (group_idx < params.big_groups_) {
      ++sk_cluster_tiles_in_group;
    }

    auto sk_tiles_in_group = sk_cluster_tiles_in_group * params.get_cluster_size();
    auto k_tiles_in_group = sk_tiles_in_group * params.divmod_tiles_per_output_tile_.divisor;
    auto k_tiles_per_unit_in_group = params.divmod_sk_units_per_group_.divide(k_tiles_in_group); 

    return {k_tiles_in_group, k_tiles_per_unit_in_group};
  }

  CUTLASS_DEVICE
  static uint32_t
  compute_big_units(Params const& params, GroupKTiling const& group_k_tiling, bool is_split_k) {
    if (is_split_k) {
      return params.big_units_;
    } else { // Stream-K / Data-Parallel
      auto sk_units_per_group = params.divmod_sk_units_per_group_.divisor;
      // Determine whether we are in a "big unit" within the group, that will process
      // an additional K chunk in the group.
      auto big_units_in_group = params.div_cluster_size(
        group_k_tiling.k_tiles - (group_k_tiling.k_tiles_per_unit * sk_units_per_group));
      return big_units_in_group;
    }
  }

  // Sets the current stream-K work to compute within work_tile_info. If new_unit is true, work_tile_info
  // is populated as a new unit of work. Otherwise, state existing in work_tile_info (e.g., remaining
  // iterations) is used to find the next tile in the current work unit.
  CUTLASS_DEVICE
  static void
  assign_work(
    Params const& params,
    uint64_t linear_idx,
    WorkTileInfo& work_tile_info) {

    work_tile_info.linear_idx = linear_idx;

    uint64_t output_tile_id = linear_idx;
    if (linear_idx >= params.units_per_problem_ * params.splits_) {
      // Separate-reduction work
      auto cluster_size = params.get_cluster_size();
      // Divide up the linearized separate reduction units into clusters
      auto cluster_linear_reduction_unit_idx = params.div_cluster_size((linear_idx - params.units_per_problem_));
      uint64_t cluster_tile_idx, epi_subtile_idx;
      params.divmod_epilogue_subtile_(cluster_tile_idx, epi_subtile_idx, cluster_linear_reduction_unit_idx);
      // Bring the linearized tile ID back into the space of tiles, rather than clusters
      output_tile_id = cluster_tile_idx * cluster_size;

      work_tile_info.setup_separate_reduction(epi_subtile_idx);
    }
    else if (linear_idx >= params.sk_units_ && params.splits_ == 1) {
      // Data-parallel work
      output_tile_id = linear_idx - params.sk_units_ + params.sk_tiles_;
      work_tile_info.K_idx = 0;
      work_tile_info.k_tile_count = params.divmod_tiles_per_output_tile_.divisor;
      work_tile_info.k_tile_remaining = params.divmod_tiles_per_output_tile_.divisor;
    }
    else {
      // In the CUTLASS 2.x implementation of stream K, stream-K work is assigned to each stream-K
      // threadblock individually. For the most part, the set of K iterations corresponding to stream-K
      // work was divided amongst stream-K threadblocks, and a threadblock determined which tile
      // it would compute a (potentially-partial) output tile for based on the space of k iterations
      // assigned to it. This often results in stream-K threadblocks processing tiles with different
      // offsets in the K dimension from one another. This can reduce locality, but is lmitied to the
      // (generally few) waves of threadblocks assigned to compute stream-K work.
      //
      // With the introduction of threadblock clusters, there is additional benefit to maintaining
      // locality in the K dimension: shared portions of operands can be multicasted to threadblocks
      // within a cluster. Thus, we would like to ensure that the assignment of stream-K work to
      // threadblocks respects the ability to perform multicasting.
      //
      // To do so, we divide up the linearized stream-K units into clusters and share the same K
      // offsets for work within clusters.

      auto cluster_linear_work_idx = params.div_cluster_size(linear_idx);

      uint64_t group_idx;
      params.divmod_sk_groups_(cluster_linear_work_idx, group_idx, cluster_linear_work_idx);

      uint64_t split;
      params.divmod_clusters_mnl_(split, cluster_linear_work_idx, cluster_linear_work_idx);

      auto group_k_tiling = compute_group_k_tiling(params, group_idx);

      bool is_split_k = params.splits_ > 1;
      auto big_unit_cmp_lhs = is_split_k ? split : cluster_linear_work_idx;
      auto big_unit_cmp_rhs = compute_big_units(params, group_k_tiling, is_split_k);
      auto tiles_per_output = params.divmod_tiles_per_output_tile_.divisor;
      auto linear_idx_mult = is_split_k ? tiles_per_output : group_k_tiling.k_tiles_per_unit;
      auto k_tiles_per_split = is_split_k ? params.k_tiles_per_sk_unit_ : group_k_tiling.k_tiles_per_unit;

      // Determine the starting k iteration computed by this stream-K work unit
      uint32_t unit_iter_start = (linear_idx_mult * cluster_linear_work_idx) +
                                 (k_tiles_per_split * split);

      // Adjust the starting position and number of k iterations for "big units," which
      // compute one extra iteration. If there are any big units, they will be the first
      // in the linearized ID space.
      auto k_tiles_in_my_split = k_tiles_per_split;
      if (big_unit_cmp_lhs < big_unit_cmp_rhs) {
        // Since the "big units" are the first units in the linearized ID space, each
        // of the units preceding this big unit computed one extra iteration. Thus,
        // we must offset our start iteration by the number of units that precede
        // the current unit in the linearized ID space.
        unit_iter_start += big_unit_cmp_lhs;
        ++k_tiles_in_my_split;
      }
      else {
        // Increment by one for each of the big clusters (since all big units precede this unit)
        unit_iter_start += big_unit_cmp_rhs;
      }

      if (!is_split_k) {
        // Adjust the unit starting position and number of tiles to avoid
        // computing splits of size less than min_iters_per_sk_unit_
        int unused, start_tile_k_tile;
        params.divmod_tiles_per_output_tile_(unused, start_tile_k_tile, unit_iter_start);
        if (start_tile_k_tile < Params::min_iters_per_sk_unit_) {
          // Starting K tile is in range [0, Params::min_iters_per_sk_unit_), which means that another
          // stream-K unit will be computing a split with fewer than Params::min_iters_per_sk_unit_ K tiles.
          // Adjust our work to take over these K tiles.
          unit_iter_start -= start_tile_k_tile;
          k_tiles_in_my_split += start_tile_k_tile;
        }
        else if (start_tile_k_tile > (params.divmod_tiles_per_output_tile_.divisor - Params::min_iters_per_sk_unit_)) {
          // Starting K tile is within the final Params::min_iters_per_sk_unit_ K tiles of some output tile,
          // which means that this unit will compute a split with fewer than Params::min_iters_per_sk_unit_ K tiles.
          // Adjust our work to shed these K tiles to a neighboring stream-K unit that will compute more consecutive K tiles.
          auto adjustment_tiles = (params.divmod_tiles_per_output_tile_.divisor - start_tile_k_tile);
          unit_iter_start += adjustment_tiles;
          k_tiles_in_my_split -= adjustment_tiles;
        }
      }

      if (work_tile_info.k_tile_count == 0) {
        // This is a new unit

        if (!is_split_k) {
          //
          // Adjust the unit ending position and number of tiles to avoid
          // computing splits of size less than min_iters_per_sk_unit_
          //

          // Begin by assuming that no adjustment is needed
          auto initial_unit_iter_end = unit_iter_start + k_tiles_in_my_split;

          int unused, end_tile_k_tile;
          params.divmod_tiles_per_output_tile_(unused, end_tile_k_tile, initial_unit_iter_end);

          if (end_tile_k_tile < Params::min_iters_per_sk_unit_) {
            // Ending K tile is within the first Params::min_iters_per_sk_unit_ K tiles of some output tile,
            // which means that this unit will compute a split with fewer than Params::min_iters_per_sk_unit_ K tiles.
            // Adjust our work to shed these K tiles to a neighboring stream-K unit that will compute more consecutive K tiles.
            k_tiles_in_my_split -= end_tile_k_tile;
          }
          else if (end_tile_k_tile > (params.divmod_tiles_per_output_tile_.divisor - Params::min_iters_per_sk_unit_)) {
            // Ending K tile is within the final Params::min_iters_per_sk_unit_ K tiles of some output tile,
            // which means that some other unit will compute a split with fewer than Params::min_iters_per_sk_unit_ K tiles.
            // Adjust our work to take on these K tiles.
            k_tiles_in_my_split += (params.divmod_tiles_per_output_tile_.divisor - end_tile_k_tile);
          }
        }

        work_tile_info.k_tile_remaining = k_tiles_in_my_split;
      }

      uint32_t unit_iter_end = unit_iter_start + work_tile_info.k_tile_remaining - 1;

      // Find the output tile corresponding to the final k tile covered by this
      // work unit. Stream-K work units will work backwards in terms of the tiles they
      // are responsible computing. This is beneficial because the final (partial)
      // tile computed by a stream-K block is typically the beginning of the output
      // tile, while the beginning (partial) tile is typically the ending of another
      // output tile. Since ending portions of an output tile must reduce across
      // other work units computing portions of that output tile, it is preferable
      // for them to be computed later, so as to reduce the likelihood of blocking
      // on other work.

      auto output_tile_id_in_group = params.divmod_tiles_per_output_tile_.divide(unit_iter_end);
      uint32_t output_tile_iter_start = output_tile_id_in_group * params.divmod_tiles_per_output_tile_.divisor;
      uint32_t output_tile_iter_end = output_tile_iter_start + params.divmod_tiles_per_output_tile_.divisor;

      // Convert the output tile from the linearized space within each group to the
      // overall linearized space.
      output_tile_id = (output_tile_id_in_group * params.divmod_sk_groups_.divisor) + group_idx;

      // Bring the linearized tile ID back into the space of tiles, rather than clusters
      output_tile_id *= params.get_cluster_size();

      auto [cta_m_in_cluster, cta_n_in_cluster, _] = cute::block_id_in_cluster();

      // The final linearized tile ID is in units of the cluster dimension over which we rasterize.
      if (params.raster_order_ == RasterOrder::AlongN) {
        output_tile_id += cta_n_in_cluster * params.divmod_cluster_shape_minor_.divisor;
      }
      else {
        output_tile_id += cta_m_in_cluster * params.divmod_cluster_shape_minor_.divisor;
      }

      // The unit's starting k iteration in the current tile is either the starting
      // iteration for the tile as a whole, or the starting k iteration for the unit
      // as a whole (if the latter is greater than the former).
      uint32_t tile_iter_start = max(output_tile_iter_start, unit_iter_start);

      // Similarly, the unit's ending k iteration (exclusive) is either the end of
      // the current tile it is assigned, or the ending iteration of the unit as a whole
      // (if the latter is less than the former).
      uint32_t tile_iter_end = min(output_tile_iter_end, unit_iter_end + 1);

      // Set the k offset to be the starting k tile for this output tile
      work_tile_info.K_idx = static_cast<int32_t>(tile_iter_start - output_tile_iter_start);
      work_tile_info.k_tile_count = tile_iter_end - tile_iter_start;
    }

    uint64_t work_idx_l, remainder;
    params.divmod_batch_(work_idx_l, remainder, output_tile_id);

    uint64_t cta_per_grid_dim = params.divmod_cluster_shape_minor_.divide(remainder);

    auto [work_idx_m, work_idx_n] = UnderlyingScheduler::get_work_idx_m_and_n(
                                          cta_per_grid_dim,
                                          params.divmod_cluster_shape_major_,
                                          params.divmod_cluster_shape_minor_,
                                          params.divmod_cluster_blk_major_,
                                          params.log_swizzle_size_,
                                          params.raster_order_
                                        );

    // Set the M, N, and L block offsets
    work_tile_info.M_idx = work_idx_m;
    work_tile_info.N_idx = work_idx_n;
    work_tile_info.L_idx = static_cast<int32_t>(work_idx_l);
  }

  // Returns the starting and ending peer ID of this tile
  CUTLASS_HOST_DEVICE
  static auto
  tile_peer_range(Params const& params, int32_t linear_idx, uint32_t tile_idx, uint32_t relative_cur_k_tile) {
    bool is_split_k = params.splits_ > 1;
    auto tile_idx_in_cluster_path = params.div_cluster_size(tile_idx);
    auto cluster_linear_work_idx = params.div_cluster_size(linear_idx);

    auto start_k_tile = params.divmod_tiles_per_output_tile_.divisor * tile_idx_in_cluster_path;
    auto end_k_tile = start_k_tile + params.divmod_tiles_per_output_tile_.divisor - 1;
    auto cur_k_tile = relative_cur_k_tile + start_k_tile;

    auto group_idx = params.divmod_sk_groups_.divide(cluster_linear_work_idx);
    auto group_k_tiling = compute_group_k_tiling(params, group_idx);
    auto big_units = compute_big_units(params, group_k_tiling, is_split_k);
    auto big_unit_k_tiles = big_units * (params.k_tiles_per_sk_unit_ + 1);

    auto adjust_unit = [&](uint32_t k_tile, uint32_t unit_idx, uint32_t unit_k_start, uint32_t k_tiles_per_unit) {
      auto unit_k_end = unit_k_start + k_tiles_per_unit;
      if (k_tile - start_k_tile < Params::min_iters_per_sk_unit_ &&
          unit_k_end - start_k_tile < Params::min_iters_per_sk_unit_) {
        // k_tile is within the first min_iters_per_sk_unit_ K tiles of this output tile,
        // and the stream-K unit computes fewer than min_iters_per_sk_unit_ K tiles for this
        // output tile. This work will thus be subsumed by the next stream-K unit.
        ++unit_idx;
      }

      if (end_k_tile + 1 - k_tile < Params::min_iters_per_sk_unit_ &&
          end_k_tile + 1 - unit_k_start < Params::min_iters_per_sk_unit_) {
        // k_tile is within the last min_iters_per_sk_unit_ K tiles of this output tile,
        // and the stream-K unit computes fewer than min_iters_per_sk_unit_ K tiles for this
        // output tile. This work will thus be subsumed by the previous stream-K unit.
        --unit_idx;
      }

      return unit_idx;
    };

    // Lambda to find the ID of the stream-K unit that computes this K tile
    auto find_unit = [&](uint32_t k_tile) {
      if (k_tile < big_unit_k_tiles) {
        // The tile is within the "big unit range"
        auto k_tiles_per_unit = params.k_tiles_per_sk_unit_ + 1;
        auto unit_idx = k_tile / k_tiles_per_unit;
        return static_cast<uint64_t>(adjust_unit(k_tile, unit_idx, unit_idx * k_tiles_per_unit, k_tiles_per_unit));
      }
      else {
        // The tile is after the "big unit range." Account for this by finding the "normal unit"
        // that it belongs to, and then offsetting by the number of big units
        auto k_tiles_per_unit = params.k_tiles_per_sk_unit_;
        auto unit_idx = ((k_tile - big_unit_k_tiles) / params.k_tiles_per_sk_unit_) + (big_units);
        return static_cast<uint64_t>(adjust_unit(k_tile, unit_idx,  unit_idx * k_tiles_per_unit + big_units, k_tiles_per_unit));
      }
    };

    return cute::make_tuple(find_unit(start_k_tile), find_unit(cur_k_tile), find_unit(end_k_tile));
  }
};

} // namespace cutlass::gemm::kernel::detail
