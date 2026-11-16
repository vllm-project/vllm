#pragma once

// Error codes and the state the host API shares.

#include <ATen/hip/HIPContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "kernels/all_reduce.cuh"
#include "kernels/all_reduce_add_attn_res_rms_norm.cuh"
#include "kernels/all_reduce_add_rms_norm.cuh"
#include "kernels/common.cuh"
#include "kernels/reduce_scatter.cuh"

namespace hip_custom_communicator {

namespace Quantization {
constexpr int exact = 0;
constexpr int int4 = 1;
}  // namespace Quantization

constexpr int kPackBytes = 16;
constexpr std::array<at::ScalarType, 2> kDtypes = {at::ScalarType::Half,
                                                   at::ScalarType::BFloat16};
constexpr std::array<int, 3> kWorlds = {2, 4, 8};

enum class Error : int {
  not_contiguous = 0,
  dtype_not_built = 1,
  world_not_built = 2,
  row_not_packs = 3,
  partials_too_small = 4,
  grid_not_resident = 5,
  staging_too_small = 6,
  device_not_built = 7,
  weight_not_built = 9,
  no_such_kernel = 10,
  not_built = 12,
  launch_out_of_range = 13,
  field_not_positive = 14,
  field_not_this_templates = 15,
  inner_stride_not_one = 17,
  row_stride_not_packs = 18,
  no_such_quantization = 19,
  quantization_not_supported = 20,
  block_too_large = 21,
  lds_too_large = 22,
  scratch_too_large = 23,
  unfenced_too_small = 24,
  pack_out_of_range = 25,
  hip_failed = 26,
  not_on_device = 27,
  shape_mismatch = 28,
  dtype_mismatch = 29,
  not_aligned = 30,
  inputs_overlap_outputs = 31,
  field_out_of_range = 32,
  view_past_region = 33,
  tables_full = 34,
  ranks_disagree = 35,
  not_open = 36,
  field_missing = 37,
  no_such_group = 38,
  exchange_failed = 39,
  peer_not_accessible = 40,
  peers_not_opened = 41,
};

enum class AllReduceAddRmsNormAlgorithm : int {
  one_shot_pull,
  two_shot_pull_rows,
  two_shot_pull_columns,
  two_shot_push,
  one_shot_pull_split,
  one_shot_push_split,
  one_shot_push_staged_split,
  two_shot_pull_split,
  two_shot_push_split,
};

enum class AddRmsNormAlgorithm : int { rows };
enum class AddAttnResRmsNormAlgorithm : int { rows };

template <typename ALGORITHM>
struct Instance {
  ALGORITHM algorithm;
  int block_x, block_y, block_z, waves_per_eu, tile_m, tile_n, tile_k;
  int quantization, pack_bytes;
  int world, wave_size;
  at::ScalarType dtype, weight_dtype;
  bool contiguous, chunked, tail, quantize_int4, residual;
  void* kernel;
};

template <typename T, typename W>
using AllReduceKernel =
    decltype(&all_reduce<AllReduceAlgorithm{}, false, T, 2, 1, 1, 64, 64, 16, 1,
                         false, false>);
template <typename T, typename W>
using AllReduceAddRmsNormKernel =
    decltype(&all_reduce_add_rms_norm<AllReduceAddRmsNormKernelAlgorithm{},
                                      false, false, T, W, 2, 1, 1, 64, 64, 16,
                                      1, false, false>);
template <typename T, typename W>
using AllReduceAddAttnResRmsNormKernel =
    decltype(&all_reduce_add_attn_res_rms_norm<
             AllReduceAddAttnResRmsNormAlgorithm{}, false, T, 2, 1, 1, 1, 64,
             64, 16, 1, false, false>);
template <typename T, typename W>
using AddRmsNormKernel =
    decltype(&add_rms_norm<false, T, W, 1, 1, 64, 64, 16, 1, false>);
template <typename T, typename W>
using AddAttnResRmsNormKernel =
    decltype(&add_attn_res_rms_norm<T, 1, 1, 1, 64, 64, 16, 1, false>);

#include "hip_custom_communicator_instances.inc"
#include "hip_custom_communicator_builds.inc"

struct InputTables {
  void** device = nullptr;
  int64_t slots = 0;

  int64_t next_slot = 1;

  std::vector<std::pair<int64_t, const void*>> pending;

  std::map<std::vector<char>, char*> opened;
};

struct SymmetricMemory {
  int64_t max_blocks = 0, unfenced_blocks = 0, unfenced_slot_bytes = 0;

  int64_t signals_bytes = 0, staging_bytes = 0, partials_bytes = 0,
          scales_bytes = 0, unfenced_bytes = 0;

  uint32_t* signals = nullptr;
  char* staging = nullptr;
  char* partials = nullptr;
  float* scales = nullptr;
  char* unfenced = nullptr;

  uint32_t* const* signals_ptr_array = nullptr;
  char* const* staging_ptr_array = nullptr;
  char* const* partials_ptr_array = nullptr;
  float* const* scales_ptr_array = nullptr;
  char* const* unfenced_ptr_array = nullptr;

  std::vector<char*> peers;
};

struct DeviceProperties {
  int index = 0;
  std::string arch;
  int wave_size = 0;
  int64_t clock_khz = 0;
  int64_t lds_bytes = 0, compute_units = 0, threads_a_cu = 0,
          scratch_a_lane = 0;
};

struct CommsState {
  c10::intrusive_ptr<c10d::ProcessGroup> group;
  double timeout_seconds = 0;
  int rank = 0;
  int world = 0;
  int64_t max_rows = 0;
  uint64_t timeout_ticks = 0;
  DeviceProperties device;
  SymmetricMemory symmetric;

  int32_t* unfenced_epochs = nullptr;
  int64_t unfenced_epoch_words = 0;
  uint32_t* status_device = nullptr;
  InputTables input_tables;
};

}  // namespace hip_custom_communicator

using OpenWire = std::tuple<std::optional<int64_t>, int64_t, std::string>;

using ErrorCode = std::optional<int64_t>;
