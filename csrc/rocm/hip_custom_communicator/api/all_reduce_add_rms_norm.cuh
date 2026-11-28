#pragma once

// Host side of the all-reduce with RMSNorm: checks the call, then launches its
// kernel.

#include "../../ops.h"
#include "../types.cuh"
#include "../utils.h"
#include "../kernels/all_reduce.cuh"
#include "../kernels/all_reduce_add_rms_norm.cuh"

ErrorCode hip_custom_communicator_all_reduce_add_rms_norm(
    int64_t state, torch::Tensor& input,
    const std::optional<torch::Tensor>& residual, torch::Tensor& weight,
    torch::Tensor& output, const std::optional<torch::Tensor>& residual_output,
    double eps, const std::string& algorithm_name, int64_t grid_x,
    int64_t grid_y, int64_t grid_z, int64_t block_x, int64_t block_y,
    int64_t block_z, int64_t waves_per_eu, int64_t tile_m, int64_t tile_n,
    const std::string& quantization_name, int64_t pack_bytes,
    std::optional<int64_t> add_rms_norm_grid_x,
    std::optional<int64_t> add_rms_norm_grid_y,
    std::optional<int64_t> add_rms_norm_grid_z,
    std::optional<int64_t> add_rms_norm_block_x,
    std::optional<int64_t> add_rms_norm_block_y,
    std::optional<int64_t> add_rms_norm_block_z,
    std::optional<int64_t> add_rms_norm_waves_per_eu,
    std::optional<int64_t> add_rms_norm_tile_m,
    std::optional<int64_t> add_rms_norm_tile_n,
    std::optional<int64_t> add_rms_norm_pack_bytes) {
  using namespace hip_custom_communicator;
  using namespace hip_custom_communicator::internal;

  if (state == 0) return error_code(Error::not_open);
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  if (!input.is_cuda() || !weight.is_cuda() || !output.is_cuda())
    return error_code(Error::not_on_device);

  if (input.dim() != 2 || input.size(0) < 1 || input.size(1) < 1 ||
      output.sizes() != input.sizes() || weight.dim() != 1 ||
      weight.numel() != input.size(1))
    return error_code(Error::shape_mismatch);
  if (output.scalar_type() != input.scalar_type())
    return error_code(Error::dtype_mismatch);

  const bool add = residual.has_value();
  if (residual_output.has_value() != add)
    return error_code(Error::shape_mismatch);
  if (add && (!residual->is_cuda() || !residual_output->is_cuda()))
    return error_code(Error::not_on_device);
  if (add && (residual->sizes() != input.sizes() ||
              residual_output->sizes() != input.sizes()))
    return error_code(Error::shape_mismatch);
  if (add && (residual->scalar_type() != input.scalar_type() ||
              residual_output->scalar_type() != input.scalar_type()))
    return error_code(Error::dtype_mismatch);
  for (const int64_t v : {waves_per_eu, tile_m, tile_n, pack_bytes, grid_x,
                          grid_y, grid_z, block_x, block_y, block_z})
    if (v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max())
      return error_code(Error::field_out_of_range);

  AllReduceAddRmsNormAlgorithm op_algorithm;
  if (algorithm_name == "one_shot_pull")
    op_algorithm = AllReduceAddRmsNormAlgorithm::one_shot_pull;
  else if (algorithm_name == "two_shot_pull_rows")
    op_algorithm = AllReduceAddRmsNormAlgorithm::two_shot_pull_rows;
  else if (algorithm_name == "two_shot_pull_columns")
    op_algorithm = AllReduceAddRmsNormAlgorithm::two_shot_pull_columns;
  else if (algorithm_name == "two_shot_push")
    op_algorithm = AllReduceAddRmsNormAlgorithm::two_shot_push;
  else if (algorithm_name == "one_shot_pull_split")
    op_algorithm = AllReduceAddRmsNormAlgorithm::one_shot_pull_split;
  else if (algorithm_name == "one_shot_push_split")
    op_algorithm = AllReduceAddRmsNormAlgorithm::one_shot_push_split;
  else if (algorithm_name == "one_shot_push_staged_split")
    op_algorithm = AllReduceAddRmsNormAlgorithm::one_shot_push_staged_split;
  else if (algorithm_name == "two_shot_pull_split")
    op_algorithm = AllReduceAddRmsNormAlgorithm::two_shot_pull_split;
  else if (algorithm_name == "two_shot_push_split")
    op_algorithm = AllReduceAddRmsNormAlgorithm::two_shot_push_split;
  else
    return error_code(Error::no_such_kernel);

  bool split = false;
  AllReduceAddRmsNormKernelAlgorithm algorithm{};
  AllReduceAlgorithm all_reduce_algorithm{};
  switch (op_algorithm) {
    case AllReduceAddRmsNormAlgorithm::one_shot_pull:
      algorithm = AllReduceAddRmsNormKernelAlgorithm::one_shot_pull;
      break;
    case AllReduceAddRmsNormAlgorithm::two_shot_pull_rows:
      algorithm = AllReduceAddRmsNormKernelAlgorithm::two_shot_pull_rows;
      break;
    case AllReduceAddRmsNormAlgorithm::two_shot_pull_columns:
      algorithm = AllReduceAddRmsNormKernelAlgorithm::two_shot_pull_columns;
      break;
    case AllReduceAddRmsNormAlgorithm::two_shot_push:
      algorithm = AllReduceAddRmsNormKernelAlgorithm::two_shot_push;
      break;
    case AllReduceAddRmsNormAlgorithm::one_shot_pull_split:
      split = true, all_reduce_algorithm = AllReduceAlgorithm::one_shot_pull;
      break;
    case AllReduceAddRmsNormAlgorithm::one_shot_push_split:
      split = true, all_reduce_algorithm = AllReduceAlgorithm::one_shot_push;
      break;
    case AllReduceAddRmsNormAlgorithm::one_shot_push_staged_split:
      split = true,
      all_reduce_algorithm = AllReduceAlgorithm::one_shot_push_staged;
      break;
    case AllReduceAddRmsNormAlgorithm::two_shot_pull_split:
      split = true, all_reduce_algorithm = AllReduceAlgorithm::two_shot_pull;
      break;
    case AllReduceAddRmsNormAlgorithm::two_shot_push_split:
      split = true, all_reduce_algorithm = AllReduceAlgorithm::two_shot_push;
      break;
  }
  for (const std::optional<int64_t>& field :
       {add_rms_norm_grid_x, add_rms_norm_grid_y, add_rms_norm_grid_z,
        add_rms_norm_block_x, add_rms_norm_block_y, add_rms_norm_block_z,
        add_rms_norm_waves_per_eu, add_rms_norm_tile_m, add_rms_norm_tile_n,
        add_rms_norm_pack_bytes}) {
    if (!split && field) return error_code(Error::field_not_this_templates);
    if (split && !field) return error_code(Error::field_missing);
  }
  if (split)
    for (const int64_t v :
         {*add_rms_norm_grid_x, *add_rms_norm_grid_y, *add_rms_norm_grid_z,
          *add_rms_norm_block_x, *add_rms_norm_block_y, *add_rms_norm_block_z,
          *add_rms_norm_waves_per_eu, *add_rms_norm_tile_m,
          *add_rms_norm_tile_n, *add_rms_norm_pack_bytes})
      if (v < std::numeric_limits<int>::min() ||
          v > std::numeric_limits<int>::max())
        return error_code(Error::field_out_of_range);
  const void* input_ptr = input.data_ptr();
  const int64_t input_stride_m = input.stride(0),
                input_stride_n = input.stride(1);
  const int64_t input_size_m = input.size(0), input_size_n = input.size(1);
  const void* weight_ptr = weight.data_ptr();
  const int64_t weight_stride_n = weight.stride(0),
                weight_size_n = weight.size(0);

  const void* residual_ptr = add ? residual->data_ptr() : nullptr;
  const int64_t residual_stride_m = add ? residual->stride(0) : 0,
                residual_stride_n = add ? residual->stride(1) : 0,
                residual_size_m = add ? residual->size(0) : 0,
                residual_size_n = add ? residual->size(1) : 0;
  void* residual_output_ptr = add ? residual_output->data_ptr() : nullptr;
  const int64_t residual_output_stride_m = add ? residual_output->stride(0) : 0,
                residual_output_stride_n = add ? residual_output->stride(1) : 0,
                residual_output_size_m = add ? residual_output->size(0) : 0,
                residual_output_size_n = add ? residual_output->size(1) : 0;
  void* output_ptr = output.data_ptr();
  const int64_t output_stride_m = output.stride(0),
                output_stride_n = output.stride(1);
  const int64_t output_size_m = output.size(0), output_size_n = output.size(1);
  const at::ScalarType dtype = input.scalar_type();
  const at::ScalarType weight_dtype = weight.scalar_type();
  const int64_t e = input.element_size();

  const int64_t m = input_size_m, n = input_size_n;
  const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y),
                  static_cast<unsigned>(grid_z));
  const dim3 block(static_cast<unsigned>(block_x),
                   static_cast<unsigned>(block_y),
                   static_cast<unsigned>(block_z));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();

  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const bool int4_config = quantization_name == "int4";

  if (grid_y != 1 || grid_z != 1 || grid_x < 1 ||
      grid_x > s.symmetric.max_blocks)
    return error_code(Error::launch_out_of_range);
  if (tile_m < 1 || tile_n < 1) return error_code(Error::field_not_positive);

  if (pack_bytes < 2 || pack_bytes > kPackBytes ||
      (pack_bytes & (pack_bytes - 1)) != 0)
    return error_code(Error::pack_out_of_range);
  if (int4_config && pack_bytes != kPackBytes)
    return error_code(Error::quantization_not_supported);

  if (std::find(kDtypes.begin(), kDtypes.end(), dtype) == kDtypes.end())
    return error_code(Error::dtype_not_built);
  if (pack_bytes < e) return error_code(Error::pack_out_of_range);
  if (n * e % pack_bytes != 0) return error_code(Error::row_not_packs);

  if (reinterpret_cast<uintptr_t>(weight_ptr) %
          (kPackBytes / e * weight.element_size()) !=
      0)
    return error_code(Error::not_aligned);

  if (input_stride_n != 1 || output_stride_n != 1 || weight_stride_n != 1 ||
      (add && (residual_stride_n != 1 || residual_output_stride_n != 1)))
    return error_code(Error::inner_stride_not_one);
  if (input_stride_m * e % pack_bytes != 0 ||
      output_stride_m * e % pack_bytes != 0 ||
      (add && (residual_stride_m * e % pack_bytes != 0 ||
               residual_output_stride_m * e % pack_bytes != 0)))
    return error_code(Error::row_stride_not_packs);

  if (m > 0 && n > 0) {
    const auto first = [](const void* p) {
      return static_cast<const char*>(p);
    };
    const auto last = [&](const void* p, int64_t stride_m, int64_t stride_n,
                          int64_t rows, int64_t elem) {
      return first(p) + ((rows - 1) * stride_m + (n - 1) * stride_n + 1) * elem;
    };
    const auto meet = [&](const void* a, const char* a_last, const void* b,
                          const char* b_last) {
      return a != nullptr && b != nullptr && first(a) < b_last &&
             first(b) < a_last;
    };
    const char* input_last =
        last(input_ptr, input_stride_m, input_stride_n, m, e);
    const char* weight_last =
        last(weight_ptr, 0, weight_stride_n, 1, weight.element_size());
    const char* residual_last =
        add ? last(residual_ptr, residual_stride_m, residual_stride_n, m, e)
            : nullptr;
    const char* output_last =
        last(output_ptr, output_stride_m, output_stride_n, m, e);
    const char* residual_output_last =
        add ? last(residual_output_ptr, residual_output_stride_m,
                   residual_output_stride_n, m, e)
            : nullptr;
    for (const auto& [out, out_last] :
         {std::pair{static_cast<const void*>(output_ptr), output_last},
          std::pair{static_cast<const void*>(residual_output_ptr),
                    residual_output_last}})
      if (meet(input_ptr, input_last, out, out_last) ||
          meet(weight_ptr, weight_last, out, out_last) ||
          meet(residual_ptr, residual_last, out, out_last))
        return error_code(Error::inputs_overlap_outputs);
    if (meet(output_ptr, output_last, residual_output_ptr,
             residual_output_last))
      return error_code(Error::inputs_overlap_outputs);
  }

  const bool quantize_int4 = int4_config;
  const int shared_memory_bytes = 0;

  void* kernel = nullptr;
  void* rms_norm_kernel = nullptr;
  dim3 rms_norm_grid, rms_norm_block;
  bool in_place = false;
  int64_t staging_need = 0, partials_need = 0;
  int64_t partials_stride_m = 0, partials_stride_n = 0, partials_size_m = 0,
          partials_size_n = 0;
  int64_t staging_stride_m = 0, staging_stride_n = 0, staging_size_m = 0,
          staging_size_n = 0;
  int64_t unfenced_stride_slot = 0, unfenced_stride_m = 0,
          unfenced_stride_n = 0, unfenced_size_slot = 0, unfenced_size_m = 0,
          unfenced_size_n = 0;
  if (split) {
    in_place = (all_reduce_algorithm == AllReduceAlgorithm::one_shot_pull ||
                all_reduce_algorithm == AllReduceAlgorithm::two_shot_pull) &&
               !quantize_int4;

    const bool two_shot =
        all_reduce_algorithm == AllReduceAlgorithm::two_shot_pull ||
        all_reduce_algorithm == AllReduceAlgorithm::two_shot_push;
    const bool contiguous = two_shot && input_stride_m == n * input_stride_n &&
                            output_stride_m == n * output_stride_n;
    const bool tail = quantize_int4 && (two_shot ? m * n % kInt4Group != 0
                                                 : n % kInt4Group != 0);

    if (all_reduce_algorithm == AllReduceAlgorithm::one_shot_push) {
      const int64_t line_pack =
          std::min<int64_t>(pack_bytes, kLinePayloadBytes);
      const int64_t row_lines =
          quantize_int4
              ? (tile_n + kInt4Group - 1) / kInt4Group * kInt4GroupPacks
              : (tile_n * e + line_pack - 1) / line_pack;
      if (row_lines > s.symmetric.unfenced_slot_bytes / kLinePayloadBytes ||
          grid_x * tile_m > s.symmetric.unfenced_blocks)
        return error_code(Error::unfenced_too_small);
    }

    for (const auto& i : kAllReduceInstances)
      if (i.algorithm == all_reduce_algorithm && i.block_x == block_x &&
          i.block_y == block_y && i.block_z == block_z &&
          i.waves_per_eu == waves_per_eu && i.tile_m == tile_m &&
          i.tile_n == tile_n &&
          i.quantization ==
              (int4_config ? Quantization::int4 : Quantization::exact) &&
          i.pack_bytes == pack_bytes && i.world == s.world &&
          i.wave_size == s.device.wave_size && i.dtype == dtype &&
          i.contiguous == contiguous && i.tail == tail &&
          i.quantize_int4 == quantize_int4)
        kernel = i.kernel;
    if (kernel == nullptr) return error_code(Error::not_built);

    hipFuncAttributes attributes;
    if (hipFuncGetAttributes(&attributes, kernel) != hipSuccess)
      return error_code(Error::hip_failed);
    if (block_x > attributes.maxThreadsPerBlock)
      return error_code(Error::block_too_large);
    if (static_cast<int64_t>(attributes.sharedSizeBytes) + shared_memory_bytes >
        s.device.lds_bytes)
      return error_code(Error::lds_too_large);
    if (static_cast<int64_t>(attributes.localSizeBytes) >
        s.device.scratch_a_lane)
      return error_code(Error::scratch_too_large);

    int blocks_a_cu = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_a_cu, kernel, static_cast<int>(block_x),
            shared_memory_bytes) != hipSuccess)
      return error_code(Error::hip_failed);
    if (grid_x > blocks_a_cu * s.device.compute_units)
      return error_code(Error::grid_not_resident);

    const int64_t packs = n * e / pack_bytes;
    const int64_t slice_bytes = (packs + s.world - 1) / s.world * pack_bytes;
    const int64_t slot = (m * packs + s.world - 1) / s.world * pack_bytes / e;

    const int64_t share =
        ((m * n + kInt4Group - 1) / kInt4Group + s.world - 1) / s.world *
        kInt4Group;
    staging_need = quantize_int4 ? 0 : m * n * e;
    partials_stride_m = slice_bytes / e, partials_stride_n = 1,
    partials_size_n = slice_bytes / e;
    switch (all_reduce_algorithm) {
      case AllReduceAlgorithm::two_shot_pull:
      case AllReduceAlgorithm::two_shot_push:
        staging_need = m * n * e, partials_need = m * slice_bytes;
        if (quantize_int4) {
          partials_stride_m = share, partials_size_m = 1,
          partials_size_n = share;
          staging_stride_m = s.world * share, staging_stride_n = 1,
          staging_size_m = 1, staging_size_n = s.world * share;
          staging_need =
              std::max(staging_need, int4_payload_bytes(1, s.world * share));
          partials_need = std::max(partials_need, int4_payload_bytes(1, share));
        } else {
          partials_size_m = m;
          if (all_reduce_algorithm == AllReduceAlgorithm::two_shot_push) {
            staging_stride_m = slot, staging_stride_n = 1,
            staging_size_m = s.world, staging_size_n = slot;
            staging_need = std::max(staging_need, s.world * slot * e);
          }
        }
        break;
      case AllReduceAlgorithm::one_shot_pull:

        if (quantize_int4) {
          partials_stride_m = 0, partials_size_n = 0;
          staging_stride_n = 1, staging_size_m = 1,
          staging_size_n = (int4_payload_bytes(m, n) + e - 1) / e;
          staging_need = int4_payload_bytes(m, n);
        }
        break;
      case AllReduceAlgorithm::one_shot_push:

        unfenced_stride_slot = 2 * s.symmetric.unfenced_blocks *
                               (s.symmetric.unfenced_slot_bytes / e);
        unfenced_stride_m = s.symmetric.unfenced_slot_bytes / e,
        unfenced_stride_n = 1;
        unfenced_size_slot = s.world,
        unfenced_size_m = 2 * s.symmetric.unfenced_blocks,
        unfenced_size_n = s.symmetric.unfenced_slot_bytes / e;
        break;
      case AllReduceAlgorithm::one_shot_push_staged: {
        const int64_t payload =
            quantize_int4 ? int4_payload_bytes(m, n) : m * n * e;
        const int64_t slot_bytes = quantize_int4
                                       ? (payload + kInt4ScalesAlign - 1) /
                                             kInt4ScalesAlign * kInt4ScalesAlign
                                       : payload;
        partials_stride_m = 0, partials_size_n = 0;
        staging_stride_n = 1, staging_size_m = 1,
        staging_size_n = (s.world * slot_bytes + e - 1) / e;
        staging_need = s.world * slot_bytes;
        break;
      }
    }
    rms_norm_grid = dim3(static_cast<unsigned>(*add_rms_norm_grid_x),
                         static_cast<unsigned>(*add_rms_norm_grid_y),
                         static_cast<unsigned>(*add_rms_norm_grid_z));
    rms_norm_block = dim3(static_cast<unsigned>(*add_rms_norm_block_x),
                          static_cast<unsigned>(*add_rms_norm_block_y),
                          static_cast<unsigned>(*add_rms_norm_block_z));

    if (*add_rms_norm_grid_y != 1 || *add_rms_norm_grid_z != 1 ||
        *add_rms_norm_grid_x < 1)
      return error_code(Error::launch_out_of_range);
    if (*add_rms_norm_tile_m < 1 || *add_rms_norm_tile_n < 1)
      return error_code(Error::field_not_positive);

    if (*add_rms_norm_pack_bytes < 2 || *add_rms_norm_pack_bytes > kPackBytes ||
        (*add_rms_norm_pack_bytes & (*add_rms_norm_pack_bytes - 1)) != 0 ||
        *add_rms_norm_pack_bytes < e)
      return error_code(Error::pack_out_of_range);
    if (n * e % *add_rms_norm_pack_bytes != 0)
      return error_code(Error::row_not_packs);

    if (reinterpret_cast<uintptr_t>(weight_ptr) %
            (kPackBytes / e * weight.element_size()) !=
        0)
      return error_code(Error::not_aligned);

    if (output_stride_n != 1 || weight_stride_n != 1)
      return error_code(Error::inner_stride_not_one);
    if (output_stride_m * e % *add_rms_norm_pack_bytes != 0 ||
        (add && (residual_stride_m * e % *add_rms_norm_pack_bytes != 0 ||
                 residual_output_stride_m * e % *add_rms_norm_pack_bytes != 0)))
      return error_code(Error::row_stride_not_packs);

    const bool rms_norm_chunked = n > *add_rms_norm_tile_n;
    for (const auto& i : kAddRmsNormInstances)
      if (i.residual == add && i.block_x == *add_rms_norm_block_x &&
          i.block_y == *add_rms_norm_block_y &&
          i.block_z == *add_rms_norm_block_z &&
          i.waves_per_eu == *add_rms_norm_waves_per_eu &&
          i.tile_m == *add_rms_norm_tile_m &&
          i.tile_n == *add_rms_norm_tile_n &&
          i.quantization == Quantization::exact &&
          i.pack_bytes == *add_rms_norm_pack_bytes &&
          i.wave_size == s.device.wave_size && i.dtype == dtype &&
          i.weight_dtype == weight_dtype && i.chunked == rms_norm_chunked)
        rms_norm_kernel = i.kernel;
    if (rms_norm_kernel == nullptr) return error_code(Error::not_built);

    hipFuncAttributes rms_norm_attributes;
    if (hipFuncGetAttributes(&rms_norm_attributes, rms_norm_kernel) !=
        hipSuccess)
      return error_code(Error::hip_failed);
    if (*add_rms_norm_block_x > rms_norm_attributes.maxThreadsPerBlock)
      return error_code(Error::block_too_large);
    if (static_cast<int64_t>(rms_norm_attributes.sharedSizeBytes) +
            shared_memory_bytes >
        s.device.lds_bytes)
      return error_code(Error::lds_too_large);
    if (static_cast<int64_t>(rms_norm_attributes.localSizeBytes) >
        s.device.scratch_a_lane)
      return error_code(Error::scratch_too_large);
  } else {
    in_place = !quantize_int4;
    const bool chunked = n > tile_n;
    const bool tail =
        quantize_int4 && (n % kInt4Group != 0 || n / kInt4Group % s.world != 0);

    for (const auto& i : kAllReduceAddRmsNormInstances)
      if (i.algorithm == algorithm && i.residual == add &&
          i.block_x == block_x && i.block_y == block_y &&
          i.block_z == block_z && i.waves_per_eu == waves_per_eu &&
          i.tile_m == tile_m && i.tile_n == tile_n &&
          i.quantization ==
              (int4_config ? Quantization::int4 : Quantization::exact) &&
          i.pack_bytes == pack_bytes && i.world == s.world &&
          i.wave_size == s.device.wave_size && i.dtype == dtype &&
          i.weight_dtype == weight_dtype && i.chunked == chunked &&
          i.tail == tail && i.quantize_int4 == quantize_int4)
        kernel = i.kernel;
    if (kernel == nullptr) return error_code(Error::not_built);

    hipFuncAttributes attributes;
    if (hipFuncGetAttributes(&attributes, kernel) != hipSuccess)
      return error_code(Error::hip_failed);
    if (block_x > attributes.maxThreadsPerBlock)
      return error_code(Error::block_too_large);
    if (static_cast<int64_t>(attributes.sharedSizeBytes) + shared_memory_bytes >
        s.device.lds_bytes)
      return error_code(Error::lds_too_large);
    if (static_cast<int64_t>(attributes.localSizeBytes) >
        s.device.scratch_a_lane)
      return error_code(Error::scratch_too_large);

    int blocks_a_cu = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_a_cu, kernel, static_cast<int>(block_x),
            shared_memory_bytes) != hipSuccess)
      return error_code(Error::hip_failed);
    if (grid_x > blocks_a_cu * s.device.compute_units)
      return error_code(Error::grid_not_resident);

    const bool two_shot =
        algorithm != AllReduceAddRmsNormKernelAlgorithm::one_shot_pull;
    staging_stride_m = n, staging_stride_n = 1, staging_size_m = m,
    staging_size_n = n;
    partials_stride_m = n, partials_stride_n = 1,
    partials_size_m = two_shot ? m : 0, partials_size_n = n;

    staging_need = m * n * e, partials_need = two_shot ? m * n * e : 0;
    if (quantize_int4) {
      staging_need = std::max(staging_need, int4_payload_bytes(m, n));
      partials_need = std::max(partials_need, int4_payload_bytes(m, n));
    }
  }

  const int64_t staging_end =
      staging_size_m == 0 || staging_size_n == 0
          ? 0
          : ((staging_size_m - 1) * staging_stride_m +
             (staging_size_n - 1) * staging_stride_n + 1) *
                e;
  const int64_t partials_end =
      partials_size_m == 0 || partials_size_n == 0
          ? 0
          : ((partials_size_m - 1) * partials_stride_m +
             (partials_size_n - 1) * partials_stride_n + 1) *
                e;
  if (std::max(staging_need, staging_end) > s.symmetric.staging_bytes)
    return error_code(Error::staging_too_small);
  if (std::max(partials_need, partials_end) > s.symmetric.partials_bytes)
    return error_code(Error::partials_too_small);

  void* const* peer_inputs = nullptr;
  int peer_inputs_size = 0;
  int64_t peer_inputs_stride_m = staging_stride_m,
          peer_inputs_stride_n = staging_stride_n;
  if (in_place) {
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream, &capture) != hipSuccess)
      return error_code(Error::hip_failed);
    InputTables& tables = s.input_tables;
    if (capture == hipStreamCaptureStatusActive) {
      if (tables.next_slot >= tables.slots)
        return error_code(Error::tables_full);
      tables.pending.emplace_back(tables.next_slot, input_ptr);
      peer_inputs = tables.device + tables.next_slot * s.world;
      peer_inputs_stride_m = input_stride_m;
      ++tables.next_slot;
    } else {
      if (hipMemcpy2DAsync(s.symmetric.staging, static_cast<size_t>(n * e),
                           input_ptr, static_cast<size_t>(input_stride_m * e),
                           static_cast<size_t>(n * e), static_cast<size_t>(m),
                           hipMemcpyDeviceToDevice, stream) != hipSuccess)
        return error_code(Error::hip_failed);
      peer_inputs = tables.device;
      peer_inputs_stride_m = n;
    }
    peer_inputs_size = s.world, peer_inputs_stride_n = 1;
  }

  if (split) {
    const auto launch = [&]<typename T, typename W>() {
      const auto typed = reinterpret_cast<AllReduceKernel<T, W>>(kernel);
      typed<<<grid, block, shared_memory_bytes, stream>>>(
          static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
          static_cast<int>(input_size_m), static_cast<int>(input_size_n),
          reinterpret_cast<const T* const*>(peer_inputs), peer_inputs_size,
          peer_inputs_stride_m, peer_inputs_stride_n,
          static_cast<T*>(output_ptr), output_stride_m, output_stride_n,
          static_cast<int>(output_size_m), static_cast<int>(output_size_n),
          s.symmetric.partials_ptr_array, s.world, partials_stride_m,
          partials_stride_n, static_cast<int>(partials_size_m),
          static_cast<int>(partials_size_n), s.symmetric.staging_ptr_array,
          s.world, staging_stride_m, staging_stride_n,
          static_cast<int>(staging_size_m), static_cast<int>(staging_size_n),
          s.symmetric.unfenced_ptr_array, s.world, unfenced_stride_slot,
          unfenced_stride_m, unfenced_stride_n,
          static_cast<int>(unfenced_size_slot),
          static_cast<int>(unfenced_size_m), static_cast<int>(unfenced_size_n),
          s.unfenced_epochs, s.unfenced_epoch_words, int64_t{1},
          static_cast<int>(s.symmetric.unfenced_blocks),
          static_cast<int>(s.unfenced_epoch_words),
          s.symmetric.signals_ptr_array, s.world,
          s.symmetric.max_blocks * s.world, int64_t{s.world}, int64_t{1},
          int64_t{2}, s.symmetric.max_blocks, int64_t{s.world}, s.status_device,
          int64_t{1}, int64_t{kStatusFields}, s.rank, s.timeout_ticks);
    };
    if (dtype == at::ScalarType::BFloat16)
      launch.template operator()<c10::BFloat16, c10::BFloat16>();
    else
      launch.template operator()<c10::Half, c10::Half>();

    const auto launch_add_rms_norm = [&]<typename T, typename W>() {
      const auto typed =
          reinterpret_cast<AddRmsNormKernel<T, W>>(rms_norm_kernel);
      typed<<<rms_norm_grid, rms_norm_block, shared_memory_bytes, stream>>>(
          static_cast<const T*>(output_ptr), output_stride_m, output_stride_n,
          static_cast<int>(output_size_m), static_cast<int>(output_size_n),
          static_cast<const T*>(residual_ptr), residual_stride_m,
          residual_stride_n, static_cast<int>(residual_size_m),
          static_cast<int>(residual_size_n), static_cast<const W*>(weight_ptr),
          weight_stride_n, static_cast<int>(weight_size_n),
          static_cast<T*>(output_ptr), output_stride_m, output_stride_n,
          static_cast<int>(output_size_m), static_cast<int>(output_size_n),
          static_cast<T*>(residual_output_ptr), residual_output_stride_m,
          residual_output_stride_n, static_cast<int>(residual_output_size_m),
          static_cast<int>(residual_output_size_n), eps);
    };
    if (dtype == at::ScalarType::BFloat16 &&
        weight_dtype == at::ScalarType::Float)
      launch_add_rms_norm.template operator()<c10::BFloat16, float>();
    else if (dtype == at::ScalarType::BFloat16)
      launch_add_rms_norm.template operator()<c10::BFloat16, c10::BFloat16>();
    else if (weight_dtype == at::ScalarType::Float)
      launch_add_rms_norm.template operator()<c10::Half, float>();
    else
      launch_add_rms_norm.template operator()<c10::Half, c10::Half>();
  } else {
    const auto launch = [&]<typename T, typename W>() {
      const auto typed =
          reinterpret_cast<AllReduceAddRmsNormKernel<T, W>>(kernel);
      typed<<<grid, block, shared_memory_bytes, stream>>>(
          static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
          static_cast<int>(input_size_m), static_cast<int>(input_size_n),
          reinterpret_cast<const T* const*>(peer_inputs), peer_inputs_size,
          peer_inputs_stride_m, peer_inputs_stride_n,
          static_cast<const T*>(residual_ptr), residual_stride_m,
          residual_stride_n, static_cast<int>(residual_size_m),
          static_cast<int>(residual_size_n), static_cast<const W*>(weight_ptr),
          weight_stride_n, static_cast<int>(weight_size_n),
          static_cast<T*>(output_ptr), output_stride_m, output_stride_n,
          static_cast<int>(output_size_m), static_cast<int>(output_size_n),
          static_cast<T*>(residual_output_ptr), residual_output_stride_m,
          residual_output_stride_n, static_cast<int>(residual_output_size_m),
          static_cast<int>(residual_output_size_n),
          s.symmetric.partials_ptr_array, s.world, partials_stride_m,
          partials_stride_n, static_cast<int>(partials_size_m),
          static_cast<int>(partials_size_n), s.symmetric.staging_ptr_array,
          s.world, staging_stride_m, staging_stride_n,
          static_cast<int>(staging_size_m), static_cast<int>(staging_size_n),
          s.symmetric.signals_ptr_array, s.world,
          s.symmetric.max_blocks * s.world, int64_t{s.world}, int64_t{1},
          int64_t{2}, s.symmetric.max_blocks, int64_t{s.world}, s.status_device,
          int64_t{1}, int64_t{kStatusFields}, s.rank, s.timeout_ticks, eps);
    };
    if (dtype == at::ScalarType::BFloat16 &&
        weight_dtype == at::ScalarType::Float)
      launch.template operator()<c10::BFloat16, float>();
    else if (dtype == at::ScalarType::BFloat16)
      launch.template operator()<c10::BFloat16, c10::BFloat16>();
    else if (weight_dtype == at::ScalarType::Float)
      launch.template operator()<c10::Half, float>();
    else
      launch.template operator()<c10::Half, c10::Half>();
  }

  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}
