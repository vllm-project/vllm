#pragma once

// Host side of the all-reduce with AttnRes: checks the call, then launches its
// kernel. The kernel can also write an FP8 (OCP e4m3fn) copy of the output and
// its float32 per-row scale (quant_output, quant_scale), and skip the bf16
// output (write_output).

#include "../../ops.h"
#include "../types.cuh"
#include "../utils.h"
#include "../kernels/all_reduce.cuh"
#include "../kernels/all_reduce_add_attn_res_rms_norm.cuh"

namespace hip_custom_communicator::internal {

// The tensors of one AttnRes call. `input` is what the rows kernel reads: the
// caller's input, or the split path's reduced copy.
struct AttnResCall {
  const torch::Tensor& input;
  const std::optional<torch::Tensor>& prefix;
  const torch::Tensor& blocks;
  const torch::Tensor& norm_weight;
  const torch::Tensor& qk_weight;
  const std::optional<torch::Tensor>& output_norm_weight;
  const torch::Tensor& prefix_output;
  const torch::Tensor& output;
  const std::optional<torch::Tensor>& block_output;
  const std::optional<torch::Tensor>& quant_output;
  const std::optional<torch::Tensor>& quant_scale;
  bool write_output;

  bool quantizes() const { return quant_output.has_value(); }
  uint8_t* quant_ptr() const {
    return quantizes() ? static_cast<uint8_t*>(quant_output->data_ptr())
                       : nullptr;
  }
  int64_t quant_stride_m() const {
    return quantizes() ? quant_output->stride(0) : 0;
  }
  float* quant_scale_ptr() const {
    return quantizes() ? static_cast<float*>(quant_scale->data_ptr()) : nullptr;
  }
  int64_t quant_scale_stride_m() const {
    return quantizes() ? quant_scale->stride(0) : 0;
  }
};

// The checks of every AttnRes call, before a kernel is chosen.
inline ErrorCode check_attn_res_call(const AttnResCall& c, int64_t num_blocks) {
  const torch::Tensor& input = c.input;
  if (!input.is_cuda() || !c.norm_weight.is_cuda() || !c.qk_weight.is_cuda() ||
      !c.prefix_output.is_cuda() || !c.output.is_cuda() || !c.blocks.is_cuda())
    return error_code(Error::not_on_device);

  if (input.dim() != 2 || input.size(0) < 1 || input.size(1) < 1)
    return error_code(Error::shape_mismatch);
  const int64_t width = input.size(1);
  if (c.prefix_output.sizes() != input.sizes() ||
      c.output.sizes() != input.sizes() || c.blocks.dim() != 3 ||
      c.blocks.size(0) != input.size(0) || c.blocks.size(2) != width ||
      c.blocks.stride(2) != 1 || num_blocks < 0 ||
      num_blocks > c.blocks.size(1) || c.norm_weight.dim() != 1 ||
      c.norm_weight.numel() != width || c.qk_weight.dim() != 1 ||
      c.qk_weight.numel() != width)
    return error_code(Error::shape_mismatch);
  std::vector<const torch::Tensor*> same = {&c.blocks, &c.norm_weight,
                                            &c.qk_weight, &c.prefix_output,
                                            &c.output};
  for (const std::optional<torch::Tensor>* o :
       {&c.prefix, &c.output_norm_weight, &c.block_output})
    if (*o) {
      if (!(*o)->is_cuda()) return error_code(Error::not_on_device);
      same.push_back(&**o);
    }
  for (const torch::Tensor* t : same)
    if (t->scalar_type() != input.scalar_type())
      return error_code(Error::dtype_mismatch);
  if ((c.prefix && c.prefix->sizes() != input.sizes()) ||
      (c.block_output && (c.block_output->sizes() != input.sizes() ||
                          c.block_output->stride(1) != 1)) ||
      (c.output_norm_weight && (c.output_norm_weight->dim() != 1 ||
                                c.output_norm_weight->numel() != width)))
    return error_code(Error::shape_mismatch);

  // The FP8 copy: e4m3fn codes shaped as the input, a float32 scale per row.
  if (c.quant_output.has_value() != c.quant_scale.has_value())
    return error_code(Error::field_missing);
  if (!c.write_output && !c.quantizes())
    return error_code(Error::field_missing);
  if (c.quantizes()) {
    if (!c.quant_output->is_cuda() || !c.quant_scale->is_cuda())
      return error_code(Error::not_on_device);
    if (c.quant_output->scalar_type() != at::ScalarType::Float8_e4m3fn ||
        c.quant_scale->scalar_type() != at::ScalarType::Float)
      return error_code(Error::dtype_mismatch);
    if (c.quant_output->sizes() != input.sizes() ||
        c.quant_scale->dim() != 2 || c.quant_scale->size(0) != input.size(0) ||
        c.quant_scale->size(1) != 1)
      return error_code(Error::shape_mismatch);
    if (c.quant_output->stride(1) != 1)
      return error_code(Error::inner_stride_not_one);
  }

  if (reinterpret_cast<uintptr_t>(c.blocks.data_ptr()) % kPackBytes != 0)
    return error_code(Error::not_aligned);

  const torch::Tensor sources = c.blocks.narrow(1, 0, num_blocks);

  struct Span {
    uintptr_t first, last;
    bool rows;
    uintptr_t stride, width;
  };
  const auto span_of = [](const torch::Tensor& t) {
    const uintptr_t first = reinterpret_cast<uintptr_t>(t.data_ptr());
    const int64_t elem = t.element_size();
    int64_t last = 0;
    for (int64_t d = 0; d < t.dim(); ++d) last += (t.size(d) - 1) * t.stride(d);
    Span span{first, first + static_cast<uintptr_t>((last + 1) * elem), false,
              0, 0};
    if (t.dim() < 2 || t.stride(t.dim() - 1) != 1) return span;
    int64_t run = 1;
    for (int64_t d = t.dim() - 1; d >= 1; --d) {
      if (t.stride(d) != run) return span;
      run *= t.size(d);
    }
    if (t.stride(0) < run) return span;
    span.rows = true;
    span.stride = static_cast<uintptr_t>(t.stride(0) * elem);
    span.width = static_cast<uintptr_t>(run * elem);
    return span;
  };
  const auto opt = [](const std::optional<torch::Tensor>& t) {
    return t ? &*t : nullptr;
  };
  for (const torch::Tensor* i : std::initializer_list<const torch::Tensor*>{
           &input, opt(c.prefix), &sources, &c.norm_weight, &c.qk_weight,
           opt(c.output_norm_weight)})
    for (const torch::Tensor* o : std::initializer_list<const torch::Tensor*>{
             &c.prefix_output, &c.output, opt(c.block_output),
             opt(c.quant_output), opt(c.quant_scale)}) {
      if (i == nullptr || o == nullptr || i->numel() == 0 || o->numel() == 0)
        continue;
      const Span a = span_of(*i), b = span_of(*o);
      if (a.last <= b.first || b.last <= a.first) continue;
      if (!(a.rows && b.rows && a.stride == b.stride))
        return error_code(Error::inputs_overlap_outputs);

      const uintptr_t at = a.first % a.stride, bt = b.first % b.stride;
      const auto meets = [&](uintptr_t x, uintptr_t xw, uintptr_t y) {
        return (y + a.stride - x) % a.stride < xw;
      };
      if (meets(at, a.width, bt) || meets(bt, b.width, at))
        return error_code(Error::inputs_overlap_outputs);
    }
  return std::nullopt;
}

// The FP8 copy is written a pack (pack_bytes / element bytes codes) at a time,
// and only by the unchunked kernels, which hold a whole row.
inline ErrorCode check_quant_launch(const AttnResCall& c, int64_t pack_bytes,
                                    bool chunked) {
  if ((c.quantizes() || !c.write_output) && chunked)
    return error_code(Error::quantization_not_supported);
  if (!c.quantizes()) return std::nullopt;
  const int64_t codes = pack_bytes / c.input.element_size();
  if (c.quant_output->stride(0) % codes != 0 ||
      reinterpret_cast<uintptr_t>(c.quant_output->data_ptr()) % codes != 0)
    return error_code(Error::row_stride_not_packs);
  return std::nullopt;
}

// The rows kernel (AddAttnResRmsNorm, world 0): the split path's second kernel.
struct RowsLaunch {
  void* kernel = nullptr;
  dim3 grid, block;
  int shared_bytes = 0;
};

inline ErrorCode prepare_attn_res_rows(
    RowsLaunch& launch, const AttnResCall& c, int64_t num_blocks,
    int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x,
    int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m,
    int64_t tile_n, int64_t tile_k, int64_t pack_bytes, int wave_size,
    int64_t lds_bytes, int64_t scratch_a_lane) {
  const torch::Tensor& input = c.input;
  const int64_t n = input.size(1), e = input.element_size();
  const at::ScalarType dtype = input.scalar_type();
  if (grid_y != 1 || grid_z != 1 || grid_x < 1)
    return error_code(Error::launch_out_of_range);
  if (tile_m < 1 || tile_n < 1 || tile_k < 1)
    return error_code(Error::field_not_positive);
  if (pack_bytes < 2 || pack_bytes > kPackBytes ||
      (pack_bytes & (pack_bytes - 1)) != 0 || pack_bytes < e)
    return error_code(Error::pack_out_of_range);
  if (n * e % pack_bytes != 0) return error_code(Error::row_not_packs);
  if (input.stride(1) != 1 || c.blocks.stride(2) != 1 ||
      c.norm_weight.stride(0) != 1 || c.qk_weight.stride(0) != 1 ||
      c.prefix_output.stride(1) != 1 || c.output.stride(1) != 1 ||
      (c.prefix && c.prefix->stride(1) != 1) ||
      (c.output_norm_weight && c.output_norm_weight->stride(0) != 1) ||
      (c.block_output && c.block_output->stride(1) != 1))
    return error_code(Error::inner_stride_not_one);
  if (input.stride(0) * e % pack_bytes != 0 ||
      c.blocks.stride(0) * e % pack_bytes != 0 ||
      c.blocks.stride(1) * e % pack_bytes != 0 ||
      c.prefix_output.stride(0) * e % pack_bytes != 0 ||
      c.output.stride(0) * e % pack_bytes != 0 ||
      (c.prefix && c.prefix->stride(0) * e % pack_bytes != 0) ||
      (c.block_output && c.block_output->stride(0) * e % pack_bytes != 0))
    return error_code(Error::row_stride_not_packs);

  const bool chunked = n > tile_n;
  if (const ErrorCode error = check_quant_launch(c, pack_bytes, chunked))
    return error;
  const int shared_bytes =
      chunked ? static_cast<int>(tile_m * (num_blocks + 1) * sizeof(float)) : 0;
  void* kernel = nullptr;
  for (const auto& i : kAddAttnResRmsNormInstances)
    if (i.algorithm == AddAttnResRmsNormAlgorithm::rows &&
        i.block_x == block_x && i.block_y == block_y && i.block_z == block_z &&
        i.waves_per_eu == waves_per_eu && i.tile_m == tile_m &&
        i.tile_n == tile_n && i.tile_k == tile_k &&
        i.quantization == Quantization::exact && i.pack_bytes == pack_bytes &&
        i.world == 0 && i.wave_size == wave_size && i.dtype == dtype &&
        i.weight_dtype == dtype && i.contiguous == false &&
        i.chunked == chunked && i.tail == false && i.quantize_int4 == false)
      kernel = i.kernel;
  if (kernel == nullptr) return error_code(Error::not_built);

  hipFuncAttributes attributes;
  if (hipFuncGetAttributes(&attributes, kernel) != hipSuccess)
    return error_code(Error::hip_failed);
  if (block_x > attributes.maxThreadsPerBlock)
    return error_code(Error::block_too_large);
  if (static_cast<int64_t>(attributes.sharedSizeBytes) + shared_bytes >
      lds_bytes)
    return error_code(Error::lds_too_large);
  if (static_cast<int64_t>(attributes.localSizeBytes) > scratch_a_lane)
    return error_code(Error::scratch_too_large);

  launch.kernel = kernel;
  launch.grid = dim3(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y),
                     static_cast<unsigned>(grid_z));
  launch.block =
      dim3(static_cast<unsigned>(block_x), static_cast<unsigned>(block_y),
           static_cast<unsigned>(block_z));
  launch.shared_bytes = shared_bytes;
  return std::nullopt;
}

inline void launch_attn_res_rows(const RowsLaunch& launch, const AttnResCall& c,
                                 int64_t num_blocks, double eps,
                                 double output_eps, hipStream_t stream) {
  const auto ptr = [](const std::optional<torch::Tensor>& t) -> void* {
    return t ? t->data_ptr() : nullptr;
  };
  const auto stride = [](const std::optional<torch::Tensor>& t, int d) {
    return t ? t->stride(d) : int64_t{0};
  };
  const auto size = [](const std::optional<torch::Tensor>& t, int d) {
    return t ? static_cast<int>(t->size(d)) : 0;
  };
  const auto run = [&]<typename T, typename W>() {
    const auto typed =
        reinterpret_cast<AddAttnResRmsNormKernel<T, W>>(launch.kernel);
    typed<<<launch.grid, launch.block, launch.shared_bytes, stream>>>(
        static_cast<const T*>(c.input.data_ptr()), c.input.stride(0),
        c.input.stride(1), static_cast<int>(c.input.size(0)),
        static_cast<int>(c.input.size(1)),
        static_cast<const T*>(ptr(c.prefix)), stride(c.prefix, 0),
        stride(c.prefix, 1), size(c.prefix, 0), size(c.prefix, 1),
        static_cast<const T*>(c.blocks.data_ptr()), c.blocks.stride(0),
        c.blocks.stride(1), c.blocks.stride(2),
        static_cast<int>(c.blocks.size(0)), static_cast<int>(c.blocks.size(1)),
        static_cast<int>(c.blocks.size(2)),
        static_cast<const T*>(c.norm_weight.data_ptr()), c.norm_weight.stride(0),
        static_cast<int>(c.norm_weight.size(0)),
        static_cast<const T*>(c.qk_weight.data_ptr()), c.qk_weight.stride(0),
        static_cast<int>(c.qk_weight.size(0)),
        static_cast<const T*>(ptr(c.output_norm_weight)),
        stride(c.output_norm_weight, 0), size(c.output_norm_weight, 0),
        static_cast<T*>(c.prefix_output.data_ptr()), c.prefix_output.stride(0),
        c.prefix_output.stride(1), static_cast<int>(c.prefix_output.size(0)),
        static_cast<int>(c.prefix_output.size(1)),
        c.write_output ? static_cast<T*>(c.output.data_ptr()) : nullptr,
        c.output.stride(0), c.output.stride(1),
        static_cast<int>(c.output.size(0)), static_cast<int>(c.output.size(1)),
        static_cast<T*>(ptr(c.block_output)), stride(c.block_output, 0),
        stride(c.block_output, 1), size(c.block_output, 0),
        size(c.block_output, 1), num_blocks, eps, output_eps, c.quant_ptr(),
        c.quant_stride_m(), c.quant_scale_ptr(), c.quant_scale_stride_m());
  };
  if (c.input.scalar_type() == at::ScalarType::BFloat16)
    run.template operator()<c10::BFloat16, c10::BFloat16>();
  else
    run.template operator()<c10::Half, c10::Half>();
}

}  // namespace hip_custom_communicator::internal

ErrorCode hip_custom_communicator_all_reduce_add_attn_res_rms_norm(
    int64_t state, torch::Tensor& input,
    const std::optional<torch::Tensor>& prefix, torch::Tensor& blocks,
    torch::Tensor& norm_weight, torch::Tensor& qk_weight,
    const std::optional<torch::Tensor>& output_norm_weight,
    torch::Tensor& prefix_output, torch::Tensor& output,
    const std::optional<torch::Tensor>& block_output, int64_t num_blocks,
    double eps, double output_eps, const std::string& algorithm_name,
    int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t block_x,
    int64_t block_y, int64_t block_z, int64_t waves_per_eu, int64_t tile_m,
    int64_t tile_n, int64_t tile_k, const std::string& quantization_name,
    int64_t pack_bytes, std::optional<int64_t> add_attn_res_rms_norm_grid_x,
    std::optional<int64_t> add_attn_res_rms_norm_grid_y,
    std::optional<int64_t> add_attn_res_rms_norm_grid_z,
    std::optional<int64_t> add_attn_res_rms_norm_block_x,
    std::optional<int64_t> add_attn_res_rms_norm_block_y,
    std::optional<int64_t> add_attn_res_rms_norm_block_z,
    std::optional<int64_t> add_attn_res_rms_norm_waves_per_eu,
    std::optional<int64_t> add_attn_res_rms_norm_tile_m,
    std::optional<int64_t> add_attn_res_rms_norm_tile_n,
    std::optional<int64_t> add_attn_res_rms_norm_tile_k,
    std::optional<int64_t> add_attn_res_rms_norm_pack_bytes,
    const std::optional<torch::Tensor>& quant_output,
    const std::optional<torch::Tensor>& quant_scale, bool write_output) {
  using namespace hip_custom_communicator;
  using namespace hip_custom_communicator::internal;

  if (state == 0) return error_code(Error::not_open);

  const AttnResCall call{input,         prefix,       blocks,
                         norm_weight,   qk_weight,    output_norm_weight,
                         prefix_output, output,       block_output,
                         quant_output,  quant_scale,  write_output};
  if (const ErrorCode error = check_attn_res_call(call, num_blocks))
    return error;

  for (const int64_t v : {waves_per_eu, tile_m, tile_n, tile_k, pack_bytes,
                          grid_x, grid_y, grid_z, block_x, block_y, block_z})
    if (v < std::numeric_limits<int>::min() ||
        v > std::numeric_limits<int>::max())
      return error_code(Error::field_out_of_range);

  const bool split = algorithm_name.ends_with("_split");
  for (const std::optional<int64_t>& field :
       {add_attn_res_rms_norm_grid_x, add_attn_res_rms_norm_grid_y,
        add_attn_res_rms_norm_grid_z, add_attn_res_rms_norm_block_x,
        add_attn_res_rms_norm_block_y, add_attn_res_rms_norm_block_z,
        add_attn_res_rms_norm_waves_per_eu, add_attn_res_rms_norm_tile_m,
        add_attn_res_rms_norm_tile_n, add_attn_res_rms_norm_tile_k,
        add_attn_res_rms_norm_pack_bytes}) {
    if (!split && field) return error_code(Error::field_not_this_templates);
    if (split && !field) return error_code(Error::field_missing);
    if (field && (*field < std::numeric_limits<int>::min() ||
                  *field > std::numeric_limits<int>::max()))
      return error_code(Error::field_out_of_range);
  }
  CommsState& s = *reinterpret_cast<CommsState*>(state);
  const void* input_ptr = input.data_ptr();
  const int64_t input_stride_m = input.stride(0);
  const int64_t input_stride_n = input.stride(1);
  const int64_t input_size_m = input.size(0);
  const int64_t input_size_n = input.size(1);
  const void* prefix_ptr = prefix ? prefix->data_ptr() : nullptr;
  const int64_t prefix_stride_m = prefix ? prefix->stride(0) : 0;
  const int64_t prefix_stride_n = prefix ? prefix->stride(1) : 0;
  const int64_t prefix_size_m = prefix ? prefix->size(0) : 0;
  const int64_t prefix_size_n = prefix ? prefix->size(1) : 0;
  const void* blocks_ptr = blocks.data_ptr();
  const int64_t blocks_stride_m = blocks.stride(0);
  const int64_t blocks_stride_r = blocks.stride(1);
  const int64_t blocks_stride_n = blocks.stride(2);
  const int64_t blocks_size_m = blocks.size(0);
  const int64_t blocks_size_r = blocks.size(1);
  const int64_t blocks_size_n = blocks.size(2);
  const void* norm_weight_ptr = norm_weight.data_ptr();
  const int64_t norm_weight_stride_n = norm_weight.stride(0);
  const int64_t norm_weight_size_n = norm_weight.size(0);
  const void* qk_weight_ptr = qk_weight.data_ptr();
  const int64_t qk_weight_stride_n = qk_weight.stride(0);
  const int64_t qk_weight_size_n = qk_weight.size(0);
  const void* output_norm_weight_ptr =
      output_norm_weight ? output_norm_weight->data_ptr() : nullptr;
  const int64_t output_norm_weight_stride_n =
      output_norm_weight ? output_norm_weight->stride(0) : 0;
  const int64_t output_norm_weight_size_n =
      output_norm_weight ? output_norm_weight->size(0) : 0;
  void* prefix_output_ptr = prefix_output.data_ptr();
  const int64_t prefix_output_stride_m = prefix_output.stride(0);
  const int64_t prefix_output_stride_n = prefix_output.stride(1);
  const int64_t prefix_output_size_m = prefix_output.size(0);
  const int64_t prefix_output_size_n = prefix_output.size(1);
  void* output_ptr = output.data_ptr();
  const int64_t output_stride_m = output.stride(0);
  const int64_t output_stride_n = output.stride(1);
  const int64_t output_size_m = output.size(0);
  const int64_t output_size_n = output.size(1);
  void* block_output_ptr = block_output ? block_output->data_ptr() : nullptr;
  const int64_t block_output_stride_m =
      block_output ? block_output->stride(0) : 0;
  const int64_t block_output_stride_n =
      block_output ? block_output->stride(1) : 0;
  const int64_t block_output_size_m = block_output ? block_output->size(0) : 0;
  const int64_t block_output_size_n = block_output ? block_output->size(1) : 0;
  const at::ScalarType dtype = input.scalar_type();
  const dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y),
                  static_cast<unsigned>(grid_z));
  const dim3 block(static_cast<unsigned>(block_x),
                   static_cast<unsigned>(block_y),
                   static_cast<unsigned>(block_z));
  const hipStream_t stream = at::cuda::getCurrentCUDAStream();
  if (!split) {
    AllReduceAddAttnResRmsNormAlgorithm algorithm;
    if (algorithm_name == "one_shot_pull")
      algorithm = AllReduceAddAttnResRmsNormAlgorithm::one_shot_pull;
    else if (algorithm_name == "two_shot_pull_rows")
      algorithm = AllReduceAddAttnResRmsNormAlgorithm::two_shot_pull_rows;
    else if (algorithm_name == "two_shot_pull_columns")
      algorithm = AllReduceAddAttnResRmsNormAlgorithm::two_shot_pull_columns;
    else if (algorithm_name == "two_shot_push")
      algorithm = AllReduceAddAttnResRmsNormAlgorithm::two_shot_push;
    else
      return error_code(Error::no_such_kernel);
    if (quantization_name != "exact" && quantization_name != "int4")
      return error_code(Error::no_such_quantization);
    const int quantization =
        quantization_name == "int4" ? Quantization::int4 : Quantization::exact;
    const int64_t m = input_size_m, n = input_size_n;
    const int64_t e = input.element_size();
    const int64_t pack = pack_bytes;

    if (std::find(kDtypes.begin(), kDtypes.end(), dtype) == kDtypes.end())
      return error_code(Error::dtype_not_built);
    if (grid_y != 1 || grid_z != 1 || grid_x < 1 ||
        grid_x > s.symmetric.max_blocks)
      return error_code(Error::launch_out_of_range);
    if (tile_m < 1 || tile_n < 1 || tile_k < 1)
      return error_code(Error::field_not_positive);
    if (pack < 2 || pack > kPackBytes || (pack & (pack - 1)) != 0 || pack < e)
      return error_code(Error::pack_out_of_range);
    if (n * e % pack != 0) return error_code(Error::row_not_packs);
    const bool int4 = quantization == Quantization::int4;
    if (int4 && pack != kPackBytes)
      return error_code(Error::quantization_not_supported);

    const bool two_shot =
        algorithm != AllReduceAddAttnResRmsNormAlgorithm::one_shot_pull;
    const int64_t partials = two_shot ? m * n * e : 0;
    if (int4 && int4_payload_bytes(m, n) > s.symmetric.staging_bytes)
      return error_code(Error::staging_too_small);
    if (int4 && two_shot &&
        int4_payload_bytes(1, m * n) > s.symmetric.partials_bytes)
      return error_code(Error::partials_too_small);
    if (partials > s.symmetric.partials_bytes)
      return error_code(Error::partials_too_small);
    if (m * n * e > s.symmetric.staging_bytes)
      return error_code(Error::staging_too_small);

    if (input_stride_n != 1 || blocks_stride_n != 1 ||
        norm_weight_stride_n != 1 || qk_weight_stride_n != 1 ||
        prefix_output_stride_n != 1 || output_stride_n != 1)
      return error_code(Error::inner_stride_not_one);
    if (input_stride_m * e % pack != 0 || blocks_stride_m * e % pack != 0 ||
        blocks_stride_r * e % pack != 0 ||
        prefix_output_stride_m * e % pack != 0 ||
        output_stride_m * e % pack != 0)
      return error_code(Error::row_stride_not_packs);

    if (prefix_ptr != nullptr && prefix_stride_n != 1)
      return error_code(Error::inner_stride_not_one);
    if (prefix_ptr != nullptr && prefix_stride_m * e % pack != 0)
      return error_code(Error::row_stride_not_packs);
    if (output_norm_weight_ptr != nullptr && output_norm_weight_stride_n != 1)
      return error_code(Error::inner_stride_not_one);
    if (block_output_ptr != nullptr && block_output_stride_n != 1)
      return error_code(Error::inner_stride_not_one);
    if (block_output_ptr != nullptr && block_output_stride_m * e % pack != 0)
      return error_code(Error::row_stride_not_packs);

    const int shared_memory_bytes =
        n > tile_n ? static_cast<int>(tile_m * (num_blocks + 1) * sizeof(float))
                   : 0;

    const bool chunked = n > tile_n;
    if (const ErrorCode error = check_quant_launch(call, pack, chunked))
      return error;

    const bool tail = quantization == Quantization::int4 &&
                      (n % kInt4Group != 0 || n / kInt4Group % s.world != 0);
    void* kernel = nullptr;

    for (const auto& i : kAllReduceAddAttnResRmsNormInstances)
      if (i.algorithm == algorithm && i.block_x == block_x &&
          i.block_y == block_y && i.block_z == block_z &&
          i.waves_per_eu == waves_per_eu && i.tile_m == tile_m &&
          i.tile_n == tile_n && i.tile_k == tile_k &&
          i.quantization == quantization && i.pack_bytes == pack &&
          i.world == s.world && i.wave_size == s.device.wave_size &&
          i.dtype == dtype && i.weight_dtype == dtype &&
          i.contiguous == false && i.chunked == chunked && i.tail == tail &&
          i.quantize_int4 ==
              (quantization == Quantization::int4 && pack == kPackBytes))
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

    const int64_t partials_stride_m = n, partials_stride_n = 1,
                  partials_size_m = two_shot && n > 0 ? m : 0,
                  partials_size_n = n;
    const int64_t staging_stride_m = n, staging_stride_n = 1,
                  staging_size_m = m, staging_size_n = n;

    const bool in_place = quantization == Quantization::exact;

    void* const* table = nullptr;
    int table_size = 0;
    int64_t table_stride_m = staging_stride_m,
            table_stride_n = staging_stride_n;
    if (in_place) {
      hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
      if (hipStreamIsCapturing(stream, &capture) != hipSuccess)
        return error_code(Error::hip_failed);
      InputTables& tables = s.input_tables;
      if (capture == hipStreamCaptureStatusActive) {
        if (tables.next_slot >= tables.slots)
          return error_code(Error::tables_full);
        tables.pending.emplace_back(tables.next_slot, input_ptr);
        table = tables.device + tables.next_slot * s.world;
        table_stride_m = input_stride_m;
        ++tables.next_slot;
      } else {
        if (hipMemcpy2DAsync(s.symmetric.staging, static_cast<size_t>(n * e),
                             input_ptr, static_cast<size_t>(input_stride_m * e),
                             static_cast<size_t>(n * e), static_cast<size_t>(m),
                             hipMemcpyDeviceToDevice, stream) != hipSuccess)
          return error_code(Error::hip_failed);
        table = tables.device;
        table_stride_m = n;
      }
      table_size = s.world, table_stride_n = 1;
    }

    const auto launch = [&]<typename T, typename W>() {
      const auto typed =
          reinterpret_cast<AllReduceAddAttnResRmsNormKernel<T, W>>(kernel);
      typed<<<grid, block, shared_memory_bytes, stream>>>(
          static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
          static_cast<int>(input_size_m), static_cast<int>(input_size_n),
          reinterpret_cast<const T* const*>(table), table_size, table_stride_m,
          table_stride_n, static_cast<const T*>(prefix_ptr), prefix_stride_m,
          prefix_stride_n, static_cast<int>(prefix_size_m),
          static_cast<int>(prefix_size_n), static_cast<const T*>(blocks_ptr),
          blocks_stride_m, blocks_stride_r, blocks_stride_n,
          static_cast<int>(blocks_size_m), static_cast<int>(blocks_size_r),
          static_cast<int>(blocks_size_n),
          static_cast<const T*>(norm_weight_ptr), norm_weight_stride_n,
          static_cast<int>(norm_weight_size_n),
          static_cast<const T*>(qk_weight_ptr), qk_weight_stride_n,
          static_cast<int>(qk_weight_size_n),
          static_cast<const T*>(output_norm_weight_ptr),
          output_norm_weight_stride_n,
          static_cast<int>(output_norm_weight_size_n),
          static_cast<T*>(prefix_output_ptr), prefix_output_stride_m,
          prefix_output_stride_n, static_cast<int>(prefix_output_size_m),
          static_cast<int>(prefix_output_size_n),
          write_output ? static_cast<T*>(output_ptr) : nullptr,
          output_stride_m, output_stride_n, static_cast<int>(output_size_m),
          static_cast<int>(output_size_n), static_cast<T*>(block_output_ptr),
          block_output_stride_m, block_output_stride_n,
          static_cast<int>(block_output_size_m),
          static_cast<int>(block_output_size_n), s.symmetric.partials_ptr_array,
          s.world, partials_stride_m, partials_stride_n,
          static_cast<int>(partials_size_m), static_cast<int>(partials_size_n),
          s.symmetric.staging_ptr_array, s.world, staging_stride_m,
          staging_stride_n, static_cast<int>(staging_size_m),
          static_cast<int>(staging_size_n), s.symmetric.signals_ptr_array,
          s.world, s.symmetric.max_blocks * s.world, int64_t{s.world},
          int64_t{1}, int64_t{2}, s.symmetric.max_blocks, int64_t{s.world},
          s.status_device, int64_t{1}, int64_t{kStatusFields}, s.rank,
          s.timeout_ticks, num_blocks, eps, output_eps, call.quant_ptr(),
          call.quant_stride_m(), call.quant_scale_ptr(),
          call.quant_scale_stride_m());
    };
    if (dtype == at::ScalarType::BFloat16)
      launch.template operator()<c10::BFloat16, c10::BFloat16>();
    else
      launch.template operator()<c10::Half, c10::Half>();

    if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
    return std::nullopt;
  }

  const std::string body = algorithm_name.substr(0, algorithm_name.size() - 6);
  AllReduceAlgorithm all_reduce_algorithm;
  if (body == "one_shot_pull")
    all_reduce_algorithm = AllReduceAlgorithm::one_shot_pull;
  else if (body == "one_shot_push")
    all_reduce_algorithm = AllReduceAlgorithm::one_shot_push;
  else if (body == "one_shot_push_staged")
    all_reduce_algorithm = AllReduceAlgorithm::one_shot_push_staged;
  else if (body == "two_shot_pull")
    all_reduce_algorithm = AllReduceAlgorithm::two_shot_pull;
  else if (body == "two_shot_push")
    all_reduce_algorithm = AllReduceAlgorithm::two_shot_push;
  else
    return error_code(Error::no_such_kernel);
  if (quantization_name != "exact" && quantization_name != "int4")
    return error_code(Error::no_such_quantization);
  const bool int4_config = quantization_name == "int4";
  const bool quantize_int4 = int4_config;
  const int64_t m = input_size_m, n = input_size_n;
  const int64_t e = input.element_size();

  torch::Tensor summed;
  try {
    summed = torch::empty_like(input);
  } catch (const std::exception&) {
    return error_code(Error::hip_failed);
  }
  void* summed_ptr = summed.data_ptr();
  const int64_t summed_stride_m = summed.stride(0),
                summed_stride_n = summed.stride(1);
  const int64_t summed_size_m = summed.size(0), summed_size_n = summed.size(1);

  if (std::find(kDtypes.begin(), kDtypes.end(), dtype) == kDtypes.end())
    return error_code(Error::dtype_not_built);
  if (grid_y != 1 || grid_z != 1 || grid_x < 1 ||
      grid_x > s.symmetric.max_blocks)
    return error_code(Error::launch_out_of_range);
  if (tile_m < 1 || tile_n < 1) return error_code(Error::field_not_positive);
  if (pack_bytes < 2 || pack_bytes > kPackBytes ||
      (pack_bytes & (pack_bytes - 1)) != 0 || pack_bytes < e)
    return error_code(Error::pack_out_of_range);
  if (int4_config && pack_bytes != kPackBytes)
    return error_code(Error::quantization_not_supported);
  if (n * e % pack_bytes != 0) return error_code(Error::row_not_packs);
  if (input_stride_n != 1) return error_code(Error::inner_stride_not_one);
  if (input_stride_m * e % pack_bytes != 0)
    return error_code(Error::row_stride_not_packs);
  void* kernel = nullptr;
  bool in_place = false;
  int64_t staging_need = 0, partials_need = 0;
  int64_t partials_stride_m = 0, partials_stride_n = 0, partials_size_m = 0,
          partials_size_n = 0;
  int64_t staging_stride_m = 0, staging_stride_n = 0, staging_size_m = 0,
          staging_size_n = 0;
  int64_t unfenced_stride_slot = 0, unfenced_stride_m = 0,
          unfenced_stride_n = 0, unfenced_size_slot = 0, unfenced_size_m = 0,
          unfenced_size_n = 0;

  in_place = (all_reduce_algorithm == AllReduceAlgorithm::one_shot_pull ||
              all_reduce_algorithm == AllReduceAlgorithm::two_shot_pull) &&
             !quantize_int4;

  const bool two_shot =
      all_reduce_algorithm == AllReduceAlgorithm::two_shot_pull ||
      all_reduce_algorithm == AllReduceAlgorithm::two_shot_push;
  const bool contiguous = two_shot && input_stride_m == n * input_stride_n &&
                          summed_stride_m == n * summed_stride_n;
  const bool tail = quantize_int4 &&
                    (two_shot ? m * n % kInt4Group != 0 : n % kInt4Group != 0);

  if (all_reduce_algorithm == AllReduceAlgorithm::one_shot_push) {
    const int64_t line_pack = std::min<int64_t>(pack_bytes, kLinePayloadBytes);
    const int64_t row_lines =
        quantize_int4 ? (tile_n + kInt4Group - 1) / kInt4Group * kInt4GroupPacks
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
  if (static_cast<int64_t>(attributes.sharedSizeBytes) + 0 > s.device.lds_bytes)
    return error_code(Error::lds_too_large);
  if (static_cast<int64_t>(attributes.localSizeBytes) > s.device.scratch_a_lane)
    return error_code(Error::scratch_too_large);

  int blocks_a_cu = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_a_cu, kernel, static_cast<int>(block_x), 0) != hipSuccess)
    return error_code(Error::hip_failed);
  if (grid_x > blocks_a_cu * s.device.compute_units)
    return error_code(Error::grid_not_resident);

  const int64_t packs = n * e / pack_bytes;
  const int64_t slice_bytes = (packs + s.world - 1) / s.world * pack_bytes;
  const int64_t slot = (m * packs + s.world - 1) / s.world * pack_bytes / e;

  const int64_t share = ((m * n + kInt4Group - 1) / kInt4Group + s.world - 1) /
                        s.world * kInt4Group;
  staging_need = quantize_int4 ? 0 : m * n * e;
  partials_stride_m = slice_bytes / e, partials_stride_n = 1,
  partials_size_n = slice_bytes / e;
  switch (all_reduce_algorithm) {
    case AllReduceAlgorithm::two_shot_pull:
    case AllReduceAlgorithm::two_shot_push:
      staging_need = m * n * e, partials_need = m * slice_bytes;
      if (quantize_int4) {
        partials_stride_m = share, partials_size_m = 1, partials_size_n = share;
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

  // The rows kernel reads the reduced copy: check and find it before the
  // all-reduce runs.
  const AttnResCall rows_call{summed,        prefix,       blocks,
                              norm_weight,   qk_weight,    output_norm_weight,
                              prefix_output, output,       block_output,
                              quant_output,  quant_scale,  write_output};
  RowsLaunch rows;
  if (const ErrorCode error = prepare_attn_res_rows(
          rows, rows_call, num_blocks, *add_attn_res_rms_norm_grid_x,
          *add_attn_res_rms_norm_grid_y, *add_attn_res_rms_norm_grid_z,
          *add_attn_res_rms_norm_block_x, *add_attn_res_rms_norm_block_y,
          *add_attn_res_rms_norm_block_z, *add_attn_res_rms_norm_waves_per_eu,
          *add_attn_res_rms_norm_tile_m, *add_attn_res_rms_norm_tile_n,
          *add_attn_res_rms_norm_tile_k, *add_attn_res_rms_norm_pack_bytes,
          s.device.wave_size, s.device.lds_bytes, s.device.scratch_a_lane))
    return error;

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

  const auto launch_all_reduce = [&]<typename T, typename W>() {
    const auto typed = reinterpret_cast<AllReduceKernel<T, W>>(kernel);
    typed<<<grid, block, 0, stream>>>(
        static_cast<const T*>(input_ptr), input_stride_m, input_stride_n,
        static_cast<int>(input_size_m), static_cast<int>(input_size_n),
        reinterpret_cast<const T* const*>(peer_inputs), peer_inputs_size,
        peer_inputs_stride_m, peer_inputs_stride_n, static_cast<T*>(summed_ptr),
        summed_stride_m, summed_stride_n, static_cast<int>(summed_size_m),
        static_cast<int>(summed_size_n), s.symmetric.partials_ptr_array,
        s.world, partials_stride_m, partials_stride_n,
        static_cast<int>(partials_size_m), static_cast<int>(partials_size_n),
        s.symmetric.staging_ptr_array, s.world, staging_stride_m,
        staging_stride_n, static_cast<int>(staging_size_m),
        static_cast<int>(staging_size_n), s.symmetric.unfenced_ptr_array,
        s.world, unfenced_stride_slot, unfenced_stride_m, unfenced_stride_n,
        static_cast<int>(unfenced_size_slot), static_cast<int>(unfenced_size_m),
        static_cast<int>(unfenced_size_n), s.unfenced_epochs,
        s.unfenced_epoch_words, int64_t{1},
        static_cast<int>(s.symmetric.unfenced_blocks),
        static_cast<int>(s.unfenced_epoch_words), s.symmetric.signals_ptr_array,
        s.world, s.symmetric.max_blocks * s.world, int64_t{s.world}, int64_t{1},
        int64_t{2}, s.symmetric.max_blocks, int64_t{s.world}, s.status_device,
        int64_t{1}, int64_t{kStatusFields}, s.rank, s.timeout_ticks);
  };
  if (dtype == at::ScalarType::BFloat16)
    launch_all_reduce.template operator()<c10::BFloat16, c10::BFloat16>();
  else
    launch_all_reduce.template operator()<c10::Half, c10::Half>();
  launch_attn_res_rows(rows, rows_call, num_blocks, eps, output_eps, stream);

  if (hipGetLastError() != hipSuccess) return error_code(Error::hip_failed);
  return std::nullopt;
}
