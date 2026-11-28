// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <optional>

#include "kernels.h"

namespace octave_rdna4 {
namespace {

using at::Tensor;
constexpr int kHeadDim = 256;

Activation activation(const Tensor& tensor, at::Device device,
                      at::ScalarType dtype) {
  TORCH_CHECK_VALUE(
      tensor.is_cuda() && tensor.device() == device &&
          tensor.scalar_type() == dtype &&
          (dtype == at::kHalf || dtype == at::kBFloat16) && tensor.dim() == 3 &&
          tensor.size(2) == kHeadDim && tensor.stride(2) == 1,
      "Expected (tokens, heads, 256) FP16 or BF16 on one ROCm device");
  return {tensor.const_data_ptr(), tensor.stride(0), tensor.stride(1)};
}

PagedCache paged_cache(const Tensor& tensor, at::Device device, int64_t fmt) {
  const int slot_bytes = cache_slot_bytes(fmt);
  TORCH_CHECK_VALUE(slot_bytes > 0, "Unsupported RDNA4 Octave format ", fmt);
  TORCH_CHECK_VALUE(
      tensor.device() == device && tensor.scalar_type() == at::kByte &&
          tensor.dim() == 4 && tensor.size(3) == slot_bytes &&
          tensor.size(1) > 0 && tensor.size(2) > 0 && tensor.stride(3) == 1 &&
          tensor.stride(0) % 4 == 0 && tensor.stride(1) % 4 == 0 &&
          tensor.stride(2) % 4 == 0 &&
          reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 4 == 0,
      "Expected aligned uint8 cache (pages, KV heads, page size, slot)");
  return {tensor.data_ptr<uint8_t>(),
          tensor.stride(0),
          tensor.stride(1),
          tensor.stride(2),
          static_cast<int>(tensor.size(2)),
          static_cast<int>(tensor.size(1))};
}

template <typename T>
T* vector_data(const Tensor& tensor, int64_t length, at::Device device) {
  TORCH_CHECK_VALUE(
      tensor.dim() == 1 && tensor.size(0) == length && tensor.is_contiguous() &&
          tensor.device() == device &&
          tensor.scalar_type() == c10::CppTypeToScalarType<T>::value,
      "Expected a contiguous ", c10::CppTypeToScalarType<T>::value,
      " vector with ", length, " entries on ", device);
  return tensor.data_ptr<T>();
}

const uint32_t* sign_words(const Tensor& signs, at::Device device) {
  return reinterpret_cast<const uint32_t*>(
      vector_data<int32_t>(signs, kHeadDim / 32, device));
}

const int32_t* int_rows(const Tensor& tensor, at::Device device,
                        const char* name) {
  TORCH_CHECK_VALUE(
      tensor.dim() == 2 && tensor.stride(1) == 1 &&
          tensor.scalar_type() == at::kInt && tensor.device() == device,
      "Expected ", name, " with contiguous int32 rows on ", device);
  return tensor.const_data_ptr<int32_t>();
}

void check_buffer(const Tensor& tensor, at::IntArrayRef shape,
                  at::ScalarType dtype, at::Device device) {
  TORCH_CHECK_VALUE(tensor.sizes() == shape && tensor.scalar_type() == dtype &&
                        tensor.device() == device && tensor.is_contiguous(),
                    "Expected contiguous ", dtype, " buffer of shape ", shape,
                    " on ", device);
}

Shadow shadow_table(const Tensor& table, const Tensor& tags,
                    const Tensor* locks, at::Device device,
                    at::ScalarType dtype, int64_t heads) {
  TORCH_CHECK_VALUE(table.dim() == 3, "Expected a 3D exact K/V table");
  const int64_t entries = table.size(0);
  TORCH_CHECK_VALUE(entries > 0 && (entries & (entries - 1)) == 0,
                    "Expected a power-of-two exact K/V table");
  check_buffer(table, {entries, heads, 2 * kHeadDim}, dtype, device);
  return {table.data_ptr(), vector_data<int32_t>(tags, entries, device),
          locks ? vector_data<int32_t>(*locks, entries, device) : nullptr,
          static_cast<int>(entries)};
}

struct AttentionInputs {
  Activation query;
  PagedCache cache;
  const int32_t* table;
  const uint32_t* signs;
  float* out;
  float* partials;
  int tokens, heads, dtype;
};

AttentionInputs attention_inputs(const Tensor& out, const Tensor& query,
                                 const Tensor& cache, const Tensor& block_table,
                                 const std::optional<Tensor>& scratch,
                                 const Tensor& signs, int64_t splits,
                                 int64_t fmt, int max_splits) {
  const auto device = query.device();
  const auto dtype = query.scalar_type();
  const auto q = activation(query, device, dtype);
  const auto pages = paged_cache(cache, device, fmt);
  const auto tokens = query.size(0), heads = query.size(1);
  TORCH_CHECK_VALUE(heads > 0 && heads % pages.heads == 0 && splits >= 1 &&
                        splits <= max_splits,
                    "Invalid RDNA4 Octave GQA group or split count");
  check_buffer(out, query.sizes(), at::kFloat, device);
  float* partials = nullptr;
  if (splits > 1) {
    TORCH_CHECK_VALUE(scratch.has_value(), "Split attention requires scratch");
    check_buffer(*scratch, {tokens, heads, splits, kHeadDim + 2}, at::kFloat,
                 device);
    partials = scratch->data_ptr<float>();
  }
  return {q,
          pages,
          int_rows(block_table, device, "page table"),
          sign_words(signs, device),
          out.data_ptr<float>(),
          partials,
          static_cast<int>(tokens),
          static_cast<int>(heads),
          dtype == at::kBFloat16};
}

void check_chunk(const Tensor& key, const Tensor& value, const Tensor& table,
                 const AttentionInputs& args, int64_t requests) {
  TORCH_CHECK_VALUE(
      (key.sizes() ==
       at::IntArrayRef{args.tokens, args.cache.heads, kHeadDim}) &&
          value.sizes() == key.sizes(),
      "Expected one unquantized chunk K/V row per query");
  TORCH_CHECK_VALUE(
      requests == table.size(0) && (args.tokens == 0 || requests > 0),
      "Prefill sequence lengths must match the page table");
}

void check_launch(int status) {
  TORCH_CHECK(status == hipSuccess, "RDNA4 Octave launch failed: ",
              hipGetErrorString(static_cast<hipError_t>(status)));
}

void cache_store(const Tensor& key, const Tensor& value, const Tensor& cache,
                 const Tensor& slot_mapping, const Tensor& signs, int64_t fmt,
                 bool use_geometry) {
  const auto device = key.device();
  const auto dtype = key.scalar_type();
  const auto k = activation(key, device, dtype),
             v = activation(value, device, dtype);
  const auto pages = paged_cache(cache, device, fmt);
  const auto tokens = slot_mapping.numel();
  TORCH_CHECK_VALUE(key.sizes() == value.sizes() &&
                        key.size(1) == pages.heads && tokens <= key.size(0),
                    "Cache head count or slot count does not match K/V");
  const c10::cuda::CUDAGuard guard(device);
  const auto launch = use_geometry ? geometry::launch_store : launch_store;
  check_launch(launch(k, v, pages,
                      vector_data<int64_t>(slot_mapping, tokens, device),
                      sign_words(signs, device), tokens, dtype == at::kBFloat16,
                      fmt, c10::cuda::getCurrentCUDAStream().stream()));
}

void decode(const Tensor& out, const Tensor& query, const Tensor& cache,
            const Tensor& block_table, const Tensor& q_to_req,
            const Tensor& q_to_klen, const std::optional<Tensor>& scratch,
            const Tensor& signs, double scale, int64_t splits, int64_t fmt,
            bool use_geometry) {
  const auto x = attention_inputs(out, query, cache, block_table, scratch,
                                  signs, splits, fmt, 256);
  const auto device = query.device();
  const c10::cuda::CUDAGuard guard(device);
  const auto launch = use_geometry ? geometry::launch_decode : launch_decode;
  check_launch(launch(x.out, x.partials, x.query, x.cache, x.table,
                      block_table.stride(0),
                      vector_data<int32_t>(q_to_req, x.tokens, device),
                      vector_data<int32_t>(q_to_klen, x.tokens, device),
                      x.signs, x.tokens, x.heads, splits, scale, x.dtype, fmt,
                      c10::cuda::getCurrentCUDAStream().stream()));
}

void prefill(const Tensor& out, const Tensor& query, const Tensor& key,
             const Tensor& value, const Tensor& cache,
             const Tensor& block_table, const Tensor& starts,
             const Tensor& lengths, const std::optional<Tensor>& scratch,
             const Tensor& signs, double scale, int64_t splits, int64_t fmt,
             bool use_geometry) {
  const auto x = attention_inputs(out, query, cache, block_table, scratch,
                                  signs, splits, fmt, 256);
  const auto device = query.device();
  const auto dtype = query.scalar_type();
  const auto k = activation(key, device, dtype),
             v = activation(value, device, dtype);
  const auto requests = lengths.numel();
  check_chunk(key, value, block_table, x, requests);
  const c10::cuda::CUDAGuard guard(device);
  const auto launch = use_geometry ? geometry::launch_prefill : launch_prefill;
  check_launch(launch(x.out, x.partials, x.query, k, v, x.cache, x.table,
                      block_table.stride(0),
                      vector_data<int32_t>(starts, requests + 1, device),
                      vector_data<int32_t>(lengths, requests, device), requests,
                      x.signs, x.tokens, x.heads, splits, scale, x.dtype, fmt,
                      c10::cuda::getCurrentCUDAStream().stream()));
}

void prefill_expanded(const Tensor& out, const Tensor& query, const Tensor& key,
                      const Tensor& value, const Tensor& cache,
                      const Tensor& block_table, const Tensor& starts,
                      const Tensor& lengths, const Tensor& key_tmp,
                      const Tensor& value_tmp, const Tensor& query_tmp,
                      const Tensor& signs, double scale, int64_t fmt,
                      const std::optional<Tensor>& repair_buffer,
                      const std::optional<Tensor>& correction_buffer) {
  const auto x = attention_inputs(out, query, cache, block_table, std::nullopt,
                                  signs, 1, fmt, 1);
  const auto device = query.device();
  const auto dtype = query.scalar_type();
  const auto k = activation(key, device, dtype),
             v = activation(value, device, dtype);
  const auto requests = lengths.numel();
  check_chunk(key, value, block_table, x, requests);
  TORCH_CHECK_VALUE(std::isfinite(scale) && scale > 0 && key_tmp.dim() == 4,
                    "Invalid RDNA4 expanded prefill arguments");
  const auto capacity = key_tmp.size(2);
  TORCH_CHECK_VALUE(capacity > 0 && capacity % 32 == 0,
                    "Expected capacity aligned to 32 KV tokens");
  const int64_t kv_shape[] = {requests, x.cache.heads, capacity, kHeadDim};
  check_buffer(key_tmp, kv_shape, at::kHalf, device);
  check_buffer(value_tmp, kv_shape, at::kHalf, device);
  check_buffer(query_tmp, query.sizes(), at::kHalf, device);
  TORCH_CHECK_VALUE(
      reinterpret_cast<uintptr_t>(key_tmp.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(value_tmp.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(query_tmp.data_ptr()) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(x.out) % 32 == 0,
      "Expected aligned RDNA4 expanded prefill buffers");
  if (x.tokens == 0) return;
  const auto* start_data = vector_data<int32_t>(starts, requests + 1, device);
  const auto* length_data = vector_data<int32_t>(lengths, requests, device);
  const c10::cuda::CUDAGuard guard(device);
  const auto repair =
      repair_buffer
          ? *repair_buffer
          : at::empty({x.tokens, x.heads, 2}, query.options().dtype(at::kByte));
  check_buffer(repair, {x.tokens, x.heads, 2}, at::kByte, device);
  // Half-precision key residuals followed by 32 FP32 maxima per KV head.
  const int64_t correction_bytes =
      (capacity * kHeadDim * 2 + 32 * sizeof(float)) * requests * x.cache.heads;
  const auto correction =
      correction_buffer
          ? *correction_buffer
          : at::empty({correction_bytes}, query.options().dtype(at::kByte));
  check_buffer(correction, {correction_bytes}, at::kByte, device);
  TORCH_CHECK_VALUE(
      reinterpret_cast<uintptr_t>(correction.data_ptr()) % 16 == 0,
      "Expected aligned expanded key correction buffer");
  query_tmp.copy_(query);
  check_launch(launch_prefill_expanded(
      x.out, key_tmp.data_ptr(), value_tmp.data_ptr(),
      query_tmp.const_data_ptr(), x.query, k, v, x.cache, x.table,
      block_table.stride(0), start_data, length_data, requests, capacity,
      x.signs, x.tokens, x.heads, scale, x.dtype, fmt, correction.data_ptr(),
      repair.data_ptr<uint8_t>(), c10::cuda::getCurrentCUDAStream().stream()));
}

void prefill_streamed(const Tensor& out, const Tensor& query, const Tensor& key,
                      const Tensor& value, const Tensor& cache,
                      const Tensor& table, const Tensor& starts,
                      const Tensor& lengths, const Tensor& partials,
                      const Tensor& staging, const Tensor& chunk_lse,
                      const Tensor& repair, const Tensor& signs, double scale,
                      int64_t fmt) {
  const auto x = attention_inputs(out, query, cache, table, std::nullopt, signs,
                                  1, fmt, 1);
  const auto device = query.device();
  const auto k = activation(key, device, query.scalar_type());
  const auto v = activation(value, device, query.scalar_type());
  const int requests = lengths.numel();
  check_chunk(key, value, table, x, requests);
  TORCH_CHECK_VALUE(partials.dim() == 4 && partials.size(0) >= 1 &&
                        partials.size(0) <= 32 && x.cache.page_size % 16 == 0 &&
                        std::isfinite(scale) && scale > 0,
                    "Invalid streamed prefill geometry");
  const int splits = partials.size(0);
  check_buffer(partials, {splits, x.tokens, x.heads, kHeadDim + 2}, at::kFloat,
               device);
  const int64_t query_bytes = int64_t(x.tokens) * x.heads * 2 * kHeadDim;
  const int64_t chunk_bytes =
      int64_t(x.tokens / 32 + requests) * 32 * x.cache.heads * kHeadDim * 4;
  check_buffer(staging, {std::max(query_bytes, chunk_bytes)}, at::kChar,
               device);
  if (chunk_lse.dim() == 3)
    check_buffer(chunk_lse, {2, x.tokens, x.heads}, at::kFloat, device);
  else
    check_buffer(chunk_lse, {x.tokens, x.heads}, at::kFloat, device);
  check_buffer(repair, {x.tokens, x.heads}, at::kByte, device);
  TORCH_CHECK_VALUE(reinterpret_cast<uintptr_t>(staging.data_ptr()) % 16 == 0 &&
                        reinterpret_cast<uintptr_t>(x.out) % 32 == 0,
                    "Expected aligned streamed prefill buffers");
  const auto* start_data = vector_data<int32_t>(starts, requests + 1, device);
  const auto* length_data = vector_data<int32_t>(lengths, requests, device);
  if (!x.tokens) return;
  const c10::cuda::CUDAGuard guard(device);
  check_launch(launch_prefill_streamed(
      x.out, x.query, k, v, x.cache, x.table, table.stride(0),
      table.size(1) * x.cache.page_size, start_data, length_data, requests,
      x.signs, x.tokens, x.heads, scale, x.dtype, fmt, splits,
      partials.data_ptr<float>(), staging.data_ptr<int8_t>(),
      chunk_lse.data_ptr<float>(), repair.data_ptr<uint8_t>(),
      c10::cuda::getCurrentCUDAStream().stream()));
}

void window_store(const Tensor& key, const Tensor& value,
                  const Tensor& slot_mapping, const Tensor& positions,
                  const Tensor& wtab, const Tensor& wtags, const Tensor& wlocks,
                  const Tensor& stab, const Tensor& stags,
                  const Tensor& slocks) {
  const auto device = key.device();
  const auto dtype = key.scalar_type();
  const auto k = activation(key, device, dtype),
             v = activation(value, device, dtype);
  const auto tokens = slot_mapping.numel(), heads = key.size(1);
  TORCH_CHECK_VALUE(key.sizes() == value.sizes() && tokens <= key.size(0),
                    "Exact K/V rows and slot counts must match");
  const auto recent = shadow_table(wtab, wtags, &wlocks, device, dtype, heads);
  const auto sinks = shadow_table(stab, stags, &slocks, device, dtype, heads);
  const c10::cuda::CUDAGuard guard(device);
  check_launch(geometry::launch_window_store(
      k, v, vector_data<int64_t>(slot_mapping, tokens, device),
      vector_data<int32_t>(positions, tokens, device), recent, sinks, tokens,
      heads, dtype == at::kBFloat16,
      c10::cuda::getCurrentCUDAStream().stream()));
}

void decode_sparse(const Tensor& out, const Tensor& query, const Tensor& cache,
                   const Tensor& block_table, const Tensor& q_to_req,
                   const Tensor& indices, const std::optional<Tensor>& scratch,
                   const Tensor& signs, double scale, int64_t splits,
                   int64_t fmt, const std::optional<Tensor>& positions,
                   const std::optional<Tensor>& wtab,
                   const std::optional<Tensor>& wtags,
                   const std::optional<Tensor>& stab,
                   const std::optional<Tensor>& stags) {
  const auto x = attention_inputs(out, query, cache, block_table, scratch,
                                  signs, splits, fmt, 1024);
  const auto device = query.device();
  const auto dtype = query.scalar_type();
  const auto* selected = int_rows(indices, device, "sparse index table");
  TORCH_CHECK_VALUE(indices.size(0) == x.tokens,
                    "Expected one sparse index row per query");
  const int32_t* position_data = nullptr;
  Shadow recent{}, sinks{};
  if (positions) {
    TORCH_CHECK_VALUE(wtab && wtags && stab && stags,
                      "Exact sparse attention requires both tagged tables");
    position_data = vector_data<int32_t>(*positions, x.tokens, device);
    recent = shadow_table(*wtab, *wtags, nullptr, device, dtype, x.cache.heads);
    sinks = shadow_table(*stab, *stags, nullptr, device, dtype, x.cache.heads);
  }
  const c10::cuda::CUDAGuard guard(device);
  check_launch(geometry::launch_decode_sparse(
      x.out, x.partials, x.query, x.cache, x.table, block_table.stride(0),
      block_table.size(1), block_table.size(0), cache.size(0),
      vector_data<int32_t>(q_to_req, x.tokens, device), selected,
      indices.stride(0), indices.size(1), x.signs, position_data, recent, sinks,
      x.tokens, x.heads, splits, scale, x.dtype, fmt,
      c10::cuda::getCurrentCUDAStream().stream()));
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def(
      "octave_rdna4_cache_store(Tensor key, Tensor value, Tensor(a!) cache, "
      "Tensor slot_mapping, Tensor signs, int fmt, bool use_geometry) -> ()");
  ops.impl("octave_rdna4_cache_store", at::kCUDA, &cache_store);
  ops.def(
      "octave_rdna4_decode(Tensor(a!) out, Tensor query, Tensor cache, "
      "Tensor block_table, Tensor q_to_req, Tensor q_to_klen, "
      "Tensor(b!)? scratch, Tensor signs, float scale, int splits, int fmt, "
      "bool use_geometry) -> ()");
  ops.impl("octave_rdna4_decode", at::kCUDA, &decode);
  ops.def(
      "octave_rdna4_prefill(Tensor(a!) out, Tensor query, Tensor key, "
      "Tensor value, Tensor cache, Tensor block_table, Tensor starts, "
      "Tensor lengths, Tensor(b!)? scratch, Tensor signs, float scale, "
      "int splits, int fmt, bool use_geometry) -> ()");
  ops.impl("octave_rdna4_prefill", at::kCUDA, &prefill);
  ops.def(
      "octave_rdna4_prefill_expanded(Tensor(a!) out, Tensor query, Tensor key, "
      "Tensor value, Tensor cache, Tensor block_table, Tensor starts, "
      "Tensor lengths, Tensor(b!) key_tmp, Tensor(c!) value_tmp, "
      "Tensor(d!) query_tmp, Tensor signs, float scale, int fmt, Tensor(e!)? "
      "repair=None, Tensor(f!)? correction=None) -> ()");
  ops.impl("octave_rdna4_prefill_expanded", at::kCUDA, &prefill_expanded);
  ops.def(
      "octave_rdna4_prefill_streamed(Tensor(a!) out, Tensor query, Tensor key, "
      "Tensor value, Tensor cache, Tensor table, Tensor starts, Tensor "
      "lengths, "
      "Tensor(b!) partials, Tensor(c!) q8, Tensor(d!) stats, Tensor(e!) "
      "repair, "
      "Tensor signs, float scale, int fmt) -> ()");
  ops.impl("octave_rdna4_prefill_streamed", at::kCUDA, &prefill_streamed);
  ops.def(
      "octave_rdna4_window_store(Tensor key, Tensor value, Tensor "
      "slot_mapping, "
      "Tensor positions, Tensor(a!) wtab, Tensor(b!) wtags, Tensor(c!) wlocks, "
      "Tensor(d!) stab, Tensor(e!) stags, Tensor(f!) slocks) -> ()");
  ops.impl("octave_rdna4_window_store", at::kCUDA, &window_store);
  ops.def(
      "octave_rdna4_decode_sparse(Tensor(a!) out, Tensor query, Tensor cache, "
      "Tensor block_table, Tensor q_to_req, Tensor indices, Tensor(b!)? "
      "scratch, "
      "Tensor signs, float scale, int splits, int fmt, Tensor? positions=None, "
      "Tensor? wtab=None, Tensor? wtags=None, Tensor? stab=None, "
      "Tensor? stags=None) -> ()");
  ops.impl("octave_rdna4_decode_sparse", at::kCUDA, &decode_sparse);
}
}  // namespace octave_rdna4
