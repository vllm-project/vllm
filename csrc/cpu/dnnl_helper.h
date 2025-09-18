#ifndef DNNL_HELPER_H
#define DNNL_HELPER_H

#include <optional>
#include <cassert>

#include "oneapi/dnnl/dnnl.hpp"

namespace c10 {
struct BFloat16;
struct Half;
}  // namespace c10

namespace dnnl {
namespace impl {
struct memory_storage_t;
struct matmul_pd_t;
struct matmul_desc_t;
}  // namespace impl
}  // namespace dnnl
struct dnnl_memory_desc;

template <typename KT, typename VT>
class DNNLPrimitiveCache;

template <typename T>
struct DNNLType {
  static constexpr dnnl::memory::data_type type =
      dnnl::memory::data_type::undef;
};

template <>
struct DNNLType<int8_t> {
  static constexpr dnnl::memory::data_type type = dnnl::memory::data_type::s8;
};

template <>
struct DNNLType<int32_t> {
  static constexpr dnnl::memory::data_type type = dnnl::memory::data_type::s32;
};

template <>
struct DNNLType<float> {
  static constexpr dnnl::memory::data_type type = dnnl::memory::data_type::f32;
};

template <>
struct DNNLType<c10::BFloat16> {
  static constexpr dnnl::memory::data_type type = dnnl::memory::data_type::bf16;
};

template <>
struct DNNLType<c10::Half> {
  static constexpr dnnl::memory::data_type type = dnnl::memory::data_type::f16;
};

template <typename T>
constexpr inline dnnl::memory::data_type get_dnnl_type() {
  return DNNLType<std::decay_t<T>>::type;
}

class DNNLScratchPadManager {
 public:
  static constexpr size_t allocation_unit = 4 * 1024 * 1024;  // 4KB

  static DNNLScratchPadManager* get_dnnl_scratchpad_manager();

  DNNLScratchPadManager();

  template <typename T>
  T* get_data() {
    return reinterpret_cast<T*>(ptr_);
  }

  static size_t round(size_t size) {
    return ((size + allocation_unit - 1) / allocation_unit) * allocation_unit;
  }

  void realloc(size_t new_size);

 private:
  size_t size_;
  void* ptr_;
};

class DNNLMatMulPrimitiveHandler {
 public:
  virtual ~DNNLMatMulPrimitiveHandler() = default;

 protected:
  struct Args {
    dnnl_dim_t b_n_size;
    dnnl_dim_t b_n_stride;
    dnnl_dim_t b_k_size;
    dnnl_dim_t b_k_stride;
    void* b_ptr;
    dnnl::memory::data_type c_type;
    size_t primitive_cache_size;
  };

 protected:
  DNNLMatMulPrimitiveHandler(const Args& args, dnnl::memory::data_type b_type);

  void prepack_weight(void* original_b_ptr,
                      dnnl::memory::desc b_target_mem_desc);

  void set_runtime_memory_ptr(size_t index, dnnl_memory* memory_ptr);

  std::pair<dnnl::impl::memory_storage_t*, dnnl_memory_desc*>
  get_runtime_memory_ptr(size_t index);

 protected:
  const dnnl_dim_t b_n_size_;
  const dnnl_dim_t b_n_stride_;
  const dnnl_dim_t b_k_size_;
  const dnnl_dim_t b_k_stride_;
  dnnl::memory::data_type b_type_;
  dnnl::memory::data_type c_type_;
  std::unordered_map<int, dnnl::memory> memory_cache_;
  std::vector<std::pair<dnnl::impl::memory_storage_t*, dnnl_memory_desc*>>
      runtime_memory_ptrs_;
  dnnl::memory::desc b_target_mem_desc_;
  int64_t primitive_cache_size_;
};

class W8A8MatMulPrimitiveHandler : public DNNLMatMulPrimitiveHandler {
 public:
  enum class QuantizationStrategy { PER_TOKEN, PER_TENSOR, PER_OUTPUT_CHANNEL };

  struct Args : public DNNLMatMulPrimitiveHandler::Args {
    bool use_a_zero_point;
    QuantizationStrategy a_quantization_strategy;
    QuantizationStrategy b_quantization_strategy;
    float* b_scales_ptr;
  };

  struct ClassMatmulCacheKey {
    dnnl_dim_t b_n_size;
    dnnl_dim_t b_k_size;
    QuantizationStrategy a_qs;
    QuantizationStrategy b_qs;
    bool use_azp;
    dnnl::memory::data_type c_type;

    friend bool operator==(const ClassMatmulCacheKey& l,
                           const ClassMatmulCacheKey& r);
  };

  struct MSizeCacheKey {
    dnnl_dim_t a_m_size;
    bool use_bias;
    dnnl::memory::data_type bias_type;

    friend bool operator==(const MSizeCacheKey& l, const MSizeCacheKey& r);
  };

  using MSizeCache = DNNLPrimitiveCache<MSizeCacheKey, dnnl::matmul>;
  using ClassMatmulCache =
      DNNLPrimitiveCache<ClassMatmulCacheKey, std::shared_ptr<MSizeCache>>;

  struct ExecArgs : public MSizeCacheKey {
    const int8_t* a_ptr;
    const float* a_scales_ptr;
    const int32_t* a_zero_points_ptr;
    const void* bias_ptr;
    void* c_ptr;
  };

 public:
  W8A8MatMulPrimitiveHandler(const Args& args);

  QuantizationStrategy get_input_scale_strategy() const { return a_qs_; }

  bool get_input_use_zero_point() const { return use_azp_; }

  void execute(ExecArgs& args);

 private:
  dnnl::matmul::primitive_desc create_primitive_desc(const MSizeCacheKey& key,
                                                     bool first_time);

  void init_runtime_memory_cache(const Args& args);

  dnnl::matmul get_matmul_cache(const MSizeCacheKey& key);

 private:
  const bool use_azp_;
  const QuantizationStrategy a_qs_;
  const QuantizationStrategy b_qs_;
  std::shared_ptr<MSizeCache> m_size_cache_;
};

class MatMulPrimitiveHandler : public DNNLMatMulPrimitiveHandler {
 public:
  struct Args : public DNNLMatMulPrimitiveHandler::Args {
    dnnl::memory::data_type ab_type;
  };

  struct ClassMatmulCacheKey {
    dnnl_dim_t b_n_size;
    dnnl_dim_t b_k_size;

    friend bool operator==(const ClassMatmulCacheKey& l,
                           const ClassMatmulCacheKey& r);
  };

  struct MSizeCacheKey {
    dnnl_dim_t a_m_size;
    dnnl_dim_t a_m_stride;
    bool use_bias;
    dnnl::memory::data_type bias_type;

    friend bool operator==(const MSizeCacheKey& l, const MSizeCacheKey& r);
  };

  using MSizeCache = DNNLPrimitiveCache<MSizeCacheKey, dnnl::matmul>;
  using ClassMatmulCache =
      DNNLPrimitiveCache<ClassMatmulCacheKey, std::shared_ptr<MSizeCache>>;

  struct ExecArgs : public MSizeCacheKey {
    const void* a_ptr;
    const void* bias_ptr;
    void* c_ptr;
  };

 public:
  MatMulPrimitiveHandler(const Args& args);

  void execute(ExecArgs& args);

 private:
  dnnl::matmul::primitive_desc create_primitive_desc(const MSizeCacheKey& key,
                                                     bool first_time);

  void init_runtime_memory_cache(const Args& args);

  dnnl::matmul get_matmul_cache(const MSizeCacheKey& key);

 private:
  std::shared_ptr<MSizeCache> m_size_cache_;
};

#endif
