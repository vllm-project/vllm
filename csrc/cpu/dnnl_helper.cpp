#include <list>
#include <optional>

#include "common/memory_desc.hpp"
#include "common/memory.hpp"

#include "dnnl_helper.h"

static dnnl::engine& default_engine() {
  static dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  return engine;
}

static dnnl::stream& default_stream() {
  static dnnl::stream stream(default_engine());
  return stream;
}

void release_dnnl_matmul_handler(int64_t handler) {
  DNNLMatMulPrimitiveHandler* ptr =
      reinterpret_cast<DNNLMatMulPrimitiveHandler*>(handler);
  delete ptr;
}

DNNLScratchPadManager::DNNLScratchPadManager() : size_(0), ptr_(nullptr) {
  this->realloc(allocation_unit * 128);
}

void DNNLScratchPadManager::realloc(size_t new_size) {
  new_size = round(new_size);
  if (new_size > size_) {
    ptr_ = std::aligned_alloc(64, new_size);
    size_ = new_size;
  }
}

DNNLScratchPadManager* DNNLScratchPadManager::get_dnnl_scratchpad_manager() {
  static DNNLScratchPadManager manager;
  return &manager;
}

template <typename KT, typename VT>
class DNNLPrimitiveCache {
 public:
  using cache_value_t = std::pair<KT, VT>;
  using result_value_t = VT;
  using container_t = std::list<cache_value_t>;
  using value_iterator_t = typename container_t::iterator;
  using map_t = std::unordered_map<KT, value_iterator_t>;
  using creator_t = VT (*)();

 public:
  DNNLPrimitiveCache(size_t capacity)
      : capacity_(capacity),
        values_(),
        key_to_value_(std::min(256lu, capacity)) {
    assert(capacity > 0);
  }

  template <typename F>
  result_value_t get_or_create(const KT& key, F&& creator) {
    std::optional<value_iterator_t> value = get_value(key);
    if (value.has_value()) {
      return value.value()->second;
    } else {
      return add_value({key, creator()})->second;
    }
  }

  size_t size() const { return values_.size(); }

 private:
  void dump_data() {
    std::stringstream ss;
    ss << "table_id: " << std::hex << reinterpret_cast<size_t>(this) << std::dec
       << "\n";
    ss << "container: [";
    for (auto&& iter : values_) {
      ss << "(" << iter.first << ", " << std::hex
         << reinterpret_cast<size_t>(iter.second.get()) << "), " << std::dec;
    }
    ss << "]\n";

    ss << "map: [";
    for (auto&& iter : key_to_value_) {
      ss << "(" << iter.first << ", " << iter.second->first << ", " << std::hex
         << reinterpret_cast<size_t>(iter.second->second.get()) << std::dec
         << "), ";
    }
    ss << "]\n";
    std::printf("%s\n", ss.str().c_str());
  }

  value_iterator_t add_value(cache_value_t&& new_value) {
    if (size() == capacity_) {
      cache_value_t& last_item = values_.back();
      key_to_value_.erase(last_item.first);
      values_.pop_back();
    }

    auto& added_value_ = values_.emplace_front(std::move(new_value));
    key_to_value_.emplace(added_value_.first, values_.begin());
    return values_.begin();
  }

  std::optional<value_iterator_t> get_value(const KT& key) {
    if (key_to_value_.size() > 0 && key == values_.begin()->first) {
      return values_.begin();
    }

    auto value_map_iterator = key_to_value_.find(key);
    if (value_map_iterator != key_to_value_.end()) {
      values_.splice(values_.begin(), values_, value_map_iterator->second);
      return value_map_iterator->second;
    } else {
      return {};
    }
  }

 private:
  const size_t capacity_;
  container_t values_;
  map_t key_to_value_;
};

DNNLMatMulPrimitiveHandler::DNNLMatMulPrimitiveHandler(
    const Args& args, dnnl::memory::data_type b_type)
    : b_n_size_(args.b_n_size),
      b_n_stride_(args.b_n_stride),
      b_k_size_(args.b_k_size),
      b_k_stride_(args.b_k_stride),
      b_type_(b_type),
      c_type_(args.c_type),
      runtime_memory_ptrs_(8),
      primitive_cache_size_(args.primitive_cache_size) {
  assert(primitive_cache_size_ > 0);
}

void DNNLMatMulPrimitiveHandler::prepack_weight(
    void* original_b_ptr, dnnl::memory::desc b_target_mem_desc) {
  dnnl::memory::desc original_b_md({b_k_size_, b_n_size_}, b_type_,
                                   {b_k_stride_, b_n_stride_});
  dnnl::memory original_weight(original_b_md, default_engine(), original_b_ptr);
  dnnl::memory packed_weight(b_target_mem_desc, default_engine());
  {
    dnnl::reorder(original_weight, packed_weight)
        .execute(default_stream(), original_weight, packed_weight);
    default_stream().wait();
  }
  memory_cache_[DNNL_ARG_WEIGHTS] = packed_weight;
  b_target_mem_desc_ = b_target_mem_desc;
}

void DNNLMatMulPrimitiveHandler::set_runtime_memory_ptr(
    size_t index, dnnl_memory* memory_ptr) {
  dnnl::impl::memory_storage_t* mem_storage_ptr = memory_ptr->memory_storage();
  dnnl_memory_desc* mem_desc = const_cast<dnnl_memory_desc*>(memory_ptr->md());
  runtime_memory_ptrs_[index] = {mem_storage_ptr, mem_desc};
}

std::pair<dnnl::impl::memory_storage_t*, dnnl_memory_desc*>
DNNLMatMulPrimitiveHandler::get_runtime_memory_ptr(size_t index) {
  return runtime_memory_ptrs_[index];
}

namespace std {
template <>
struct hash<W8A8MatMulPrimitiveHandler::ClassMatmulCacheKey> {
  size_t operator()(
      const W8A8MatMulPrimitiveHandler::ClassMatmulCacheKey& val) const {
    return hash<dnnl_dim_t>()(val.b_n_size) ^ hash<dnnl_dim_t>()(val.b_k_size) ^
           hash<int>()(static_cast<int>(val.a_qs)) ^
           hash<int>()(static_cast<int>(val.b_qs)) ^ hash<bool>()(val.use_azp) ^
           hash<int>()(static_cast<int>(val.c_type));
  }
};

template <>
struct hash<W8A8MatMulPrimitiveHandler::MSizeCacheKey> {
  size_t operator()(
      const W8A8MatMulPrimitiveHandler::MSizeCacheKey& val) const {
    return hash<dnnl_dim_t>()(val.a_m_size) ^ hash<bool>()(val.use_bias) ^
           hash<int>()(static_cast<int>(val.bias_type));
  }
};

template <>
struct hash<MatMulPrimitiveHandler::ClassMatmulCacheKey> {
  size_t operator()(
      const MatMulPrimitiveHandler::ClassMatmulCacheKey& val) const {
    return hash<dnnl_dim_t>()(val.b_n_size) ^ hash<dnnl_dim_t>()(val.b_k_size);
  }
};

template <>
struct hash<MatMulPrimitiveHandler::MSizeCacheKey> {
  size_t operator()(const MatMulPrimitiveHandler::MSizeCacheKey& val) const {
    return hash<dnnl_dim_t>()(val.a_m_size) ^
           hash<dnnl_dim_t>()(val.a_m_stride) ^ hash<bool>()(val.use_bias) ^
           hash<int>()(static_cast<int>(val.bias_type));
  }
};
}  // namespace std

bool operator==(const W8A8MatMulPrimitiveHandler::ClassMatmulCacheKey& l,
                const W8A8MatMulPrimitiveHandler::ClassMatmulCacheKey& r) {
  return l.b_n_size == r.b_n_size && l.b_k_size == r.b_k_size &&
         l.a_qs == r.a_qs && l.b_qs == r.b_qs && l.use_azp == r.use_azp &&
         l.c_type == r.c_type;
}

bool operator==(const W8A8MatMulPrimitiveHandler::MSizeCacheKey& l,
                const W8A8MatMulPrimitiveHandler::MSizeCacheKey& r) {
  return l.use_bias == r.use_bias && l.a_m_size == r.a_m_size &&
         l.bias_type == r.bias_type;
}

bool operator==(const MatMulPrimitiveHandler::ClassMatmulCacheKey& l,
                const MatMulPrimitiveHandler::ClassMatmulCacheKey& r) {
  return l.b_n_size == r.b_n_size && l.b_k_size == r.b_k_size;
}

bool operator==(const MatMulPrimitiveHandler::MSizeCacheKey& l,
                const MatMulPrimitiveHandler::MSizeCacheKey& r) {
  return l.a_m_size == r.a_m_size && l.a_m_stride == r.a_m_stride &&
         l.use_bias == r.use_bias && l.bias_type == r.bias_type;
}

static std::shared_ptr<W8A8MatMulPrimitiveHandler::MSizeCache>
get_w8a8_class_primitive_cache(
    const W8A8MatMulPrimitiveHandler::ClassMatmulCacheKey& key,
    int64_t cache_size) {
  static W8A8MatMulPrimitiveHandler::ClassMatmulCache cache(128);
  assert(cache_size > 0);
  return cache.get_or_create(key, [&]() {
    return std::make_shared<W8A8MatMulPrimitiveHandler::MSizeCache>(cache_size);
  });
}

W8A8MatMulPrimitiveHandler::W8A8MatMulPrimitiveHandler(const Args& args)
    : DNNLMatMulPrimitiveHandler(
          static_cast<const DNNLMatMulPrimitiveHandler::Args&>(args),
          dnnl::memory::data_type::s8),
      use_azp_(args.use_a_zero_point),
      a_qs_(args.a_quantization_strategy),
      b_qs_(args.b_quantization_strategy),
      m_size_cache_(nullptr) {
  assert(a_qs_ != QuantizationStrategy::PER_OUTPUT_CHANNEL);
  assert(b_qs_ != QuantizationStrategy::PER_TOKEN);
  if (a_qs_ == QuantizationStrategy::PER_TOKEN) {
    assert(!use_azp_);
  };
  prepack_weight(args.b_ptr,
                 create_primitive_desc(
                     MSizeCacheKey{.a_m_size = DNNL_RUNTIME_DIM_VAL,
                                   .use_bias = false,
                                   .bias_type = dnnl::memory::data_type::undef},
                     true)
                     .weights_desc());
  init_runtime_memory_cache(args);
}

void W8A8MatMulPrimitiveHandler::execute(ExecArgs& args) {
  auto&& [a_storage, a_mem_desc] = get_runtime_memory_ptr(0);
  auto&& [c_storage, c_mem_desc] = get_runtime_memory_ptr(1);
  a_storage->set_data_handle((void*)args.a_ptr);
  a_mem_desc->dims[0] = args.a_m_size;
  c_storage->set_data_handle((void*)args.c_ptr);
  c_mem_desc->dims[0] = args.a_m_size;

  if (a_qs_ == QuantizationStrategy::PER_TENSOR) {
    auto&& [a_scale_storage, a_scale_mem_desc] = get_runtime_memory_ptr(2);
    a_scale_storage->set_data_handle((void*)args.a_scales_ptr);
  }
  if (use_azp_) {
    auto&& [a_zero_point_storage, a_zero_point_mem_desc] =
        get_runtime_memory_ptr(3);
    a_zero_point_storage->set_data_handle((void*)args.a_zero_points_ptr);
  }

  if (args.use_bias) {
    auto&& [bias_storage, bias_mem_desc] = get_runtime_memory_ptr(4);
    bias_storage->set_data_handle((void*)args.bias_ptr);
  }

  dnnl::matmul matmul = get_matmul_cache(args);

  auto&& [scratchpad_storage, scratchpad_mem_desc] = get_runtime_memory_ptr(5);
  scratchpad_storage->set_data_handle(
      DNNLScratchPadManager::get_dnnl_scratchpad_manager()->get_data<void>());

  matmul.execute(default_stream(), memory_cache_);
  default_stream().wait();
}

dnnl::matmul W8A8MatMulPrimitiveHandler::get_matmul_cache(
    const MSizeCacheKey& key) {
  if (m_size_cache_.get() == nullptr) {
    ClassMatmulCacheKey key = {.b_n_size = b_n_size_,
                               .b_k_size = b_k_size_,
                               .a_qs = a_qs_,
                               .b_qs = b_qs_,
                               .use_azp = use_azp_,
                               .c_type = c_type_};
    m_size_cache_ = get_w8a8_class_primitive_cache(key, primitive_cache_size_);
  }

  return m_size_cache_->get_or_create(key, [&]() {
    dnnl::matmul::primitive_desc desc = this->create_primitive_desc(key, false);
    auto manager = DNNLScratchPadManager::get_dnnl_scratchpad_manager();
    manager->realloc(desc.scratchpad_desc().get_size());
    return dnnl::matmul(desc);
  });
}

void W8A8MatMulPrimitiveHandler::init_runtime_memory_cache(const Args& args) {
  memory_cache_[DNNL_ARG_SRC] = dnnl::memory({{1, b_k_size_},
                                              dnnl::memory::data_type::s8,
                                              dnnl::memory::format_tag::ab},
                                             default_engine(), nullptr);
  set_runtime_memory_ptr(0, memory_cache_[DNNL_ARG_SRC].get());
  memory_cache_[DNNL_ARG_DST] =
      dnnl::memory({{1, b_n_size_}, c_type_, dnnl::memory::format_tag::ab},
                   default_engine(), nullptr);
  set_runtime_memory_ptr(1, memory_cache_[DNNL_ARG_DST].get());

  // For PER_TOKEN, scales will be applied in outside epilogue
  if (a_qs_ == QuantizationStrategy::PER_TENSOR) {
    memory_cache_[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = dnnl::memory(
        {{1}, dnnl::memory::data_type::f32, {1}}, default_engine(), nullptr);
    set_runtime_memory_ptr(
        2, memory_cache_[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC].get());
    if (use_azp_) {
      memory_cache_[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC] = dnnl::memory(
          {{1}, dnnl::memory::data_type::s32, {1}}, default_engine(), nullptr);
      set_runtime_memory_ptr(
          3, memory_cache_[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC].get());
    }
  }

  if (b_qs_ == QuantizationStrategy::PER_TENSOR) {
    memory_cache_[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] =
        dnnl::memory({{1}, dnnl::memory::data_type::f32, {1}}, default_engine(),
                     (void*)args.b_scales_ptr);
  } else if (b_qs_ == QuantizationStrategy::PER_OUTPUT_CHANNEL) {
    memory_cache_[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] =
        dnnl::memory({{b_n_size_}, dnnl::memory::data_type::f32, {1}},
                     default_engine(), (void*)args.b_scales_ptr);
  }

  memory_cache_[DNNL_ARG_BIAS] =
      dnnl::memory({{b_n_size_}, dnnl::memory::data_type::f32, {1}},
                   default_engine(), nullptr);
  set_runtime_memory_ptr(4, memory_cache_[DNNL_ARG_BIAS].get());

  memory_cache_[DNNL_ARG_SCRATCHPAD] =
      dnnl::memory({{b_n_size_}, dnnl::memory::data_type::f32, {1}},
                   default_engine(), nullptr);
  set_runtime_memory_ptr(5, memory_cache_[DNNL_ARG_SCRATCHPAD].get());
}

dnnl::matmul::primitive_desc W8A8MatMulPrimitiveHandler::create_primitive_desc(
    const MSizeCacheKey& key, bool first_time) {
  dnnl::memory::desc a_md({key.a_m_size, b_k_size_},
                          dnnl::memory::data_type::s8,
                          dnnl::memory::format_tag::ab);
  dnnl::memory::desc b_md;
  if (first_time) {
    b_md =
        dnnl::memory::desc({b_k_size_, b_n_size_}, dnnl::memory::data_type::s8,
                           dnnl::memory::format_tag::any);
  } else {
    b_md = b_target_mem_desc_;
  }
  dnnl::memory::desc c_md({key.a_m_size, b_n_size_}, c_type_,
                          dnnl::memory::format_tag::ab);

  dnnl::primitive_attr attr;

  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  // For PER_TOKEN, scales will be applied in outside epilogue
  if (a_qs_ == QuantizationStrategy::PER_TENSOR) {
    attr.set_scales_mask(DNNL_ARG_SRC, 0);
    if (use_azp_) {
      attr.set_zero_points_mask(DNNL_ARG_SRC, 0);
    }
  }

  if (b_qs_ == QuantizationStrategy::PER_TENSOR) {
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
  } else if (b_qs_ == QuantizationStrategy::PER_OUTPUT_CHANNEL) {
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, 2);
  }

  if (key.use_bias) {
    // For PER_TOKEN, bias will be applied in epilogue
    assert(a_qs_ == QuantizationStrategy::PER_TENSOR);
    dnnl::memory::desc bias_md({1, b_n_size_}, key.bias_type, {b_n_size_, 1});
    return dnnl::matmul::primitive_desc(default_engine(), a_md, b_md, bias_md,
                                        c_md, attr);
  } else {
    return dnnl::matmul::primitive_desc(default_engine(), a_md, b_md, c_md,
                                        attr);
  }
}

MatMulPrimitiveHandler::MatMulPrimitiveHandler(const Args& args)
    : DNNLMatMulPrimitiveHandler(
          static_cast<DNNLMatMulPrimitiveHandler::Args>(args), args.ab_type),
      m_size_cache_(nullptr) {
  assert(ab_type_ == dnnl::memory::data_type::f32 ||
         ab_type_ == dnnl::memory::data_type::bf16 ||
         ab_type_ == dnnl::memory::data_type::f16);
  prepack_weight(args.b_ptr,
                 create_primitive_desc(
                     MSizeCacheKey{.a_m_size = DNNL_RUNTIME_DIM_VAL,
                                   .a_m_stride = DNNL_RUNTIME_DIM_VAL,
                                   .use_bias = false,
                                   .bias_type = dnnl::memory::data_type::undef},
                     true)
                     .weights_desc());
  init_runtime_memory_cache(args);
}

static std::shared_ptr<MatMulPrimitiveHandler::MSizeCache>
get_matul_class_primitive_cache(
    const MatMulPrimitiveHandler::ClassMatmulCacheKey& key,
    int64_t cache_size) {
  static MatMulPrimitiveHandler::ClassMatmulCache cache(128);
  assert(cache_size > 0);
  return cache.get_or_create(key, [&]() {
    return std::make_shared<MatMulPrimitiveHandler::MSizeCache>(cache_size);
  });
}

void MatMulPrimitiveHandler::execute(ExecArgs& args) {
  auto&& [a_storage, a_mem_desc] = get_runtime_memory_ptr(0);
  auto&& [c_storage, c_mem_desc] = get_runtime_memory_ptr(1);
  a_storage->set_data_handle((void*)args.a_ptr);
  a_mem_desc->dims[0] = args.a_m_size;
  a_mem_desc->format_desc.blocking.strides[0] = args.a_m_stride;
  c_storage->set_data_handle((void*)args.c_ptr);
  c_mem_desc->dims[0] = args.a_m_size;

  if (args.use_bias) {
    auto&& [bias_storage, bias_mem_desc] = get_runtime_memory_ptr(2);
    bias_storage->set_data_handle((void*)args.bias_ptr);
  }

  dnnl::matmul matmul = get_matmul_cache(args);

  auto&& [scratchpad_storage, scratchpad_mem_desc] = get_runtime_memory_ptr(3);
  scratchpad_storage->set_data_handle(
      DNNLScratchPadManager::get_dnnl_scratchpad_manager()->get_data<void>());

  matmul.execute(default_stream(), memory_cache_);
  default_stream().wait();
}

dnnl::matmul MatMulPrimitiveHandler::get_matmul_cache(
    const MSizeCacheKey& key) {
  if (m_size_cache_.get() == nullptr) {
    ClassMatmulCacheKey key = {.b_n_size = b_n_size_, .b_k_size = b_k_size_};
    m_size_cache_ = get_matul_class_primitive_cache(key, primitive_cache_size_);
  }
  return m_size_cache_->get_or_create(key, [&]() {
    dnnl::matmul::primitive_desc desc = this->create_primitive_desc(key, false);
    auto manager = DNNLScratchPadManager::get_dnnl_scratchpad_manager();
    manager->realloc(desc.scratchpad_desc().get_size());
    return dnnl::matmul(desc);
  });
}

dnnl::matmul::primitive_desc MatMulPrimitiveHandler::create_primitive_desc(
    const MSizeCacheKey& key, bool first_time) {
  dnnl::memory::desc a_md;
  dnnl::memory::desc b_md;
  if (first_time) {
    a_md = dnnl::memory::desc({key.a_m_size, b_k_size_}, b_type_,
                              dnnl::memory::format_tag::ab);
    b_md = dnnl::memory::desc({b_k_size_, b_n_size_}, b_type_,
                              dnnl::memory::format_tag::any);
  } else {
    a_md = dnnl::memory::desc({key.a_m_size, b_k_size_}, b_type_,
                              {key.a_m_stride, 1});
    b_md = b_target_mem_desc_;
  }
  dnnl::memory::desc c_md({key.a_m_size, b_n_size_}, c_type_,
                          dnnl::memory::format_tag::ab);

  dnnl::primitive_attr attr;
  attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  if (key.use_bias) {
    dnnl::memory::desc bias_md({1, b_n_size_}, key.bias_type, {b_n_size_, 1});
    return dnnl::matmul::primitive_desc(default_engine(), a_md, b_md, bias_md,
                                        c_md, attr);
  } else {
    return dnnl::matmul::primitive_desc(default_engine(), a_md, b_md, c_md,
                                        attr);
  }
}

void MatMulPrimitiveHandler::init_runtime_memory_cache(const Args& args) {
  memory_cache_[DNNL_ARG_SRC] = dnnl::memory(
      {{1, b_k_size_}, b_type_, {b_k_size_, 1}}, default_engine(), nullptr);
  set_runtime_memory_ptr(0, memory_cache_[DNNL_ARG_SRC].get());
  memory_cache_[DNNL_ARG_DST] =
      dnnl::memory({{1, b_n_size_}, c_type_, dnnl::memory::format_tag::ab},
                   default_engine(), nullptr);
  set_runtime_memory_ptr(1, memory_cache_[DNNL_ARG_DST].get());

  memory_cache_[DNNL_ARG_BIAS] =
      dnnl::memory({{b_n_size_}, dnnl::memory::data_type::f32, {1}},
                   default_engine(), nullptr);
  set_runtime_memory_ptr(2, memory_cache_[DNNL_ARG_BIAS].get());

  memory_cache_[DNNL_ARG_SCRATCHPAD] =
      dnnl::memory({{b_n_size_}, dnnl::memory::data_type::f32, {1}},
                   default_engine(), nullptr);
  set_runtime_memory_ptr(3, memory_cache_[DNNL_ARG_SCRATCHPAD].get());
}
