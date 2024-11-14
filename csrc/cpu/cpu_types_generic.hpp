#ifndef CPU_TYPES_GENERIC_HPP
#define CPU_TYPES_GENERIC_HPP

#include <memory>
#include <torch/torch.h>

#include "cpu_types_arm.hpp"

namespace vec_op {

template <typename T> struct Vec {
  constexpr static int get_elem_num() { return T::VEC_ELEM_NUM; }
};

struct FP32Vec8;
struct FP32Vec16;
struct INT32Vec16;

/****************************************/
/*               FP16Vec8               */
/****************************************/
struct FP16Vec8 : public Vec<FP16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  explicit FP16Vec8(const void* ptr)
      : impl_(FP16Vec8Impl(ptr)) {}

  explicit FP16Vec8(const FP32Vec8&);

  void save(void* ptr) const { impl_.save(ptr); }

  FP16Vec8Impl impl_;
};

/****************************************/
/*               FP16Vec16              */
/****************************************/
struct FP16Vec16 : public Vec<FP16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  explicit FP16Vec16(const void* ptr)
      : impl_(FP16Vec16Impl(ptr)) {}

  explicit FP16Vec16(const FP32Vec16&);

  void save(void* ptr) const { impl_.save(ptr); }

  void save(void* ptr, const int elem_num) const { impl_.save(ptr, elem_num); }

  FP16Vec16Impl impl_;
};

/****************************************/
/*               BF16Vec8               */
/****************************************/
struct BF16Vec8 : public Vec<BF16Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  explicit BF16Vec8(const void* ptr)
      : impl_(BF16Vec8Impl(ptr)) {}

  explicit BF16Vec8(const FP32Vec8&);

  void save(void* ptr) const { impl_.save(ptr); }

  BF16Vec8Impl impl_;
};

/****************************************/
/*               BF16Vec16              */
/****************************************/
struct BF16Vec16 : public Vec<BF16Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  explicit BF16Vec16(const void* ptr)
      : impl_(BF16Vec16Impl(ptr)) {}

  explicit BF16Vec16(const FP32Vec16&);

  void save(void* ptr) const { impl_.save(ptr); }

  void save(void* ptr, const int elem_num) const { impl_.save(ptr, elem_num); }

  BF16Vec16Impl impl_;
};

/****************************************/
/*               BF16Vec32              */
/****************************************/
struct BF16Vec32 : public Vec<BF16Vec32> {
  constexpr static int VEC_ELEM_NUM = 32;

  explicit BF16Vec32(const void* ptr)
      : impl_(BF16Vec32Impl(ptr)) {}

  explicit BF16Vec32(const BF16Vec8& v)
      : impl_(BF16Vec32Impl(v.impl_)) {}

  void save(void* ptr) const { impl_.save(ptr); }

  BF16Vec32Impl impl_;
};

/****************************************/
/*               FP32Vec4               */
/****************************************/
struct FP32Vec4 : public Vec<FP32Vec4> {
  constexpr static int VEC_ELEM_NUM = 4;

  explicit FP32Vec4(const float* ptr)
      : impl_(FP32Vec4Impl(ptr)) {}

  explicit FP32Vec4(const float& v)
      : impl_(FP32Vec4Impl(v)) {}

  explicit FP32Vec4()
      : impl_(FP32Vec4Impl()) {}

  explicit FP32Vec4(const FP32Vec4& v)
      : impl_(FP32Vec4Impl(v.impl_)) {}

  void save(float* ptr) const { impl_.save(ptr); }

  FP32Vec4Impl impl_;
};

/****************************************/
/*               FP32Vec8               */
/****************************************/
struct FP32Vec8 : public Vec<FP32Vec8> {
  constexpr static int VEC_ELEM_NUM = 8;

  explicit FP32Vec8(const void* ptr)
      : impl_(FP32Vec8Impl(ptr)) {}

  explicit FP32Vec8(const float& v)
      : impl_(FP32Vec8Impl(v)) {}

  explicit FP32Vec8()
      : impl_(FP32Vec8Impl()) {}

  explicit FP32Vec8(const FP32Vec8& v)
      : impl_(v.impl_) {}

  explicit FP32Vec8(const FP16Vec8& v)
      : impl_(FP32Vec8Impl(v.impl_)) {}

  explicit FP32Vec8(const BF16Vec8& v)
      : impl_(FP32Vec8Impl(v.impl_)) {}

  explicit FP32Vec8(FP32Vec8Impl&& v) : impl_(FP32Vec8Impl(v)) {}

  float reduce_sum() const { return impl_.reduce_sum(); }

  FP32Vec8 exp() const { return FP32Vec8(impl_.exp()); }

  FP32Vec8 tanh() const { return FP32Vec8(impl_.tanh()); }

  FP32Vec8 er() const { return FP32Vec8(impl_.er()); }

  FP32Vec8 operator*(const FP32Vec8& b) const {
    return FP32Vec8(impl_ * b.impl_);
  }

  FP32Vec8 operator+(const FP32Vec8& b) const {
    return FP32Vec8(impl_ + b.impl_);
  }

  FP32Vec8 operator-(const FP32Vec8& b) const {
    return FP32Vec8(impl_ - b.impl_);
  }

  FP32Vec8 operator/(const FP32Vec8& b) const {
    return FP32Vec8(impl_ / b.impl_);
  }

  void save(float* ptr) const { impl_.save(ptr); }

  FP32Vec8Impl impl_;
};

/****************************************/
/*               FP32Vec16              */
/****************************************/
struct FP32Vec16 : public Vec<FP32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  explicit FP32Vec16(const void* ptr)
      : impl_(FP32Vec16Impl(ptr)) {}

  explicit FP32Vec16(float v)
      : impl_(FP32Vec16Impl(v)) {}

  explicit FP32Vec16()
      : impl_(FP32Vec16Impl()) {}

  explicit FP32Vec16(const FP32Vec4& v)
      : impl_(FP32Vec16Impl(v.impl_)) {}

  explicit FP32Vec16(const FP32Vec8& v)
      : impl_(FP32Vec16Impl(v.impl_)) {}

  explicit FP32Vec16(const FP32Vec16& v)
      : impl_(FP32Vec16Impl(v.impl_)) {}

  explicit FP32Vec16(const BF16Vec16& v)
      : impl_(FP32Vec16Impl(v.impl_)) {}

  explicit FP32Vec16(const FP16Vec16& v)
      : impl_(FP32Vec16Impl(v.impl_)) {}

  explicit FP32Vec16(const FP16Vec8& v)
      : impl_(FP32Vec16Impl(v.impl_)) {}

  explicit FP32Vec16(const BF16Vec8& v)
      : impl_(FP32Vec16Impl(v.impl_)) {}

  explicit FP32Vec16(const INT32Vec16& v);

  explicit FP32Vec16(FP32Vec16Impl&& v)
      : impl_(FP32Vec16Impl(v)) {}

  FP32Vec16 operator*(const FP32Vec16& b) const {
    return FP32Vec16(impl_ * b.impl_);
  }

  FP32Vec16 operator+(const FP32Vec16& b) const {
    return FP32Vec16(impl_ + b.impl_);
  }

  FP32Vec16 operator-(const FP32Vec16& b) const {
    return FP32Vec16(impl_ - b.impl_);
  }

  FP32Vec16 operator/(const FP32Vec16& b) const {
    return FP32Vec16(impl_ / b.impl_);
  }

  FP32Vec16 clamp(const FP32Vec16& min, const FP32Vec16& max) const {
    return FP32Vec16(impl_.clamp(min.impl_, max.impl_));
  }

  FP32Vec16 max(const FP32Vec16& b) const {
    return FP32Vec16(impl_.max(b.impl_));
  }

  FP32Vec16 max(const FP32Vec16& b, const int elem_num) const {
    return FP32Vec16(impl_.max(b.impl_, elem_num));
  }

  FP32Vec16 min(const FP32Vec16& b) const {
    return FP32Vec16(impl_.min(b.impl_));
  }

  FP32Vec16 min(const FP32Vec16& b, const int elem_num) const {
    return FP32Vec16(impl_.min(b.impl_, elem_num));
  }

  FP32Vec16 abs() const {
    return FP32Vec16(impl_.abs());
  }

  float reduce_max() const { return impl_.reduce_max(); }

  float reduce_min() const { return impl_.reduce_min(); }

  float reduce_sum() const { return impl_.reduce_sum(); }

  template <int group_size> float reduce_sub_sum(int idx) const {
    static_assert(VEC_ELEM_NUM % group_size == 0);
    return impl_.reduce_sub_sum(idx, group_size);
  }

  void save(float* ptr) const { impl_.save(ptr); }

  void save(float* ptr, const int elem_num) const { impl_.save(ptr, elem_num); }

  FP32Vec16Impl impl_;
};

/****************************************/
/*               INT32Vec16             */
/****************************************/
struct INT32Vec16 : public Vec<INT32Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  explicit INT32Vec16(const void* ptr)
      : impl_(INT32Vec16Impl(ptr)) {}

  void save(int32_t* ptr) const { impl_.save(ptr); }

  void save(int32_t* ptr, const int elem_num) const { impl_.save(ptr, elem_num); }

  INT32Vec16Impl impl_;
};

/****************************************/
/*               INT8Vec16              */
/****************************************/
struct INT8Vec16 : public Vec<INT8Vec16> {
  constexpr static int VEC_ELEM_NUM = 16;

  explicit INT8Vec16(const int8_t* ptr)
      : impl_(INT8Vec16Impl(ptr)) {}

  explicit INT8Vec16(const FP32Vec16& v)
      : impl_(INT8Vec16Impl(v.impl_)) {}

  void save(int8_t* ptr) const { impl_.save(ptr); }

  void save(int8_t* ptr, const int elem_num) const { impl_.save(ptr, elem_num); }

  INT8Vec16Impl impl_;
};

inline FP16Vec8::FP16Vec8(const FP32Vec8& v)
    : impl_(FP16Vec8Impl(v.impl_)) {}

inline FP16Vec16::FP16Vec16(const FP32Vec16& v)
    : impl_(FP16Vec16Impl(v.impl_)) {}

inline BF16Vec8::BF16Vec8(const FP32Vec8& v)
    : impl_(BF16Vec8Impl(v.impl_)) {}

inline BF16Vec16::BF16Vec16(const FP32Vec16& v)
    : impl_(BF16Vec16Impl(v.impl_)) {}

inline FP32Vec16::FP32Vec16(const INT32Vec16& v)
    : impl_(FP32Vec16Impl(v.impl_)) {}

/****************************************/
/*                  Utils               */
/****************************************/
inline void fma(FP32Vec16& acc, FP32Vec16& a, FP32Vec16& b) {
  fma_impl(acc.impl_, a.impl_, b.impl_);
}

inline void fma(FP32Vec16& acc, BF16Vec32& a, BF16Vec32& b) {
  fma_impl(acc.impl_, a.impl_, b.impl_);
}

template <typename T> void storeFP32(float v, T* ptr) {
  *ptr = v;
}

template <> inline void storeFP32<c10::Half>(float v, c10::Half* ptr) {
  storeFP32_impl(v, ptr);
}

template <> inline void storeFP32<c10::BFloat16>(float v, c10::BFloat16* ptr) {
  storeFP32_impl(v, ptr);
}

inline void prefetch(const void* addr) { prefetch_impl(addr); }

/****************************************/
/*             Specializations          */
/****************************************/
template <typename T> struct VecType { using vec_type = void; };

template <typename T> using vec_t = typename VecType<T>::vec_type;

template <> struct VecType<float> { using vec_type = FP32Vec8; };

template <> struct VecType<c10::Half> { using vec_type = FP16Vec8; };

template <> struct VecType<c10::BFloat16> { using vec_type = BF16Vec8; };

}; // namespace vec_op

#endif
