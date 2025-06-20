#ifndef DNNL_HELPER_HPP
#define DNNL_HELPER_HPP

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include "oneapi/dnnl/dnnl.hpp"

namespace {
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
};  // namespace

template <bool InputNoScale>
class DNNLPrimitiveHelper {
 public:
  // I8 input GEMM kernel (C = a_scales * A @ (b_scales * B^T) + bias)
  // A: [M, K], row-major
  // B: [K, N], column-major
  // C: [M, N], row-major
  // bias: [N], row-major, optional
  // a_scales: [MS]
  // b_scales: [NS]
  // Note: Due to the limitation of oneDNN
  // (https://github.com/oneapi-src/oneDNN/issues/1636), the quantized bias is
  // not supported.
  template <typename OutputT, typename BiasT>
  static void gemm_s8s8_jit(const int8_t* a, const int8_t* b, OutputT* c,
                            const BiasT* bias, dnnl_dim_t M, dnnl_dim_t N,
                            dnnl_dim_t K, const float* a_scales,
                            const float* b_scales, dnnl_dim_t MS,
                            dnnl_dim_t NS) {
    auto&& OutputType = get_dnnl_type<OutputT>();
    auto&& BiasType = get_dnnl_type<BiasT>();

    dnnl::memory::desc a_md({M, K}, dnnl::memory::data_type::s8, {K, 1});
    dnnl::memory::desc b_md({K, N}, dnnl::memory::data_type::s8, {1, K});
    dnnl::memory::desc c_md({M, N}, OutputType, {N, 1});

    dnnl::primitive_attr attr;
    if constexpr (!InputNoScale) {
      if (MS == 1) {
        // per-tensor
        attr.set_scales_mask(DNNL_ARG_SRC, 0);
      } else {
        // per-token
        TORCH_CHECK(false, "per-token quantization is unsupported.");
      }
    }

    if (NS == 1) {
      // per-tensor
      attr.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
    } else {
      // per-channel
      attr.set_scales_mask(DNNL_ARG_WEIGHTS, 2);
    }

    dnnl::matmul::primitive_desc matmul_pd;
    if (bias) {
      dnnl::memory::desc bias_md({1, N}, BiasType, {N, 1});
      matmul_pd = dnnl::matmul::primitive_desc(default_engine(), a_md, b_md,
                                               bias_md, c_md, attr);
    } else {
      matmul_pd = dnnl::matmul::primitive_desc(default_engine(), a_md, b_md,
                                               c_md, attr);
    }
    dnnl::matmul matmul(matmul_pd);

    auto& engine = default_engine();

    dnnl::memory a_m(a_md, engine, (void*)a);
    dnnl::memory b_m(b_md, engine, (void*)b);
    dnnl::memory c_m(c_md, engine, (void*)c);
    dnnl::memory a_scales_m({{MS}, dnnl::memory::data_type::f32, {1}}, engine,
                            (void*)a_scales);
    dnnl::memory b_scales_m({{NS}, dnnl::memory::data_type::f32, {1}}, engine,
                            (void*)b_scales);

    auto& stream = default_stream();
    if constexpr (InputNoScale) {
      if (bias) {
        dnnl::memory::desc bias_md({N}, BiasType, {1});
        dnnl::memory bias_m(bias_md, engine, (void*)bias);
        matmul.execute(
            stream, {
                        {DNNL_ARG_SRC, a_m},
                        {DNNL_ARG_WEIGHTS, b_m},
                        {DNNL_ARG_BIAS, bias_m},
                        {DNNL_ARG_DST, c_m},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, b_scales_m},
                    });
      } else {
        matmul.execute(
            stream, {
                        {DNNL_ARG_SRC, a_m},
                        {DNNL_ARG_WEIGHTS, b_m},
                        {DNNL_ARG_DST, c_m},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, b_scales_m},
                    });
      }
    } else {
      if (bias) {
        dnnl::memory::desc bias_md({N}, BiasType, {1});
        dnnl::memory bias_m(bias_md, engine, (void*)bias);
        matmul.execute(
            stream, {
                        {DNNL_ARG_SRC, a_m},
                        {DNNL_ARG_WEIGHTS, b_m},
                        {DNNL_ARG_BIAS, bias_m},
                        {DNNL_ARG_DST, c_m},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, a_scales_m},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, b_scales_m},
                    });
      } else {
        matmul.execute(
            stream, {
                        {DNNL_ARG_SRC, a_m},
                        {DNNL_ARG_WEIGHTS, b_m},
                        {DNNL_ARG_DST, c_m},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, a_scales_m},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, b_scales_m},
                    });
      }
    }
    stream.wait();
  }

 private:
  static dnnl::engine& default_engine() {
    static dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    return engine;
  }

  static dnnl::stream& default_stream() {
    static dnnl::stream stream(default_engine());
    return stream;
  }
};

#endif
