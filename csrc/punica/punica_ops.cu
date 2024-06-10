#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

#include "type_convert.h"
#include "../cuda_compat.h"
#include "bgmv/bgmv_config.h"


//====== utils ======

inline void check_shape(const torch::Tensor &a, const torch::Tensor &b,
                        const char *a_name, const char *b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ",
              a.dim(), " vs ", b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name,
                ".size(", i, ")");
  }
}

inline constexpr uint64_t pack_u32(uint32_t a, uint32_t b) {
  return (uint64_t(a) << 32) | uint64_t(b);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x)                                                        \
  TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b)                                                         \
  TORCH_CHECK(a == b, "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

//====== bgmv ======

template <typename in_T, typename out_T, typename W_T>
inline bool launch_bgmv_kernel(out_T *Y, const in_T *X, const W_T *W,
                               const int64_t *lora_indices,
                               uint32_t in_features, uint32_t out_features,
                               int64_t y_offset, int64_t full_y_size,
                               int64_t batch_size, int64_t num_layers,
                               int64_t layer_idx, float scale) {
  // NOTE(woosuk): While Punica supports various combinations of input/output
  // data types, we limit the supported data types to reduce the binary size.
  constexpr bool is_input_float = std::is_same<in_T, float>::value;
  constexpr bool is_output_float = std::is_same<out_T, float>::value;
  if (is_input_float) {
    if (!std::is_same<out_T, W_T>::value) {
      return false;
    }
  } else if (is_output_float) {
    if (!std::is_same<in_T, W_T>::value) {
      return false;
    }
  } else if (!(std::is_same<in_T, W_T>::value &&
               std::is_same<out_T, W_T>::value)) {
    return false;
  }

  switch (pack_u32(in_features, out_features)) {
#define CASE_ONESIDE(_in_T, _out_T, _W_T, feat_in, feat_out)                   \
  case pack_u32(feat_in, feat_out):                                            \
    bgmv_kernel<feat_in, feat_out>(Y, X, W, lora_indices, y_offset,            \
                                   full_y_size, batch_size, num_layers,        \
                                   layer_idx, scale);                          \
    break;
#define CASE(_in_T, _out_T, _W_T, narrow, wide)                                \
  CASE_ONESIDE(in_T, out_T, W_T, narrow, wide)                                 \
  CASE_ONESIDE(in_T, out_T, W_T, wide, narrow)

    FOR_BGMV_WIDE_NARROW(CASE, _, _, _)
    FOR_INST_BGMV_WIDE_NARROW(CASE_ONESIDE, _, _, _)
#undef CASE
#undef CASE_ONESIDE
  default:
    return false;
  }
  return true;
}

void dispatch_bgmv(torch::Tensor y, torch::Tensor x, torch::Tensor w,
                   torch::Tensor indicies, int64_t layer_idx, double scale) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w);
  CHECK_INPUT(indicies);

  CHECK_DIM(2, y);
  CHECK_DIM(2, x);
  CHECK_DIM(4, w);
  CHECK_DIM(1, indicies);

  int64_t B = x.size(0);
  int64_t h_in = x.size(1);
  int64_t h_out = y.size(1);
  int64_t num_layers = w.size(1);
  CHECK_EQ(w.size(3), h_in);
  CHECK_EQ(w.size(2), h_out);
  CHECK_EQ(indicies.size(0), x.size(0));
  CHECK_EQ(y.size(0), x.size(0));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  bool ok = false;
  if (h_in <= 128512 && h_out <= 128512) {
    // TODO: See if we can get rid of this massive nested switch
    switch (x.scalar_type()) {
    case at::ScalarType::Half:
      switch (y.scalar_type()) {
      case at::ScalarType::Half:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::BFloat16:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::Float:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      default:
        break;
      }
      break;
    case at::ScalarType::BFloat16:
      switch (y.scalar_type()) {
      case at::ScalarType::Half:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::BFloat16:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::Float:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      default:
        break;
      }
      break;
    case at::ScalarType::Float:
      switch (y.scalar_type()) {
      case at::ScalarType::Half:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::BFloat16:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::Float:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out, 0,
                                  h_out, B, num_layers, layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      default:
        break;
      }
      break;
    default:
      break;
    }
  }
  TORCH_CHECK(ok, "No suitable kernel.", " h_in=", h_in, " h_out=", h_out,
              " dtype=", x.scalar_type(), " out_dtype=", y.scalar_type());
}

void dispatch_bgmv_low_level(torch::Tensor y, torch::Tensor x, torch::Tensor w,
                             torch::Tensor indicies, int64_t layer_idx,
                             double scale, int64_t h_in, int64_t h_out,
                             int64_t y_offset) {
  CHECK_INPUT(y);
  CHECK_INPUT(x);
  CHECK_INPUT(w);
  CHECK_INPUT(indicies);

  CHECK_DIM(2, y);
  CHECK_DIM(2, x);
  CHECK_DIM(4, w);
  CHECK_DIM(1, indicies);

  int64_t B = x.size(0);
  int64_t num_layers = w.size(1);
  int64_t full_y_size = y.size(1);
  CHECK_EQ(w.size(3), h_in);
  CHECK_EQ(w.size(2), h_out);
  CHECK_EQ(indicies.size(0), x.size(0));
  CHECK_EQ(y.size(0), x.size(0));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  bool ok = false;
  if (h_in <= 128512 && h_out <= 128512) {
    // TODO: See if we can get rid of this massive nested switch
    switch (x.scalar_type()) {
    case at::ScalarType::Half:
      switch (y.scalar_type()) {
      case at::ScalarType::Half:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::BFloat16:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::Float:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<nv_half *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      default:
        break;
      }
      break;
    case at::ScalarType::BFloat16:
      switch (y.scalar_type()) {
      case at::ScalarType::Half:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::BFloat16:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::Float:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<nv_bfloat16 *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      default:
        break;
      }
      break;
    case at::ScalarType::Float:
      switch (y.scalar_type()) {
      case at::ScalarType::Half:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_half *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::BFloat16:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<nv_bfloat16 *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      case at::ScalarType::Float:
        switch (w.scalar_type()) {
        case at::ScalarType::Half:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_half *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        case at::ScalarType::BFloat16:
          ok = launch_bgmv_kernel(static_cast<float *>(y.data_ptr()),
                                  static_cast<float *>(x.data_ptr()),
                                  static_cast<nv_bfloat16 *>(w.data_ptr()),
                                  indicies.data_ptr<int64_t>(), h_in, h_out,
                                  y_offset, full_y_size, B, num_layers,
                                  layer_idx, scale);
          break;
        default:
          break;
        }
        break;
      default:
        break;
      }
      break;
    default:
      break;
    }
  }
  TORCH_CHECK(ok, "No suitable kernel.", " h_in=", h_in, " h_out=", h_out,
              " dtype=", x.scalar_type(), " out_dtype=", y.scalar_type());
}
