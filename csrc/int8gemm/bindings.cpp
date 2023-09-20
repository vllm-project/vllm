#include "include/bmm.h"
#include "include/fused.h"
#include "include/linear.h"
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_relu_a8_w8_b8_o8", &linear_relu_a8_w8_b8_o8,
        "Linear ReLU (INT8)");
  m.def("linear_a8_w8_b32_o32", &linear_a8_w8_b32_o32, "Linear (INT32)");
  m.def("linear_a8_w8_bfp32_ofp32", &linear_a8_w8_bfp32_ofp32,
        "Linear (I8-OFP32)");
  m.def("linear_a8_w8_b32_o32_with_scaling", &linear_a8_w8_b32_o32_with_scaling,
        "Linear (INT32) with scaling");
  m.def("linear_a8_w8_b8_o8", &linear_a8_w8_b8_o8, "Linear (INT8)");
  m.def("dq_add_layernorm_q", &dq_add_layernorm_q,
        "DQ + Add + LayerNorm (INT8)");
  m.def("bmm_s8t_s8n_s8t", &bmm_s8t_s8n_s8t, "BMM (INT8 IO) A x B.T");
  m.def("bmm_s8t_s8n_f32t", &bmm_s8t_s8n_f32t, "BMM (INT8 I FP32 O) A x B.T");
  m.def("bmm_s8t_s8n_s32t", &bmm_s8t_s8n_s32t,
        "BMM (INT8 In Int32 Out) A x B.T");
}
