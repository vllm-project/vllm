/*
  gemm methods are adapted from ft
*/
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "cublasAlgoMap.h"
#include "cublasINT8MMWrapper.h"
#include "transform_layout.h"

class I8CUGEMM {
private:
  cublasINT8MMWrapper *int8_gemm_wrapper = nullptr;
  // const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

public:
  I8CUGEMM();
  ~I8CUGEMM();

  void linear_a8_w8_o32(torch::Tensor &input, torch::Tensor &weight,
                        torch::Tensor &output);
  void linear_a8_w8_o32_(torch::Tensor &input, torch::Tensor &weight,
                        torch::Tensor &output);                     
  void linear_a8_w8_o8(torch::Tensor &input, torch::Tensor &weight,
                       torch::Tensor &output, float alpha);
  void linear_a8_w8_o8_(torch::Tensor &input, torch::Tensor &weight,
                       torch::Tensor &output, float alpha);
  void linear_a8_w8_ofp32(torch::Tensor &input, torch::Tensor &weight,
                       torch::Tensor &output, float alpha);
  void transform_row_to_col32(torch::Tensor &input, torch::Tensor &out);
  void transform_col32_to_row(torch::Tensor &input, torch::Tensor &out);
  void transform_row_to_ampere(torch::Tensor &input, torch::Tensor &out);
  void transform_row_to_turing(torch::Tensor &input, torch::Tensor &out);

};
I8CUGEMM::I8CUGEMM() {
  // cublasAlgoMap *cublas_algo_map = new cublasAlgoMap("igemm_config.in");
  cublasAlgoMap *cublas_algo_map = new cublasAlgoMap();
  std::mutex *cublas_wrapper_mutex = new std::mutex();
  bool use_ORDER_COL32_2R_4R4 = true;

  // const cudaStream_t stream;
  // const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cublasLtHandle_t cublaslt_handle;
//   cudaStreamCreate(&stream);
  cublasLtCreate(&cublaslt_handle);

  int8_gemm_wrapper =
      new cublasINT8MMWrapper(cublaslt_handle, this->stream, cublas_algo_map,
                              cublas_wrapper_mutex, use_ORDER_COL32_2R_4R4);
}

I8CUGEMM::~I8CUGEMM() {}

void I8CUGEMM::linear_a8_w8_o32(torch::Tensor &input,  // INT8
                              torch::Tensor &weight, // INT8
                              torch::Tensor &out // INT32
) {
  int m = input.size(0);
  int n = weight.size(0);
  int k = input.size(1);

  // Set data types
  int8_t *input_ptr = input.data_ptr<int8_t>();
  int8_t *weight_ptr = weight.data_ptr<int8_t>();
  int32_t *output_ptr = out.data_ptr<int32_t>();

  int8_gemm_wrapper->Gemm(output_ptr, 1, m, n, k, 0, 0, 0, input_ptr,
                          weight_ptr);
}

void I8CUGEMM::linear_a8_w8_o32_(torch::Tensor &input,  // INT8
                              torch::Tensor &weight, // INT8
                              torch::Tensor &out // INT32
) {
  int m = input.size(0);
  int n = weight.size(0);
  int k = input.size(1);

  // Set data types
  int8_t *input_ptr = input.data_ptr<int8_t>();
  int8_t *weight_ptr = weight.data_ptr<int8_t>();
  int32_t *output_ptr = out.data_ptr<int32_t>();

  int8_gemm_wrapper->Gemm_(output_ptr, 1, m, n, k, 0, 0, 0, input_ptr,
                          weight_ptr);
}

void I8CUGEMM::linear_a8_w8_o8(torch::Tensor &input,  // INT8
                             torch::Tensor &weight, // INT8
                             torch::Tensor &out,    // INT8
                             float alpha // FP32
) {
  int m = input.size(0);
  int n = weight.size(0);
  int k = input.size(1);

  // Set data types
  int8_t *input_ptr = input.data_ptr<int8_t>();
  int8_t *weight_ptr = weight.data_ptr<int8_t>();
  int8_t *output_ptr = out.data_ptr<int8_t>();

  int8_gemm_wrapper->Gemm(output_ptr, 1, m, n, k, 0, 0, 0, alpha, input_ptr,
                          weight_ptr);
}

void I8CUGEMM::linear_a8_w8_o8_(torch::Tensor &input,  // INT8
                             torch::Tensor &weight, // INT8
                             torch::Tensor &out,    // INT8
                             float alpha // FP32
) {
  int m = input.size(0);
  int n = weight.size(0);
  int k = input.size(1);

  // Set data types
  int8_t *input_ptr = input.data_ptr<int8_t>();
  int8_t *weight_ptr = weight.data_ptr<int8_t>();
  int8_t *output_ptr = out.data_ptr<int8_t>();

  int8_gemm_wrapper->Gemm_(output_ptr, 1, m, n, k, 0, 0, 0, alpha, input_ptr,
                          weight_ptr);
}

void I8CUGEMM::linear_a8_w8_ofp32(torch::Tensor &input,  // INT8
                             torch::Tensor &weight, // INT8
                             torch::Tensor &out,    // INT8
                             float alpha // FP32
) {
  int m = input.size(0);
  int n = weight.size(0);
  int k = input.size(1);

  // Set data types
  int8_t *input_ptr = input.data_ptr<int8_t>();
  int8_t *weight_ptr = weight.data_ptr<int8_t>();
  float *output_ptr = out.data_ptr<float>();

  int8_gemm_wrapper->Gemm_f(output_ptr, 1, m, n, k, 0, 0, 0, alpha, input_ptr,
                          weight_ptr);
}

void I8CUGEMM::transform_row_to_col32(torch::Tensor &input, torch::Tensor &out) {
  int m = input.size(0);
  int n = input.size(1);
  int m_ = out.size(0);
  int n_ = out.size(1);

  assert(m == m_);
  assert(n == n_);
  // const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int8_t *input_ptr = input.data_ptr<int8_t>();
  int8_t *out_ptr = out.data_ptr<int8_t>();
  invokeRowMajorToCOL32(out_ptr, input_ptr, m, n, this->stream);
  // invokeRowMajorToCOL32(out_ptr, input_ptr, m, n, stream);
}

void I8CUGEMM::transform_col32_to_row(torch::Tensor &input, torch::Tensor &out) {
  int m = input.size(0);
  int n = input.size(1);
  int m_ = out.size(0);
  int n_ = out.size(1);

  assert(m == m_);
  assert(n == n_);
  // const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int8_t *input_ptr = input.data_ptr<int8_t>();
  int8_t *out_ptr = out.data_ptr<int8_t>();
  invokeCOL32ToRowMajor(out_ptr, input_ptr, m, n, this->stream);
  // invokeCOL32ToRowMajor(out_ptr, input_ptr, m, n, stream);
}

void I8CUGEMM::transform_row_to_ampere(torch::Tensor &input, torch::Tensor &out) {
  int m = input.size(0);
  int n = input.size(1);
  int m_ = out.size(0);
  int n_ = out.size(1);

  assert(m == m_);
  assert(n == n_);
  // const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int8_t *input_ptr = input.data_ptr<int8_t>();
  int8_t *out_ptr = out.data_ptr<int8_t>();
  invokeRowMajorToAmpere(out_ptr, input_ptr, m, n, this->stream);
  // invokeCOL32ToRowMajor(out_ptr, input_ptr, m, n, stream);
}

void I8CUGEMM::transform_row_to_turing(torch::Tensor &input, torch::Tensor &out) {
  int m = input.size(0);
  int n = input.size(1);
  int m_ = out.size(0);
  int n_ = out.size(1);

  assert(m == m_);
  assert(n == n_);
  // const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int8_t *input_ptr = input.data_ptr<int8_t>();
  int8_t *out_ptr = out.data_ptr<int8_t>();
  invokeRowMajorToTuring(out_ptr, input_ptr, m, n, this->stream);
  // invokeCOL32ToRowMajor(out_ptr, input_ptr, m, n, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<I8CUGEMM>(m, "I8CUGEMM")
      .def(pybind11::init<>())
      .def("linear_a8_w8_o32", &I8CUGEMM::linear_a8_w8_o32)
      .def("linear_a8_w8_o8", &I8CUGEMM::linear_a8_w8_o8)
      .def("linear_a8_w8_o8_", &I8CUGEMM::linear_a8_w8_o8_)
      .def("linear_a8_w8_o32_", &I8CUGEMM::linear_a8_w8_o32_)
      .def("linear_a8_w8_ofp32", &I8CUGEMM::linear_a8_w8_ofp32)
      .def("transform_row_to_col32", &I8CUGEMM::transform_row_to_col32)
      .def("transform_col32_to_row", &I8CUGEMM::transform_col32_to_row)
      .def("transform_row_to_ampere", &I8CUGEMM::transform_row_to_ampere)
      .def("transform_row_to_turing", &I8CUGEMM::transform_row_to_turing);
}
