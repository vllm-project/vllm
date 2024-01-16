/*
  gemm methods are adapted from ft
*/
#include <ATen/cuda/CUDAContext.h>
#include "cublasAlgoMap.h"
#include "cublasINT8MMWrapper.h"

class I8CUGEMM {
private:
  cublasINT8MMWrapper *int8_gemm_wrapper = nullptr;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

public:
  I8CUGEMM();
  ~I8CUGEMM();

  void linear_a8_w8_o32(
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& output);
  void linear_a8_w8_o32_(
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& output);                     
  void linear_a8_w8_o8(
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& output,
    float alpha);
  void linear_a8_w8_o8_(
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& output,
    float alpha);
};
I8CUGEMM::I8CUGEMM() {
  // cublasAlgoMap *cublas_algo_map = new cublasAlgoMap("igemm_config.in");
  cublasAlgoMap *cublas_algo_map = new cublasAlgoMap();
  std::mutex *cublas_wrapper_mutex = new std::mutex();
  bool use_ORDER_COL32_2R_4R4 = true;

  cublasLtHandle_t cublaslt_handle;
  cublasLtCreate(&cublaslt_handle);

  int8_gemm_wrapper = new cublasINT8MMWrapper(
    cublaslt_handle,
    this->stream,
    cublas_algo_map,
    cublas_wrapper_mutex,
    use_ORDER_COL32_2R_4R4);
}

I8CUGEMM::~I8CUGEMM() {}

void I8CUGEMM::linear_a8_w8_o32(
  torch::Tensor& input,  // INT8
  torch::Tensor& weight, // INT8
  torch::Tensor& out // INT32
) {
  int m = input.size(0);
  int n = weight.size(0);
  int k = input.size(1);

  // Set data types
  int8_t* input_ptr = input.data_ptr<int8_t>();
  int8_t* weight_ptr = weight.data_ptr<int8_t>();
  int32_t* output_ptr = out.data_ptr<int32_t>();

  int8_gemm_wrapper->Gemm(output_ptr, 1, m, n, k, 0, 0, 0, input_ptr,
                          weight_ptr);
}

void I8CUGEMM::linear_a8_w8_o32_(
  torch::Tensor& input,  // INT8
  torch::Tensor& weight, // INT8
  torch::Tensor& out // INT32
) {
  int m = input.size(0);
  int n = weight.size(0);
  int k = input.size(1);

  // Set data types
  int8_t* input_ptr = input.data_ptr<int8_t>();
  int8_t* weight_ptr = weight.data_ptr<int8_t>();
  int32_t* output_ptr = out.data_ptr<int32_t>();

  int8_gemm_wrapper->Gemm_(output_ptr, 1, m, n, k, 0, 0, 0, input_ptr,
                          weight_ptr);
}

void I8CUGEMM::linear_a8_w8_o8(
  torch::Tensor& input,  // INT8
  torch::Tensor& weight, // INT8
  torch::Tensor& out,    // INT8
  float alpha // FP32
) {
  int m = input.size(0);
  int n = weight.size(0);
  int k = input.size(1);

  // Set data types
  int8_t* input_ptr = input.data_ptr<int8_t>();
  int8_t* weight_ptr = weight.data_ptr<int8_t>();
  int8_t* output_ptr = out.data_ptr<int8_t>();

  int8_gemm_wrapper->Gemm(output_ptr, 1, m, n, k, 0, 0, 0, alpha, input_ptr,
                          weight_ptr);
}

void I8CUGEMM::linear_a8_w8_o8_(
  torch::Tensor& input,  // INT8
  torch::Tensor& weight, // INT8
  torch::Tensor& out,    // INT8
  float alpha // FP32
) {
  int m = input.size(0);
  int n = weight.size(0);
  int k = input.size(1);

  // Set data types
  int8_t* input_ptr = input.data_ptr<int8_t>();
  int8_t* weight_ptr = weight.data_ptr<int8_t>();
  int8_t* output_ptr = out.data_ptr<int8_t>();

  int8_gemm_wrapper->Gemm_(output_ptr, 1, m, n, k, 0, 0, 0, alpha, input_ptr,
                          weight_ptr);
}
