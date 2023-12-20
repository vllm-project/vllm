#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

void vecquant2matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
);

void vecquant2matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant2matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
}

void vecquant3matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
);

void vecquant3matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
}

void vecquant4matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
);

void vecquant4matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
}

void vecquant8matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
);

void vecquant8matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant8matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
}


// old

void vecquant2matmul_cuda_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
); 

void vecquant2matmul_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant2matmul_cuda_old(vec, mat, mul, scales, zeros,groupsize);
}

void vecquant3matmul_cuda_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
); 

void vecquant3matmul_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_cuda_old(vec, mat, mul, scales, zeros, groupsize);
}

void vecquant4matmul_cuda_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
); 

void vecquant4matmul_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_cuda_old(vec, mat, mul, scales, zeros, groupsize);
}

void vecquant8matmul_cuda_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
); 

void vecquant8matmul_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant8matmul_cuda_old(vec, mat, mul, scales, zeros, groupsize);
}

void vecquant2matmul_faster_cuda_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
); 

void vecquant2matmul_faster_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant2matmul_faster_cuda_old(vec, mat, mul, scales, zeros, groupsize, vec_height);
}

void vecquant3matmul_faster_cuda_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
); 

void vecquant3matmul_faster_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_faster_cuda_old(vec, mat, mul, scales, zeros, groupsize, vec_height);
}

void vecquant4matmul_faster_cuda_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
); 

void vecquant4matmul_faster_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize, int vec_height
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_faster_cuda_old(vec, mat, mul, scales, zeros, groupsize, vec_height);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant2matmul", &vecquant2matmul, "Vector 2-bit Quantized Matrix Multiplication (CUDA) (desc_act)");
  m.def("vecquant3matmul", &vecquant3matmul, "Vector 3-bit Quantized Matrix Multiplication (CUDA) (desc_act)");
  m.def("vecquant4matmul", &vecquant4matmul, "Vector 4-bit Quantized Matrix Multiplication (CUDA) (desc_act)");
  m.def("vecquant8matmul", &vecquant8matmul, "Vector 8-bit Quantized Matrix Multiplication (CUDA) (desc_act)");
  
  m.def("vecquant2matmul_old", &vecquant2matmul_old, "Vector 2-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant3matmul_old", &vecquant3matmul_old, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant4matmul_old", &vecquant4matmul_old, "Vector 4-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant8matmul_old", &vecquant8matmul_old, "Vector 8-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant2matmul_faster_old", &vecquant2matmul_faster_old, "Vector 2-bit Quantized Matrix Multiplication (CUDA), faster version");
  m.def("vecquant3matmul_faster_old", &vecquant3matmul_faster_old, "Vector 3-bit Quantized Matrix Multiplication (CUDA), faster version");
  m.def("vecquant4matmul_faster_old", &vecquant4matmul_faster_old, "Vector 4-bit Quantized Matrix Multiplication (CUDA), faster version");
}