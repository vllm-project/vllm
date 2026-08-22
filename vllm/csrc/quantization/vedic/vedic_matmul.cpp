// SPDX-License-Identifier: Apache-2.0
// Vedic 4-bit matmul implementation for vLLM MLA attention
#include <torch/extension.h>
#include <cstdint>
#include <vector>
#include <cmath>

inline uint8_t unpack_nibble(int32_t packed, int j) {
  return (packed >> (j * 4)) & 0xF;
}

torch::Tensor vedic_4bit_matmul(
    torch::Tensor A,         // float32, [M, K], CPU
    torch::Tensor B_packed,  // int32, [N, K/8], CPU (packed nibbles)
    double B_scale           // scale used in packing
) {
  TORCH_CHECK(A.device().is_cpu(), "A must be on CPU");
  TORCH_CHECK(B_packed.device().is_cpu(), "B_packed must be on CPU");
  TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
  TORCH_CHECK(B_packed.dtype() == torch::kInt32, "B_packed must be int32");
  TORCH_CHECK(A.dim() == 2 && B_packed.dim() == 2, "Invalid tensor dims");

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t N = B_packed.size(0);
  int64_t K8 = B_packed.size(1);
  TORCH_CHECK(K8 * 8 == K, "K must be divisible by 8 and match B_packed");

  auto A_cont = A.contiguous();
  auto Bp_cont = B_packed.contiguous();

  const float* A_ptr = A_cont.data_ptr<float>();
  const int32_t* Bp_ptr = Bp_cont.data_ptr<int32_t>();

  torch::Tensor C = torch::empty({M, N}, A.options());
  float* C_ptr = C.data_ptr<float>();

  // Unpack all B into an N x K vector of uint8 (0..15)
  std::vector<uint8_t> B_unpacked;
  B_unpacked.resize((size_t)N * (size_t)K);

  for (int64_t n = 0; n < N; ++n) {
    const int32_t* row = Bp_ptr + n * K8;
    for (int64_t k8 = 0; k8 < K8; ++k8) {
      int32_t packed = row[k8];
      for (int j = 0; j < 8; ++j) {
        uint8_t q = unpack_nibble(packed, j);
        B_unpacked[n * K + k8 * 8 + j] = q;
      }
    }
  }

  const float offset = 7.5f;
  for (int64_t m = 0; m < M; ++m) {
    const float* arow = A_ptr + m * K;
    float* crow = C_ptr + m * N;
    for (int64_t n = 0; n < N; ++n) {
      double acc = 0.0;
      const uint8_t* brow = &B_unpacked[n * K];
      for (int64_t k = 0; k < K; ++k) {
        float bval = ((float)brow[k] - offset) * (float)B_scale;
        acc += (double)arow[k] * (double)bval;
      }
      crow[n] = (float)acc;
    }
  }

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vedic_4bit_matmul", &vedic_4bit_matmul, "Vedic 4-bit matmul");
}

