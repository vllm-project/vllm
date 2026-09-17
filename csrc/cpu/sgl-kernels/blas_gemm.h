// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/native/CPUBlas.h>

// Unlike brgemm, PyTorch does not publicly expose at::native::cpublas::gemm
// If OpenBLS is available in the PyTorch wheel, we rely on it for fast
// bf16:bf16->fp32 GEMMs Otherwise, we fall back to PyTorch reference BLAS path.
#if defined(VLLM_HAS_OPENBLAS)
extern "C" void sbgemm_(char* transa, char* transb, int* m, int* n, int* k,
                        float* alpha, const at::BFloat16* a, int* lda,
                        const at::BFloat16* b, int* ldb, float* beta, float* c,
                        int* ldc);

extern "C" void sgemm_(char* transa, char* transb, int* m, int* n, int* k,
                       float* alpha, const float* a, int* lda, const float* b,
                       int* ldb, float* beta, float* c, int* ldc);

inline char blas_transpose(at::native::TransposeType trans) {
  switch (trans) {
    case at::native::TransposeType::NoTranspose:
      return 'n';
    case at::native::TransposeType::Transpose:
      return 't';
    case at::native::TransposeType::ConjTranspose:
      return 'c';
  }
  return 'n';
}

inline void blas_gemm(at::native::TransposeType transa,
                      at::native::TransposeType transb, int64_t m, int64_t n,
                      int64_t k, float alpha, const at::BFloat16* a,
                      int64_t lda, const at::BFloat16* b, int64_t ldb,
                      float beta, float* c, int64_t ldc) {
  char transa_ = blas_transpose(transa);
  char transb_ = blas_transpose(transb);
  int m_ = static_cast<int>(m);
  int n_ = static_cast<int>(n);
  int k_ = static_cast<int>(k);
  int lda_ = static_cast<int>(lda);
  int ldb_ = static_cast<int>(ldb);
  int ldc_ = static_cast<int>(ldc);
  sbgemm_(&transa_, &transb_, &m_, &n_, &k_, &alpha, a, &lda_, b, &ldb_, &beta,
          c, &ldc_);
}

inline void blas_gemm(at::native::TransposeType transa,
                      at::native::TransposeType transb, int64_t m, int64_t n,
                      int64_t k, float alpha, const float* a, int64_t lda,
                      const float* b, int64_t ldb, float beta, float* c,
                      int64_t ldc) {
  char transa_ = blas_transpose(transa);
  char transb_ = blas_transpose(transb);
  int m_ = static_cast<int>(m);
  int n_ = static_cast<int>(n);
  int k_ = static_cast<int>(k);
  int lda_ = static_cast<int>(lda);
  int ldb_ = static_cast<int>(ldb);
  int ldc_ = static_cast<int>(ldc);
  sgemm_(&transa_, &transb_, &m_, &n_, &k_, &alpha, a, &lda_, b, &ldb_, &beta,
         c, &ldc_);
}

inline void blas_gemm(at::native::TransposeType, at::native::TransposeType,
                      int64_t, int64_t, int64_t, float, const at::Half*,
                      int64_t, const at::Half*, int64_t, float, float*,
                      int64_t) {
  TORCH_CHECK(false, "CPU OpenBLAS hgemm is not available.");
}
#else
template <typename scalar_t>
inline void blas_gemm(at::native::TransposeType transa,
                      at::native::TransposeType transb, int64_t m, int64_t n,
                      int64_t k, float alpha, const scalar_t* a, int64_t lda,
                      const scalar_t* b, int64_t ldb, float beta, float* c,
                      int64_t ldc) {
  auto gemm = at::native::cpublas::gemm_no_downcast_stub.DEFAULT;
  gemm(c10::CppTypeToScalarType<scalar_t>::value, transa, transb, m, n, k,
       at::Scalar(alpha), a, lda, b, ldb, at::Scalar(beta), c, ldc);
}
#endif