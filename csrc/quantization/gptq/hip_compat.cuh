// Adapted from turboderp exllama: https://github.com/turboderp/exllama

#ifndef _hip_compat_cuh
#define _hip_compat_cuh

// Workaround for a bug in hipamd, backported from upstream.
__device__ __forceinline__ __half __compat_hrcp(__half x) {
    return __half_raw{
        static_cast<_Float16>(__builtin_amdgcn_rcph(static_cast<__half_raw>(x).data))};
}

__device__ __forceinline__ __half2 __compat_h2rcp(__half2 x) {
    return _Float16_2{static_cast<_Float16>(__builtin_amdgcn_rcph(x.x)),
        static_cast<_Float16>(__builtin_amdgcn_rcph(x.y))};
}

#define hrcp __compat_hrcp
#define h2rcp __compat_h2rcp

// Workaround for hipify_python using rocblas instead of hipblas.
__host__ __forceinline__ hipblasStatus_t __compat_hipblasHgemm(hipblasHandle_t    handle,
                                                               hipblasOperation_t transA,
                                                               hipblasOperation_t transB,
                                                               int                m,
                                                               int                n,
                                                               int                k,
                                                               const half*        alpha,
                                                               const half*        AP,
                                                               int                lda,
                                                               const half*        BP,
                                                               int                ldb,
                                                               const half*        beta,
                                                               half*              CP,
                                                               int                ldc) {
    return hipblasHgemm(handle, transA, transB, m, n, k,
                        reinterpret_cast<const hipblasHalf *>(alpha),
                        reinterpret_cast<const hipblasHalf *>(AP), lda,
                        reinterpret_cast<const hipblasHalf *>(BP), ldb,
                        reinterpret_cast<const hipblasHalf *>(beta),
                        reinterpret_cast<hipblasHalf *>(CP), ldc);
}

#define rocblas_handle hipblasHandle_t
#define rocblas_operation_none HIPBLAS_OP_N
#define rocblas_get_stream hipblasGetStream
#define rocblas_set_stream hipblasSetStream
#define rocblas_hgemm __compat_hipblasHgemm

#endif
