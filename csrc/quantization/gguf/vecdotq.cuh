// copied and adapted from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/vecdotq.cuh
// and https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmq.cu
static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

static __device__ __forceinline__ int get_int_from_int8(const int8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32); // assume at least 2 byte alignment
    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;
    return x32;
}

static __device__ __forceinline__ int get_int_from_uint8(const uint8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32); // assume at least 2 byte alignment
    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;
    return x32;
}

static __device__ __forceinline__ int get_int_from_int8_aligned(const int8_t * x8, const int & i32) {
    return *((const int *) (x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}

static __device__ __forceinline__ int get_int_from_uint8_aligned(const uint8_t * x8, const int & i32) {
    return *((const int *) (x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = __dp4a(vi0, u[2*i+0], sumi);
        sumi = __dp4a(vi1, u[2*i+1], sumi);
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
#endif
}

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm4, const half2 & ds8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = __dp4a(vi0, u[2*i+0], sumi);
        sumi = __dp4a(vi1, u[2*i+1], sumi);
    }

    const float2 tmp = __half22float2(__hmul2(dm4, ds8));
    const float d4d8 = tmp.x;
    const float m4s8 = tmp.y;

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
#endif
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl(
    const int * vl, const int * vh, const int * u, const float & d5, const half2 & ds8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = __dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = __dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 16 from each quant value
    return d5 * (sumi * ds8f.x - (16*vdr/QI5_0) * ds8f.y);
#endif
}


#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_1_q8_1_impl(
    const int * vl, const int * vh, const int * u, const half2 & dm5, const half2 & ds8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = __dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = __dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

    const float2 tmp = __half22float2(__hmul2(dm5, ds8));
    const float d5d8 = tmp.x;
    const float m5s8 = tmp.y;

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    return sumi*d5d8 + m5s8 / (QI5_1 / vdr);
#endif
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const float & d8_0, const float & d8_1) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = __dp4a(v[i], u[i], sumi);
    }
    return d8_0*d8_1 * sumi;
#endif
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm8, const half2 & ds8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = __dp4a(v[i], u[i], sumi);
    }

    const float2 tmp = __half22float2(__hmul2(dm8, ds8));
    const float d8d8 = tmp.x;
    const float m8s8 = tmp.y;

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    return sumi*d8d8 + m8s8 / (QI8_1 / vdr);
#endif
}

#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q2_K_Q8_1_MMQ  2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float * __restrict__ d8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (__dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * __dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
    }

    const float2 dm2f = __half22float2(dm2);

    return dm2f.x*sumf_d - dm2f.y*sumf_m;
#endif
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float & d8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    int sumi_d = 0;
    int sumi_m = 0;

#pragma unroll
    for (int i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        int sumi_d_sc = 0;

        const int sc = scales[i0 / (QI8_1/2)];

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = __dp4a(v[i], u[i], sumi_d_sc); // SIMD dot product
            sumi_m    = __dp4a(m,    u[i], sumi_m); // multiply sum of q8_1 values with m
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    const float2 dm2f = __half22float2(dm2);

    return d8 * (dm2f.x*sumi_d - dm2f.y*sumi_m);
#endif
}

#define VDR_Q3_K_Q8_1_MMVQ 1
#define VDR_Q3_K_Q8_1_MMQ  2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const int & scale_offset, const float & d3, const float * __restrict__ d8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2*i;

        const int isc_low = isc % (QK_K/32);
        const int sc_shift_low = 4 * (isc / (QK_K/32));
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K/64);
        const int sc_shift_high = 2 * (isc / (QK_K/64));
        const int sc_high = ((scales[(QK_K/32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;

        const int vil = (vl >> (2*i)) & 0x03030303;

        const int vih = ((vh >> i) << 2) & 0x04040404;

        const int vi = __vsubss4(vil, vih);

        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d3 * sumf;
#endif
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d3, const float & d8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    int sumi = 0;

#pragma unroll
    for (int i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        int sumi_sc = 0;

        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = __dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    return d3*d8 * sumi;
#endif
}

#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q4_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = __dp4a(v1i, u[2*i+1], __dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
        const int dot2 = __dp4a(0x01010101, u[2*i+1], __dp4a(0x01010101, u[2*i+0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const float2 dm4f = __half22float2(dm4);
    return dm4f.x*sumf_d - dm4f.y*sumf_m;
#endif
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = __dp4a((v[j] >> (4*i)) & 0x0F0F0F0F, u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
#endif
}

#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMQ  8

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = __dp4a(v0i, u[2*i+0], __dp4a(v1i, u[2*i+1], 0)); // SIMD dot product
        const int dot2 = __dp4a(0x01010101, u[2*i+0], __dp4a(0x01010101, u[2*i+1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    const float2 dm5f = __half22float2(dm5);
    return dm5f.x*sumf_d - dm5f.y*sumf_m;
#endif
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = __dp4a(v[i*QI8_1 + j], u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
#endif
}

#define VDR_Q6_K_Q8_1_MMVQ 1
#define VDR_Q6_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
#endif
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ sc,
    const float & d6, const float * __restrict__ d8) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    float sumf_d = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
        for (int i = i0; i < i0 + 2; ++i) {
            sumi_d.x = __dp4a(v[2*i+0], u[2*i+0], sumi_d.x); // SIMD dot product
            sumi_d.x = __dp4a(v[2*i+1], u[2*i+1], sumi_d.x); // SIMD dot product

            sumi_d.y = __dp4a(v[2*i+4], u[2*i+4], sumi_d.y); // SIMD dot product
            sumi_d.y = __dp4a(v[2*i+5], u[2*i+5], sumi_d.y); // SIMD dot product
        }

        sumf_d += d8[i0/4] * (sc[i0/2+0]*sumi_d.x + sc[i0/2+1]*sumi_d.y);
    }

    return d6 * sumf_d;
#endif
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq;

    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2*VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, __half2float(bq4_0->d), bq8_1->ds);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int  tile_x_qs[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF/QI4_0) + mmq_y/QI4_0];
    *x_ql = tile_x_qs;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI4_0;
    const int kqsx = k % QI4_0;

    const block_q4_0 * bx0 = (const block_q4_0 *) vx;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbx;
        x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        // x_dmf[i * (WARP_SIZE_GGUF/QI4_0) + i / QI4_0 + kbx] = bxi->d;
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI4_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + i_offset * QI4_0 + k / blocks_per_tile_x_row;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE_GGUF/QI4_0) + i / QI4_0 + kbxd] = __half2float(bxi->d);
    }
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    (void)x_qh; (void)x_sc;

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const float * x_dmf = (const float *) x_dm;

    int u[2*VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE_GGUF + (kyqs + l)         % WARP_SIZE_GGUF];
        u[2*l+1] = y_qs[j * WARP_SIZE_GGUF + (kyqs + l + QI4_0) % WARP_SIZE_GGUF];
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE_GGUF + 1) + k], u, x_dmf[i * (WARP_SIZE_GGUF/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (WARP_SIZE_GGUF/QI8_1) + (2*k/QI8_1) % (WARP_SIZE_GGUF/QI8_1)]);
}

static __device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q4_1 * bq4_1 = (const block_q4_1 *) vbq;

    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2*VDR_Q4_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]    = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_1(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_qs[mmq_y * (WARP_SIZE_GGUF) +     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE_GGUF/QI4_1) + mmq_y/QI4_1];
    *x_ql = tile_x_qs;
    *x_dm = tile_x_dm;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_1(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI4_1;
    const int kqsx = k % QI4_1;

    const block_q4_1 * bx0 = (const block_q4_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbx;
        x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI4_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        int i = i0 + i_offset * QI4_1 + k / blocks_per_tile_x_row;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dm[i * (WARP_SIZE_GGUF/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}

static __device__ __forceinline__ float vec_dot_q4_1_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    int u[2*VDR_Q4_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE_GGUF + (kyqs + l)         % WARP_SIZE_GGUF];
        u[2*l+1] = y_qs[j * WARP_SIZE_GGUF + (kyqs + l + QI4_1) % WARP_SIZE_GGUF];
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE_GGUF + 1) + k], u, x_dm[i * (WARP_SIZE_GGUF/QI4_1) + i/QI4_1 + k/QI4_1],
         y_ds[j * (WARP_SIZE_GGUF/QI8_1) + (2*k/QI8_1) % (WARP_SIZE_GGUF/QI8_1)]);
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q5_0 * bq5_0 = (const block_q5_0 *) vbq;

    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int  u[2*VDR_Q5_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]    = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }

    return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, __half2float(bq5_0->d), bq8_1->ds);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int  tile_x_ql[mmq_y * (2*WARP_SIZE_GGUF)     + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF/QI5_0) + mmq_y/QI5_0];

    *x_ql = tile_x_ql;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI5_0;
    const int kqsx = k % QI5_0;

    const block_q5_0 * bx0 = (const block_q5_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }
        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbx;
        const int ql = get_int_from_uint8(bxi->qs, kqsx);
        const int qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0     = __vsubss4(qs0, 0x10101010); // subtract 16

        x_ql[i * (2*WARP_SIZE_GGUF + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1     = __vsubss4(qs1, 0x10101010); // subtract 16

        x_ql[i * (2*WARP_SIZE_GGUF + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI5_0;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        int i = i0 + i_offset * QI5_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE_GGUF/QI5_0) + i / QI5_0 + kbxd] = __half2float(bxi->d);
    }
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE_GGUF/QI5_0) + i/QI5_0 + k/QI5_0;
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    int u[2*VDR_Q5_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE_GGUF + (kyqs + l)         % WARP_SIZE_GGUF];
        u[2*l+1] = y_qs[j * WARP_SIZE_GGUF + (kyqs + l + QI5_0) % WARP_SIZE_GGUF];
    }

    return vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE_GGUF + 1) + 2 * k], u, x_dmf[index_bx], y_df[j * (WARP_SIZE_GGUF/QI8_1) + (2*k/QI8_1) % (WARP_SIZE_GGUF/QI8_1)]);
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q5_1 * bq5_1 = (const block_q5_1 *) vbq;

    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int  u[2*VDR_Q5_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]   = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]   = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }

    return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_1(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE_GGUF)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE_GGUF/QI5_1) + mmq_y/QI5_1];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_1(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI5_1;
    const int kqsx = k % QI5_1;

    const block_q5_1 * bx0 = (const block_q5_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (k % QI5_1));

        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_ql[i * (2*WARP_SIZE_GGUF + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_ql[i * (2*WARP_SIZE_GGUF + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI5_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        int i = i0 + i_offset * QI5_1 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE_GGUF/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE_GGUF/QI5_1) + + i/QI5_1 + k/QI5_1;

    int u[2*VDR_Q5_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE_GGUF + (kyqs + l)         % WARP_SIZE_GGUF];
        u[2*l+1] = y_qs[j * WARP_SIZE_GGUF + (kyqs + l + QI5_1) % WARP_SIZE_GGUF];
    }

    return vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE_GGUF + 1) + 2 * k], u, x_dm[index_bx], y_ds[j * (WARP_SIZE_GGUF/QI8_1) + (2*k/QI8_1) % (WARP_SIZE_GGUF/QI8_1)]);
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, __half2float(bq8_0->d), __low2float(bq8_1->ds));
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q8_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int  tile_x_qs[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF/QI8_0) + mmq_y/QI8_0];

    *x_ql = tile_x_qs;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q8_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI8_0;
    const int kqsx = k % QI8_0;
    float * x_dmf = (float *) x_dm;

    const block_q8_0 * bx0 = (const block_q8_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }
        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbx;
        x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI8_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        int i = i0 + i_offset * QI8_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }
        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE_GGUF/QI8_0) + i / QI8_0 + kbxd] = __half2float(bxi->d);
    }
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE_GGUF + 1) + k], &y_qs[j * WARP_SIZE_GGUF + k], x_dmf[i * (WARP_SIZE_GGUF/QI8_0) + i/QI8_0 + k/QI8_0],
         y_df[j * (WARP_SIZE_GGUF/QI8_1) + k/QI8_1]);
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q2_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE_GGUF/QI2_K) + mmq_y/QI2_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE_GGUF/4)     + mmq_y/4];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q2_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI2_K;
    const int kqsx = k % QI2_K;

    const block_q2_K * bx0 = (const block_q2_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }
        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbx;
        x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI2_K;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI2_K) {
        int i = (i0 + i_offset * QI2_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }
        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dm[i * (WARP_SIZE_GGUF/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE_GGUF/4);

        if (need_check) {
            i = min(i, i_max);
        }
        const block_q2_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE_GGUF/4)) / (QI2_K/4);
        x_sc[i * (WARP_SIZE_GGUF/4) + i / 4 + k % (WARP_SIZE_GGUF/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    const int kbx = k / QI2_K;
    const int ky  = (k % QI2_K) * QR2_K;
    const float * y_df = (const float *) y_ds;

    int v[QR2_K*VDR_Q2_K_Q8_1_MMQ];

    const int kqsx = i * (WARP_SIZE_GGUF + 1) + kbx*QI2_K + (QI2_K/2) * (ky/(2*QI2_K)) + ky % (QI2_K/2);
    const int shift = 2 * ((ky % (2*QI2_K)) / (QI2_K/2));

#pragma unroll
    for (int l = 0; l < QR2_K*VDR_Q2_K_Q8_1_MMQ; ++l) {
        v[l] = (x_ql[kqsx + l] >> shift) & 0x03030303;
    }

    const uint8_t * scales = ((const uint8_t *) &x_sc[i * (WARP_SIZE_GGUF/4) + i/4 + kbx*4]) + ky/4;

    const int index_y = j * WARP_SIZE_GGUF + (QR2_K*k) % WARP_SIZE_GGUF;
    return vec_dot_q2_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dm[i * (WARP_SIZE_GGUF/QI2_K) + i/QI2_K + kbx], y_df[index_y/QI8_1]);
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q3_K * bq3_K = (const block_q3_K *) vbq;

    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const float d = __half2float(bq3_K->d);

    const int vl = get_int_from_uint8(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    int    u[QR3_K];
    float d8[QR3_K];

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q3_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE_GGUF/QI3_K) + mmq_y/QI3_K];
    __shared__ int   tile_x_qh[mmq_y * (WARP_SIZE_GGUF/2)     + mmq_y/2];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE_GGUF/4)     + mmq_y/4];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_qh = tile_x_qh;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q3_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI3_K;
    const int kqsx = k % QI3_K;

    const block_q3_K * bx0 = (const block_q3_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbx;
        x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI3_K;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        int i = (i0 + i_offset * QI3_K + k / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE_GGUF/QI3_K) + i / QI3_K + kbxd] = __half2float(bxi->d);
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        int i = i0 + i_offset * 2 + k / (WARP_SIZE_GGUF/2);
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE_GGUF/2)) / (QI3_K/2);
        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_SIZE_GGUF/2) + i / 2 + k % (WARP_SIZE_GGUF/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE_GGUF/4);
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE_GGUF/4)) / (QI3_K/4);

        const int ksc = k % (QI3_K/4);

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_from_uint8(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_from_uint8(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

        x_sc[i * (WARP_SIZE_GGUF/4) + i / 4 + k % (WARP_SIZE_GGUF/4)] = sc;
    }
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    const int kbx  = k / QI3_K;
    const int ky  = (k % QI3_K) * QR3_K;
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    const int8_t * scales = ((const int8_t *) (x_sc + i * (WARP_SIZE_GGUF/4) + i/4 + kbx*4)) + ky/4;

    int v[QR3_K*VDR_Q3_K_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < QR3_K*VDR_Q3_K_Q8_1_MMQ; ++l) {
        const int kqsx = i * (WARP_SIZE_GGUF + 1) + kbx*QI3_K + (QI3_K/2) * (ky/(2*QI3_K)) + ky % (QI3_K/2);
        const int shift = 2 * ((ky % 32) / 8);
        const int vll = (x_ql[kqsx + l] >> shift) & 0x03030303;

        const int vh = x_qh[i * (WARP_SIZE_GGUF/2) + i/2 + kbx * (QI3_K/2) + (ky+l)%8] >> ((ky+l) / 8);
        const int vlh = (vh << 2) & 0x04040404;

        v[l] = __vsubss4(vll, vlh);
    }

    const int index_y = j * WARP_SIZE_GGUF + (k*QR3_K) % WARP_SIZE_GGUF;
    return vec_dot_q3_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dmf[i * (WARP_SIZE_GGUF/QI3_K) + i/QI3_K + kbx], y_df[index_y/QI8_1]);
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_q4_K * bq4_K = (const block_q4_K *) vbq;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE_GGUF/QI4_K) + mmq_y/QI4_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE_GGUF/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI4_K; // == 0 if QK_K == 256
    const int kqsx = k % QI4_K; // == k if QK_K == 256

    const block_q4_K * bx0 = (const block_q4_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbx;
        x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI4_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        int i = (i0 + i_offset * QI4_K + k / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dm[i * (WARP_SIZE_GGUF/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE_GGUF/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE_GGUF/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = k % (WARP_SIZE_GGUF/8);
        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE_GGUF/8) + i / 8 + ksc] = scales8;
    }
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    (void)x_qh;

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE_GGUF/8) + i/8 + k/16]) + 2*((k % 16) / 8);

    const int index_y = j * WARP_SIZE_GGUF + (QR4_K*k) % WARP_SIZE_GGUF;
    return vec_dot_q4_K_q8_1_impl_mmq(&x_ql[i * (WARP_SIZE_GGUF + 1) + k], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE_GGUF/QI4_K) + i/QI4_K], &y_ds[index_y/QI8_1]);
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q5_K * bq5_K = (const block_q5_K *) vbq;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_offset/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE_GGUF)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE_GGUF/QI5_K) + mmq_y/QI5_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE_GGUF/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI5_K; // == 0 if QK_K == 256
    const int kqsx = k % QI5_K; // == k if QK_K == 256

    const block_q5_K * bx0 = (const block_q5_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR5_K*kqsx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + (QI5_K/4);

        x_ql[i * (2*WARP_SIZE_GGUF + 1) + kq0] = ql0 | qh0;
        x_ql[i * (2*WARP_SIZE_GGUF + 1) + kq1] = ql1 | qh1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI5_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        int i = (i0 + i_offset * QI5_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dm[i * (WARP_SIZE_GGUF/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE_GGUF/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE_GGUF/8)) / (QI5_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = k % (WARP_SIZE_GGUF/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE_GGUF/8) + i / 8 + ksc] = scales8;
    }
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE_GGUF/8) + i/8 + k/16]) + 2 * ((k % 16) / 8);

    const int index_x = i * (QR5_K*WARP_SIZE_GGUF + 1) +  QR5_K*k;
    const int index_y = j * WARP_SIZE_GGUF             + (QR5_K*k) % WARP_SIZE_GGUF;
    return vec_dot_q5_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE_GGUF/QI5_K) + i/QI5_K], &y_ds[index_y/QI8_1]);
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_from_uint8(bq6_K->ql, iqs);
    const int vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, __half2float(bq6_K->d), d8);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q6_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE_GGUF)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE_GGUF/QI6_K) + mmq_y/QI6_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE_GGUF/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q6_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI6_K; // == 0 if QK_K == 256
    const int kqsx = k % QI6_K; // == k if QK_K == 256

    const block_q6_K * bx0 = (const block_q6_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR6_K*kqsx;

        const int ql = get_int_from_uint8(bxi->ql, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        const int qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        const int qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        const int kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        const int kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        x_ql[i * (2*WARP_SIZE_GGUF + 1) + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_ql[i * (2*WARP_SIZE_GGUF + 1) + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI6_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE_GGUF/QI6_K) + i / QI6_K + kbxd] = __half2float(bxi->d);
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE_GGUF/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE_GGUF/8)) / 4;

        x_sc[i * (WARP_SIZE_GGUF/8) + i / 8 + k % (WARP_SIZE_GGUF/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    const int8_t * sc = ((const int8_t *) &x_sc[i * (WARP_SIZE_GGUF/8) + i/8 + k/8]);

    const int index_x = i * (QR6_K*WARP_SIZE_GGUF + 1) +  QR6_K*k;
    const int index_y = j * WARP_SIZE_GGUF             + (QR6_K*k) % WARP_SIZE_GGUF;
    return vec_dot_q6_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, x_dmf[i * (WARP_SIZE_GGUF/QI6_K) + i/QI6_K], &y_df[index_y/QI8_1]);
}

static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq;

    const int ib32 = iqs;
    const uint16_t * q2 = bq2->qs + 4*ib32;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = q2[2] | (q2[3] << 16);
    int sumi = 0;
    for (int l = 0; l < 4; ++l) {
        const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
        const uint8_t  signs = ksigns_iq2xs[aux32 & 127];
        for (int j = 0; j < 8; ++j) {
            sumi += q8[j] * grid[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        }
        q8 += 8;
        aux32 >>= 7;
    }
    const float d = __half2float(bq2->d) * (0.5f + aux32) * __half2float(bq8_1[ib32].ds.x) * 0.25f;
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq;

    const int ib32 = iqs;
    const uint16_t * q2 = bq2->qs + 4*ib32;
    const int8_t   * q8 = bq8_1[ib32].qs;
    const uint8_t ls1 = bq2->scales[ib32] & 0xf;
    const uint8_t ls2 = bq2->scales[ib32] >>  4;
    int sumi1 = 0;
    for (int l = 0; l < 2; ++l) {
        const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
        const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
        for (int j = 0; j < 8; ++j) {
            sumi1 += q8[j] * grid[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        }
        q8 += 8;
    }
    int sumi2 = 0;
    for (int l = 2; l < 4; ++l) {
        const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
        const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
        for (int j = 0; j < 8; ++j) {
            sumi2 += q8[j] * grid[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        }
        q8 += 8;
    }
    const float d = __half2float(bq2->d) * __half2float(bq8_1[ib32].ds.x) * 0.25f;
    return d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
}

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    const block_iq2_s * bq2 = (const block_iq2_s *) vbq;

    const int ib32 = iqs;
    const int8_t  * q8 = bq8_1[ib32].qs;
    const uint8_t * signs = bq2->qs + QK_K/8 + 4*ib32;
    const uint8_t ls1 = bq2->scales[ib32] & 0xf;
    const uint8_t ls2 = bq2->scales[ib32] >>  4;
    int sumi1 = 0;
    for (int l = 0; l < 2; ++l) {
        const uint32_t * grid = (const uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        const uint32_t signs0 = __vcmpeq4(((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201);
        const uint32_t signs1 = __vcmpeq4(((signs[l] >>  4) * 0x01010101) & 0x08040201, 0x08040201);
        const int grid_l = __vsub4(grid[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid[1] ^ signs1, signs1);
        sumi1 = __dp4a(grid_l, *((const int *)q8 + 0), sumi1);
        sumi1 = __dp4a(grid_h, *((const int *)q8 + 1), sumi1);
        q8 += 8;
    }
    int sumi2 = 0;
    for (int l = 2; l < 4; ++l) {
        const uint32_t * grid = (const uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        const uint32_t signs0 = __vcmpeq4(((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201);
        const uint32_t signs1 = __vcmpeq4(((signs[l] >>  4) * 0x01010101) & 0x08040201, 0x08040201);
        const int grid_l = __vsub4(grid[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid[1] ^ signs1, signs1);
        sumi2 = __dp4a(grid_l, *((const int *)q8 + 0), sumi2);
        sumi2 = __dp4a(grid_h, *((const int *)q8 + 1), sumi2);
        q8 += 8;
    }
    const float d = __half2float(bq2->d) * __low2float(bq8_1[ib32].ds) * 0.25f;
    return d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#endif
}

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    const block_iq3_xxs * bq2 = (const block_iq3_xxs *) vbq;

    const int ib32 = iqs;
    const uint8_t  * q3 = bq2->qs + 8*ib32;
    const uint16_t * gas = (const uint16_t *)(bq2->qs + QK_K/4) + 2*ib32;
    const int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = gas[0] | (gas[1] << 16);
    int sumi = 0;
    for (int l = 0; l < 4; ++l) {
        const uint32_t * grid1 = iq3xxs_grid + q3[2*l+0];
        const uint32_t * grid2 = iq3xxs_grid + q3[2*l+1];
        const uint32_t * signs = (const uint32_t *)(ksigns64 + (aux32 & 127));
        const int grid_l = __vsub4(grid1[0] ^ signs[0], signs[0]);
        const int grid_h = __vsub4(grid2[0] ^ signs[1], signs[1]);
        sumi = __dp4a(grid_l, *((int *)q8+0), sumi);
        sumi = __dp4a(grid_h, *((int *)q8+1), sumi);
        q8 += 8;
        aux32 >>= 7;
    }
    const float d = __half2float(bq2->d) * (0.5f + aux32) * __low2float(bq8_1[ib32].ds) * 0.5f;
    return d * sumi;
#endif
}

static __device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    const block_iq3_s * bq2 = (const block_iq3_s *) vbq;

    const int ib32 = iqs;
    const uint8_t  * qs = bq2->qs + 8*ib32;
    const int8_t   * q8 = bq8_1[ib32].qs;
    int sumi = 0;
    for (int l = 0; l < 4; ++l) {
        const uint32_t * grid1 = iq3xs_grid + (qs[2*l+0] | ((bq2->qh[ib32] << (8 - 2*l)) & 256));
        const uint32_t * grid2 = iq3xs_grid + (qs[2*l+1] | ((bq2->qh[ib32] << (7 - 2*l)) & 256));
        uint32_t signs0 = __vcmpeq4(((bq2->signs[4*ib32+l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201);
        uint32_t signs1 = __vcmpeq4(((bq2->signs[4*ib32+l] >>  4) * 0x01010101) & 0x08040201, 0x08040201);
        const int grid_l = __vsub4(grid1[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid2[0] ^ signs1, signs1);
        sumi = __dp4a(grid_l, *((int *)q8+0), sumi);
        sumi = __dp4a(grid_h, *((int *)q8+1), sumi);
        q8 += 8;
    }
    const float d = __half2float(bq2->d) * (0.5f + ((bq2->scales[ib32/2] >> 4*(ib32%2)) & 0xf)) * __low2float(bq8_1[ib32].ds) * 0.5f;
    return d * sumi;
#endif
}

static __device__ __forceinline__ float vec_dot_iq1_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    const block_iq1_s * bq1 = (const block_iq1_s *) vbq;

    const int       qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq1->qh[iqs];

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = iq1s_grid_gpu[qs[l0/2] | (((qh >> 3*(l0/2)) & 0x07) << 8)];

        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;

        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);

        sumi = __dp4a(grid0, u0, sumi);
        sumi = __dp4a(grid1, u1, sumi);
    }

    const float  d1q   = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float  delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);
    const float2 ds    = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x*sumi + ds.y*delta);
#endif
}

static __device__ __forceinline__ float vec_dot_iq1_m_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM

    const block_iq1_m * bq1 = (const block_iq1_m *) vbq;

    const int       qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    int   sumi[2] = {0};
    float sumf[2] = {0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2*iqs + l0/4] >> (4 * ((l0/2) % 2));

        const int grid = iq1s_grid_gpu[qs[l0/2] | ((qhl & 0x07) << 8)];

        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;

        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);

        sumi[l0/4] = __dp4a(grid0, u0, sumi[l0/4]);
        sumi[l0/4] = __dp4a(grid1, u1, sumi[l0/4]);

        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f*IQ1M_DELTA/0x08);
        int sumy = 0;
        sumy = __dp4a(u0, 0x01010101, sumy);
        sumy = __dp4a(u1, 0x01010101, sumy);
        sumf[l0/4] += delta*sumy;
    }

    const uint16_t * sc = (const uint16_t *) bq1->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * __low2float(bq8_1[iqs].ds);

    const int tmp = sc[iqs/2] >> (6*(iqs%2));
    const int sc0 = 2*((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2*((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
#endif
}

static __device__ __forceinline__ void get_int_from_table_16(const uint32_t & q4, const uint8_t * values,
        int & val1, int & val2) {

    uint32_t aux32; const uint8_t * q8 = (const uint8_t *)&aux32;
    aux32 = q4 & 0x0f0f0f0f;
    uint16_t v1 = values[q8[0]] | (values[q8[1]] << 8);
    uint16_t v2 = values[q8[2]] | (values[q8[3]] << 8);
    val1 = v1 | (v2 << 16);
    aux32 = (q4 >> 4) & 0x0f0f0f0f;
    v1 = values[q8[0]] | (values[q8[1]] << 8);
    v2 = values[q8[2]] | (values[q8[3]] << 8);
    val2 = v1 | (v2 << 16);
}

static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM

    const block_iq4_nl * bq = (const block_iq4_nl *) vbq;

    const uint16_t * q4 = (const uint16_t *)bq->qs + 2*iqs;
    const int32_t  * q8 = (const int32_t  *)bq8_1->qs + iqs;

    const uint8_t * values = (const uint8_t *)kvalues_iq4nl;

    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        const uint32_t aux = q4[2*l] | (q4[2*l+1] << 16);
        get_int_from_table_16(aux, values, v1, v2);
        sumi1 = __dp4a(v1, q8[l+0], sumi1);
        sumi2 = __dp4a(v2, q8[l+4], sumi2);
    }
    const float d = __half2float(bq->d) * __low2float(bq8_1->ds);
    return d * (sumi1 + sumi2);
#endif
}


static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 610 || defined USE_ROCM
    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq;
    const uint8_t * values = (const uint8_t *)kvalues_iq4nl;

    // iqs is 0...7
    const int ib32 = iqs;
    const int32_t  * q8 = (const int *)bq8_1[ib32].qs;
    const uint32_t * q4 = (const uint32_t *)bq4->qs + 4*ib32;
    const int8_t ls = ((bq4->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((bq4->scales_h >> 2*ib32) & 3) << 4);
    const float d = __half2float(bq4->d) * (ls - 32) * __low2float(bq8_1[ib32].ds);
    int v1, v2;
    int sumi1 = 0, sumi2 = 0;
    for (int j = 0; j < 4; ++j) {
        get_int_from_table_16(q4[j], values, v1, v2);
        sumi1 = __dp4a(v1, q8[j+0], sumi1);
        sumi2 = __dp4a(v2, q8[j+4], sumi2);
    }
    return d * (sumi1 + sumi2);
#endif
}

// ========================= IQ4_NL MMQ =========================
// IQ4_NL is structurally identical to Q4_0 (QK=32, QR=2, QI=4) but uses
// a 16-entry lookup table (kvalues_iq4nl) to convert 4-bit indices to signed int8.
// Tiles store raw packed nibble data (same as Q4_0); the lookup is applied in vec_dot.
// need_sum=false because the looked-up values are already signed (no bias correction).

#define VDR_IQ4_NL_Q8_1_MMQ 4

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_iq4_nl(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_qs[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF/QI4_NL) + mmq_y/QI4_NL];

    *x_ql = tile_x_qs;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_nl(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI4_NL;
    const int kqsx = k % QI4_NL;

    const block_iq4_nl * bx0 = (const block_iq4_nl *) vx;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq4_nl * bxi = bx0 + i*blocks_per_row + kbx;
        x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI4_NL;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_NL) {
        int i = i0 + i_offset * QI4_NL + k / blocks_per_tile_x_row;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq4_nl * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE_GGUF/QI4_NL) + i / QI4_NL + kbxd] = __half2float(bxi->d);
    }
}

static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    (void)x_qh; (void)x_sc;

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const float * x_dmf = (const float *) x_dm;

    int u[2*VDR_IQ4_NL_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_IQ4_NL_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE_GGUF + (kyqs + l)         % WARP_SIZE_GGUF];
        u[2*l+1] = y_qs[j * WARP_SIZE_GGUF + (kyqs + l + QI4_NL) % WARP_SIZE_GGUF];
    }

    int sumi = 0;

    // IQ4_NL lookup table values (same as kvalues_iq4nl in ggml-common.h).
    const int8_t iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

#pragma unroll
    for (int l = 0; l < VDR_IQ4_NL_Q8_1_MMQ; ++l) {
        int v1, v2;
        get_int_from_table_16(x_ql[i * (WARP_SIZE_GGUF + 1) + k + l], (const uint8_t *)iq4nl, v1, v2);
        sumi = __dp4a(v1, u[2*l+0], sumi);
        sumi = __dp4a(v2, u[2*l+1], sumi);
    }

    // need_sum=true: y_ds is stored as half2 (d, sum). We only need the d component
    // since IQ4_NL values are already signed (no unsigned-to-signed bias correction).
    const float2 ds8 = __half22float2(y_ds[j * (WARP_SIZE_GGUF/QI8_1) + (2*k/QI8_1) % (WARP_SIZE_GGUF/QI8_1)]);

    return x_dmf[i * (WARP_SIZE_GGUF/QI4_NL) + i/QI4_NL + k/QI4_NL] * ds8.x * sumi;
}

// ========================= IQ4_XS MMQ =========================
// IQ4_XS is a 256-element super-block (QK_K=256) with 8 sub-blocks of 32 elements.
// Each sub-block has a 6-bit scale: 4 low bits from scales_l + 2 high bits from scales_h.
// For MMQ, we use Q4_K-like tiling: QR=2, QI=32 (blocks_per_warp=1).
// This ensures all 128 bytes of qs are loaded into shared memory per step.
// The MMVQ constants (QR4_XS=8, QI4_XS=8) are NOT used for MMQ.

#define QR_IQ4_XS_MMQ 2
#define QI_IQ4_XS_MMQ (QK_K / (4 * QR_IQ4_XS_MMQ))
#define VDR_IQ4_XS_Q8_1_MMQ 4

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_iq4_xs(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF/QI_IQ4_XS_MMQ) + mmq_y/QI_IQ4_XS_MMQ];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE_GGUF/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = (half2 *) tile_x_d;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq4_xs(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    const int kbx  = k / QI_IQ4_XS_MMQ;
    const int kqsx = k % QI_IQ4_XS_MMQ;

    const block_iq4_xs * bx0 = (const block_iq4_xs *) vx;
    float * x_dmf = (float *) x_dm;

    // Load qs data (packed nibbles, 128 bytes per block)
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq4_xs * bxi = bx0 + i*blocks_per_row + kbx;
        x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    // Load global d scale (1 float per block)
    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI_IQ4_XS_MMQ;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI_IQ4_XS_MMQ) {
        int i = (i0 + i_offset * QI_IQ4_XS_MMQ + k / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq4_xs * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE_GGUF/QI_IQ4_XS_MMQ) + i / QI_IQ4_XS_MMQ + kbxd] = __half2float(bxi->d);
    }

    // Load sub-block scales (8 per block, decoded to int8, packed 4 per int32)
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE_GGUF/8)) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq4_xs * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE_GGUF/8)) / (QI_IQ4_XS_MMQ/8);

        const int ksc = k % (WARP_SIZE_GGUF/8);

        // Pack 4 decoded sub-block scales per int32 (only need ksc=0,1 for 8 scales)
        if (ksc < 2) {
            int scales_packed = 0;
#pragma unroll
            for (int sb = 0; sb < 4; sb++) {
                const int ib32 = ksc * 4 + sb;
                const int8_t ls = ((bxi->scales_l[ib32/2] >> (4*(ib32%2))) & 0xf)
                                | (((bxi->scales_h >> (2*ib32)) & 3) << 4);
                ((int8_t *)&scales_packed)[sb] = ls - 32;
            }
            x_sc[i * (WARP_SIZE_GGUF/8) + i / 8 + ksc] = scales_packed;
        }
    }
}

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    (void)x_qh;

    const float * x_dmf = (const float *) x_dm;

    // IQ4_NL lookup table values (same as kvalues_iq4nl in ggml-common.h).
    const int8_t iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

    // Sub-block index: each sub-block = 4 int32 of packed nibble data = 32 elements
    const int ib32 = k / 4;
    const int8_t * sc_bytes = (const int8_t *)&x_sc[i * (WARP_SIZE_GGUF/8) + i/8 + ib32/4];
    const float sub_scale = (float)sc_bytes[ib32 % 4];

    const int index_y = j * WARP_SIZE_GGUF + (QR_IQ4_XS_MMQ*k) % WARP_SIZE_GGUF;

    int sumi = 0;

#pragma unroll
    for (int l = 0; l < VDR_IQ4_XS_Q8_1_MMQ; ++l) {
        int v1, v2;
        get_int_from_table_16(x_ql[i * (WARP_SIZE_GGUF + 1) + k + l], (const uint8_t *)iq4nl, v1, v2);
        sumi = __dp4a(v1, y_qs[index_y + l], sumi);
        sumi = __dp4a(v2, y_qs[index_y + l + 4], sumi);
    }

    // need_sum=true: y_ds is stored as half2 (d, sum). We only need the d component
    // since IQ4_NL values are already signed (no unsigned-to-signed bias correction).
    const float2 ds8 = __half22float2(y_ds[j * (WARP_SIZE_GGUF/QI8_1) + (QR_IQ4_XS_MMQ*k/QI8_1) % (WARP_SIZE_GGUF/QI8_1)]);

    return x_dmf[i * (WARP_SIZE_GGUF/QI_IQ4_XS_MMQ) + i/QI_IQ4_XS_MMQ]
         * sub_scale
         * ds8.x
         * sumi;
}

// ========================= IQ3_S MMQ =========================
// IQ3_S: QK_K=256, qs[64], qh[8], signs[32], scales[4], d(half).
// Custom QR=2, QI=32 for MMQ (1 block/warp, like IQ4_XS_MMQ).
// tile_x_ql[0..15]: qs (64 bytes), tile_x_ql[16..23]: signs (32 bytes).
// tile_x_qh: qh (8 bytes). tile_x_sc: scales (4 bytes). tile_x_dm: d.
// Grid lookup (iq3xs_grid[512]) in vec_dot.

#define QR_IQ3_S_MMQ 2
#define QI_IQ3_S_MMQ (QK_K / (4 * QR_IQ3_S_MMQ))
#define VDR_IQ3_S_Q8_1_MMQ 4

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_iq3_s(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF/QI_IQ3_S_MMQ) + mmq_y/QI_IQ3_S_MMQ];
    __shared__ int   tile_x_qh[mmq_y * (WARP_SIZE_GGUF/4)     + mmq_y/4];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE_GGUF/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = (half2 *) tile_x_d;
    *x_qh = tile_x_qh;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_s(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    const int kbx  = k / QI_IQ3_S_MMQ;  // 0 (1 block per warp)
    const int kqsx = k % QI_IQ3_S_MMQ;  // 0..31

    const block_iq3_s * bx0 = (const block_iq3_s *) vx;
    float * x_dmf = (float *) x_dm;

    // Load qs (64 bytes = 16 int32, slots 0..15) and signs (32 bytes = 8 int32, slots 16..23)
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq3_s * bxi = bx0 + i*blocks_per_row + kbx;
        if (kqsx < 16) {
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        } else if (kqsx < 24) {
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->signs, kqsx - 16);
        } else {
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = 0;
        }
    }

    // Load d scale
    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI_IQ3_S_MMQ;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI_IQ3_S_MMQ) {
        int i = (i0 + i_offset * QI_IQ3_S_MMQ + k / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq3_s * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE_GGUF/QI_IQ3_S_MMQ) + i / QI_IQ3_S_MMQ + kbxd] = __half2float(bxi->d);
    }

    // Load qh (8 bytes = 2 int32)
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = (i0 + i_offset * 4 + k / (WARP_SIZE_GGUF/4)) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const int kqh = k % (WARP_SIZE_GGUF/4);
        if (kqh < 2) {
            const block_iq3_s * bxi = bx0 + i*blocks_per_row + kbx;
            x_qh[i * (WARP_SIZE_GGUF/4) + i / 4 + kqh] = get_int_from_uint8(bxi->qh, kqh);
        }
    }

    // Load scales (4 bytes = 1 int32)
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE_GGUF/8)) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const int ksc = k % (WARP_SIZE_GGUF/8);
        if (ksc < 1) {
            const block_iq3_s * bxi = bx0 + i*blocks_per_row + kbx;
            x_sc[i * (WARP_SIZE_GGUF/8) + i / 8 + ksc] = get_int_from_uint8(bxi->scales, ksc);
        }
    }
}

static __device__ __forceinline__ float vec_dot_iq3_s_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    const float * x_dmf = (const float *) x_dm;
    const int ib32 = k / 4;  // sub-block index (same for all VDR iterations since k is multiple of 4)

    // Read qs base for this ib32 (8 bytes of grid indices)
    const uint8_t * qs = (const uint8_t *)&x_ql[i * (WARP_SIZE_GGUF + 1) + ib32 * 2];
    // Read qh byte for this ib32
    const uint8_t * qh_all = (const uint8_t *)&x_qh[i * (WARP_SIZE_GGUF/4) + i/4];
    const uint8_t qh_val = qh_all[ib32];
    // Signs base
    const uint8_t * all_signs = (const uint8_t *)&x_ql[i * (WARP_SIZE_GGUF + 1) + 16];
    // Per-sub-block scale
    const uint8_t * sc_bytes = (const uint8_t *)&x_sc[i * (WARP_SIZE_GGUF/8) + i/8];
    const float sub_scale = 0.5f + ((sc_bytes[ib32/2] >> (4*(ib32%2))) & 0xf);

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_IQ3_S_Q8_1_MMQ; ++l) {
        const int sub = l;  // k%4=0 always, so (k+l)%4 = l

        // Grid lookup: 9-bit index
        const int grid1_val = iq3xs_grid[qs[2*sub+0] | ((qh_val << (8 - 2*sub)) & 256)];
        const int grid2_val = iq3xs_grid[qs[2*sub+1] | ((qh_val << (7 - 2*sub)) & 256)];

        // Sign extraction
        const uint8_t sign_byte = all_signs[4*ib32 + sub];
        const uint32_t signs0 = __vcmpeq4(((sign_byte & 0xf) * 0x01010101) & 0x08040201, 0x08040201);
        const uint32_t signs1 = __vcmpeq4(((sign_byte >>  4) * 0x01010101) & 0x08040201, 0x08040201);
        const int grid_l = __vsub4(grid1_val ^ signs0, signs0);
        const int grid_h = __vsub4(grid2_val ^ signs1, signs1);

        const int index_y = j * WARP_SIZE_GGUF + (QR_IQ3_S_MMQ * (k + l)) % WARP_SIZE_GGUF;
        sumi = __dp4a(grid_l, y_qs[index_y + 0], sumi);
        sumi = __dp4a(grid_h, y_qs[index_y + 1], sumi);
    }

    const float2 ds8 = __half22float2(y_ds[j * (WARP_SIZE_GGUF/QI8_1) + (QR_IQ3_S_MMQ*k/QI8_1) % (WARP_SIZE_GGUF/QI8_1)]);

    return x_dmf[i * (WARP_SIZE_GGUF/QI_IQ3_S_MMQ) + i/QI_IQ3_S_MMQ]
         * sub_scale * 0.5f
         * ds8.x * sumi;
}

// ========================= IQ3_XXS MMQ =========================
// IQ3_XXS: QK_K=256, qs[3*(QK_K/8)=96 bytes]: first 64 = grid indices, last 32 = gas (signs+scales).
// Grid: iq3xxs_grid[256] (uint32). Sign via ksigns64. Scale from gas aux bits.
// Same custom QR=2, QI=32 approach.

#define QR_IQ3_XXS_MMQ 2
#define QI_IQ3_XXS_MMQ (QK_K / (4 * QR_IQ3_XXS_MMQ))
#define VDR_IQ3_XXS_Q8_1_MMQ 4

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_iq3_xxs(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    // tile_x_ql: slots 0..15 = grid indices (64 bytes), slots 16..23 = gas data (32 bytes)
    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF/QI_IQ3_XXS_MMQ) + mmq_y/QI_IQ3_XXS_MMQ];

    *x_ql = tile_x_ql;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq3_xxs(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    (void)x_qh; (void)x_sc;

    const int kbx  = k / QI_IQ3_XXS_MMQ;
    const int kqsx = k % QI_IQ3_XXS_MMQ;

    const block_iq3_xxs * bx0 = (const block_iq3_xxs *) vx;
    float * x_dmf = (float *) x_dm;

    // Load grid indices (64 bytes, slots 0..15) and gas data (32 bytes, slots 16..23)
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq3_xxs * bxi = bx0 + i*blocks_per_row + kbx;
        if (kqsx < 16) {
            // Grid indices: first 64 bytes of qs
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        } else if (kqsx < 24) {
            // Gas data (signs+scales): bytes 64..95 of qs
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qs + QK_K/4, kqsx - 16);
        } else {
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = 0;
        }
    }

    // Load d scale
    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI_IQ3_XXS_MMQ;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI_IQ3_XXS_MMQ) {
        int i = (i0 + i_offset * QI_IQ3_XXS_MMQ + k / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq3_xxs * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE_GGUF/QI_IQ3_XXS_MMQ) + i / QI_IQ3_XXS_MMQ + kbxd] = __half2float(bxi->d);
    }
}

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    (void)x_qh; (void)x_sc;

    const float * x_dmf = (const float *) x_dm;

    const int ib32 = k / 4;

    // Grid indices for this ib32
    const uint8_t * q3 = (const uint8_t *)&x_ql[i * (WARP_SIZE_GGUF + 1) + ib32 * 2];

    // Gas data for this ib32 (signs + scale packed in 4 bytes per ib32)
    const uint16_t * gas = (const uint16_t *)&x_ql[i * (WARP_SIZE_GGUF + 1) + 16 + ib32];
    uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float sub_scale = 0.5f + (aux32 >> 28);

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_IQ3_XXS_Q8_1_MMQ; ++l) {
        const int sub = l;

        // Extract signs for this sub-group
        uint32_t signs_val = aux32 >> (7 * sub);
        const uint32_t * signs = (const uint32_t *)(ksigns64 + (signs_val & 127));

        // Grid lookup
        const uint32_t * grid1 = iq3xxs_grid + q3[2*sub+0];
        const uint32_t * grid2 = iq3xxs_grid + q3[2*sub+1];

        // Apply signs
        const int grid_l = __vsub4(grid1[0] ^ signs[0], signs[0]);
        const int grid_h = __vsub4(grid2[0] ^ signs[1], signs[1]);

        // Q8 data
        const int index_y = j * WARP_SIZE_GGUF + (QR_IQ3_XXS_MMQ * (k + l)) % WARP_SIZE_GGUF;
        sumi = __dp4a(grid_l, y_qs[index_y + 0], sumi);
        sumi = __dp4a(grid_h, y_qs[index_y + 1], sumi);
    }

    const float2 ds8 = __half22float2(y_ds[j * (WARP_SIZE_GGUF/QI8_1) + (QR_IQ3_XXS_MMQ*k/QI8_1) % (WARP_SIZE_GGUF/QI8_1)]);

    return x_dmf[i * (WARP_SIZE_GGUF/QI_IQ3_XXS_MMQ) + i/QI_IQ3_XXS_MMQ]
         * sub_scale * 0.5f
         * ds8.x * sumi;
}

// ========================= IQ2_XXS MMQ =========================
// IQ2_XXS: QK_K=256, qs[QK_K/8=32 uint16]. Grid: iq2xxs_grid[256] (uint64).
// Per ib32: 4 uint16 from qs. First 2 = grid indices (8-bit each). Last 2 = aux (signs+scale).

#define QR_IQ2_XXS_MMQ 2
#define QI_IQ2_XXS_MMQ (QK_K / (4 * QR_IQ2_XXS_MMQ))
#define VDR_IQ2_XXS_Q8_1_MMQ 4

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_iq2_xxs(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    // qs data: 32 uint16 = 64 bytes = 16 int32 in slots 0..15
    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF/QI_IQ2_XXS_MMQ) + mmq_y/QI_IQ2_XXS_MMQ];

    *x_ql = tile_x_ql;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_xxs(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    (void)x_qh; (void)x_sc;

    const int kbx  = k / QI_IQ2_XXS_MMQ;
    const int kqsx = k % QI_IQ2_XXS_MMQ;

    const block_iq2_xxs * bx0 = (const block_iq2_xxs *) vx;
    float * x_dmf = (float *) x_dm;

    // Load qs data (32 uint16 = 64 bytes = 16 int32)
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq2_xxs * bxi = bx0 + i*blocks_per_row + kbx;
        if (kqsx < 16) {
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_b2(bxi->qs, kqsx);
        } else {
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = 0;
        }
    }

    // Load d scale
    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI_IQ2_XXS_MMQ;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI_IQ2_XXS_MMQ) {
        int i = (i0 + i_offset * QI_IQ2_XXS_MMQ + k / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq2_xxs * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE_GGUF/QI_IQ2_XXS_MMQ) + i / QI_IQ2_XXS_MMQ + kbxd] = __half2float(bxi->d);
    }
}

static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    (void)x_qh; (void)x_sc;

    const float * x_dmf = (const float *) x_dm;

    const int ib32 = k / 4;

    // Read the 4 uint16 for this ib32
    const uint16_t * q2 = (const uint16_t *)&x_ql[i * (WARP_SIZE_GGUF + 1) + ib32 * 2];
    const uint8_t * aux8 = (const uint8_t *)q2;
    uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float sub_scale = 0.5f + (aux32 >> 28);

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_IQ2_XXS_Q8_1_MMQ; ++l) {
        const int sub = l;
        const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[sub]);
        const uint8_t  signs = ksigns_iq2xs[(aux32 >> (7*sub)) & 127];

        const int index_y = j * WARP_SIZE_GGUF + (QR_IQ2_XXS_MMQ * (k + l)) % WARP_SIZE_GGUF;
        const int8_t * q8 = (const int8_t *)&y_qs[index_y];

        for (int jj = 0; jj < 8; ++jj) {
            sumi += q8[jj] * grid[jj] * (signs & kmask_iq2xs[jj] ? -1 : 1);
        }
    }

    const float2 ds8 = __half22float2(y_ds[j * (WARP_SIZE_GGUF/QI8_1) + (QR_IQ2_XXS_MMQ*k/QI8_1) % (WARP_SIZE_GGUF/QI8_1)]);

    return x_dmf[i * (WARP_SIZE_GGUF/QI_IQ2_XXS_MMQ) + i/QI_IQ2_XXS_MMQ]
         * sub_scale * 0.25f
         * ds8.x * sumi;
}

// ========================= IQ2_XS MMQ =========================
// IQ2_XS: QK_K=256, qs[QK_K/8=32 uint16], scales[QK_K/32=8 uint8]. Grid: iq2xs_grid[512] (uint64).
// Per ib32: 4 uint16 from qs. Each uint16: low 9 bits = grid index, upper 7 = signs.

#define QR_IQ2_XS_MMQ 2
#define QI_IQ2_XS_MMQ (QK_K / (4 * QR_IQ2_XS_MMQ))
#define VDR_IQ2_XS_Q8_1_MMQ 4

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_iq2_xs(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF/QI_IQ2_XS_MMQ) + mmq_y/QI_IQ2_XS_MMQ];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE_GGUF/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = (half2 *) tile_x_d;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_xs(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    (void)x_qh;

    const int kbx  = k / QI_IQ2_XS_MMQ;
    const int kqsx = k % QI_IQ2_XS_MMQ;

    const block_iq2_xs * bx0 = (const block_iq2_xs *) vx;
    float * x_dmf = (float *) x_dm;

    // Load qs data (32 uint16 = 64 bytes = 16 int32)
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq2_xs * bxi = bx0 + i*blocks_per_row + kbx;
        if (kqsx < 16) {
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_b2(bxi->qs, kqsx);
        } else {
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = 0;
        }
    }

    // Load d scale
    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI_IQ2_XS_MMQ;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI_IQ2_XS_MMQ) {
        int i = (i0 + i_offset * QI_IQ2_XS_MMQ + k / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq2_xs * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE_GGUF/QI_IQ2_XS_MMQ) + i / QI_IQ2_XS_MMQ + kbxd] = __half2float(bxi->d);
    }

    // Load scales (8 bytes = 2 int32)
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE_GGUF/8)) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const int ksc = k % (WARP_SIZE_GGUF/8);
        if (ksc < 2) {
            const block_iq2_xs * bxi = bx0 + i*blocks_per_row + kbx;
            x_sc[i * (WARP_SIZE_GGUF/8) + i / 8 + ksc] = get_int_from_uint8(bxi->scales, ksc);
        }
    }
}

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    (void)x_qh;

    const float * x_dmf = (const float *) x_dm;

    const int ib32 = k / 4;

    // Read the 4 uint16 for this ib32
    const uint16_t * q2 = (const uint16_t *)&x_ql[i * (WARP_SIZE_GGUF + 1) + ib32 * 2];

    // Per-sub-block scales
    const uint8_t * sc_bytes = (const uint8_t *)&x_sc[i * (WARP_SIZE_GGUF/8) + i/8];

    float sumf = 0;
#pragma unroll
    for (int l = 0; l < VDR_IQ2_XS_Q8_1_MMQ; ++l) {
        const int sub = l;

        // Grid lookup: 9-bit index, 7-bit signs
        const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[sub] & 511));
        const uint8_t  signs = ksigns_iq2xs[q2[sub] >> 9];

        const int index_y = j * WARP_SIZE_GGUF + (QR_IQ2_XS_MMQ * (k + l)) % WARP_SIZE_GGUF;
        const int8_t * q8 = (const int8_t *)&y_qs[index_y];

        int sumi = 0;
        for (int jj = 0; jj < 8; ++jj) {
            sumi += q8[jj] * grid[jj] * (signs & kmask_iq2xs[jj] ? -1 : 1);
        }

        const uint8_t ls = (sub < 2) ? (sc_bytes[ib32] & 0xf) : (sc_bytes[ib32] >> 4);
        sumf += (0.5f + ls) * sumi;
    }

    const float2 ds8 = __half22float2(y_ds[j * (WARP_SIZE_GGUF/QI8_1) + (QR_IQ2_XS_MMQ*k/QI8_1) % (WARP_SIZE_GGUF/QI8_1)]);

    return x_dmf[i * (WARP_SIZE_GGUF/QI_IQ2_XS_MMQ) + i/QI_IQ2_XS_MMQ]
         * 0.25f * ds8.x * sumf;
}

// ========================= IQ2_S MMQ =========================
// IQ2_S: QK_K=256, qs[QK_K/4=64], qh[QK_K/32=8], scales[QK_K/32=8]. Grid: iq2s_grid[1024] (uint64).
// Per ib32: qs[4] + qh[1] + signs[4] + scales[1].
// 10-bit index: 8 from qs + 2 from qh. Signs from dedicated array at qs+64.

#define QR_IQ2_S_MMQ 2
#define QI_IQ2_S_MMQ (QK_K / (4 * QR_IQ2_S_MMQ))
#define VDR_IQ2_S_Q8_1_MMQ 4

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_iq2_s(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    // slots 0..15: qs (64 bytes), slots 16..23: signs (32 bytes)
    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF/QI_IQ2_S_MMQ) + mmq_y/QI_IQ2_S_MMQ];
    __shared__ int   tile_x_qh[mmq_y * (WARP_SIZE_GGUF/4)     + mmq_y/4];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE_GGUF/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = (half2 *) tile_x_d;
    *x_qh = tile_x_qh;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq2_s(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    const int kbx  = k / QI_IQ2_S_MMQ;
    const int kqsx = k % QI_IQ2_S_MMQ;

    const block_iq2_s * bx0 = (const block_iq2_s *) vx;
    float * x_dmf = (float *) x_dm;

    // Load qs (64 bytes, slots 0..15) and signs (32 bytes at qs+64, slots 16..23)
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq2_s * bxi = bx0 + i*blocks_per_row + kbx;
        if (kqsx < 16) {
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        } else if (kqsx < 24) {
            // qs layout: [0..31] grid indices, [32..63] sign bytes (signs start at qs + QK_K/8)
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qs + QK_K/8, kqsx - 16);
        } else {
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = 0;
        }
    }

    // Load d
    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI_IQ2_S_MMQ;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI_IQ2_S_MMQ) {
        int i = (i0 + i_offset * QI_IQ2_S_MMQ + k / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq2_s * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE_GGUF/QI_IQ2_S_MMQ) + i / QI_IQ2_S_MMQ + kbxd] = __half2float(bxi->d);
    }

    // Load qh (8 bytes = 2 int32)
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = (i0 + i_offset * 4 + k / (WARP_SIZE_GGUF/4)) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const int kqh = k % (WARP_SIZE_GGUF/4);
        if (kqh < 2) {
            const block_iq2_s * bxi = bx0 + i*blocks_per_row + kbx;
            x_qh[i * (WARP_SIZE_GGUF/4) + i / 4 + kqh] = get_int_from_uint8(bxi->qh, kqh);
        }
    }

    // Load scales (8 bytes = 2 int32)
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE_GGUF/8)) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const int ksc = k % (WARP_SIZE_GGUF/8);
        if (ksc < 2) {
            const block_iq2_s * bxi = bx0 + i*blocks_per_row + kbx;
            x_sc[i * (WARP_SIZE_GGUF/8) + i / 8 + ksc] = get_int_from_uint8(bxi->scales, ksc);
        }
    }
}

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    const float * x_dmf = (const float *) x_dm;

    const int ib32 = k / 4;

    // Grid indices from qs[0..31] (stored in tile slots 0..7 as int32)
    const uint8_t * qs = (const uint8_t *)&x_ql[i * (WARP_SIZE_GGUF + 1)];

    // qh for the 10-bit index
    const uint8_t * qh_all = (const uint8_t *)&x_qh[i * (WARP_SIZE_GGUF/4) + i/4];
    const uint8_t qh_val = qh_all[ib32];

    // Signs from qs[32..63] (stored in tile slots 16..23)
    const uint8_t * sign_bytes = (const uint8_t *)&x_ql[i * (WARP_SIZE_GGUF + 1) + 16];

    // Per-sub-block scales
    const uint8_t * sc_bytes = (const uint8_t *)&x_sc[i * (WARP_SIZE_GGUF/8) + i/8];

    float sumf = 0;
#pragma unroll
    for (int l = 0; l < VDR_IQ2_S_Q8_1_MMQ; ++l) {
        const int sub = l;

        // 10-bit grid index
        const uint32_t * grid = (const uint32_t *)(iq2s_grid + (qs[4*ib32+sub] | ((qh_val << (8-2*sub)) & 0x300)));

        const uint8_t sign_byte = sign_bytes[4*ib32 + sub];
        uint32_t signs0 = __vcmpeq4(((sign_byte & 0xf) * 0x01010101) & 0x08040201, 0x08040201);
        uint32_t signs1 = __vcmpeq4(((sign_byte >>  4) * 0x01010101) & 0x08040201, 0x08040201);
        const int grid_l = __vsub4(grid[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid[1] ^ signs1, signs1);

        const int index_y = j * WARP_SIZE_GGUF + (QR_IQ2_S_MMQ * (k + l)) % WARP_SIZE_GGUF;

        int sumi = 0;
        sumi = __dp4a(grid_l, y_qs[index_y + 0], sumi);
        sumi = __dp4a(grid_h, y_qs[index_y + 1], sumi);

        const uint8_t ls = (sub < 2) ? (sc_bytes[ib32] & 0xf) : (sc_bytes[ib32] >> 4);
        sumf += (0.5f + ls) * sumi;
    }

    const float2 ds8 = __half22float2(y_ds[j * (WARP_SIZE_GGUF/QI8_1) + (QR_IQ2_S_MMQ*k/QI8_1) % (WARP_SIZE_GGUF/QI8_1)]);

    return x_dmf[i * (WARP_SIZE_GGUF/QI_IQ2_S_MMQ) + i/QI_IQ2_S_MMQ]
         * 0.25f * ds8.x * sumf;
}

// ========================= IQ1_S MMQ =========================
// IQ1_S: QK_K=256, qs[QK_K/8=32], qh[QK_K/32=8 uint16]. Grid: iq1s_grid_gpu[2048] (uint64).
// 11-bit grid index (8 from qs + 3 from qh). need_sum=true (delta correction).
// Grid values are packed nibbles: (grid>>0)&0x0F0F0F0F, (grid>>4)&0x0F0F0F0F.

#define QR_IQ1_S_MMQ 2
#define QI_IQ1_S_MMQ (QK_K / (4 * QR_IQ1_S_MMQ))
#define VDR_IQ1_S_Q8_1_MMQ 4

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_iq1_s(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    // slots 0..7: qs (32 bytes = 8 int32)
    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF/QI_IQ1_S_MMQ) + mmq_y/QI_IQ1_S_MMQ];
    __shared__ int   tile_x_qh[mmq_y * (WARP_SIZE_GGUF/4)     + mmq_y/4]; // qh: 16 bytes = 4 int32

    *x_ql = tile_x_ql;
    *x_dm = (half2 *) tile_x_d;
    *x_qh = tile_x_qh;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq1_s(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    (void)x_sc;

    const int kbx  = k / QI_IQ1_S_MMQ;
    const int kqsx = k % QI_IQ1_S_MMQ;

    const block_iq1_s * bx0 = (const block_iq1_s *) vx;
    float * x_dmf = (float *) x_dm;

    // Load qs (32 bytes = 8 int32, slots 0..7)
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq1_s * bxi = bx0 + i*blocks_per_row + kbx;
        if (kqsx < 8) {
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_b2(bxi->qs, kqsx);
        } else {
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = 0;
        }
    }

    // Load d scale
    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI_IQ1_S_MMQ;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI_IQ1_S_MMQ) {
        int i = (i0 + i_offset * QI_IQ1_S_MMQ + k / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq1_s * bxi = bx0 + i*blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE_GGUF/QI_IQ1_S_MMQ) + i / QI_IQ1_S_MMQ + kbxd] = __half2float(bxi->d);
    }

    // Load qh (8 uint16 = 16 bytes = 4 int32)
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = (i0 + i_offset * 4 + k / (WARP_SIZE_GGUF/4)) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const int kqh = k % (WARP_SIZE_GGUF/4);
        if (kqh < 4) {
            const block_iq1_s * bxi = bx0 + i*blocks_per_row + kbx;
            x_qh[i * (WARP_SIZE_GGUF/4) + i / 4 + kqh] = get_int_b2(bxi->qh, kqh);
        }
    }
}

static __device__ __forceinline__ float vec_dot_iq1_s_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    (void)x_sc;

    const float * x_dmf = (const float *) x_dm;

    const int iqs = k / 4;

    // Read qs (4 bytes per iqs group)
    const int qs_packed = x_ql[i * (WARP_SIZE_GGUF + 1) + iqs];
    const uint8_t * qs = (const uint8_t *)&qs_packed;

    // Read qh for this iqs group
    const uint16_t * qh_all = (const uint16_t *)&x_qh[i * (WARP_SIZE_GGUF/4) + i/4];
    const int qh = qh_all[iqs];

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_IQ1_S_Q8_1_MMQ; ++l) {
        const int sub = l;

        // 11-bit grid index: qs[sub] | (3 bits from qh << 8)
        const int grid = iq1s_grid_gpu[qs[sub] | (((qh >> 3*sub) & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;

        const int index_y = j * WARP_SIZE_GGUF + (QR_IQ1_S_MMQ * (k + l)) % WARP_SIZE_GGUF;
        sumi = __dp4a(grid0, y_qs[index_y + 0], sumi);
        sumi = __dp4a(grid1, y_qs[index_y + 1], sumi);
    }

    const float d1q = x_dmf[i * (WARP_SIZE_GGUF/QI_IQ1_S_MMQ) + i/QI_IQ1_S_MMQ]
                    * (((qh >> 11) & 0x0E) + 1);
    const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);

    // need_sum=true: y_ds stores half2(d, sum)
    const float2 ds = __half22float2(y_ds[j * (WARP_SIZE_GGUF/QI8_1) + (QR_IQ1_S_MMQ*k/QI8_1) % (WARP_SIZE_GGUF/QI8_1)]);

    return d1q * (ds.x * sumi + ds.y * delta);
}

// ========================= IQ1_M MMQ =========================
// IQ1_M: QK_K=256, qs[QK_K/8=32], qh[QK_K/16=16], scales[QK_K/32=8]. No d field.
// Scale reconstructed from iq1m_scale_t union. Grid: iq1s_grid_gpu[2048].
// need_sum=false but delta correction computed internally.

#define QR_IQ1_M_MMQ 2
#define QI_IQ1_M_MMQ (QK_K / (4 * QR_IQ1_M_MMQ))
#define VDR_IQ1_M_Q8_1_MMQ 4

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_iq1_m(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    // slots 0..7: qs (32 bytes), slots 8..11: qh (16 bytes)
    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE_GGUF)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF/QI_IQ1_M_MMQ) + mmq_y/QI_IQ1_M_MMQ];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE_GGUF/8)     + mmq_y/8]; // scales: 8 bytes = 2 int32

    *x_ql = tile_x_ql;
    *x_dm = (half2 *) tile_x_d;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_iq1_m(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    (void)x_qh;

    const int kbx  = k / QI_IQ1_M_MMQ;
    const int kqsx = k % QI_IQ1_M_MMQ;

    const block_iq1_m * bx0 = (const block_iq1_m *) vx;
    float * x_dmf = (float *) x_dm;

    // Load qs (32 bytes = 8 int32, slots 0..7) and qh (16 bytes = 4 int32, slots 8..11)
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq1_m * bxi = bx0 + i*blocks_per_row + kbx;
        if (kqsx < 8) {
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_b4(bxi->qs, kqsx);
        } else if (kqsx < 12) {
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qh, kqsx - 8);
        } else {
            x_ql[i * (WARP_SIZE_GGUF + 1) + k] = 0;
        }
    }

    // Load reconstructed d scale (from iq1m_scale_t)
    const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI_IQ1_M_MMQ;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI_IQ1_M_MMQ) {
        int i = (i0 + i_offset * QI_IQ1_M_MMQ + k / blocks_per_tile_x_row) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const block_iq1_m * bxi = bx0 + i*blocks_per_row + kbxd;
        const uint16_t * sc = (const uint16_t *)bxi->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        x_dmf[i * (WARP_SIZE_GGUF/QI_IQ1_M_MMQ) + i / QI_IQ1_M_MMQ + kbxd] = __half2float(scale.f16);
    }

    // Load scales (8 bytes = 2 int32)
#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE_GGUF/8)) % mmq_y;
        if (need_check) {
            i = min(i, i_max);
        }
        const int ksc = k % (WARP_SIZE_GGUF/8);
        if (ksc < 2) {
            const block_iq1_m * bxi = bx0 + i*blocks_per_row + kbx;
            x_sc[i * (WARP_SIZE_GGUF/8) + i / 8 + ksc] = get_int_from_uint8(bxi->scales, ksc);
        }
    }
}

static __device__ __forceinline__ float vec_dot_iq1_m_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    (void)x_qh;

    const float * x_dmf = (const float *) x_dm;

    const int iqs = k / 4;

    // Read qs (4 bytes per iqs group)
    const int qs_packed = x_ql[i * (WARP_SIZE_GGUF + 1) + iqs];
    const uint8_t * qs = (const uint8_t *)&qs_packed;

    // Read qh (stored at slots 8..11)
    const uint8_t * qh_all = (const uint8_t *)&x_ql[i * (WARP_SIZE_GGUF + 1) + 8];

    // Per-group scale from scales array
    const uint16_t * sc = (const uint16_t *)&x_sc[i * (WARP_SIZE_GGUF/8) + i/8];
    const int tmp = sc[iqs/2] >> (6*(iqs%2));

    float sumf = 0;
#pragma unroll
    for (int l = 0; l < VDR_IQ1_M_Q8_1_MMQ; ++l) {
        const int sub = l;
        const int l0 = sub * 2;
        const int qhl = qh_all[2*iqs + l0/4] >> (4 * ((l0/2) % 2));

        const int grid = iq1s_grid_gpu[qs[l0/2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;

        const int index_y = j * WARP_SIZE_GGUF + (QR_IQ1_M_MMQ * (k + l)) % WARP_SIZE_GGUF;
        const int u0 = y_qs[index_y + 0];
        const int u1 = y_qs[index_y + 1];

        int sumi = 0;
        sumi = __dp4a(grid0, u0, sumi);
        sumi = __dp4a(grid1, u1, sumi);

        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f*IQ1M_DELTA/0x08);
        int sumy = 0;
        sumy = __dp4a(u0, 0x01010101, sumy);
        sumy = __dp4a(u1, 0x01010101, sumy);

        const int sc_val = (sub < 2) ? (2*((tmp >> 0) & 0x07) + 1) : (2*((tmp >> 3) & 0x07) + 1);
        sumf += sc_val * (sumi + delta * sumy);
    }

    // need_sum=false: y_ds stores float d (not half2)
    const float * y_df = (const float *) y_ds;
    const float d8 = y_df[j * (WARP_SIZE_GGUF/QI8_1) + (QR_IQ1_M_MMQ*k/QI8_1) % (WARP_SIZE_GGUF/QI8_1)];

    return x_dmf[i * (WARP_SIZE_GGUF/QI_IQ1_M_MMQ) + i/QI_IQ1_M_MMQ]
         * d8 * sumf;
}
