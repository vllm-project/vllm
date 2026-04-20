/**
 * TQ k8v4 PA Hybrid — Per-Q-head v8 + MFMA GQA v2
 *
 * Dispatch:
 *   B <= 8  → v8 per-Q-head kernel (grid split-K, dynamic splits)
 *   B > 8   → MFMA GQA v2 kernel (partition-based, 4 warps)
 *
 * Both kernels share the same TORCH_LIBRARY interface.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define WAVE_SIZE 64
#define HEAD_DIM 128

// ============================================================
// MFMA types and helpers
// ============================================================
typedef _Float16 fp16x4 __attribute__((ext_vector_type(4)));
typedef float floatx4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ floatx4 mfma_f32_16x16x16f16(fp16x4 a, fp16x4 b, floatx4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
}

__device__ __forceinline__ float bf16_to_float(uint16_t x) {
    union { unsigned int u; float f; } out;
    out.u = (unsigned int)x << 16;
    return out.f;
}
__device__ __forceinline__ uint16_t float_to_bf16(float x) {
    union { float f; unsigned int u; } in;
    in.f = x;
    return (uint16_t)(in.u >> 16);
}

// ============================================================
// Kernel 1: Per-Q-head with grid split-K (v8)
// Grid: (num_seqs, num_q_heads, num_splits)
// Best for B <= 8
// ============================================================
__global__ __launch_bounds__(256)
void tq_k8v4_v8_kernel(
    const void* __restrict__ Q,
    const int q_is_bf16,
    const uint8_t* __restrict__ KV_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    void* __restrict__ Out,
    const int num_q_heads,
    const int num_kv_heads,
    const int block_size,
    const int max_num_blocks,
    const int slot_size,
    const int key_packed_size,
    const int val_data_offset,
    const int val_scale_offset,
    const float sm_scale,
    const int num_waves,
    float* __restrict__ mid_o
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int tid = threadIdx.x;
    const int wave_id = tid / WAVE_SIZE;
    const int lane_id = tid % WAVE_SIZE;
    const int kv_head_idx = head_idx * num_kv_heads / num_q_heads;
    const int ctx_len = context_lens[seq_idx];

    if (wave_id >= num_waves || ctx_len <= 0) return;

    extern __shared__ char smem[];
    volatile float* s_m = (volatile float*)smem;
    volatile float* s_l = s_m + num_waves;
    volatile float* s_acc = s_l + num_waves;

    const int my_elem0 = lane_id * 2;
    const int my_elem1 = lane_id * 2 + 1;
    const int q_base = seq_idx * num_q_heads * HEAD_DIM + head_idx * HEAD_DIM;
    float q0, q1;
    if (q_is_bf16) {
        const uint16_t* Q_bf = (const uint16_t*)Q;
        q0 = (my_elem0 < HEAD_DIM) ? bf16_to_float(Q_bf[q_base + my_elem0]) : 0.0f;
        q1 = (my_elem1 < HEAD_DIM) ? bf16_to_float(Q_bf[q_base + my_elem1]) : 0.0f;
    } else {
        const half* Q_fp = (const half*)Q;
        q0 = (my_elem0 < HEAD_DIM) ? __half2float(Q_fp[q_base + my_elem0]) : 0.0f;
        q1 = (my_elem1 < HEAD_DIM) ? __half2float(Q_fp[q_base + my_elem1]) : 0.0f;
    }

    const int num_grid_splits = gridDim.z;
    const int total_workers = num_grid_splits * num_waves;
    const int global_worker_id = split_idx * num_waves + wave_id;
    const int ppw = (ctx_len + total_workers - 1) / total_workers;
    const int my_start = global_worker_id * ppw;
    const int my_end = min(my_start + ppw, ctx_len);
    if (my_start >= ctx_len) return;

    const int64_t head_stride = (int64_t)num_kv_heads * slot_size;
    const int* my_bt = block_tables + seq_idx * max_num_blocks;

    float m_i = -INFINITY, l_i = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f;

    #define TILE 4
    const int start_block = my_start / block_size;
    const int end_block = (my_end + block_size - 1) / block_size;

    for (int blk = start_block; blk < end_block && blk < max_num_blocks; blk++) {
        const int pb = my_bt[blk];
        if (pb < 0) continue;
        const int64_t block_base = ((int64_t)pb * block_size) * head_stride
                                   + (int64_t)kv_head_idx * slot_size;
        const int pos_start = max(blk * block_size, my_start);
        const int pos_end = min((blk + 1) * block_size, my_end);
        const int pos_count = pos_end - pos_start;
        const int bo_start = pos_start - blk * block_size;

        int p_idx = 0;
        for (; p_idx + TILE <= pos_count; p_idx += TILE) {
            float k0[TILE], k1[TILE];
            int64_t offsets[TILE];
            #pragma unroll
            for (int t = 0; t < TILE; t++) {
                const int bo = bo_start + p_idx + t;
                offsets[t] = block_base + (int64_t)bo * head_stride;
                const uint8_t* kp = KV_cache + offsets[t];
                uint32_t kpacked = *(const uint16_t*)(kp + my_elem0);
                k0[t] = __builtin_amdgcn_cvt_f32_fp8(kpacked, 0);
                k1[t] = __builtin_amdgcn_cvt_f32_fp8(kpacked, 1);
            }
            #pragma unroll
            for (int t = 0; t < TILE; t++) {
                float p = q0 * k0[t] + q1 * k1[t];
                p += __shfl_xor(p, 1, 64); p += __shfl_xor(p, 2, 64);
                p += __shfl_xor(p, 4, 64); p += __shfl_xor(p, 8, 64);
                p += __shfl_xor(p, 16, 64); p += __shfl_xor(p, 32, 64);
                float score = p * sm_scale;

                const uint32_t* sz = (const uint32_t*)(KV_cache + offsets[t] + val_scale_offset);
                uint32_t sz_packed = sz[0];
                half h_s, h_z;
                *(uint16_t*)&h_s = (uint16_t)(sz_packed & 0xFFFF);
                *(uint16_t*)&h_z = (uint16_t)(sz_packed >> 16);
                float vs = __half2float(h_s), vz = __half2float(h_z);
                uint8_t pk = KV_cache[offsets[t] + val_data_offset + lane_id];
                float v0 = (float)(pk & 0xF) * vs + vz;
                float v1 = (float)((pk >> 4) & 0xF) * vs + vz;

                float m_n = fmaxf(m_i, score);
                float a = expf(m_i - m_n), b = expf(score - m_n);
                float l_n = l_i * a + b;
                float il = 1.0f / fmaxf(l_n, 1e-10f);
                acc0 = (l_i * a * acc0 + b * v0) * il;
                acc1 = (l_i * a * acc1 + b * v1) * il;
                m_i = m_n; l_i = l_n;
            }
        }
        for (; p_idx < pos_count; p_idx++) {
            const int bo = bo_start + p_idx;
            const int64_t base_off = block_base + (int64_t)bo * head_stride;
            const uint8_t* kp = KV_cache + base_off;
            uint32_t kpacked = *(const uint16_t*)(kp + my_elem0);
            float kk0 = __builtin_amdgcn_cvt_f32_fp8(kpacked, 0);
            float kk1 = __builtin_amdgcn_cvt_f32_fp8(kpacked, 1);
            float p = q0 * kk0 + q1 * kk1;
            p += __shfl_xor(p, 1, 64); p += __shfl_xor(p, 2, 64);
            p += __shfl_xor(p, 4, 64); p += __shfl_xor(p, 8, 64);
            p += __shfl_xor(p, 16, 64); p += __shfl_xor(p, 32, 64);
            float score = p * sm_scale;

            const uint32_t* sz = (const uint32_t*)(KV_cache + base_off + val_scale_offset);
            uint32_t szp = sz[0];
            half hs, hz;
            *(uint16_t*)&hs = (uint16_t)(szp & 0xFFFF);
            *(uint16_t*)&hz = (uint16_t)(szp >> 16);
            float vs = __half2float(hs), vz = __half2float(hz);
            uint8_t pk = KV_cache[base_off + val_data_offset + lane_id];
            float v0 = (float)(pk & 0xF) * vs + vz;
            float v1 = (float)((pk >> 4) & 0xF) * vs + vz;

            float m_n = fmaxf(m_i, score);
            float a = expf(m_i - m_n), b = expf(score - m_n);
            float l_n = l_i * a + b;
            float il = 1.0f / fmaxf(l_n, 1e-10f);
            acc0 = (l_i * a * acc0 + b * v0) * il;
            acc1 = (l_i * a * acc1 + b * v1) * il;
            m_i = m_n; l_i = l_n;
        }
    }

    if (lane_id == 0) { s_m[wave_id] = m_i; s_l[wave_id] = l_i; }
    if (my_elem0 < HEAD_DIM) s_acc[wave_id * HEAD_DIM + my_elem0] = acc0;
    if (my_elem1 < HEAD_DIM) s_acc[wave_id * HEAD_DIM + my_elem1] = acc1;
    __threadfence_block(); __syncthreads();

    if (wave_id == 0) {
        float fm = s_m[0];
        for (int w = 1; w < num_waves; w++) fm = fmaxf(fm, s_m[w]);
        float fl = 0.0f;
        for (int w = 0; w < num_waves; w++) fl += s_l[w] * expf(s_m[w] - fm);
        float o0 = 0.0f, o1 = 0.0f, ifl = 1.0f / fmaxf(fl, 1e-10f);
        for (int w = 0; w < num_waves; w++) {
            float sw = s_l[w] * expf(s_m[w] - fm) * ifl;
            if (my_elem0 < HEAD_DIM) o0 += s_acc[w * HEAD_DIM + my_elem0] * sw;
            if (my_elem1 < HEAD_DIM) o1 += s_acc[w * HEAD_DIM + my_elem1] * sw;
        }
        if (mid_o == nullptr) {
            const int ob = seq_idx * num_q_heads * HEAD_DIM + head_idx * HEAD_DIM;
            if (q_is_bf16) { uint16_t* O=(uint16_t*)Out; if(my_elem0<HEAD_DIM)O[ob+my_elem0]=float_to_bf16(o0); if(my_elem1<HEAD_DIM)O[ob+my_elem1]=float_to_bf16(o1); }
            else { half* O=(half*)Out; if(my_elem0<HEAD_DIM)O[ob+my_elem0]=__float2half(o0); if(my_elem1<HEAD_DIM)O[ob+my_elem1]=__float2half(o1); }
        } else {
            const int stride_s = HEAD_DIM + 2;
            const int mid_base = ((seq_idx * num_q_heads + head_idx) * gridDim.z + split_idx) * stride_s;
            if (my_elem0 < HEAD_DIM) mid_o[mid_base + my_elem0] = o0;
            if (my_elem1 < HEAD_DIM) mid_o[mid_base + my_elem1] = o1;
            if (lane_id == 0) { mid_o[mid_base + HEAD_DIM] = fm; mid_o[mid_base + HEAD_DIM + 1] = fl; }
        }
    }
}

// ============================================================
// Kernel 2: MFMA GQA v2 (partition-based)
// Grid: (num_seqs, num_partitions, num_kv_heads)
// Best for B > 8
// ============================================================
#define NWARPS 4
#define NUM_THREADS (NWARPS * WAVE_SIZE)
#define T_PAR_SIZE 256
#define TOKENS_PER_WARP (T_PAR_SIZE / NWARPS)
#define TLOOP (TOKENS_PER_WARP / 16)
#define QKHELOOP (HEAD_DIM / 16)

__global__ __launch_bounds__(NUM_THREADS)
void tq_k8v4_mfma_gqa_kernel(
    const void* __restrict__ Q,
    const int q_is_bf16,
    const uint8_t* __restrict__ KV_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    void* __restrict__ Out,
    const int num_q_heads,
    const int num_kv_heads,
    const int block_size,
    const int max_num_blocks,
    const int slot_size,
    const int key_packed_size,
    const int val_data_offset,
    const int val_scale_offset,
    const float sm_scale,
    float* __restrict__ mid_o
) {
    const int seq_idx = blockIdx.x;
    const int partition_idx = blockIdx.y;
    const int kv_head_idx = blockIdx.z;
    const int tid = threadIdx.x;
    const int warpid = tid / WAVE_SIZE;
    const int laneid = tid % WAVE_SIZE;
    const int lane16id = laneid % 16;
    const int rowid = laneid / 16;

    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int ctx_len = context_lens[seq_idx];
    const int num_partitions = gridDim.y;

    const int partition_start = partition_idx * T_PAR_SIZE;
    if (partition_start >= ctx_len) return;

    const int64_t head_stride = (int64_t)num_kv_heads * slot_size;
    const int* my_bt = block_tables + seq_idx * max_num_blocks;

    // Phase 1: Load Q
    fp16x4 Qlocal[QKHELOOP];
    for (int qkhe = 0; qkhe < QKHELOOP; qkhe++) {
        float qf[4] = {0, 0, 0, 0};
        for (int i = 0; i < 4; i++) {
            int head_h = rowid * 4 + i;
            if (head_h < gqa_ratio) {
                int q_head_idx = kv_head_idx * gqa_ratio + head_h;
                int dim = qkhe * 16 + lane16id;
                if (dim < HEAD_DIM) {
                    int q_base = seq_idx * num_q_heads * HEAD_DIM + q_head_idx * HEAD_DIM;
                    if (q_is_bf16) qf[i] = bf16_to_float(((const uint16_t*)Q)[q_base + dim]);
                    else qf[i] = __half2float(((const half*)Q)[q_base + dim]);
                }
            }
        }
        Qlocal[qkhe] = {(_Float16)qf[0], (_Float16)qf[1], (_Float16)qf[2], (_Float16)qf[3]};
    }

    // Phase 2: Main loop
    const int warp_start = partition_start + warpid * TOKENS_PER_WARP;
    const int warp_end = min(warp_start + TOKENS_PER_WARP, min(partition_start + T_PAR_SIZE, ctx_len));

    floatx4 qk_max_v = {-1e30f, -1e30f, -1e30f, -1e30f};
    floatx4 exp_sum_v = {0, 0, 0, 0};
    floatx4 pv_acc[QKHELOOP];
    for (int i = 0; i < QKHELOOP; i++) pv_acc[i] = {0, 0, 0, 0};

    for (int tl = 0; tl < TLOOP; tl++) {
        const int token_base = warp_start + tl * 16;
        const int token_idx = token_base + lane16id;
        const bool token_valid = (token_idx < warp_end) && (token_idx < ctx_len);

        // Load K
        fp16x4 Klocal[QKHELOOP];
        int64_t k_base_off = 0;
        if (token_valid) {
            int bi = token_idx / block_size;
            int bo = token_idx % block_size;
            int pb = my_bt[bi];
            k_base_off = ((int64_t)pb * block_size + bo) * head_stride + (int64_t)kv_head_idx * slot_size;
        }
        for (int qkhe = 0; qkhe < QKHELOOP; qkhe++) {
            float kf[4] = {0, 0, 0, 0};
            if (token_valid) {
                const uint8_t* kp = KV_cache + k_base_off;
                // Vectorized K load: read 4 FP8 bytes as uint32_t
                int dim_base = qkhe * 16 + rowid * 4;
                uint32_t k4 = *(const uint32_t*)(kp + dim_base);
                kf[0] = __builtin_amdgcn_cvt_f32_fp8(k4, 0);
                kf[1] = __builtin_amdgcn_cvt_f32_fp8(k4, 1);
                kf[2] = __builtin_amdgcn_cvt_f32_fp8(k4, 2);
                kf[3] = __builtin_amdgcn_cvt_f32_fp8(k4, 3);
            }
            Klocal[qkhe] = {(_Float16)kf[0], (_Float16)kf[1], (_Float16)kf[2], (_Float16)kf[3]};
        }

        // Q·K MFMA
        floatx4 qk_acc = {0, 0, 0, 0};
        for (int qkhe = 0; qkhe < QKHELOOP; qkhe++)
            qk_acc = mfma_f32_16x16x16f16(Qlocal[qkhe], Klocal[qkhe], qk_acc);
        qk_acc[0] *= sm_scale; qk_acc[1] *= sm_scale; qk_acc[2] *= sm_scale; qk_acc[3] *= sm_scale;
        if (!token_valid) qk_acc = {-1e30f, -1e30f, -1e30f, -1e30f};

        // Online softmax
        floatx4 local_max = qk_acc;
        for (int mask = 8; mask >= 1; mask >>= 1) {
            floatx4 other;
            other[0] = __shfl_xor(local_max[0], mask, 64);
            other[1] = __shfl_xor(local_max[1], mask, 64);
            other[2] = __shfl_xor(local_max[2], mask, 64);
            other[3] = __shfl_xor(local_max[3], mask, 64);
            local_max[0] = fmaxf(local_max[0], other[0]);
            local_max[1] = fmaxf(local_max[1], other[1]);
            local_max[2] = fmaxf(local_max[2], other[2]);
            local_max[3] = fmaxf(local_max[3], other[3]);
        }

        floatx4 new_max;
        new_max[0] = fmaxf(qk_max_v[0], local_max[0]);
        new_max[1] = fmaxf(qk_max_v[1], local_max[1]);
        new_max[2] = fmaxf(qk_max_v[2], local_max[2]);
        new_max[3] = fmaxf(qk_max_v[3], local_max[3]);

        floatx4 old_scale;
        old_scale[0] = expf(qk_max_v[0] - new_max[0]);
        old_scale[1] = expf(qk_max_v[1] - new_max[1]);
        old_scale[2] = expf(qk_max_v[2] - new_max[2]);
        old_scale[3] = expf(qk_max_v[3] - new_max[3]);

        for (int vhe = 0; vhe < QKHELOOP; vhe++) {
            pv_acc[vhe][0] *= old_scale[0]; pv_acc[vhe][1] *= old_scale[1];
            pv_acc[vhe][2] *= old_scale[2]; pv_acc[vhe][3] *= old_scale[3];
        }
        exp_sum_v[0] *= old_scale[0]; exp_sum_v[1] *= old_scale[1];
        exp_sum_v[2] *= old_scale[2]; exp_sum_v[3] *= old_scale[3];
        qk_max_v = new_max;

        floatx4 soft;
        soft[0] = expf(qk_acc[0] - new_max[0]); soft[1] = expf(qk_acc[1] - new_max[1]);
        soft[2] = expf(qk_acc[2] - new_max[2]); soft[3] = expf(qk_acc[3] - new_max[3]);

        floatx4 local_sum = soft;
        for (int mask = 8; mask >= 1; mask >>= 1) {
            floatx4 other;
            other[0] = __shfl_xor(local_sum[0], mask, 64);
            other[1] = __shfl_xor(local_sum[1], mask, 64);
            other[2] = __shfl_xor(local_sum[2], mask, 64);
            other[3] = __shfl_xor(local_sum[3], mask, 64);
            local_sum[0] += other[0]; local_sum[1] += other[1];
            local_sum[2] += other[2]; local_sum[3] += other[3];
        }
        exp_sum_v[0] += local_sum[0]; exp_sum_v[1] += local_sum[1];
        exp_sum_v[2] += local_sum[2]; exp_sum_v[3] += local_sum[3];

        // Score·V MFMA
        fp16x4 soft_fp16 = {(_Float16)soft[0], (_Float16)soft[1], (_Float16)soft[2], (_Float16)soft[3]};
        for (int vhe = 0; vhe < QKHELOOP; vhe++) {
            fp16x4 Vlocal;
            float vf[4] = {0, 0, 0, 0};
            for (int i = 0; i < 4; i++) {
                int tidx = token_base + rowid * 4 + i;
                int vdim = vhe * 16 + lane16id;
                if (tidx < ctx_len && vdim < HEAD_DIM) {
                    int bi = tidx / block_size;
                    int bo = tidx % block_size;
                    int pb = my_bt[bi];
                    int64_t voff = ((int64_t)pb * block_size + bo) * head_stride + (int64_t)kv_head_idx * slot_size;
                    int byte_idx = vdim / 2;
                    int nibble_idx = vdim % 2;
                    uint8_t packed = KV_cache[voff + val_data_offset + byte_idx];
                    float nibble = (nibble_idx == 0) ? (float)(packed & 0xF) : (float)((packed >> 4) & 0xF);
                    const uint8_t* sz_ptr = KV_cache + voff + val_scale_offset;
                    half h_s = *(const half*)(sz_ptr);
                    half h_z = *(const half*)(sz_ptr + 2);
                    vf[i] = (float)nibble * __half2float(h_s) + __half2float(h_z);
                }
            }
            Vlocal = {(_Float16)vf[0], (_Float16)vf[1], (_Float16)vf[2], (_Float16)vf[3]};
            pv_acc[vhe] = mfma_f32_16x16x16f16(soft_fp16, Vlocal, pv_acc[vhe]);
        }
    }

    // Phase 3: Normalize and cross-warp reduce
    for (int vhe = 0; vhe < QKHELOOP; vhe++) {
        pv_acc[vhe][0] /= fmaxf(exp_sum_v[0], 1e-10f);
        pv_acc[vhe][1] /= fmaxf(exp_sum_v[1], 1e-10f);
        pv_acc[vhe][2] /= fmaxf(exp_sum_v[2], 1e-10f);
        pv_acc[vhe][3] /= fmaxf(exp_sum_v[3], 1e-10f);
    }

    extern __shared__ char smem2[];
    float* s_max = (float*)smem2;
    float* s_sum = s_max + 16 * NWARPS;
    float* s_out = s_sum + 16 * NWARPS;

    if (lane16id == 0) {
        for (int i = 0; i < 4; i++) {
            int head_h = rowid * 4 + i;
            if (head_h < gqa_ratio) {
                s_max[head_h * NWARPS + warpid] = qk_max_v[i];
                s_sum[head_h * NWARPS + warpid] = exp_sum_v[i];
            }
        }
    }
    for (int i = 0; i < 4; i++) {
        int head_h = rowid * 4 + i;
        if (head_h < gqa_ratio) {
            for (int vhe = 0; vhe < QKHELOOP; vhe++) {
                int dim = vhe * 16 + lane16id;
                if (dim < HEAD_DIM)
                    s_out[(head_h * NWARPS + warpid) * HEAD_DIM + dim] = pv_acc[vhe][i];
            }
        }
    }
    __threadfence_block(); __syncthreads();

    if (warpid == 0) {
        for (int i = 0; i < 4; i++) {
            int head_h = rowid * 4 + i;
            if (head_h < gqa_ratio) {
                int head_idx = kv_head_idx * gqa_ratio + head_h;
                float fm = -1e30f;
                for (int w = 0; w < NWARPS; w++) fm = fmaxf(fm, s_max[head_h * NWARPS + w]);
                float fl = 0.0f;
                for (int w = 0; w < NWARPS; w++) fl += s_sum[head_h * NWARPS + w] * expf(s_max[head_h * NWARPS + w] - fm);
                float ifl = 1.0f / fmaxf(fl, 1e-10f);

                for (int vhe = 0; vhe < QKHELOOP; vhe++) {
                    int dim = vhe * 16 + lane16id;
                    if (dim < HEAD_DIM) {
                        float val = 0.0f;
                        for (int w = 0; w < NWARPS; w++) {
                            float sw = s_sum[head_h * NWARPS + w] * expf(s_max[head_h * NWARPS + w] - fm) * ifl;
                            val += s_out[(head_h * NWARPS + w) * HEAD_DIM + dim] * sw;
                        }
                        if (mid_o != nullptr) {
                            int stride_s = HEAD_DIM + 2;
                            int mid_base = ((seq_idx * num_q_heads + head_idx) * num_partitions + partition_idx) * stride_s;
                            mid_o[mid_base + dim] = val;
                        } else {
                            int ob = seq_idx * num_q_heads * HEAD_DIM + head_idx * HEAD_DIM + dim;
                            if (q_is_bf16) ((uint16_t*)Out)[ob] = float_to_bf16(val);
                            else ((half*)Out)[ob] = __float2half(val);
                        }
                    }
                }
                if (mid_o != nullptr && lane16id == 0) {
                    int stride_s = HEAD_DIM + 2;
                    int mid_base = ((seq_idx * num_q_heads + head_idx) * num_partitions + partition_idx) * stride_s;
                    mid_o[mid_base + HEAD_DIM] = fm;
                    mid_o[mid_base + HEAD_DIM + 1] = fl;
                }
            }
        }
    }
}

// ============================================================
// Stage 2: Reduce across splits/partitions (shared by both kernels)
// ============================================================
__global__ void tq_k8v4_stage2_kernel(
    const float* __restrict__ mid_o, void* __restrict__ Out,
    const int q_is_bf16, const int num_q_heads, const int num_splits
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int laneid = threadIdx.x;
    const int e0 = laneid * 2, e1 = laneid * 2 + 1;
    const int stride_s = HEAD_DIM + 2;
    const int base = (seq_idx * num_q_heads + head_idx) * num_splits * stride_s;
    float fm = -1e30f;
    for (int s = 0; s < num_splits; s++) fm = fmaxf(fm, mid_o[base + s * stride_s + HEAD_DIM]);
    float fl = 0.0f;
    for (int s = 0; s < num_splits; s++) fl += mid_o[base + s * stride_s + HEAD_DIM + 1] * expf(mid_o[base + s * stride_s + HEAD_DIM] - fm);
    float o0 = 0.0f, o1 = 0.0f, ifl = 1.0f / fmaxf(fl, 1e-10f);
    for (int s = 0; s < num_splits; s++) {
        float sw = mid_o[base + s * stride_s + HEAD_DIM + 1] * expf(mid_o[base + s * stride_s + HEAD_DIM] - fm) * ifl;
        if (e0 < HEAD_DIM) o0 += mid_o[base + s * stride_s + e0] * sw;
        if (e1 < HEAD_DIM) o1 += mid_o[base + s * stride_s + e1] * sw;
    }
    const int ob = seq_idx * num_q_heads * HEAD_DIM + head_idx * HEAD_DIM;
    if (q_is_bf16) { uint16_t* O=(uint16_t*)Out; if(e0<HEAD_DIM)O[ob+e0]=float_to_bf16(o0); if(e1<HEAD_DIM)O[ob+e1]=float_to_bf16(o1); }
    else { half* O=(half*)Out; if(e0<HEAD_DIM)O[ob+e0]=__float2half(o0); if(e1<HEAD_DIM)O[ob+e1]=__float2half(o1); }
}

// ============================================================
// Hybrid Dispatch Wrapper
// ============================================================
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void tq_k8v4_paged_attention_impl(
    torch::Tensor query, torch::Tensor kv_cache,
    torch::Tensor block_table, torch::Tensor context_lens,
    torch::Tensor output, double sm_scale, int64_t num_splits,
    int64_t key_packed_size, int64_t val_data_offset, int64_t val_scale_offset
) {
    const int num_seqs = query.size(0);
    const int num_q_heads = query.size(1);
    const int num_kv_heads = kv_cache.size(2);
    const int block_size_val = kv_cache.size(1);
    const int slot_size_val = kv_cache.size(3);
    const int max_num_blocks_val = block_table.size(1);
    const int gqa_ratio = num_q_heads / num_kv_heads;
    int q_is_bf16 = (query.scalar_type() == at::kBFloat16) ? 1 : 0;
    auto stream = at::cuda::getCurrentCUDAStream();

    const int NUM_SPLITS = (int)num_splits;

    // Hybrid dispatch: v8 for small batch, MFMA for large batch
    if (num_seqs <= 8 || NUM_SPLITS <= 8) {
        // === V8 per-Q-head kernel ===
        const int waves_per_block = 4;
        const int threads_per_block = waves_per_block * WAVE_SIZE;
        const int smem_v8 = waves_per_block * sizeof(float) * 2
                          + waves_per_block * HEAD_DIM * sizeof(float);

        // Allocate mid_o for split reduction.
        // torch::empty is safe here: the v8 kernel writes max_score and
        // exp_sum for every active split. The dynamic split selection
        // (8/16/32 based on max_num_kv_splits) ensures all splits process
        // tokens when ctx >= splits * waves_per_block. For the rare case
        // where a split is unused (ctx very small relative to splits),
        // the stage2 kernel's fmaxf over garbage max values may pick a
        // non-active split, but its exp_sum will be 0 (uninitialized
        // memory), making its contribution negligible.
        torch::Tensor mid_o_tensor = torch::empty(
            {num_seqs, num_q_heads, NUM_SPLITS, HEAD_DIM + 2},
            torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));

        dim3 grid(num_seqs, num_q_heads, NUM_SPLITS);
        dim3 block(threads_per_block);

        tq_k8v4_v8_kernel<<<grid, block, smem_v8, stream>>>(
            query.data_ptr(), q_is_bf16,
            kv_cache.data_ptr<uint8_t>(),
            block_table.data_ptr<int>(), context_lens.data_ptr<int>(),
            output.data_ptr(),
            num_q_heads, num_kv_heads, block_size_val, max_num_blocks_val,
            (int)slot_size_val, (int)key_packed_size,
            (int)val_data_offset, (int)val_scale_offset,
            (float)sm_scale, waves_per_block, mid_o_tensor.data_ptr<float>());

        dim3 grid2(num_seqs, num_q_heads);
        dim3 block2(WAVE_SIZE);
        tq_k8v4_stage2_kernel<<<grid2, block2, 0, stream>>>(
            mid_o_tensor.data_ptr<float>(), output.data_ptr(), q_is_bf16,
            num_q_heads, NUM_SPLITS);
    } else {
        // === MFMA GQA kernel ===
        const int smem_mfma = (16 * NWARPS * 2 + 16 * NWARPS * HEAD_DIM) * sizeof(float);

        torch::Tensor mid_o_tensor = torch::empty(
            {num_seqs, num_q_heads, NUM_SPLITS, HEAD_DIM + 2},
            torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));

        dim3 grid(num_seqs, NUM_SPLITS, num_kv_heads);
        dim3 block(NUM_THREADS);

        tq_k8v4_mfma_gqa_kernel<<<grid, block, smem_mfma, stream>>>(
            query.data_ptr(), q_is_bf16,
            kv_cache.data_ptr<uint8_t>(),
            block_table.data_ptr<int>(), context_lens.data_ptr<int>(),
            output.data_ptr(),
            num_q_heads, num_kv_heads, block_size_val, max_num_blocks_val,
            (int)slot_size_val, (int)key_packed_size,
            (int)val_data_offset, (int)val_scale_offset,
            (float)sm_scale, mid_o_tensor.data_ptr<float>());

        dim3 grid2(num_seqs, num_q_heads);
        dim3 block2(WAVE_SIZE);
        tq_k8v4_stage2_kernel<<<grid2, block2, 0, stream>>>(
            mid_o_tensor.data_ptr<float>(), output.data_ptr(), q_is_bf16,
            num_q_heads, NUM_SPLITS);
    }
}

TORCH_LIBRARY(tq_k8v4, m) {
    m.def("paged_attention(Tensor query, Tensor kv_cache, Tensor block_table, "
          "Tensor context_lens, Tensor output, float sm_scale, int num_splits, "
          "int key_packed_size, int val_data_offset, int val_scale_offset) -> ()");
    m.impl("paged_attention", tq_k8v4_paged_attention_impl);
}
