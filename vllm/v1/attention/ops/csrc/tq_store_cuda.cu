// SPDX-License-Identifier: Apache-2.0
// TurboQuant fused CUDA store kernel.
//
// Single kernel: normalize → rotate(Pi) → bucketize → reconstruct
//                → residual → QJL(S) → pack(MSE+QJL+norms) → value quant → scatter
//
// Optimizations:
// - Warp-shuffle reductions for norms (no shared memory atomics)
// - Tiled GEMV with shared memory for rotation + QJL projection
// - __launch_bounds__ for register pressure tuning
// - Single kernel launch for the entire store pipeline
// - Warp-shuffle packing for MSE/QJL/value bit-packing

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cmath>

#define FULL_MASK 0xFFFFFFFF

// ---- Configuration ----
#ifndef TQ_STORE_HEAD_DIM
#define TQ_STORE_HEAD_DIM 256
#endif

constexpr int D = TQ_STORE_HEAD_DIM;
constexpr int THREADS = D;
constexpr int NUM_WARPS = THREADS / 32;

// TQ3: 2-bit MSE, 4 centroids
constexpr int MSE_BITS = 2;
constexpr int MSE_BYTES = D / 4;   // 2-bit: 4 per byte, D must be multiple of 4
constexpr int QJL_BYTES = D / 8;   // 1-bit: 8 per byte, D must be multiple of 8

// Tile size for GEMV
constexpr int TILE_K = 32;

// ---- Warp-level reduce sum ----
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

// ---- Block-level reduce sum via shared memory ----
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    val = warp_reduce_sum(val);

    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }

    __syncthreads();
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

// ---- Block-level reduce min/max via shared memory ----
// NOTE: Cross-warp reduction uses serial loop instead of __shfl_down_sync
// because SM 12.0 (Blackwell) hangs on partial-warp shuffles with fminf/fmaxf
// in the conditional (warp_id==0 && lane_id<NUM_WARPS) path.
__device__ __forceinline__ void block_reduce_minmax(
    float val, float* smem, float& out_min, float& out_max
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    float v_min = val, v_max = val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float o_min = __shfl_down_sync(FULL_MASK, v_min, offset);
        float o_max = __shfl_down_sync(FULL_MASK, v_max, offset);
        v_min = fminf(v_min, o_min);
        v_max = fmaxf(v_max, o_max);
    }

    if (lane_id == 0) {
        smem[warp_id] = v_min;
        smem[warp_id + NUM_WARPS] = v_max;
    }
    __syncthreads();

    // Serial cross-warp reduction by thread 0
    if (threadIdx.x == 0) {
        v_min = smem[0]; v_max = smem[NUM_WARPS];
        #pragma unroll
        for (int i = 1; i < NUM_WARPS; i++) {
            v_min = fminf(v_min, smem[i]);
            v_max = fmaxf(v_max, smem[i + NUM_WARPS]);
        }
        smem[0] = v_min;
        smem[1] = v_max;
    }
    __syncthreads();
    out_min = smem[0];
    out_max = smem[1];
}

// ---- Warp-shuffle pack: gather N values within groups of N lanes ----
// For groups of 4 (MSE 2-bit packing) or 8 (QJL sign packing),
// all group members are within the same warp (since D=256, each warp has 32 lanes,
// and groups of 4 or 8 naturally fit within a warp).
__device__ __forceinline__ int shuffle_pack_4(int shifted_val) {
    // Pack 4 lanes: lane%4 == {0,1,2,3} within a group
    int lane_id = threadIdx.x % 32;
    int base = (lane_id / 4) * 4;
    int packed = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        packed |= __shfl_sync(FULL_MASK, shifted_val, base + i);
    }
    return packed;
}

__device__ __forceinline__ int shuffle_pack_8(int shifted_val) {
    int lane_id = threadIdx.x % 32;
    int base = (lane_id / 8) * 8;
    int packed = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        packed |= __shfl_sync(FULL_MASK, shifted_val, base + i);
    }
    return packed;
}

// =====================================================================
// Main fused store kernel
// =====================================================================
template <bool NO_QJL, int VQB>
__global__ void __launch_bounds__(THREADS, 4)
tq_fused_store_kernel(
    const __half* __restrict__ key_ptr,       // [N, H, D] half
    const __half* __restrict__ value_ptr,     // [N, H, D] half
    uint8_t* __restrict__ kv_cache,           // flat byte array
    const int32_t* __restrict__ slot_mapping, // [N]
    const float* __restrict__ PiT_ptr,        // [D, D] float32
    const float* __restrict__ Pi_S_T_ptr,     // [D, D] float32
    const float* __restrict__ centroids,      // [4] float32
    const float* __restrict__ midpoints,      // [3] float32
    int N, int H, int block_size,
    int64_t stride_cache_block,
    int stride_cache_pos,
    int stride_cache_head,
    int key_packed_size,
    int value_packed_size
) {
    int pid = blockIdx.x;
    int token_idx = pid / H;
    int head_idx = pid % H;
    int tid = threadIdx.x;

    int slot = slot_mapping[token_idx];
    if (slot < 0) return;

    int blk = slot / block_size;
    int off = slot % block_size;
    int64_t slot_base = (int64_t)blk * stride_cache_block
                      + off * stride_cache_pos
                      + head_idx * stride_cache_head;

    // Shared memory: [NUM_WARPS * 2] for reduction + [D] for vector broadcast
    extern __shared__ float smem[];
    float* reduce_smem = smem;
    float* vec_smem = smem + NUM_WARPS * 2;  // D floats for GEMV broadcast

    // ================================================================
    // STEP 1: Load key and normalize
    // ================================================================
    int kv_base = (token_idx * H + head_idx) * D;
    float x = __half2float(key_ptr[kv_base + tid]);

    float norm_sq = block_reduce_sum(x * x, reduce_smem);
    float norm_val = sqrtf(norm_sq + 1e-16f);
    float x_hat = x / norm_val;

    // ================================================================
    // STEP 2: Rotate (tiled GEMV) or skip
    // ================================================================
    // STEP 2: Rotate (tiled GEMV)
    // Write x_hat to shared memory for broadcast access
    vec_smem[tid] = x_hat;
    __syncthreads();

    // y[tid] = sum_k x_hat[k] * PiT[k, tid]
    float accum_rot = 0.0f;
    #pragma unroll 8
    for (int k = 0; k < D; k += TILE_K) {
        #pragma unroll
        for (int ki = 0; ki < TILE_K; ki++) {
            accum_rot += vec_smem[k + ki] * PiT_ptr[(k + ki) * D + tid];
        }
    }
    float y = accum_rot;
    __syncthreads();

    // ================================================================
    // STEP 3: Bucketize
    // ================================================================
    float mp0 = midpoints[0], mp1 = midpoints[1], mp2 = midpoints[2];
    float c0 = centroids[0], c1 = centroids[1], c2 = centroids[2], c3 = centroids[3];

    int idx = (y < mp0) ? 0 : (y < mp1) ? 1 : (y < mp2) ? 2 : 3;
    float y_hat = (idx == 0) ? c0 : (idx == 1) ? c1 : (idx == 2) ? c2 : c3;

    // ================================================================
    // STEP 4: Residual + norm
    // ================================================================
    float r = y - y_hat;
    float gamma_sq = block_reduce_sum(r * r, reduce_smem);
    float gamma_val = sqrtf(gamma_sq + 1e-16f);

    // ================================================================
    // STEP 5: QJL projection (optional)
    // ================================================================
    float projected = 0.0f;
    if constexpr (!NO_QJL) {
        vec_smem[tid] = r;
        __syncthreads();

        float accum = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < D; k += TILE_K) {
            #pragma unroll
            for (int ki = 0; ki < TILE_K; ki++) {
                accum += vec_smem[k + ki] * Pi_S_T_ptr[(k + ki) * D + tid];
            }
        }
        projected = accum;
        __syncthreads();
    }

    // ================================================================
    // STEP 6: Pack MSE indices (2-bit, 4 per byte)
    // ================================================================
    {
        int sub = tid % 4;
        int byte_pos = tid / 4;
        int shifted = idx << (sub * 2);
        int packed = shuffle_pack_4(shifted);
        if (sub == 0 && byte_pos < MSE_BYTES) {
            kv_cache[slot_base + byte_pos] = (uint8_t)(packed & 0xFF);
        }
    }

    // ================================================================
    // STEP 7: Pack QJL signs (1-bit, 8 per byte)
    // ================================================================
    if constexpr (!NO_QJL) {
        int sign_bit = (projected >= 0.0f) ? 1 : 0;
        int bit_pos = tid % 8;
        int byte_pos = tid / 8;
        int shifted = sign_bit << bit_pos;
        int packed = shuffle_pack_8(shifted);
        if (bit_pos == 0 && byte_pos < QJL_BYTES) {
            kv_cache[slot_base + MSE_BYTES + byte_pos] = (uint8_t)(packed & 0xFF);
        }
    } else {
        int bit_pos = tid % 8;
        int byte_pos = tid / 8;
        if (bit_pos == 0 && byte_pos < QJL_BYTES) {
            kv_cache[slot_base + MSE_BYTES + byte_pos] = 0;
        }
    }

    // ================================================================
    // STEP 8: Store norms
    // ================================================================
    if (tid == 0) {
        int norm_off = MSE_BYTES + QJL_BYTES;
        __half vn_h = __float2half(norm_val);
        __half gm_h = __float2half(gamma_val);
        // Write as uint16 (2 bytes each, little-endian)
        *reinterpret_cast<uint16_t*>(&kv_cache[slot_base + norm_off]) =
            *reinterpret_cast<uint16_t*>(&vn_h);
        *reinterpret_cast<uint16_t*>(&kv_cache[slot_base + norm_off + 2]) =
            *reinterpret_cast<uint16_t*>(&gm_h);
    }

    // ================================================================
    // STEP 9: Value quantization + store
    // ================================================================
    float v = __half2float(value_ptr[kv_base + tid]);

    if constexpr (VQB == 8) {
        // FP8 E4M3: cast via half → truncate
        // Safe approach: convert to fp8 via saturation
        __half v_h = __float2half(v);
        // On SM89+ (Ada/Blackwell), use native FP8. Otherwise approximate.
        #if __CUDA_ARCH__ >= 890
        __nv_fp8_e4m3 v_fp8 = __nv_fp8_e4m3(v);
        uint8_t v_u8 = *reinterpret_cast<uint8_t*>(&v_fp8);
        #else
        // Approximate FP8 E4M3: just store high byte of fp16
        uint16_t v_u16 = *reinterpret_cast<uint16_t*>(&v_h);
        uint8_t v_u8 = (uint8_t)((v_u16 >> 8) & 0xFF);
        #endif
        kv_cache[slot_base + key_packed_size + tid] = v_u8;

    } else if constexpr (VQB == 4) {
        // 4-bit uniform quantization
        float v_min, v_max;
        block_reduce_minmax(v, reduce_smem, v_min, v_max);

        float v_scale = fmaxf((v_max - v_min) / 15.0f, 1e-8f);
        int q = __float2int_rn((v - v_min) / v_scale);
        q = max(0, min(15, q));

        // Pack pairs: even=low nibble, odd=high nibble
        int sub = tid % 2;
        int byte_pos = tid / 2;
        int shifted = sub ? (q << 4) : q;

        // Shuffle within pair (both lanes in same warp)
        int lane_id = threadIdx.x % 32;
        int partner = lane_id ^ 1;  // XOR with 1 flips bit 0 (even↔odd)
        int partner_val = __shfl_sync(FULL_MASK, shifted, partner);
        int packed = shifted | partner_val;

        int val_data_bytes = D / 2;
        if (sub == 0 && byte_pos < val_data_bytes) {
            kv_cache[slot_base + key_packed_size + byte_pos] = (uint8_t)(packed & 0xFF);
        }

        // Scale + zero
        if (tid == 0) {
            int sc_off = key_packed_size + val_data_bytes;
            __half sc_h = __float2half(v_scale);
            __half zr_h = __float2half(v_min);
            *reinterpret_cast<uint16_t*>(&kv_cache[slot_base + sc_off]) =
                *reinterpret_cast<uint16_t*>(&sc_h);
            *reinterpret_cast<uint16_t*>(&kv_cache[slot_base + sc_off + 2]) =
                *reinterpret_cast<uint16_t*>(&zr_h);
        }

    } else {  // VQB == 2
        float v_min, v_max;
        block_reduce_minmax(v, reduce_smem, v_min, v_max);

        float v_scale = fmaxf((v_max - v_min) / 3.0f, 1e-8f);
        int q = __float2int_rn((v - v_min) / v_scale);
        q = max(0, min(3, q));

        int sub = tid % 4;
        int byte_pos = tid / 4;
        int shifted = q << (sub * 2);
        int packed = shuffle_pack_4(shifted);

        int val_data_bytes = D / 4;
        if (sub == 0 && byte_pos < val_data_bytes) {
            kv_cache[slot_base + key_packed_size + byte_pos] = (uint8_t)(packed & 0xFF);
        }

        if (tid == 0) {
            int sc_off = key_packed_size + val_data_bytes;
            __half sc_h = __float2half(v_scale);
            __half zr_h = __float2half(v_min);
            *reinterpret_cast<uint16_t*>(&kv_cache[slot_base + sc_off]) =
                *reinterpret_cast<uint16_t*>(&sc_h);
            *reinterpret_cast<uint16_t*>(&kv_cache[slot_base + sc_off + 2]) =
                *reinterpret_cast<uint16_t*>(&zr_h);
        }
    }
}

// =====================================================================
// C++ launch wrapper
// =====================================================================
void tq_fused_store_launch(
    torch::Tensor key, torch::Tensor value,
    torch::Tensor kv_cache, torch::Tensor slot_mapping,
    torch::Tensor PiT, torch::Tensor Pi_S_T,
    torch::Tensor centroids, torch::Tensor midpoints,
    int N, int H, int block_size,
    int64_t stride_cache_block, int stride_cache_pos, int stride_cache_head,
    int key_packed_size, int value_packed_size,
    bool no_qjl, int value_quant_bits
) {
    int NH = N * H;
    if (NH <= 0) return;

    dim3 grid(NH);
    dim3 block(THREADS);

    // Shared memory: NUM_WARPS*2 for reduce + D for vector broadcast
    int smem_bytes = (NUM_WARPS * 2 + D) * sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();

    #define LAUNCH(NR, NQ, VQ) \
        tq_fused_store_kernel<NR, NQ, VQ><<<grid, block, smem_bytes, stream>>>( \
            reinterpret_cast<const __half*>(key.data_ptr<at::Half>()), \
            reinterpret_cast<const __half*>(value.data_ptr<at::Half>()), \
            kv_cache.data_ptr<uint8_t>(), \
            slot_mapping.data_ptr<int32_t>(), \
            PiT.data_ptr<float>(), \
            Pi_S_T.data_ptr<float>(), \
            centroids.data_ptr<float>(), \
            midpoints.data_ptr<float>(), \
            N, H, block_size, \
            stride_cache_block, stride_cache_pos, stride_cache_head, \
            key_packed_size, value_packed_size \
        );

    if (no_qjl) {
        if      (value_quant_bits == 8) { LAUNCH(true, 8); }
        else if (value_quant_bits == 4) { LAUNCH(true, 4); }
        else                            { LAUNCH(true, 2); }
    } else {
        if      (value_quant_bits == 8) { LAUNCH(false, 8); }
        else if (value_quant_bits == 4) { LAUNCH(false, 4); }
        else                            { LAUNCH(false, 2); }
    }
    #undef LAUNCH
}
