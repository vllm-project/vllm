#include "allspark_utils.cuh"
#include <torch/all.h>
#include "core/registration.h"
#include <cublas_v2.h>

at::Tensor as_g_workspace;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

torch::Tensor allspark_w8a16_gemm(
    torch::Tensor const& a, torch::Tensor const& b_qweight,
    torch::Tensor const& b_scales, std::optional<torch::Tensor> const& b_qzeros,
    int64_t n, int64_t group_size, int64_t sm_count, int64_t sm_version,
    int64_t CUBLAS_M_THRESHOLD, bool has_zp, bool n32k16_reorder) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "allspark_w8a16_gemm(..) requires CUDA_ARCH >= 8.0");
  return torch::empty({1, 1});
}

#else
namespace allspark {
/*
 * GemmTile manage data movement from Global Memory to Shared Memory
 * requiring N % 8 == 0， K % 16 == 0 by loading uint
 * BN is obtained by padding the original N to a multiple of 32
 * weight B is rearranged as N32K16 order,
 * i.e. a initial data block of size 32(n)x16(k) is reordered as n8k4n4k4，
 * in order to put data loaded by the same thread of 32x16 data block together
 * continuously (see
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type)
 */
template <typename FType, typename QType, int Mtile, int Ntile, int NStage,
          int BLOCK>
struct GmemTile_W8A16_PerC_MtilexNtilex32_multistage_SM8x_SplitK {
  // element num loaded by a LDG inst.
  static constexpr int LDG_ELEMENT_CNT_A = 8;
  static constexpr int LDG_ELEMENT_CNT_B = 16;
  static constexpr int WARP_SIZE = 32;
  static constexpr int M_SIZE_ONE_LOAD = (BLOCK * LDG_ELEMENT_CNT_A) / 32;
  static constexpr int N_SIZE_ONE_LOAD = (BLOCK * LDG_ELEMENT_CNT_B) / 32;

  __device__ GmemTile_W8A16_PerC_MtilexNtilex32_multistage_SM8x_SplitK(
      const SM8x_GEMM_W8A16_Splitk_Params<FType, QType>& k_params,
      const uint32_t& A_smem_addr, const uint32_t& BQ_smem_addr,
      const uint32_t& A_stage_stride, const uint32_t& BQ_stage_stride)
      : params(k_params),
        A_smem_base_addr(A_smem_addr),
        BQ_smem_base_addr(BQ_smem_addr),
        A_smem_stage_stride(A_stage_stride),
        BQ_smem_stage_stride(BQ_stage_stride) {
    this_block_A_base_ptr = params.A_ptr + blockIdx.x * Mtile * params.K +
                            blockIdx.z * params.SplitK;
    // here B is rearranged as N32K16 order, i.e. 4 continuous N-direction
    // 8(N)x16(K) size data blocks are packed together
    this_block_B_base_ptr = params.B_ptr + blockIdx.y * Ntile * params.K +
                            blockIdx.z * params.SplitK * 4;

    const auto lane_id = threadIdx.x % WARP_SIZE;

    // For matrix A, a block load/store Mtile(row) x 32(col) elements in
    // multiple iters, 8x4 warp load/store 8(row) x 32(col) elements per iter
    const auto Aldg_row_base_idx = threadIdx.x / 4;
    Aldg_col_idx = (threadIdx.x % 4) * LDG_ELEMENT_CNT_A;
    const int Aldg_base_offset = Aldg_row_base_idx * params.K + Aldg_col_idx;

    // For matrix B, a block load/store elements of (Ntile / 4) row x 128 col
    // elements of N32K16 packing in multiple iters, 4x8 warp load/store 4(row)
    // * 128(col) per iter
    Bldg_col_idx = (threadIdx.x % 8) * LDG_ELEMENT_CNT_B;
    const auto Bldg_row_base_idx = threadIdx.x / 8;
    const int Bldg_base_offset =
        Bldg_row_base_idx * params.K * 4 + Bldg_col_idx;

    this_block_A_base_ptr += Aldg_base_offset;
    this_block_B_base_ptr += Bldg_base_offset;

    const int sts_a_base_offset =
        (threadIdx.x / 4) * 32 +
        ((lane_id % 4) ^ ((lane_id / 4) % 4) ^ ((lane_id / 4) / 4)) *
            LDG_ELEMENT_CNT_A;
    const int sts_bq_base_offset =
        Bldg_row_base_idx * 32 * 4 +
        ((threadIdx.x % 8) ^ (((threadIdx.x / 8) % 2) * 4)) * LDG_ELEMENT_CNT_B;

    A_smem_base_addr += sts_a_base_offset * sizeof(FType);
    BQ_smem_base_addr += sts_bq_base_offset * sizeof(uint8_t);

    A_ldg_guard = 0;
    B_ldg_guard = 0;
  #pragma unroll
    for (int i = 0; i < (Mtile + M_SIZE_ONE_LOAD - 1) / M_SIZE_ONE_LOAD; ++i) {
      auto m_idx = blockIdx.x * Mtile + Aldg_row_base_idx + i * M_SIZE_ONE_LOAD;
      if (m_idx < params.M) {
        A_ldg_guard |= (1u << i);
      }
    }

    const int N_padded = (params.N + 31) / 32 * 32;
  #pragma unroll
    for (int i = 0; i < (Ntile + N_SIZE_ONE_LOAD - 1) / N_SIZE_ONE_LOAD; ++i) {
      auto n_idx = blockIdx.y * Ntile + (Bldg_row_base_idx / 8) * 32 +
                   i * N_SIZE_ONE_LOAD;
      if (n_idx < N_padded) {
        B_ldg_guard |= (1u << i);
      }
    }
  }

  __device__ void ldgsts_first_ktiles(const int& first_k_tile,
                                      const int& k_tiles) {
    // load first k_tile
    // load A
    const int A_src_size = Aldg_col_idx < first_k_tile ? 16 : 0;
  #pragma unroll
    for (int i = 0; i < (Mtile + M_SIZE_ONE_LOAD - 1) / M_SIZE_ONE_LOAD; ++i) {
      cp_async<16>(
          A_smem_base_addr + (i * M_SIZE_ONE_LOAD * 32) * sizeof(FType),
          this_block_A_base_ptr + i * M_SIZE_ONE_LOAD * params.K, A_src_size,
          (A_ldg_guard & (1u << i)) != 0);
    }

    // load B
    const int B_src_size = (Bldg_col_idx / 4) < first_k_tile ? 16 : 0;
  #pragma unroll
    for (int i = 0; i < (Ntile + N_SIZE_ONE_LOAD - 1) / N_SIZE_ONE_LOAD; ++i) {
      cp_async<16>(
          BQ_smem_base_addr + (i * N_SIZE_ONE_LOAD * 32) * sizeof(uint8_t),
          this_block_B_base_ptr + i * N_SIZE_ONE_LOAD * params.K, B_src_size,
          (B_ldg_guard & (1u << i)) != 0);
    }

    cp_async_commit_group();
    this_block_A_base_ptr += first_k_tile;
    this_block_B_base_ptr += (first_k_tile * 4);

    // load second to (N-stage - 1) k_tiles
    for (int stage_idx = 1; stage_idx < NStage - 1; ++stage_idx) {
      if (stage_idx < k_tiles) {
  #pragma unroll
        for (int i = 0; i < (Mtile + M_SIZE_ONE_LOAD - 1) / M_SIZE_ONE_LOAD;
             ++i) {
          cp_async<16>(A_smem_base_addr + stage_idx * A_smem_stage_stride +
                           (i * M_SIZE_ONE_LOAD * 32) * sizeof(FType),
                       this_block_A_base_ptr + i * M_SIZE_ONE_LOAD * params.K,
                       16, (A_ldg_guard & (1u << i)) != 0);
        }

  #pragma unroll
        for (int i = 0; i < (Ntile + N_SIZE_ONE_LOAD - 1) / N_SIZE_ONE_LOAD;
             ++i) {
          cp_async<16>(BQ_smem_base_addr + stage_idx * BQ_smem_stage_stride +
                           (i * N_SIZE_ONE_LOAD * 32) * sizeof(uint8_t),
                       this_block_B_base_ptr + i * N_SIZE_ONE_LOAD * params.K,
                       16, (B_ldg_guard & (1u << i)) != 0);
        }

        this_block_A_base_ptr += 32;
        this_block_B_base_ptr += (32 * 4);
      }
      cp_async_commit_group();
    }
  }

  __device__ void ldgsts(const int& sts_stage_idx) {
    const int a_stage_offset = sts_stage_idx * A_smem_stage_stride;
    const int bq_stage_offset = sts_stage_idx * BQ_smem_stage_stride;
  #pragma unroll
    for (int i = 0; i < (Mtile + M_SIZE_ONE_LOAD - 1) / M_SIZE_ONE_LOAD; ++i) {
      cp_async<16>(A_smem_base_addr + a_stage_offset +
                       (i * M_SIZE_ONE_LOAD * 32) * sizeof(FType),
                   this_block_A_base_ptr + i * M_SIZE_ONE_LOAD * params.K, 16,
                   (A_ldg_guard & (1u << i)) != 0);
    }

  #pragma unroll
    for (int i = 0; i < (Ntile + N_SIZE_ONE_LOAD - 1) / N_SIZE_ONE_LOAD; ++i) {
      cp_async<16>(BQ_smem_base_addr + bq_stage_offset +
                       (i * N_SIZE_ONE_LOAD * 32) * sizeof(uint8_t),
                   this_block_B_base_ptr + i * N_SIZE_ONE_LOAD * params.K, 16,
                   (B_ldg_guard & (1u << i)) != 0);
    }

    cp_async_commit_group();
    this_block_A_base_ptr += 32;
    this_block_B_base_ptr += (32 * 4);
  }

  const FType* this_block_A_base_ptr = nullptr;
  const QType* this_block_B_base_ptr = nullptr;

  int Aldg_col_idx;
  int Bldg_col_idx;

  uint32_t A_ldg_guard;
  uint32_t B_ldg_guard;

  uint32_t A_smem_base_addr, BQ_smem_base_addr;
  const uint32_t A_smem_stage_stride, BQ_smem_stage_stride;

  const SM8x_GEMM_W8A16_Splitk_Params<FType, QType>& params;
};

/*
 * requiring N % 8 == 0
 */
template <typename FType, typename QType, int Mtile, int Ntile, int BLOCK,
          bool EnableFuse, bool has_zp>
struct ComputeTile_W8A16_PerC_MtilexNtilex32_multistage_SM8x_SplitK {
  static constexpr int WARP_SIZE = 32;
  static constexpr int WARP_CNT = BLOCK / WARP_SIZE;
  static constexpr int WARP_NTILE = Ntile / WARP_CNT;
  static constexpr int WARP_NITER = WARP_NTILE / 8;  // hmma16816
  static_assert(WARP_NTILE == 32 or WARP_NTILE == 64,
                "now only support WARP_NTILE = 32 or 64!");

  __device__ ComputeTile_W8A16_PerC_MtilexNtilex32_multistage_SM8x_SplitK(
      const SM8x_GEMM_W8A16_Splitk_Params<FType, QType>& k_params,
      const uint32_t& A_smem_addr, const uint32_t& BQ_smem_addr,
      const uint32_t& A_stage_stride, const uint32_t& BQ_stage_stride)
      : params(k_params),
        A_smem_base_addr(A_smem_addr),
        BQ_smem_base_addr(BQ_smem_addr),
        A_smem_stage_stride(A_stage_stride),
        BQ_smem_stage_stride(BQ_stage_stride) {
    warp_id = threadIdx.x / WARP_SIZE;
    lane_id = threadIdx.x % WARP_SIZE;

    load_a_base_offset[0] =
        (lane_id % 16) * 32 +
        ((lane_id / 16) ^ (lane_id % 4) ^ ((lane_id / 4) % 2)) * 8;
    load_a_base_offset[1] =
        (lane_id % 16) * 32 +
        ((lane_id / 16 + 2) ^ (lane_id % 4) ^ ((lane_id / 4) % 2)) * 8;

    load_b_base_offset[0] =
        (lane_id / 4 + warp_id * (WARP_NTILE / 4)) * 32 * 4 +
        (lane_id % 4) * 16 + ((lane_id / 4) % 2) * 16 * 4;
    load_b_base_offset[1] =
        (lane_id / 4 + warp_id * (WARP_NTILE / 4)) * 32 * 4 +
        (lane_id % 4) * 16 + (((lane_id / 4) % 2) ^ 1) * 16 * 4;

    sts_c_base_offset = warp_id * Mtile * WARP_NTILE +
                        (lane_id / 4) * WARP_NTILE + (lane_id % 4) * 2;

    if (EnableFuse) {
      this_block_C_base_ptr =
          params.C_ptr + blockIdx.x * Mtile * params.N + blockIdx.y * Ntile;
    } else {
      this_block_C_base_ptr =
          params.C_split_ptr + blockIdx.z * params.M * params.N +
          blockIdx.x * Mtile * params.N + blockIdx.y * Ntile;
    }
    int store_thds_in_row = WARP_NTILE / 8;
    store_c_row_base_idx = lane_id / store_thds_in_row;
    store_c_col_idx = warp_id * WARP_NTILE + (lane_id % store_thds_in_row) * 8;
    store_c_base_offset = store_c_row_base_idx * params.N + store_c_col_idx;

  #pragma unroll
    for (int i = 0; i < Mtile / 16; ++i) {
  #pragma unroll
      for (int j = 0; j < WARP_NITER; ++j) {
  #pragma unroll
        for (int k = 0; k < 4; ++k) {
          C_frag[i][j][k] = 0.f;
        }
      }
    }
    params_n_idx =
        blockIdx.y * Ntile + warp_id * WARP_NTILE + (lane_id / 4) * 4;
  }

  __device__ void lds(const int& smem_stage_idx, const int& reg_buf_idx,
                      const int& k_phase_idx) {
    uint32_t A_smem_addr =
        A_smem_base_addr + A_smem_stage_stride * smem_stage_idx;
    uint32_t B_smem_addr =
        BQ_smem_base_addr + BQ_smem_stage_stride * smem_stage_idx;

  #pragma unroll
    for (int i = 0; i < Mtile / 16; ++i) {
      ldsm_4(A_frag[reg_buf_idx][i][0], A_frag[reg_buf_idx][i][1],
             A_frag[reg_buf_idx][i][2], A_frag[reg_buf_idx][i][3],
             A_smem_addr + (load_a_base_offset[k_phase_idx] + i * 16 * 32) *
                               sizeof(FType));
    }
  #pragma unroll
    for (int i = 0; i < WARP_NTILE / 32; ++i) {
      lds128(BQ_frag[reg_buf_idx][4 * i + 0], BQ_frag[reg_buf_idx][4 * i + 1],
             BQ_frag[reg_buf_idx][4 * i + 2], BQ_frag[reg_buf_idx][4 * i + 3],
             B_smem_addr + (load_b_base_offset[k_phase_idx] + i * 32 * 32) *
                               sizeof(uint8_t));
    }

  // dequant B
  #pragma unroll
    for (int i = 0; i < WARP_NITER / 2; ++i) {
      cvt_8bx4_to_16bx4_bias128(BQ_frag[reg_buf_idx][2 * i],
                                BF_frag[reg_buf_idx][2 * i]);
      if (has_zp) {
        BF_frag[reg_buf_idx][2 * i][0] =
            __hsub2(BF_frag[reg_buf_idx][2 * i][0], num2num2(B_zero[i].x));
        BF_frag[reg_buf_idx][2 * i][1] =
            __hsub2(BF_frag[reg_buf_idx][2 * i][1], num2num2(B_zero[i].x));
      }

      BF_frag[reg_buf_idx][2 * i][0] =
          __hmul2(BF_frag[reg_buf_idx][2 * i][0], num2num2(B_scale[i].x));
      BF_frag[reg_buf_idx][2 * i][1] =
          __hmul2(BF_frag[reg_buf_idx][2 * i][1], num2num2(B_scale[i].x));

      cvt_8bx4_to_16bx4_bias128(BQ_frag[reg_buf_idx][2 * i + 1],
                                BF_frag[reg_buf_idx][2 * i + 1]);
      if (has_zp) {
        BF_frag[reg_buf_idx][2 * i + 1][0] =
            __hsub2(BF_frag[reg_buf_idx][2 * i + 1][0], num2num2(B_zero[i].y));
        BF_frag[reg_buf_idx][2 * i + 1][1] =
            __hsub2(BF_frag[reg_buf_idx][2 * i + 1][1], num2num2(B_zero[i].y));
      }

      BF_frag[reg_buf_idx][2 * i + 1][0] =
          __hmul2(BF_frag[reg_buf_idx][2 * i + 1][0], num2num2(B_scale[i].y));
      BF_frag[reg_buf_idx][2 * i + 1][1] =
          __hmul2(BF_frag[reg_buf_idx][2 * i + 1][1], num2num2(B_scale[i].y));
    }
  }

  __device__ void ldg_params() {
    const int N_padded = (params.N + 31) / 32 * 32;
    // load B scale and zero_point
  #pragma unroll
    for (int i = 0; i < WARP_NTILE / 32; ++i) {
      ldg64_ca(B_scale[2 * i + 0], B_scale[2 * i + 1],
               params.B_scale_ptr + params_n_idx + i * 32,
               (params_n_idx + i * 32) < N_padded);
      if (has_zp) {
        ldg64_ca(B_zero[2 * i + 0], B_zero[2 * i + 1],
                 params.B_zero_ptr + params_n_idx + i * 32,
                 (params_n_idx + i * 32) < N_padded);
      }
    }
  }

  __device__ void mma(const int& reg_buf_idx) {
  #pragma unroll
    for (int m_idx = 0; m_idx < Mtile / 16; ++m_idx) {
  #pragma unroll
      for (int n_idx = 0; n_idx < WARP_NITER; ++n_idx) {
        hmma16816_f32<FType>(
            C_frag[m_idx][n_idx], A_frag[reg_buf_idx][m_idx],
            reinterpret_cast<uint32_t (&)[2]>(BF_frag[reg_buf_idx][n_idx]));
      }
    }
  }

  __device__ void fused_splitk_reduce() {
    // need splitk-reduce if enable splitk
    if (gridDim.z > 1) {
      auto blk_red_idx = blockIdx.x * gridDim.y + blockIdx.y;
      // Wait for all previous blocks in the splitk direction to accumulate the
      // results into C_tmp
      if (threadIdx.x == 0) {
        uint32_t* red_count_ptr = params.red_count_ptr + blk_red_idx;
        uint32_t count;
        do {
          // make sure the ld.cg inside the do-wile loop
          __threadfence_block();
          asm volatile("ld.global.cg.b32 %0, [%1];"
                       : "=r"(count)
                       : "l"(red_count_ptr));
        } while (count != blockIdx.z);
      }
      __syncthreads();

      auto C_tmp_base_offset = blk_red_idx * Mtile * Ntile + threadIdx.x * 4;
      if (blockIdx.z != 0) {
        // expecting that temporary register here reuses the previous A&B frag
        // register
        float temp_frag[Mtile / 16][WARP_NITER][4];
  #pragma unroll
        for (int m_idx = 0; m_idx < Mtile / 16; ++m_idx) {
  #pragma unroll
          for (int n_idx = 0; n_idx < WARP_NITER; ++n_idx) {
            int offset =
                C_tmp_base_offset + (m_idx * WARP_NITER + n_idx) * BLOCK * 4;
            *reinterpret_cast<int4*>(temp_frag[m_idx][n_idx]) =
                *reinterpret_cast<int4*>(params.C_tmp_ptr + offset);
          }
        }
  #pragma unroll
        for (int m_idx = 0; m_idx < Mtile / 16; ++m_idx) {
  #pragma unroll
          for (int n_idx = 0; n_idx < WARP_NITER; ++n_idx) {
  #pragma unroll
            for (int idx = 0; idx < 4; ++idx) {
              C_frag[m_idx][n_idx][idx] += temp_frag[m_idx][n_idx][idx];
            }
          }
        }
      }

      // first splitk - 1 blocks need to write partial results into C_tmp
      if (blockIdx.z != gridDim.z - 1) {
  #pragma unroll
        for (int m_idx = 0; m_idx < Mtile / 16; ++m_idx) {
  #pragma unroll
          for (int n_idx = 0; n_idx < WARP_NITER; ++n_idx) {
            int offset =
                C_tmp_base_offset + (m_idx * WARP_NITER + n_idx) * BLOCK * 4;
            asm volatile(
                "{st.global.cg.v4.b32 [%0], {%1, %2, %3, %4};}\n"
                :
                : "l"(params.C_tmp_ptr + offset), "f"(C_frag[m_idx][n_idx][0]),
                  "f"(C_frag[m_idx][n_idx][1]), "f"(C_frag[m_idx][n_idx][2]),
                  "f"(C_frag[m_idx][n_idx][3]));
          }
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
          uint32_t* red_count_ptr = params.red_count_ptr + blk_red_idx;
          atomicInc(red_count_ptr, gridDim.z);
        }
      }
    }
  }

  __device__ void stg(char* smem) {
    if (EnableFuse) {
      if (blockIdx.z != gridDim.z - 1) return;
    }
    uint32_t* C_sts_ptr =
        reinterpret_cast<uint32_t*>(smem + sts_c_base_offset * sizeof(FType));
    // C_tile sts
  #pragma unroll
    for (int m_idx = 0; m_idx < Mtile / 16; ++m_idx) {
  #pragma unroll
      for (int n_idx = 0; n_idx < WARP_NITER; ++n_idx) {
  #pragma unroll
        for (int k_idx = 0; k_idx < 2; ++k_idx) {
          FType low16 =
              ScalarType<FType>::float2num(C_frag[m_idx][n_idx][k_idx * 2]);
          FType high16 =
              ScalarType<FType>::float2num(C_frag[m_idx][n_idx][k_idx * 2 + 1]);
          uint32_t tmp = (reinterpret_cast<uint32_t&>(low16) & 0xffff) |
                         (reinterpret_cast<uint32_t&>(high16) << 16);
          int sts_offset =
              m_idx * 16 * (WARP_NTILE / 2) +
              (((lane_id / (32 / WARP_NITER)) + n_idx) % WARP_NITER) * (8 / 2) +
              k_idx * 8 * (WARP_NTILE / 2);
          C_sts_ptr[sts_offset] = tmp;
        }
      }
    }

    __syncthreads();

    FType* C_base_ptr = this_block_C_base_ptr + store_c_base_offset;
    // C_tile lds and stg
    auto m_base_idx = store_c_row_base_idx + blockIdx.x * Mtile;
    bool n_guard = (store_c_col_idx + blockIdx.y * Ntile) < params.N;
    if (WARP_NTILE == 32) {
      int lds_c_base_offset = warp_id * Mtile * WARP_NTILE +
                              (lane_id / 4) * WARP_NTILE +
                              ((lane_id % 4 + lane_id / 8) % 4) * 8;
      uint4* C_lds_ptr =
          reinterpret_cast<uint4*>(smem + lds_c_base_offset * sizeof(FType));
  #pragma unroll
      for (int i = 0; i < (Mtile / 16) * (WARP_NITER / 2); ++i) {
        uint4 stg_reg = C_lds_ptr[i * 8 * 4];
        stg128(stg_reg.x, stg_reg.y, stg_reg.z, stg_reg.w,
               C_base_ptr + i * 8 * params.N,
               (m_base_idx + i * 8) < params.M && n_guard);
      }
    } else if (WARP_NTILE == 64) {
      int lds_c_base_offset =
          warp_id * Mtile * WARP_NTILE + (lane_id / 8) * WARP_NTILE;
  #pragma unroll
      for (int i = 0; i < (Mtile / 16) * (WARP_NITER / 2); ++i) {
        int lds_c_offset = lds_c_base_offset + i * 4 * WARP_NTILE +
                           ((lane_id % 8 + lane_id / 8 + (i % 2) * 4) % 8) * 8;
        uint4 stg_reg =
            *reinterpret_cast<uint4*>(smem + lds_c_offset * sizeof(FType));
        stg128(stg_reg.x, stg_reg.y, stg_reg.z, stg_reg.w,
               C_base_ptr + i * 4 * params.N,
               (m_base_idx + i * 4) < params.M && n_guard);
      }
    }
  }

  const SM8x_GEMM_W8A16_Splitk_Params<FType, QType>& params;

  int load_a_base_offset[2];
  int load_b_base_offset[2];
  int sts_c_base_offset;

  int store_c_base_offset;

  int store_c_row_base_idx, store_c_col_idx;
  FType* this_block_C_base_ptr = nullptr;

  int params_n_idx;
  const uint32_t A_smem_base_addr, BQ_smem_base_addr;
  const uint32_t A_smem_stage_stride, BQ_smem_stage_stride;

  int lane_id;
  int warp_id;
  // first 2 denotes double buffer, second dim denotes M direction
  uint32_t A_frag[2][Mtile / 16][4];

  typename HalfType<FType>::T2 B_scale[WARP_NITER / 2];
  typename HalfType<FType>::T2 B_zero[WARP_NITER / 2];
  uint32_t BQ_frag[2][WARP_NITER];
  // first 2 denotes double buffer, second dim denotes N direction, last 2
  // denotes K direction
  typename HalfType<FType>::T2 BF_frag[2][WARP_NITER][2];
  // first dim denotes M direction, second dim denotes N direction
  float C_frag[Mtile / 16][WARP_NITER][4];
};

/*
 *  @brief W8A16 Perchannel Quantization GEMM,
 *         requires N % 8 == 0, K % 16 == 0
 *         accumulator precision: FP32
 *  @tparam FType: DataType for A, B_scale, B_zero, and C, supports half or
 * nv_bfloat16
 *  @tparam QType: DataType for B, support uint8(bias128)
 *  @tparam Mtile: M-dimensional size of the gemm block tile, supports 16, 32,
 * 48 or 64
 *  @tparam Ntile: N-dimensional size of the gemm block tile, supports 128 or
 * 256
 *  @tparam NStage: Num of stages for async copy
 *  @tparam BLOCK: BLOCK size
 *  @tparam EnableFuse: If true, use fused splitk-reduce, otherwise use
 * non-fused splitk-reduce
 *  @tparam has_zp: whether to use zero_point
 *
 *  @fparam params struct consists of following parameters:
 *      @param A_ptr: Matrix A value ptr, A = (M, K)
 *      @param B_ptr: Matrix B value ptr, B = (N32_align, K) (N32K16 special
 * format), N32_align = (N + 32 - 1) / 32 * 32
 *      @param B_scale_ptr: B_scale value ptr, B_scale = (N32_align,) (N32K16
 * special format)
 *      @param B_zero_ptr: B_zero value ptr, B_zero = (N32_align,) (N32K16
 * special format)
 *      @param C_ptr: Matrix C value ptr, C = (M, N)
 *      @param M: dimnesion m
 *      @param N: dimnesion n
 *      @param K: dimnesion k
 *      @param SplitK: split size along K-dimension
 *      @param C_split_ptr: Matrix C_split value ptr, used only in non-fused
 * splitk-reduce
 *      @param C_tmp_ptr: Matrix C_tmp value ptr, used only in fused
 * splitk-reduce
 *      @param red_count_ptr: 1-D red_count value ptr, used only in fused
 * splitk-reduce
 */
template <typename FType, typename QType, int Mtile, int Ntile, int NStage,
          int BLOCK, bool EnableFuse, bool has_zp>
__global__ void __launch_bounds__(BLOCK)
    ampere_hgemm_W8A16_perc_f16_f16_MtilexNtilex32_hmma16816_multistage_AN_BTN32K16_CN_splitk_kernel(
        const SM8x_GEMM_W8A16_Splitk_Params<FType, QType> params) {
  // A smem size = 64 * 32 * 2B/elem * 4(stage) = 16KB
  // B smem size = 128 * 32 * 1B/elem * 4(stage) = 16KB
  constexpr int smem_size_one_stage = Mtile * 32 * 2 + Ntile * 32;
  __shared__ char smem[NStage * smem_size_one_stage];
  char* A_smem = smem;
  char* BQ_smem = smem + Mtile * 32 * 2 * NStage;

  uint32_t A_smem_addr = smem_u32addr(A_smem);
  uint32_t BQ_smem_addr = smem_u32addr(BQ_smem);
  uint32_t A_smem_stage_stride = Mtile * 32 * 2;
  uint32_t BQ_smem_stage_stride = Ntile * 32;

  // initialize the data move process from GM to SMEM for this block
  GmemTile_W8A16_PerC_MtilexNtilex32_multistage_SM8x_SplitK<
      FType, QType, Mtile, Ntile, NStage, BLOCK>
      gmem_tile(params, A_smem_addr, BQ_smem_addr, A_smem_stage_stride,
                BQ_smem_stage_stride);

  int sts_stage_idx = 0;
  int lds_stage_idx = 0;

  auto tb_k_slice = blockIdx.z * params.SplitK + params.SplitK <= params.K
                        ? params.SplitK
                        : params.K - blockIdx.z * params.SplitK;
  int k_tiles = (tb_k_slice + 31) / 32;
  int first_k_tile = tb_k_slice - (k_tiles - 1) * 32;

  // load first three tiles to shared memory
  gmem_tile.ldgsts_first_ktiles(first_k_tile, k_tiles);
  sts_stage_idx += (NStage - 2);
  ComputeTile_W8A16_PerC_MtilexNtilex32_multistage_SM8x_SplitK<
      FType, QType, Mtile, Ntile, BLOCK, EnableFuse, has_zp>
      compute_tile(params, A_smem_addr, BQ_smem_addr, A_smem_stage_stride,
                   BQ_smem_stage_stride);
  compute_tile.ldg_params();
  cp_asyc_wait_group<NStage - 2>();
  __syncthreads();

  compute_tile.lds(lds_stage_idx, 0, 0);
  int reg_buf_idx = 1;

  // main loop
  for (; k_tiles > NStage - 1; --k_tiles) {
    // load next A&B tile
    sts_stage_idx = sts_stage_idx < NStage - 1 ? sts_stage_idx + 1 : 0;
    gmem_tile.ldgsts(sts_stage_idx);

  #pragma unroll
    for (int k_phase_idx = 0; k_phase_idx < 2; k_phase_idx++) {
      // dequantize next B tile
      if (k_phase_idx == 1) {
        cp_asyc_wait_group<NStage - 2>();
        __syncthreads();
        lds_stage_idx = lds_stage_idx < NStage - 1 ? lds_stage_idx + 1 : 0;
      }

      compute_tile.lds(lds_stage_idx, reg_buf_idx, (k_phase_idx + 1) % 2);

      compute_tile.mma(reg_buf_idx ^ 1);
      reg_buf_idx ^= 1;
    }
  }

  // last NStage-1 tiles
  for (; k_tiles > 0; --k_tiles) {
    cp_async_commit_group();
  #pragma unroll
    for (int k_phase_idx = 0; k_phase_idx < 2; k_phase_idx++) {
      // dequantize next B tile
      if (k_phase_idx == 1) {
        cp_asyc_wait_group<NStage - 2>();
        __syncthreads();
        lds_stage_idx = lds_stage_idx < NStage - 1 ? lds_stage_idx + 1 : 0;
      }

      compute_tile.lds(lds_stage_idx, reg_buf_idx, (k_phase_idx + 1) % 2);

      compute_tile.mma(reg_buf_idx ^ 1);
      reg_buf_idx ^= 1;
    }
  }

  if (EnableFuse) {
    compute_tile.fused_splitk_reduce();
  }
  compute_tile.stg(smem);
}

  #define __CALL_IF(MTILE, NTILE, NUM_THREADS, ENABLE_FUSE, HAS_ZP)                                     \
    else if (Mtile == MTILE && Ntile == NTILE && BLOCK == NUM_THREADS &&                                \
             enable_fuse == ENABLE_FUSE && has_zp == HAS_ZP) {                                          \
      ampere_hgemm_W8A16_perc_f16_f16_MtilexNtilex32_hmma16816_multistage_AN_BTN32K16_CN_splitk_kernel< \
          FType, QType, MTILE, NTILE, 4, NUM_THREADS, ENABLE_FUSE, HAS_ZP>                              \
          <<<grid, block, 0, stream>>>(params);                                                         \
    }

template <typename FType, typename QType>
void ampere_hgemm_W8A16_perc_f16_f16_MtilexNtilex32_mma16816_multistage_AN_BTN32K16_CN_splitk(
    const FType* A, const QType* B, const FType* B_scale, const FType* B_zero,
    FType* C, const int M, const int N, const int K, void* workspace,
    const int sm_version, const BlockTileSplitkParams& fused_gemm_params,
    cudaStream_t stream) {
  int Mtile = fused_gemm_params.Mtile;
  int grid_x = (M + Mtile - 1) / Mtile;
  int Ntile = fused_gemm_params.Ntile;
  int grid_y = (N + Ntile - 1) / Ntile;
  int SplitK = fused_gemm_params.SplitK;
  int grid_z = (K + SplitK - 1) / SplitK;

  int BLOCK = (Ntile == 256) ? 256 : 128;

  dim3 grid(grid_x, grid_y, grid_z);
  dim3 block(BLOCK);

  bool enable_fuse = fused_gemm_params.EnableFuse;
  bool has_zp = B_zero != nullptr;
  if (enable_fuse) {
    float* C_tmp = reinterpret_cast<float*>(workspace);
    uint32_t* red_count = reinterpret_cast<uint32_t*>(
        (char*)workspace + grid_x * Mtile * grid_y * Ntile * sizeof(float));
    CHECK_CUDA(cudaMemsetAsync(red_count, 0, grid_x * grid_y * sizeof(uint32_t),
                               stream));
    SM8x_GEMM_W8A16_Splitk_Params<FType, QType> params{
        A, B,      B_scale, B_zero, C,       M,     N,
        K, SplitK, 0,       -1,     nullptr, C_tmp, red_count};

    if (false) {
    }
    // Select the template parameters for kernel launch
    // according to the above settings. Tuning is not supported.
    __CALL_IF(16, 256, 256, true, false)
    __CALL_IF(32, 256, 256, true, false)
    __CALL_IF(48, 256, 256, true, false)
    __CALL_IF(64, 128, 128, true, false)
    __CALL_IF(64, 256, 256, true, false)
    __CALL_IF(16, 256, 256, true, true)
    __CALL_IF(32, 256, 256, true, true)
    __CALL_IF(48, 256, 256, true, true)
    __CALL_IF(64, 128, 128, true, true)
    __CALL_IF(64, 256, 256, true, true)
  } else {
    FType* C_split = reinterpret_cast<FType*>(workspace);
    SM8x_GEMM_W8A16_Splitk_Params<FType, QType> params{
        A, B,      B_scale, B_zero, C,       M,       N,
        K, SplitK, 0,       -1,     C_split, nullptr, nullptr};

    if (false) {
    }
    // Select the template parameters for kernel launch
    // according to the above settings. Tuning is not supported.
    __CALL_IF(16, 256, 256, false, false)
    __CALL_IF(32, 256, 256, false, false)
    __CALL_IF(48, 256, 256, false, false)
    __CALL_IF(64, 128, 128, false, false)
    __CALL_IF(64, 256, 256, false, false)
    __CALL_IF(16, 256, 256, false, true)
    __CALL_IF(32, 256, 256, false, true)
    __CALL_IF(48, 256, 256, false, true)
    __CALL_IF(64, 128, 128, false, true)
    __CALL_IF(64, 256, 256, false, true)

    // SplitK reduce
    f16_gemm_splitk_reduce(C_split, C, M, N, grid_z, stream);
  }
}

size_t allspark_qgemm_w8a16_perc_n32k16_ampere_workspace_size(
    int m, int n, int k, int sm_count,
    BlockTileSplitkParams& fused_gemm_params) {
  // Determine the block tile and splitk strategy
  int m16_times = (m + 16 - 1) / 16;
  int Mtile = m16_times <= 4 ? m16_times * 16 : 64;
  int grid_x = (m + Mtile - 1) / Mtile;
  int Ntile =
      (float(grid_x * ((n + 127) / 128)) / sm_count > 10) || (Mtile < 64) ? 256
                                                                          : 128;
  int grid_y = (n + Ntile - 1) / Ntile;
  int grid_z;

  // split-k
  const float SPLIT_THRESHOLD = 0.8;
  int n_slice;
  for (n_slice = 1; n_slice < k / 256; ++n_slice) {
    int n_block = grid_x * grid_y * n_slice;
    if (n_block >= sm_count * SPLIT_THRESHOLD &&
        (n_block % sm_count == 0 || n_block % sm_count >= sm_count * 0.5)) {
      break;
    }
  }

  int k_slice =
      (k / n_slice) % 32 == 0 ? k / n_slice : k / n_slice / 32 * 32 + 32;
  grid_z = (k + k_slice - 1) / k_slice;
  bool enable_fuse = float(grid_x * grid_y) / sm_count >= 0.5 ? 1 : 0;

  size_t ws_size;
  if (enable_fuse) {
    ws_size = grid_x * Mtile * grid_y * Ntile * sizeof(float)  // For C_tmp
              + grid_x * grid_y * sizeof(uint32_t);            // For red_count
  } else {
    ws_size = grid_z * m * n * sizeof(__half);
  }

  fused_gemm_params.Mtile = Mtile;
  fused_gemm_params.Ntile = Ntile;
  fused_gemm_params.SplitK = k_slice;
  fused_gemm_params.EnableFuse = enable_fuse;
  return ws_size;
}

// restore from N32K16 order to original N-major order
// K % 16 == 0, N % 8 == 0
// each block process 64(k) * 32(n) result elements
template <typename FT, typename QT>
__global__ void restore_N32_K16_dequantize_rhs_w8a16_perc_kernel(
    const QT* qdata, const FT* scales, const FT* zeros, FT* fdata,
    const int N_32align, const int N, const int K) {
  __shared__ FT smem[64 * 32];
  auto warp_id = threadIdx.x / 32;
  auto lane_id = threadIdx.x % 32;
  const auto src_row_idx = blockIdx.x * 8 + lane_id / 4;
  const int src_col_idx =
      blockIdx.y * 64 * 4 + warp_id * 16 * 4 + (lane_id % 4) * 16;
  const int src_offset = src_row_idx * K * 4 + src_col_idx;
  auto params_nidx = blockIdx.x * 32 + (lane_id / 4) * 4;

  QT qval_reg[16];
  const QT* pdata = qdata + src_offset;
  if (src_col_idx < (K * 4)) {
    *(reinterpret_cast<uint4*>(qval_reg)) =
        *(reinterpret_cast<const uint4*>(qdata + src_offset));
  }
  FT scale_reg[4];
  *(reinterpret_cast<uint2*>(scale_reg)) =
      *(reinterpret_cast<const uint2*>(scales + params_nidx));
  FT zero_reg[4];
  if (zeros != nullptr) {
    *(reinterpret_cast<uint2*>(zero_reg)) =
        *(reinterpret_cast<const uint2*>(zeros + params_nidx));
  }
  FT fval_reg[16];

  const int sts_base_offset =
      (warp_id * 16 + (lane_id % 4) * 2) * 32 + lane_id / 4;
  #pragma unroll
  for (int ni = 0; ni < 4; ++ni) {
    cvt_8bx4_to_16bx4_bias128(
        *reinterpret_cast<uint32_t*>(&qval_reg[ni * 4]),
        reinterpret_cast<typename HalfType<FT>::T2*>(&(fval_reg[ni * 4])));
  #pragma unroll
    for (int ki = 0; ki < 4; ++ki) {
      if (zeros != nullptr) {
        fval_reg[ni * 4 + ki] = __hsub(fval_reg[ni * 4 + ki], zero_reg[ni]);
      }
      fval_reg[ni * 4 + ki] = __hmul(fval_reg[ni * 4 + ki], scale_reg[ni]);
      int sts_offset = sts_base_offset + ((ki / 2) * 8 + (ki % 2)) * 32 +
                       ((ni + lane_id % 4) % 4) * 8;
      smem[sts_offset] = fval_reg[ni * 4 + ki];
    }
  }
  __syncthreads();

  const int lds_base_offset =
      (threadIdx.x / 4) * 32 + ((threadIdx.x % 4 + threadIdx.x / 8) % 4) * 8;
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *reinterpret_cast<uint4*>(fval_reg + i * 8) =
        *reinterpret_cast<uint4*>(smem + lds_base_offset + i * 32 * 32);
  }

  const auto dst_row_base_kidx = blockIdx.y * 64 + threadIdx.x / 4;
  const auto dst_col_nidx = blockIdx.x * 32 + (threadIdx.x % 4) * 8;
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    int dst_row_kidx = dst_row_base_kidx + i * 32;
    int dst_offset = dst_row_kidx * N + dst_col_nidx;
    if (dst_row_kidx < K && dst_col_nidx < N) {
      *reinterpret_cast<uint4*>(fdata + dst_offset) =
          *reinterpret_cast<uint4*>(fval_reg + i * 8);
    }
  }
}

template <typename FT, typename QT>
void restore_N32_K16_dequantize_rhs_w8a16(const QT* qdata, const FT* scales,
                                          const FT* zeros, FT* fdata,
                                          const int N_32align, const int N,
                                          const int K, const int GroupSize,
                                          cudaStream_t stream) {
  TORCH_CHECK(N % 8 == 0 && K % 16 == 0 && N_32align % 32 == 0,
              "Unsupported shape");
  if (GroupSize == -1) {
    const int BLOCK = 128;
    dim3 grid(N_32align / 32, ((K / 16) + 3) / 4);
    restore_N32_K16_dequantize_rhs_w8a16_perc_kernel<FT, QT>
        <<<grid, BLOCK, 0, stream>>>(qdata, scales, zeros, fdata, N_32align, N,
                                     K);
  }
  // TODO: Support SubChannel
  else {
    TORCH_CHECK(false, "Now only support PerChannel");
  }
}

template <typename FT, typename QT>
void w8a16_gemm_dq_cublas(const FT* in, const QT* rhs_qdata_ptr,
                          const FT* rhs_scales_ptr, const FT* rhs_zeros_ptr,
                          FT* out, void* workspace, const int M,
                          const int N_32align, const int N, const int K,
                          const int group_size, cudaStream_t stream,
                          cublasHandle_t handle) {
  static_assert(
      std::is_same<FT, half>::value || std::is_same<FT, nv_bfloat16>::value,
      "only float16 and bfloat16 is supported");
  // Dequant
  FT* rhs_fdata_ptr = static_cast<FT*>(workspace);
  restore_N32_K16_dequantize_rhs_w8a16(rhs_qdata_ptr, rhs_scales_ptr,
                                       rhs_zeros_ptr, rhs_fdata_ptr, N_32align,
                                       N, K, group_size, stream);
  // cuBLAS GEMM
  int lda = K;
  int ldb = N;
  int ldc = N;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudaDataType_t cuda_type;
  if (std::is_same<FT, __half>::value) {
    cuda_type = CUDA_R_16F;
  } else {
    cuda_type = CUDA_R_16BF;
  }
  CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                            rhs_fdata_ptr, cuda_type, ldb, in, cuda_type, lda,
                            &beta, out, cuda_type, ldc, CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <typename FType, typename QType>
void allspark_qgemm_w8a16_perc_ampere(
    const FType* A, const QType* B, const FType* B_scale, const FType* B_zero,
    FType* C, const int M, const int N_32align, const int N, const int K,
    void* workspace, const BlockTileSplitkParams& fused_gemm_params,
    const int group_size, int CUBLAS_M_THRESHOLD, const int sm_version,
    cudaStream_t stream, cublasHandle_t handle) {
  if (M > CUBLAS_M_THRESHOLD) {
    w8a16_gemm_dq_cublas<FType, QType>(A, B, B_scale, B_zero, C, workspace, M,
                                       N_32align, N, K, group_size, stream,
                                       handle);
  } else {
    ampere_hgemm_W8A16_perc_f16_f16_MtilexNtilex32_mma16816_multistage_AN_BTN32K16_CN_splitk<
        FType, QType>(A, B, B_scale, B_zero, C, M, N, K, workspace, sm_version,
                      fused_gemm_params, stream);
  }
}

}  // namespace allspark

torch::Tensor allspark_w8a16_gemm(
    torch::Tensor const& a, torch::Tensor const& b_qweight,
    torch::Tensor const& b_scales, std::optional<torch::Tensor> const& b_qzeros,
    int64_t n, int64_t group_size, int64_t sm_count, int64_t sm_version,
    int64_t CUBLAS_M_THRESHOLD, bool has_zp, bool n32k16_reorder) {
  // Verify device and strides
  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.is_contiguous(), "A is not contiguous");

  TORCH_CHECK(b_qweight.device().is_cuda(), "b_qweight is not on GPU");
  TORCH_CHECK(b_qweight.is_contiguous(), "b_qweight is not contiguous");

  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

  if (has_zp) {
    TORCH_CHECK(b_qzeros.value().device().is_cuda(), "b_qzeros is not on GPU");
    TORCH_CHECK(b_qzeros.value().is_contiguous(), "b_qzeros is not contiguous");
  }

  int m = a.size(0);
  int n_32align = (n + 32 - 1) / 32 * 32;
  int k = a.size(1);

  // Verify shape
  TORCH_CHECK(b_qweight.size(0) == n_32align,
              "Shape mismatch: b_qweight.size(0) = ", b_qweight.size(0),
              ", n_32align = ", n_32align);
  TORCH_CHECK(b_qweight.size(1) == k,
              "Shape mismatch: b_qweight.size(1) = ", b_qweight.size(1),
              ", k = ", k);

  TORCH_CHECK(group_size == -1, "Currently only supports group_size = -1");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  const void* a_ptr = reinterpret_cast<const void*>(a.data_ptr());
  const uint8_t* b_ptr = reinterpret_cast<const uint8_t*>(b_qweight.data_ptr());
  const void* b_scale_ptr = reinterpret_cast<const void*>(b_scales.data_ptr());
  const void* b_zero_ptr = nullptr;
  if (b_qzeros.has_value()) {
    b_zero_ptr = reinterpret_cast<const void*>(b_qzeros.value().data_ptr());
  }

  auto c_options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  torch::Tensor c = torch::empty({m, n}, c_options);
  void* c_ptr = reinterpret_cast<void*>(c.data_ptr());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

  allspark::BlockTileSplitkParams fused_gemm_params;

  size_t ws_size = 0;
  if (m > CUBLAS_M_THRESHOLD) {
    ws_size = k * n * 2;  // sizeof(f16)==2
  } else {
    ws_size = allspark::allspark_qgemm_w8a16_perc_n32k16_ampere_workspace_size(
        m, n, k, sm_count, fused_gemm_params);
  }

  auto ws_options = torch::TensorOptions().dtype(at::kChar).device(a.device());
  if (as_g_workspace.numel() <
      ws_size) {  // ws_options: kChar, so numel() is bytes
    as_g_workspace = torch::empty({long(ws_size)}, ws_options);
  }
  void* ws = reinterpret_cast<void*>(as_g_workspace.data_ptr());

  if (a.dtype() == at::ScalarType::Half) {
    allspark::allspark_qgemm_w8a16_perc_ampere<__half, uint8_t>(
        reinterpret_cast<const __half*>(a_ptr), b_ptr,
        reinterpret_cast<const __half*>(b_scale_ptr),
        reinterpret_cast<const __half*>(b_zero_ptr),
        reinterpret_cast<__half*>(c_ptr), m, n_32align, n, k, ws,
        fused_gemm_params, group_size, CUBLAS_M_THRESHOLD, sm_version, stream,
        handle);
  } else if (a.dtype() == at::ScalarType::BFloat16) {
    allspark::allspark_qgemm_w8a16_perc_ampere<__nv_bfloat16, uint8_t>(
        reinterpret_cast<const __nv_bfloat16*>(a_ptr), b_ptr,
        reinterpret_cast<const __nv_bfloat16*>(b_scale_ptr),
        reinterpret_cast<const __nv_bfloat16*>(b_zero_ptr),
        reinterpret_cast<__nv_bfloat16*>(c_ptr), m, n_32align, n, k, ws,
        fused_gemm_params, group_size, CUBLAS_M_THRESHOLD, sm_version, stream,
        handle);
  }

  return c;
}

#endif

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("allspark_w8a16_gemm", &allspark_w8a16_gemm);
}
