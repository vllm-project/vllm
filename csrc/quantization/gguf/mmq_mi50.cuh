#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define WARP_SIZE_GCN 64

// matmul tile size def
#define TILE_SIZE_COL 4  // unit: (unquantized) element
#define TILE_SIZE_ROW 4  //  unit: (unquantized) element

// how many quantize blocks (block_q8_0 and block_q8_1) in each segment
// for each tile, input data is loaded and calculated segment by segment
// since shared memory is limited.
// NOTE: actually we need to load double number of qblocks, half of them from
// qweight and half of them from x.
// NOTE : we set this to warp size of gcn arch (64) so it's convenience to use
// warp shuffle to reduce partial sum of qblocks in a segment
#define QBLOCKS_PER_SEGMENT WARP_SIZE_GCN

// each thread handle one block of vector dot in every segment
#define THREADS_PER_BLOCK TILE_SIZE_COL* TILE_SIZE_ROW* QBLOCKS_PER_SEGMENT

__device__ __forceinline__ float vec_dot_q8_0_q8_0(block_q8_0* b0,
                                                   block_q8_0* b1) {
  /*
  result = d0 * d1 * (qs_a_0*qs_1_0 + ... + qs_0_31*qs_1_31)
  in which d0 and d1 are scale factor (float16), qs_x_i is quantized weight
  (int8).

  we do int8 dot product using __builtin_amdgcn_sdot4,
  other keywords: llvm.amdgcn.sdot4, v_dot4_i32_i8

  references:
  https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Basic/BuiltinsAMDGPU.def?spm=a2ty_o01.29997173.0.0.2f5ec921WpP3rh&file=BuiltinsAMDGPU.def
  https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html
  https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/vega-7nm-shader-instruction-set-architecture.pdf
  */

  float d_q8_0 = __half2float(b0->d);
  float d_q8_1 = __half2float(b1->d);

  int32_t int_product_result = 0;
  int32_t inner_product_out = 0;

#pragma unroll
  for (int i = 0; i < QK8_0; i += 4) {
    int32_t* a = reinterpret_cast<int32_t*>(b0->qs + i);
    int32_t* b = reinterpret_cast<int32_t*>(b1->qs + i);
    int_product_result =
        __builtin_amdgcn_sdot4(*a, *b, int_product_result, false);
  }

  return d_q8_0 * d_q8_1 * (float)int_product_result;
}

__global__ void mul_mat_q8_0_mi50(block_q8_0* __restrict__ qweight,
                                    block_q8_0* __restrict__ qx,
                                    half* __restrict__ out, int ncols,
                                    int nrows_weight, int nrows_x,
                                    int num_qblocks_per_row,
                                    int num_segments_per_row,
                                    int num_invald_qblocks_per_row) {
  /*
  Arguments:
    qweight: quantized weight, q8_0.
    qx: quantiezd input, q8_0.
    out: output, half.
    ncols: num of columns of weight(unquantized) and x(unquantized), assuming
      they are equal.
    nrows_weight: num of rows of weight(unquantized).
    nrows_x: num of rows of x(unquantized).
    num_qblocks_per_row: num of block_q8_0 per row, equal to ceil(ncols/QK8_0).
    num_segments_per_row: num of segments per row,
      equal to num_qblocks_per_row/QBLOCKS_PER_SEGMENT.
      rows in qweight and x are loaded and computed segment by segment.
    num_invalid_qblocks_per_row: num of block_q8_0 that need to be padded at
      end of each row in qweight and x, to make last segment complete.
      Actually we don't pad the row but simply do no computation and pretend
      the dot product of invalid blocks as 0.


  Matmul kernel to perform quantized matrix multiplication:
      out = (qweight * x.T).T

  If we temporarily ignore quantization, element in out is:
      out_i_j = sum(qweight_j_k * x_i_k) for k in range(ncols),

  so for (i, j) in out, we need (i, :) in x and (j, :) in qweight, in which
  ":" means all elements in that dimension.


  We perform matmul tilewise, one tile (of out) per block, tile size is
  defined as TILE_SIZE_COL and TILE_SIZE_ROW, so each block handle
  TILE_SIZE_COL * TILE_SIZE_ROW output elements.

  For computation in one block:
  Since we can't assume that all input data that required for output of this
  tile can be loaded into shared memory, so we load just one segment of the
  input data, do the partial matmul, and add result to a temprary sum. Do this
  until we done for all input data requierd, and we got the final output.

  Step 1: allocate shared memory for (partial) input segments
  Step 2: allocate register for output
  Step 3:
      for input_segment_index in all_segments:
          1. load input data of this segment into shared memory
          2. do vecdot with this thread's qblock and add to partial_sum
      after the loop, we've done all vecdot,
      we can reduce partial_sum to full output sum now.
  Step 4: write output back to global memory

  Each block contains QBLOCKS_PER_SEGMENT * TILE_SIZE_COL * TILE_SIZE_ROW
  threads, each thread handles one qblock vector dot in every segment.

  Thread indexing: since qblock in a segment is contigious in global memory,
  for coalescing memory access, thread indexing rule is:
      threadIdx.x -> qblock idx in a segment
      threadIdx.y -> column coordinate of element in tile output
      threadIdx.z -> row coordinate of element in tile output

  So this kernel is expected to be launched with:
      gridDim = (ncols_out/TILE_SIZE_COL, nrows_out/TILE_SIZE_ROW)
      blockDim = (WARP_SIZE_GCN, TILE_SIZE_COL, TILE_SIZE_ROW)
  */

  int qblock_idx_in_seg = threadIdx.x;
  int tile_col_idx = threadIdx.y;
  int tile_row_idx = threadIdx.z;

  int tile_col_idx_bias = TILE_SIZE_COL * blockIdx.x;
  int tile_row_idx_bias = TILE_SIZE_ROW * blockIdx.y;

  int global_out_col_idx = tile_col_idx_bias + tile_col_idx;
  int global_out_row_idx = tile_row_idx_bias + tile_row_idx;

  int& nrows_out = nrows_x;
  int& ncols_out = nrows_weight;

  if (global_out_row_idx >= nrows_out || global_out_col_idx >= ncols_out) {
    return;
  }

  // Step1: allocate shared memory for input segments
  __shared__ block_q8_0
      smem_qweight_segment[TILE_SIZE_COL * QBLOCKS_PER_SEGMENT];
  __shared__ block_q8_0 smem_x_segment[TILE_SIZE_ROW * QBLOCKS_PER_SEGMENT];

  // Step2: allocate register for output
  float sum = 0.0f;                         // one of the final output's element
  float partial_sum_of_qblocks_in_segment;  // sum all this to get partial sum
                                            // of a segment, will be reduced
                                            // along qblock_idx_in_seg in
                                            // segment for loop

  // Step3: calculation
  // loop over segments
  for (int seg_idx = 0; seg_idx < num_segments_per_row; seg_idx++) {
    // load input data segment from global memory to shared memory
    // IDEA: if tile_row_idx == 0 and tile_col_idx == 0, latency is 2x, maybe
    // offload work to tile_idx(1,1) ?
    int num_vald_qblock =
        seg_idx != num_segments_per_row - 1
            ? QBLOCKS_PER_SEGMENT
            : QBLOCKS_PER_SEGMENT - num_invald_qblocks_per_row;
    if (qblock_idx_in_seg < num_vald_qblock) {
      if (tile_row_idx == 0) {
        // only threads with tile_row_idx == 0 load data
        // to avoid duplicated load
        smem_qweight_segment[tile_col_idx * QBLOCKS_PER_SEGMENT +
                             qblock_idx_in_seg] =
            qweight[global_out_col_idx *
                        num_qblocks_per_row          // row idx of qweight
                    + seg_idx * QBLOCKS_PER_SEGMENT  // segment idx in this row
                    + qblock_idx_in_seg  // qblock idx in this segment
        ];
      }
      if (tile_col_idx == 0) {
        // only threads with tile_col_idx == 0 load data
        // to avoid duplicated load
        smem_x_segment[tile_row_idx * QBLOCKS_PER_SEGMENT + qblock_idx_in_seg] =
            qx[global_out_row_idx * num_qblocks_per_row  // row idx of x
               + seg_idx * QBLOCKS_PER_SEGMENT  // segment idx in this row
               + qblock_idx_in_seg              // qblock idx in this segment
        ];
      }
    }
    __syncthreads();

    // do vecdot for the qblock of this thread
    if (qblock_idx_in_seg < num_vald_qblock) {
      partial_sum_of_qblocks_in_segment = vec_dot_q8_0_q8_0(
          &smem_qweight_segment[tile_col_idx * QBLOCKS_PER_SEGMENT +
                                qblock_idx_in_seg],
          &smem_x_segment[tile_row_idx * QBLOCKS_PER_SEGMENT +
                          qblock_idx_in_seg]);
    } else {
      partial_sum_of_qblocks_in_segment = 0.0f;
    }
    __syncthreads();

    // reduce partial_sum_of_qblocks_in_segment to get partial_sum_of_segments
    // assuming QBLOCKS_PER_SEGMENT == WARP_SIZE_GCN == 64
#pragma unroll
    for (int delta = 32; delta > 0; delta /= 2) {
      partial_sum_of_qblocks_in_segment += __shfl_xor_sync(
          (uint64_t)-1, partial_sum_of_qblocks_in_segment, delta, 64);
    }
    sum += partial_sum_of_qblocks_in_segment;
  }

  // write output back to global memory
  if (qblock_idx_in_seg == 0 && global_out_row_idx < nrows_out &&
      global_out_col_idx < ncols_out) {
    out[global_out_row_idx * nrows_weight + global_out_col_idx] =
        __float2half(sum);
  }
}