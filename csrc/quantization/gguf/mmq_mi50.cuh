#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define WARP_SIZE_GCN 64

// matmul tile size def
#define TILE_SIZE_COL 32 // unit: (unquantized) element
#define TILE_SIZE_ROW 32 //  unit: (unquantized) element

// how many quantize blocks (block_q8_0 and block_q8_1) in each segment
// for each tile, input data is loaded and calculated segment by segment
// since shared memory is limited.
// NOTE: actually we need to load double number of qblocks, half of them from
// qweight and half of them from x.
// NOTE : we set this to warp size of gcn arch (64) so it's convenience to use
// warp shuffle to reduce partial sum of qblocks in a segment
#define QBLOCKS_PER_SEGMENT 8

__device__ __forceinline__ void allocate_shared_memory(int32_t *&qweight_qs, half *&qweight_d,
                                                       int32_t *&x_qs, half *&x_d)
{
  __shared__ int32_t _qweight_qs[TILE_SIZE_COL * QBLOCKS_PER_SEGMENT * QI8_0];
  __shared__ int32_t _x_qs[TILE_SIZE_ROW * QBLOCKS_PER_SEGMENT * QI8_0];
  __shared__ half _qweight_d[TILE_SIZE_COL * QBLOCKS_PER_SEGMENT];
  __shared__ half _x_d[TILE_SIZE_ROW * QBLOCKS_PER_SEGMENT];

  qweight_qs = _qweight_qs;
  qweight_d = _qweight_d;
  x_qs = _x_qs;
  x_d = _x_d;
}

__device__ __forceinline__ void print_smem_qs(int32_t *qs, int row, int col)
{

  for (int r = 0; r < row; r++)
  {
    for (int c = 0; c < col; c++)
    {
      printf("%d, ", qs[r * col + c]);
    }
    printf("\n");
  }
}

__device__ __forceinline__ void print_smem_d(half *d, int row, int col)
{

  for (int r = 0; r < row; r++)
  {
    for (int c = 0; c < col; c++)
    {
      printf("%f, ", __half2float(d[r * col + c]));
    }
    printf("\n");
  }
}

__device__ __forceinline__ void print_qblock(block_q8_0 *b)
{
  printf("qs: ");
  for (int i = 0; i < 32; i++)
    printf("%d, ", b->qs[i]);
  printf("\nd: %f\n", __half2float(b->d));
}

__device__ __forceinline__ void load_qblock_to_shared_memory(block_q8_0 *qblock_start, int32_t *qs_start, half *d_start,
                                                             int &global_col_or_row_idx, int &tile_col_or_row_idx, int &num_qblocks_per_row, int &segment_idx, int &qblock_idx_in_segment)
{
  block_q8_0 *qblock = qblock_start +
                       (global_col_or_row_idx * num_qblocks_per_row +
                        segment_idx * QBLOCKS_PER_SEGMENT +
                        qblock_idx_in_segment);

  int32_t *qs = qs_start + (tile_col_or_row_idx * QBLOCKS_PER_SEGMENT + qblock_idx_in_segment) * QI8_0;

  half *d = d_start + (tile_col_or_row_idx * QBLOCKS_PER_SEGMENT + qblock_idx_in_segment);

  // #pragma unroll
  //   for (int i = 0; i < QI8_0; i++)
  //   {
  //     *(qs + i) = *reinterpret_cast<int32_t *>(qblock->qs + (i * 4));
  //     memcpy(qs + i, qblock->qs + (i * 4), sizeof(int32_t));
  //   }
  memcpy(qs, qblock->qs, QI8_0 * sizeof(int32_t));

  *d = qblock->d;

  // printf("t(%d,%d,%d), d:%f, qs:[%d,%d,%d,%d,%d,%d,%d,%d]\n", threadIdx.x, threadIdx.y, threadIdx.z, __half2float(*d), *qs, *(qs + 1), *(qs + 2), *(qs + 3), *(qs + 4), *(qs + 5), *(qs + 6), *(qs + 7));
}

__device__ __forceinline__ void get_qblock_from_shared_memory(int32_t *qs_start, half *d_start, block_q8_0 *out,
                                                              int &tile_col_or_row_idx, int &qblock_idx_in_segment)
{
  int32_t *qs = qs_start + (tile_col_or_row_idx * QBLOCKS_PER_SEGMENT + qblock_idx_in_segment) * QI8_0;
  half *d = d_start + (tile_col_or_row_idx * QBLOCKS_PER_SEGMENT + qblock_idx_in_segment);

  // for (int i = 0; i < QI8_0; i++)
  // {
  //   *reinterpret_cast<int32_t *>((out->qs) + (i * 4)) = *(qs + i);
  // }

  memcpy(out->qs, qs, QI8_0 * sizeof(int32_t));

  out->d = *d;

  // printf("t(%d,%d,%d), d:%f, qs:[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]\n", threadIdx.x, threadIdx.y, threadIdx.z, __half2float(out->d),
  //        out->qs[0], out->qs[1], out->qs[2], out->qs[3], out->qs[4], out->qs[5], out->qs[6], out->qs[7],
  //        out->qs[8], out->qs[9], out->qs[10], out->qs[11], out->qs[12], out->qs[13], out->qs[14], out->qs[15],
  //        out->qs[16], out->qs[17], out->qs[18], out->qs[19], out->qs[20], out->qs[21], out->qs[22], out->qs[23],
  //        out->qs[24], out->qs[25], out->qs[26], out->qs[27], out->qs[28], out->qs[29], out->qs[30], out->qs[31]);
}

__device__ __forceinline__ float vec_dot_q8_0_q8_0(block_q8_0 *b0,
                                                   block_q8_0 *b1)
{
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
  for (int i = 0; i < QK8_0; i += 4)
  {
    int32_t *a = reinterpret_cast<int32_t *>(b0->qs + i);
    int32_t *b = reinterpret_cast<int32_t *>(b1->qs + i);
    int_product_result =
        __builtin_amdgcn_sdot4(*a, *b, int_product_result, false);
  }

  return d_q8_0 * d_q8_1 * (float)int_product_result;
}

__global__ void mul_mat_q8_0_mi50(block_q8_0 *__restrict__ qweight,
                                  block_q8_0 *__restrict__ qx,
                                  half *__restrict__ out, int ncols,
                                  int nrows_weight, int nrows_x,
                                  int num_qblocks_per_row,
                                  int num_segments_per_row,
                                  int num_invalid_qblocks_per_row)
{
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


  We perform matmul tilewise, one tile (of out) per block, tile size is defined as TILE_SIZE_COL and TILE_SIZE_ROW,
  so each block handle TILE_SIZE_COL * TILE_SIZE_ROW output elements.

  Each block contains 32 * 32 = 1024 threads,
  since we define TILE_SIZE_ROW = 32 and TILE_SIZE_COL = 64,
  each thread must handle 2 outupt elements.

  So this kernel is expected to be launched with:
    gridDim = (ncols_out/TILE_SIZE_COL, nrows_out/TILE_SIZE_ROW)
    blockDim = (32, 32)

  Thread indexing:
      threadIdx.x -> (column coordinate / 2) of element in tile output
      threadIdx.y -> row coordinate of element in tile output


  For computation in one block:
  Since we can't assume that all input data that required for output of this
  tile can be loaded into shared memory, so we load just one segment of the
  input data, do the partial matmul, and add result to a temprary sum. Do this
  until we done for all input data requierd, and we got the final output.

  Step 1: allocate shared memory and registers
  Step 2:
      for input_segment_index in all_segments:
          1. load input data of this segment into shared memory
          2. do vecdot with loaded data and add to sum
      after the loop, we've done all vecdot
  Step 4: write output back to global memory


  */

  int tile_col_idx = threadIdx.x;
  int tile_row_idx = threadIdx.y;

  int tile_col_idx_bias = TILE_SIZE_COL * blockIdx.x;
  int tile_row_idx_bias = TILE_SIZE_ROW * blockIdx.y;

  int global_out_col_idx = tile_col_idx_bias + tile_col_idx;
  int global_out_row_idx = tile_row_idx_bias + tile_row_idx;

  int &nrows_out = nrows_x;
  int &ncols_out = nrows_weight;

  if (global_out_row_idx >= nrows_out || global_out_col_idx >= ncols_out)
  {
    return;
  }

  // Step1: allocate shared memory and registers
  // __shared__ block_q8_0 smem_qweight_segment[TILE_SIZE_COL * QBLOCKS_PER_SEGMENT];
  // __shared__ block_q8_0 smem_x_segment[TILE_SIZE_ROW * QBLOCKS_PER_SEGMENT];

  int32_t *smem_qweight_qs;
  half *smem_qweight_d;
  int32_t *smem_x_qs;
  half *smem_x_d;

  allocate_shared_memory(smem_qweight_qs, smem_qweight_d, smem_x_qs, smem_x_d);

  // __syncthreads();
  // if (threadIdx.x == 0 && threadIdx.y == 0)
  // {
  //   for (int i = 0; i < TILE_SIZE_COL * QBLOCKS_PER_SEGMENT * QI8_0; i++)
  //   {
  //     smem_qweight_qs[i] = 0;
  //     smem_x_qs[i] = 0;
  //   }
  //   for (int i = 0; i < TILE_SIZE_COL * QBLOCKS_PER_SEGMENT; i++)
  //   {
  //     smem_qweight_d[i] = __float2half(0.0f);
  //     smem_x_d[i] = __float2half(0.0f);
  //   }
  //   // printf("smem_qweight_qs ====================\n");
  //   // print_smem_qs(smem_qweight_qs, TILE_SIZE_COL, QBLOCKS_PER_SEGMENT * QI8_0);
  //   // printf("smem_qweight_d =====================\n");
  //   // print_smem_d(smem_qweight_d, TILE_SIZE_ROW, QBLOCKS_PER_SEGMENT * QI8_0);

  //   // printf("smem_x_qs ====================\n");
  //   // print_smem_qs(smem_x_qs, TILE_SIZE_COL, QBLOCKS_PER_SEGMENT * QI8_0);
  //   // printf("smem_x_d =====================\n");
  //   // print_smem_d(smem_x_d, TILE_SIZE_ROW, QBLOCKS_PER_SEGMENT * QI8_0);
  // }
  // __syncthreads();

  float sum = 0.0f; // final output's element @ (tile_row_idx, tile_col_idx)

  // Step3: calculation
  // loop over segments
  for (int segment_idx = 0; segment_idx < num_segments_per_row; segment_idx++)
  {
    __syncthreads();

    // load input data segment from global memory to shared memory
    // IDEA: if tile_row_idx == 0 and tile_col_idx == 0, latency is 2x, maybe offload work to tile_idx(1,1) ?
    int num_vald_qblock = segment_idx != num_segments_per_row - 1
                              ? QBLOCKS_PER_SEGMENT
                              : QBLOCKS_PER_SEGMENT - num_invalid_qblocks_per_row;
    for (int qblock_idx_in_segment = 0; qblock_idx_in_segment < num_vald_qblock; qblock_idx_in_segment++)
    {
      if (tile_row_idx == 0)
      {
        load_qblock_to_shared_memory(qweight, smem_qweight_qs, smem_qweight_d, global_out_col_idx, tile_col_idx, num_qblocks_per_row, segment_idx, qblock_idx_in_segment);
        //     qweight[global_out_col_idx_1 * num_qblocks_per_row // row idx of qweight
        //             + segment_idx * QBLOCKS_PER_SEGMENT            // segment idx in this row
        //             + qblock_idx_in_segment                        // qblock idx in this segment
        // ];
        // smem_qweight_segment[tile_col_idx_2 * QBLOCKS_PER_SEGMENT + qblock_idx_in_segment] =
        //     qweight[global_out_col_idx_2 * num_qblocks_per_row // row idx of qweight
        //             + segment_idx * QBLOCKS_PER_SEGMENT            // segment idx in this row
        //             + qblock_idx_in_segment                        // qblock idx in this segment
        // ];
      }
      if (tile_col_idx == 0)
      {
        load_qblock_to_shared_memory(qx, smem_x_qs, smem_x_d, global_out_row_idx, tile_row_idx, num_qblocks_per_row, segment_idx, qblock_idx_in_segment);
        // smem_x_segment[tile_row_idx * QBLOCKS_PER_SEGMENT + qblock_idx_in_segment] =
        //     qx[global_out_row_idx * num_qblocks_per_row // row idx of x
        //        + segment_idx * QBLOCKS_PER_SEGMENT          // segment idx in this row
        //        + qblock_idx_in_segment                      // qblock idx in this segment
        // ];
      }
    }
    __syncthreads();

    // if (threadIdx.x == 0 && threadIdx.y == 0)
    // {
    //   printf("smem_qweight_qs ====================\n");
    //   print_smem_qs(smem_qweight_qs, TILE_SIZE_COL, QBLOCKS_PER_SEGMENT * QI8_0);
    //   printf("smem_qweight_d =====================\n");
    //   print_smem_d(smem_qweight_d, TILE_SIZE_COL, QBLOCKS_PER_SEGMENT);

    //   printf("smem_x_qs ====================\n");
    //   print_smem_qs(smem_x_qs, TILE_SIZE_ROW, QBLOCKS_PER_SEGMENT * QI8_0);
    //   printf("smem_x_d =====================\n");
    //   print_smem_d(smem_x_d, TILE_SIZE_ROW, QBLOCKS_PER_SEGMENT);
    // }
    // __syncthreads();

    // do vecdot for the qblock of this thread
    for (int qblock_idx_in_segment = 0; qblock_idx_in_segment < num_vald_qblock; qblock_idx_in_segment++)
    {
      block_q8_0 qblock_qweight, qblock_x;
      get_qblock_from_shared_memory(smem_x_qs, smem_x_d, &qblock_x, tile_row_idx, qblock_idx_in_segment);
      get_qblock_from_shared_memory(smem_qweight_qs, smem_qweight_d, &qblock_qweight, tile_col_idx, qblock_idx_in_segment);
      // if (blockIdx.x == 1 && blockIdx.y == 1 && threadIdx.x == 0 && threadIdx.y == 0)
      // {
      //   printf("qblock_idx: %d\n", qblock_idx_in_segment);
      //   printf("qweight:\n");
      //   print_qblock(&qblock_qweight);
      //   printf("x:\n");
      //   print_qblock(&qblock_x);
      // }
      sum += vec_dot_q8_0_q8_0(&qblock_qweight, &qblock_x);
    }
  }

  // write output back to global memory
  out[global_out_row_idx * nrows_weight + global_out_col_idx] = __float2half(sum);
}