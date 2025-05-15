#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

__global__ void quantize_q8_0_mi50(half* __restrict__ x,
                                   block_q8_0* __restrict__ out, int nrows_x,
                                   int ncols_x, int ncols_x_padded,
                                   int block_q8_0_per_row) {
  /*
  quantize x to q8_0 on gcn arch

  x: input tensor (fp16) need to be quantized
  out: quantized block
  nrows_x: # of rows of x
  ncols_x: # of cols of x
  ncols_x_padded: # of cols of padded x, pad to multiple of 64
  block_q8_0_per_row: # of block_q8_0 per row

  expecting to be launched with:
  - gridDim = (nrows_x, ncols_x_padded / 64)
  - blockDim = (warpSize, 1)

  each thread corresponding to a element in the output
  e.g. if output contains 2 block_q8_0, 64 elements total, we launch 64 threads
  to handle that

  each row of x is padded to multiple of 64 (warpSize of gcn)
  e.g. if original ncols_x = 100, than ncols_x_padded = 128, which means last 28
  cols are padded for padded elements, instead of (allocate and) read pad value
  from vram, we just pretend them as 0 to save vram

  each block_q8_0 contains 32 elements, so 1 warp on gcn (64 threads) can handle
  2 block_q8_0s

  in each warp:
  1. load 64 elements (2 block_q8_0s) from x
  2. calculate delta(d) of first 32-elements and second 32-elements
  which corresponding to 1st block_q8_0 and 2nd block_q8_0
  d = max(x_i) / 127
  3. calculate 64 quantized elements(qs) with d
  qs = x_i / d = x_i * 127 / max
  4. write qs and d to output
  */

  int global_thread_idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  int global_thread_idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  if (global_thread_idx_x >= ncols_x_padded || global_thread_idx_y >= nrows_x) {
    return;
  }

  // 1. load element from x
  half xi;
  if (global_thread_idx_x < ncols_x)  // not pad elements
  {
    xi = x[ncols_x * global_thread_idx_y + global_thread_idx_x];
  } else  // padded elements
  {
    xi = __float2half(0.0f);
  }

  // 2. calculate d and sum
  half max = __habs(xi);
  half delta;

  for (int offset = 16; offset > 0; offset /= 2) {
    // broadcast max and sum to all threads handling the same block_q8_0
    max = __hmax(max, __shfl_xor_sync((uint64_t)-1, max, offset, 32));
  }
  delta = max / (half)127.0;

  // 3. calculate qs with d
  int8_t qs = 0;
  if (delta != (half)0.0f) {
      float q = __half2float(xi) / __half2float(delta);
      qs = (int8_t)roundf(q);
  }

  // 4. write qs and d to output
  int out_block_q8_0_1st_idx = blockIdx.y * block_q8_0_per_row +
                               blockIdx.x * 2;  // index of 1st out block_q8_0
  int out_block_q8_0_2nd_idx =
      out_block_q8_0_1st_idx + 1;  // index of 2nd out block_q8_0
  int local_qs_idx =
      threadIdx.x % 32;  // local qs idx of this thread in output block_q8_0
  bool thread_belongs_to_1st_block_q8_0 =
      threadIdx.x / 32 ==
      0;  // this thread belongs to 1st block_q8_0 or 2nd block_q8_0

  if (thread_belongs_to_1st_block_q8_0) {
    out[out_block_q8_0_1st_idx].qs[local_qs_idx] = qs;
  } else {
    out[out_block_q8_0_2nd_idx].qs[local_qs_idx] = qs;
  }

  // write d to output
  if (threadIdx.x == 0) {  // write 1st block of this warp
    out[out_block_q8_0_1st_idx].d = delta;
  } else if (threadIdx.x == 32) {  // write 2nd block of this wrap
    out[out_block_q8_0_2nd_idx].d = delta;
  }
}