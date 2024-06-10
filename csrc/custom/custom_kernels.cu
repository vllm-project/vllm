#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <algorithm>

constexpr int WARP_SIZE = 64;

template <typename T>
__device__ __forceinline__ T loadnt(T* addr) {
          return __builtin_nontemporal_load(addr);
}

__device__ __forceinline__ float4 load_ntmprl(const float4* addr) {
          auto addr_alias = reinterpret_cast<const float*>(addr);
          auto dat0 = loadnt(addr_alias);
          auto dat1 = loadnt(addr_alias + 1);
          auto dat2 = loadnt(addr_alias + 2);
          auto dat3 = loadnt(addr_alias + 3);
          //auto dat0 = *(addr_alias);
          //auto dat1 = *(addr_alias+1);
          //auto dat2 = *(addr_alias+2);
          //auto dat3 = *(addr_alias+3);
          return make_float4(dat0,dat1,dat2,dat3);
}

//TBlock fetches entire rows of A, and entire col of B (K dimension); assume N=1 for time being
//grid is M/A_NUM_ROWS blocks
template <int NUM_A_ROWS_PER_BLOCK>
__global__ void LLGemm1_kernel(float4 *af4, __half2 *bf4, __half2 *c, const int K) {
      __shared__ float red_smem[NUM_A_ROWS_PER_BLOCK][WARP_SIZE];
      const int row_addr = blockIdx.x * NUM_A_ROWS_PER_BLOCK * K / 8;
      const int threadid = threadIdx.x;
      const int warp = threadIdx.x / WARP_SIZE;
      const int lane = threadIdx.x % WARP_SIZE;
      const int num_warps = blockDim.x / WARP_SIZE;
      const int qwarpid = threadid/16;
      const int qthreadid = threadid%16;
      float4 rowA_elem4[NUM_A_ROWS_PER_BLOCK];
      __half2 colB_elem4x,colB_elem4y,colB_elem4z,colB_elem4w;
      float4 sum4; //[NUM_A_ROWS_PER_BLOCK];
      float acc[NUM_A_ROWS_PER_BLOCK] = {0.0};
      __half2 acch2;
      __half2 oval;

      // As we later use warp shuffle operations, we may have more threads in the block
      // than the actual available data, hence the if guard here.
      if(threadid * 8 < K) {
        #pragma unroll
        for (int i=0; i<NUM_A_ROWS_PER_BLOCK; i++) {
          // rowA_elem4[i] holds 8 * half numbers seen as a single float4.
          rowA_elem4[i] = load_ntmprl(&af4[row_addr + threadid + K / 8 * i]);
        }
      }

      colB_elem4x = bf4[threadid*4+0];
      colB_elem4y = bf4[threadid*4+1];
      colB_elem4z = bf4[threadid*4+2];
      colB_elem4w = bf4[threadid*4+3];

       __half2 Af2; __half2 Bf2; float2 S;

       auto Ah2ptr = reinterpret_cast<__half2 *>(&rowA_elem4);
       __half2 *ah2lptr;

      #pragma unroll
      for (int i=0; i<NUM_A_ROWS_PER_BLOCK; i++) {
        // Multiply-add on 8 half.
        ah2lptr = Ah2ptr+i*4;
        Af2 = *(ah2lptr);
        acch2 = __hmul2(Af2,colB_elem4x);
        Af2 = *(ah2lptr+1);
        acch2 = __hfma2(Af2,colB_elem4y,acch2);
        Af2 = *(ah2lptr+2);
        acch2 = __hfma2(Af2,colB_elem4z,acch2);
        Af2 = *(ah2lptr+3);
        acch2 = __hfma2(Af2,colB_elem4w,acch2);
        S = __half22float2(acch2);

        // See comment above concerning the if guard.
        if(threadid * 8 < K) {
            acc[i] = S.x + S.y;  // accumulation on float
        }
      }

      // all reduce accross warp.
      #pragma unroll
      for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        #pragma unroll
          for (int i=0; i<NUM_A_ROWS_PER_BLOCK; i++) {
            acc[i] += __shfl_xor(acc[i], mask);
          }
      }

      // Warp leaders store the data to shared memory.
      if (lane < NUM_A_ROWS_PER_BLOCK) {
        red_smem[lane][warp] = acc[lane];
      }

      // Make sure the data is in shared memory.
      __syncthreads();

      if (qwarpid<NUM_A_ROWS_PER_BLOCK) {
        acc[qwarpid] = qthreadid<num_warps ? red_smem[qwarpid][qthreadid] : 0.f;
        #pragma unroll
        for (int mask = 16 / 2; mask >= 1; mask /= 2) {
          acc[qwarpid] += __shfl_xor(acc[qwarpid], mask);
        }
        float oval2 = __shfl_xor(acc[qwarpid],16);

        if (threadid%WARP_SIZE ==0 or threadid%WARP_SIZE==32) {
          oval = __float22half2_rn(make_float2(acc[qwarpid],oval2));
          c[blockIdx.x*NUM_A_ROWS_PER_BLOCK/2+qwarpid/2] = oval;
        }
      }
}

// define the kernel calling code:
//template <typename T>
void LLGemm1(void *in_a, void *in_b, void *out_c, const int M, const int K, cudaStream_t stream, const int rows_per_block=4) {
      float4 *af4 = reinterpret_cast<float4*>(in_a);
      auto *bf4 = reinterpret_cast<__half2*>(in_b);
      auto *c = reinterpret_cast<__half2*>(out_c);

      // NUM_TREADS need to be a multiple of WARP_SIZE, as we are using warp shuffle operations.
      const int NUM_THREADS = K*2/16 % WARP_SIZE == 0 ? K*2/16 : K*2/16 + (WARP_SIZE - K*2/16 % WARP_SIZE);

      int NUM_BLOCKS = M/rows_per_block;

      if (rows_per_block==2) {
        LLGemm1_kernel<2><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, K);
      }
      else if (rows_per_block==4) {
        LLGemm1_kernel<4><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, K);
      }
      else if (rows_per_block==8) {
        LLGemm1_kernel<8><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, K);
      }
      else if (rows_per_block==16) {
        LLGemm1_kernel<16><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, K);
      }
      else {
        NUM_BLOCKS = M/4;
        LLGemm1_kernel<4><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(af4, bf4, c, K);
      }

        cudaError_t err = cudaGetLastError();
          if (cudaSuccess != err)
                  throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}

// instantiate the kernel template for T=float:
//template void AddGPUKernel<float>(float *in_a, float *in_b, float *out_c, const int M, const int K, cudaStream_t stream);

const unsigned int TILE_WIDTH = 32;

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
        __shared__ float sA[TILE_WIDTH][TILE_WIDTH];   // Tile size of 32x32
        __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

        int Row = blockDim.y * blockIdx.y + threadIdx.y;
        int Col = blockDim.x * blockIdx.x + threadIdx.x;
        float Cvalue = 0.0;
        sA[threadIdx.y][threadIdx.x] = 0.0;
        sB[threadIdx.y][threadIdx.x] = 0.0;

        for (int ph = 0; ph < (((numAColumns - 1) / TILE_WIDTH) + 1); ph++) {
            if ((Row < numARows) && (threadIdx.x + (ph * TILE_WIDTH)) < numAColumns) {
                sA[threadIdx.y][threadIdx.x] = A[(Row * numAColumns) + threadIdx.x + (ph * TILE_WIDTH)];
            } else {
                sA[threadIdx.y][threadIdx.x] = 0.0;
            }
            if (Col < numBColumns && (threadIdx.y + ph * TILE_WIDTH) < numBRows) {
                sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + ph * TILE_WIDTH) * numBColumns + Col];
            } else {
                sB[threadIdx.y][threadIdx.x] = 0.0;
            }
            __syncthreads();
            for (int j = 0; j < TILE_WIDTH; ++j) {
                Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
            }
        }
        if (Row < numCRows && Col < numCColumns) {
            C[Row * numCColumns + Col] = Cvalue;
        }
}


void MMGPUKernel(float *in_a, float *in_b, float *out_c, 
        int numARows, int numAColumns,
        int numBRows, int numBColumns,
        int numCRows, int numCColumns, 
        cudaStream_t stream) {

            // Initialize the grid and block dimensions 
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
            dim3 dimGrid((numCColumns / TILE_WIDTH) + 1, (numCRows / TILE_WIDTH) + 1, 1);
            //@@ Launch the GPU Kernel here
                matrixMultiplyShared <<<dimGrid, dimBlock>>>
                                                           (in_a, in_b, out_c, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

        cudaError_t err = cudaGetLastError();
          if (cudaSuccess != err)
                  throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}



template<int nThreads_per_row, int CTA, int MT0, int MT1>
__global__
__launch_bounds__(512)
void HGEMV_WFPerRow(int m, int n, const _Float16 *A, int lda, const _Float16 *x, _Float16 *y)
{
  int num_row_per_block = CTA / nThreads_per_row;
  int row_id = (blockIdx.x*num_row_per_block+threadIdx.y)*MT0;
  int inc = (gridDim.x * num_row_per_block)*MT0;

  while (row_id < m) {
    float2 sum2[MT0];

#pragma unroll
    for (int i = 0; i < MT0; ++i)
    {
       sum2[i] = {0.0,0.0};
    }

    for (int j = threadIdx.x; j < n; j += (nThreads_per_row*MT1)){
        bool is_active = j < n;
        if (is_active) {
            float2 x2[MT1>>1];
#pragma unroll
	    for(int offset = 0; offset < MT1; offset += 2)
	    {
            	x2[offset>>1] = {x[j+nThreads_per_row*offset], x[j+nThreads_per_row*(offset+1)]};
	    }
	    float2 a2[MT0][MT1>>1];
#pragma unroll
	    for (int i = 0; i < MT0; i++)
	    {
#pragma unroll
	    	for (int offset = 0; offset < MT1; offset += 2)
	    	{
            	    a2[i][offset>>1] = {A[(row_id+i)*n+j+nThreads_per_row*offset], A[(row_id+i)*n+j+nThreads_per_row*(offset+1)]};
	    	}
	    }

#pragma unroll
	    for (int i = 0; i < MT0; i++)
	    {
#pragma unroll
	    	for (int offset = 0; offset < (MT1>>1); offset++)
	    	{
	  		sum2[i] += a2[i][offset]*x2[offset];
		}
	    }

        }
    }
    float sum[MT0];
#pragma unroll
    for (int i = 0; i < MT0; i++)
    {
    	sum[i] = sum2[i].x+sum2[i].y;
    }

#pragma unroll
    for (int i = 0; i < MT0; i++)
    {
#pragma unroll 
    	for (int offset = nThreads_per_row  >> 1; offset >= 1; offset = offset >> 1) {
            sum[i] += __shfl_down(sum[i], offset, nThreads_per_row);
	}
    }
    if (threadIdx.x == 0) 
    {
#pragma unroll
	for (int i = 0; i < MT0; i++)
	{	
           y[row_id+i] = sum[i];
	}
    }
    row_id += inc;
  }
}

void LLGemmZZ(void *in_a, void *in_b, void *out_c, const int M, const int K, cudaStream_t stream, const int solidx=0) {
      //m -> M, n-> K
      dim3 grid(1024);
      dim3 block(64, 8); 
      if (solidx==0) {
        HGEMV_WFPerRow<64, 512, 4, 8><<<grid, block,0,stream>>>(M, K, reinterpret_cast<const _Float16*>(in_a), K, 
              reinterpret_cast<const _Float16*>(in_b),reinterpret_cast<_Float16*>(out_c));
      }
      else if (solidx==1) {
        HGEMV_WFPerRow<64, 512, 2, 8><<<grid, block,0,stream>>>(M, K, reinterpret_cast<const _Float16*>(in_a), K, 
              reinterpret_cast<const _Float16*>(in_b),reinterpret_cast<_Float16*>(out_c));
      }
      else if (solidx==2) {
        HGEMV_WFPerRow<64, 512, 1, 8><<<grid, block,0,stream>>>(M, K, reinterpret_cast<const _Float16*>(in_a), K, 
              reinterpret_cast<const _Float16*>(in_b),reinterpret_cast<_Float16*>(out_c));
      }
      else {
        HGEMV_WFPerRow<64, 512, 4, 8><<<grid, block,0,stream>>>(M, K, reinterpret_cast<const _Float16*>(in_a), K, 
              reinterpret_cast<const _Float16*>(in_b),reinterpret_cast<_Float16*>(out_c));
      }
        cudaError_t err = cudaGetLastError();
          if (cudaSuccess != err)
                  throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}
