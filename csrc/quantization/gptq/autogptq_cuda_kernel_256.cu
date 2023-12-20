#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// atomicAdd for double-precision floating-point numbers on hardware with
// compute capability < 6.0 from:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
// __device__ double atomicAdd(
//     double* address,
//     double val
// ) {
//   unsigned long long int* address_as_ull = (unsigned long long int*)address;
//   unsigned long long int old = *address_as_ull, assumed;
//
//   do {
//     assumed = old;
//     old = atomicCAS(
//       address_as_ull,
//       assumed,
//       __double_as_longlong(val + __longlong_as_double(assumed))
//     );
//
//   // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//   } while (assumed != old);
//
//   return __longlong_as_double(old);
// }
// #endif

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700) || defined(USE_ROCM)
// adapted from https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh

__device__ __forceinline__ void atomicAdd(c10::Half* address, c10::Half val) {
    unsigned int *address_as_ui = reinterpret_cast<unsigned int *>(reinterpret_cast<char *>(address) - (reinterpret_cast<size_t>(address) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        unsigned short hsum = reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
        hsum += val;
        old = reinterpret_cast<size_t>(address) & 2
                 ? (old & 0xffff) | (hsum << 16)
                 : (old & 0xffff0000) | hsum;
        old = atomicCAS(address_as_ui, assumed, old);

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
}
__device__ __forceinline__ void atomicAdd(__half* address, c10::Half val) {
    unsigned int * address_as_ui = (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        __half_raw hsum;
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        half tmpres = __hadd(hsum, val);
        hsum = __half_raw(tmpres);
        old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
}
#endif


template <typename scalar_t>
__global__ void VecQuant2MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
	const  	    int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	int zero_width
);

template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
	const  	    int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	int zero_width
);

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
	const  	    int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	int zero_width
);

template <typename scalar_t>
__global__ void VecQuant8MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
	const  	    int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	int zero_width
);

template <typename scalar_t>
__global__ void VecQuant2MatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
);

template <typename scalar_t>
__global__ void VecQuant3MatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
);

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
);

template <typename scalar_t>
__global__ void VecQuant8MatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
);

__global__ void VecQuant2MatMulKernelFaster_old(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const    int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
);

__global__ void VecQuant3MatMulKernelFaster_old(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const    int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
);

__global__ void VecQuant4MatMulKernelFaster_old(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const    int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
);


const int BLOCKWIDTH  = 256;
const int BLOCKHEIGHT2 =  16;
const int BLOCKHEIGHT3 =  24;
const int BLOCKHEIGHT4 =  32;
const int BLOCKHEIGHT8 =  64;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}


void vecquant2matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant2matmul_cuda", ([&] {
      VecQuant2MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(), g_idx.data<int>(),
        batch, vec_height, height, width, zero_width
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant2MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
    const   	int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	int zero_width
) {
  int h = BLOCKHEIGHT2 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  int i = width * h + w;
  int g_h = h * 16;
  int k;
  unsigned int g;
  scalar_t w_tmp;

  int z_w = w / 16;
  int z_mod = (w % 16) * 2;

  float weight[BLOCKWIDTH];

  for (k = 0; k <  BLOCKWIDTH; ++k){
	int k_w = (k / 16);
	int k_bit = (k % 16) * 2;

    g = as_int(g_idx[g_h + k]);
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scalar_t((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0x3) + 1);

    w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x3);

	weight[k] = scale * (w_tmp - zero);
  }

  scalar_t res;
  for (int b = 0; b < batch; ++b){
	res = 0;

    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
    __syncthreads();
	for (k = 0; k <  BLOCKWIDTH; ++k){
	  res += weight[k] * blockvec[k];
    }
    atomicAdd(&mul[b * width + w], res);
    __syncthreads();
  }
}

void vecquant3matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant3matmul_cuda", ([&] {
      VecQuant3MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(), g_idx.data<int>(),
        batch, vec_height, height, width, zero_width
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    const   	int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	int zero_width
) {
  int h = BLOCKHEIGHT3 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  int i = width * h + w;
  int g_h = (h / 3) * 32;
  int k;
  unsigned int g;
  scalar_t w_tmp;

  int z_w = (w / 32) * 3;
  int z_mod = w % 32;
  int z_bit;
  unsigned int z_tmp;
  if (z_mod != 10){
    if (z_mod != 21){
      z_bit = z_mod;
      if (z_bit > 21){
        z_bit -= 22;
        z_bit *= 3;
        z_bit += 2;
        z_w += 2;
      } else if (z_bit > 10){
        z_bit -= 11;
        z_bit *= 3;
        z_bit += 1;
        z_w += 1;
      } else {
        z_bit *= 3;
      }
    } else {
      z_w += 1;
    }
  }

  float weight[BLOCKWIDTH];

  for (k = 0; k <  BLOCKWIDTH; ++k){
	int k_w = (k / 32) * 3;
	int k_mod = k % 32;
	int k_bit;

	if (k_mod != 10){
	  if (k_mod != 21){
        k_bit = k_mod;
        if (k_bit > 21){
		  k_bit -= 22;
		  k_bit *= 3;
		  k_bit += 2;
		  k_w += 2;
        } else if (k_bit > 10){
		  k_bit -= 11;
		  k_bit *= 3;
		  k_bit += 1;
		  k_w += 1;
        } else {
		  k_bit *= 3;
        }
	  } else {
        k_w += 1;
	  }
	}

    g = as_int(g_idx[g_h + k]);
    scalar_t scale = scales[g * width + w];
    scalar_t zero;
    if (z_mod == 10) {
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 30) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 2) & 0x4);
      zero = scalar_t((z_tmp) + 1);
    } else if (z_mod == 21){
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 31) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 1) & 0x6);
      zero = scalar_t((z_tmp) + 1);
    } else {
      zero = scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_bit) & 0x7) + 1);
    }

    if (k_mod == 10) {
      w_tmp = (as_unsigned(mat[i + (k_w * width)]) >> 30) | ((as_unsigned(mat[i + ((k_w + 1)* width)]) << 2) & 0x4);
    } else if (k_mod == 21){
      w_tmp = (as_unsigned(mat[i + (k_w * width)]) >> 31) | ((as_unsigned(mat[i + ((k_w + 1)* width)]) << 1) & 0x6);
    } else {
      w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x7);
    }
	weight[k] = scale * (w_tmp - zero);
  }

  scalar_t res;
  for (int b = 0; b < batch; ++b){
	res = 0;

    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
    __syncthreads();
	for (k = 0; k <  BLOCKWIDTH; ++k){
	  res += weight[k] * blockvec[k];
    }
    atomicAdd(&mul[b * width + w], res);
    __syncthreads();
  }
}

void vecquant4matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant4matmul_cuda", ([&] {
      VecQuant4MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(), g_idx.data<int>(),
        batch, vec_height, height, width, zero_width
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    const   	int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	int zero_width
) {
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  int i = width * h + w;
  int g_h = h * 8;
  int k;
  unsigned int g;
  scalar_t w_tmp;


  int z_w = w / 8;
  int z_mod = (w % 8) * 4;

  float weight[BLOCKWIDTH];

  for (k = 0; k <  BLOCKWIDTH; ++k){
	int k_w = (k / 8);
	int k_bit = (k % 8) * 4;

    g = as_int(g_idx[g_h + k]);
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1);

    w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xF);

	weight[k] = scale * (w_tmp - zero);
  }

  scalar_t res;
  for (int b = 0; b < batch; ++b){
	res = 0;

    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
    __syncthreads();
	for (k = 0; k <  BLOCKWIDTH; ++k){
	  res += weight[k] * blockvec[k];
    }
    atomicAdd(&mul[b * width + w], res);
    __syncthreads();
  }
}

void vecquant8matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT8 - 1) / BLOCKHEIGHT8,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant8matmul_cuda", ([&] {
      VecQuant8MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(), g_idx.data<int>(),
        batch, vec_height, height, width, zero_width
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant8MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    const   	int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
	int zero_width
) {
  int h = BLOCKHEIGHT8 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  int i = width * h + w;
  int g_h = h * 4;
  int k;
  unsigned int g;
  scalar_t w_tmp;

  int z_w = w / 4;
  int z_mod = (w % 4) * 8;

  float weight[BLOCKWIDTH];

  for (k = 0; k <  BLOCKWIDTH; ++k){
	int k_w = (k / 4);
	int k_bit = (k % 4) * 8;

    g = as_int(g_idx[g_h + k]);
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xFF) + 1);

    w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xFF);

	weight[k] = scale * (w_tmp - zero);
  }

  scalar_t res;
  for (int b = 0; b < batch; ++b){
	res = 0;

    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
    __syncthreads();
	for (k = 0; k <  BLOCKWIDTH; ++k){
	  res += weight[k] * blockvec[k];
    }
    atomicAdd(&mul[b * width + w], res);
    __syncthreads();
  }
}


void vecquant2matmul_cuda_old(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant2matmul_cuda_old", ([&] {
      VecQuant2MatMulKernel_old<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, vec_height, height, width, zero_width, groupsize
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant2MatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT2 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  scalar_t res = 0;
  int i = width * h + w;
  int g_h = h * 16;
  int k = 0;

  int z_w = w / 16;
  int z_mod = (w % 16) * 2;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);

    int g = (g_h + k) / groupsize;
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scale * scalar_t((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0x3) + 1);

    res += (scale * scalar_t((tmp >> 0) & 0x3) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp >> 2) & 0x3) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp >> 4) & 0x3) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp >> 6) & 0x3) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp >> 8) & 0x3) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp >> 10) & 0x3) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp >> 12) & 0x3) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp >> 14) & 0x3) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp >> 16) & 0x3) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp >> 18) & 0x3) - zero) * blockvec[k + 9];
    res += (scale * scalar_t((tmp >> 20) & 0x3) - zero) * blockvec[k + 10];
    res += (scale * scalar_t((tmp >> 22) & 0x3) - zero) * blockvec[k + 11];
    res += (scale * scalar_t((tmp >> 24) & 0x3) - zero) * blockvec[k + 12];
    res += (scale * scalar_t((tmp >> 26) & 0x3) - zero) * blockvec[k + 13];
    res += (scale * scalar_t((tmp >> 28) & 0x3) - zero) * blockvec[k + 14];
    res += (scale * scalar_t((tmp >> 30) & 0x3) - zero) * blockvec[k + 15];

    i += width;
    k += 16;
  }

  atomicAdd(&mul[b * width + w], res);
}

void vecquant3matmul_cuda_old(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant3matmul_cuda_old", ([&] {
      VecQuant3MatMulKernel_old<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, vec_height, height, width, zero_width, groupsize
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant3MatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT3 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  scalar_t res = 0;
  int i = width * h + w;
  int g_h = (h / 3) * 32;
  int k = 0;

  int z_w = (w / 32) * 3;
  int z_mod = w % 32;
  int z_bit;

  if (z_mod != 10){
    if (z_mod != 21){
      z_bit = z_mod;
      if (z_bit > 21){
        z_bit -= 22;
        z_bit *= 3;
        z_bit += 2;
        z_w += 2;
      } else if (z_bit > 10){
        z_bit -= 11;
        z_bit *= 3;
        z_bit += 1;
        z_w += 1;
      } else {
        z_bit *= 3;
      }
    } else {
      z_w += 1;
    }
  }

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;
  unsigned int z_tmp;

  while (k < BLOCKWIDTH) {
    tmp1 = as_unsigned(mat[i]);

    int g = (g_h + k) / groupsize;
    scalar_t scale = scales[g * width + w];
    scalar_t zero;
    if (z_mod == 10) {
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 30) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 2) & 0x4);
      zero = scale * scalar_t((z_tmp) + 1);
    } else if (z_mod == 21){
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 31) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 1) & 0x6);
      zero = scale * scalar_t((z_tmp) + 1);
    } else {
      zero = scale * scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_bit) & 0x7) + 1);
    }

    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];

    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;

    res += (scale * scalar_t((tmp2 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp2 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp2 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp2 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp2 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp2 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp2 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp2 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp2 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp2 >> 27) & 0x7) - zero) * blockvec[k + 9];

    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;

    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];

    i += width;
    k += 10;
  }

  atomicAdd(&mul[b * width + w], res);
}

void vecquant4matmul_cuda_old(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant4matmul_cuda_old", ([&] {
      VecQuant4MatMulKernel_old<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, vec_height, height, width, zero_width, groupsize
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  scalar_t res = 0;
  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;

  int z_w = w / 8;
  int z_mod = (w % 8) * 4;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);

    int g = (g_h + k) / groupsize;
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scale * scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1);

    res += (scale * scalar_t((tmp >> 0) & 0xF) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp >> 4) & 0xF) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp >> 8) & 0xF) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp >> 12) & 0xF) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp >> 16) & 0xF) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp >> 20) & 0xF) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp >> 24) & 0xF) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp >> 28) & 0xF) - zero) * blockvec[k + 7];

    i += width;
    k += 8;
  }

  atomicAdd(&mul[b * width + w], res);
}

void vecquant8matmul_cuda_old(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT8 - 1) / BLOCKHEIGHT8,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant8matmul_cuda_old", ([&] {
      VecQuant8MatMulKernel_old<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, vec_height, height, width, zero_width, groupsize
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant8MatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT8 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  scalar_t res = 0;
  int i = width * h + w;
  int g_h = h * 4;
  int k = 0;

  int z_w = w / 4;
  int z_mod = (w % 4) * 8;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);

    int g = (g_h + k) / groupsize;
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scale * scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xFF) + 1);

    res += (scale * scalar_t((tmp >> 0) & 0xFF) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp >> 8) & 0xFF) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp >> 16) & 0xFF) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp >> 24) & 0xFF) - zero) * blockvec[k + 3];

    i += width;
    k += 4;
  }

  atomicAdd(&mul[b * width + w], res);
}


void vecquant2matmul_faster_cuda_old(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant2MatMulKernelFaster_old<<<blocks, threads>>>(
    (half2*) vec.data_ptr(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    scales.data_ptr<float>(),
    zeros.data_ptr<int>(),
    batch, vec_height, height, width, zero_width, groupsize
  );
}

__global__ void VecQuant2MatMulKernelFaster_old(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  	 int* __restrict__ zeros,
	int batch,
	int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT2 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[16][16];
  int val = threadIdx.x / 16;
  int off = threadIdx.x % 16;
  for (; val < 16; val += BLOCKWIDTH / 16) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0x3), __int2half_rn(val >> 2)
    );
  }

  int i = width * h + w;
  int g_h = h * 16;
  int k = 0;

  int z_w = w / 16;
  int z_mod = (w % 16) * 2;

  float res = 0;
  half2 res2;

  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
    int g = (g_h + (k * 2)) / groupsize;
	float scale_f = scales[g * width + w];
    half2 scale = __float2half2_rn(scale_f);
    half2 zero = __float2half2_rn(-(scale_f * (((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0x3) + 1)));

    std::memset(&res2, 0, sizeof(half2));
    tmp = as_unsigned(mat[i]);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  0) & 0xf][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  4) & 0xf][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  8) & 0xf][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 12) & 0xf][off], scale, zero), blockvec[k + 3], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 16) & 0xf][off], scale, zero), blockvec[k + 4], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 20) & 0xf][off], scale, zero), blockvec[k + 5], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 24) & 0xf][off], scale, zero), blockvec[k + 6], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 28) & 0xf][off], scale, zero), blockvec[k + 7], res2);
	i += width;
    k += 8;
    res += __low2float(res2) + __high2float(res2);
  }

  atomicAdd(&mul[b * width + w], res);
}

void vecquant3matmul_faster_cuda_old(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelFaster_old<<<blocks, threads>>>(
    (half2*) vec.data_ptr(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    scales.data_ptr<float>(),
    zeros.data_ptr<int>(),
    batch, vec_height, height, width, zero_width, groupsize
  );
}

__global__ void VecQuant3MatMulKernelFaster_old(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  	 int* __restrict__ zeros,
	int batch,
	int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT3 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[64][32];
  int val = threadIdx.x / 32;
  int off = threadIdx.x % 32;
  for (; val < 64; val += BLOCKWIDTH / 32) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0x7), __int2half_rn(val >> 3)
    );
  }

  int i = width * h + w;
  int g_h = (h / 3) * 32;
  int k = 0;

  int z_w = (w / 32) * 3;
  int z_mod = w % 32;
  int z_bit;

  if (z_mod != 10){
    if (z_mod != 21){
      z_bit = z_mod;
      if (z_bit > 21){
        z_bit -= 22;
        z_bit *= 3;
        z_bit += 2;
        z_w += 2;
      } else if (z_bit > 10){
        z_bit -= 11;
        z_bit *= 3;
        z_bit += 1;
        z_w += 1;
      } else {
        z_bit *= 3;
      }
    } else {
      z_w += 1;
    }
  }

  float res = 0;
  half2 res2;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;
  unsigned int z_tmp;

  __syncthreads();

  while (k < blockwidth2) {
    int g = (g_h + (k * 2)) / groupsize;
	float scale_f = scales[g * width + w];
    half2 scale = __float2half2_rn(scale_f);
    half2 zero;
    if (z_mod == 10) {
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 30) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 2) & 0x4);
      zero = __float2half2_rn(-(scale_f * ((z_tmp) + 1)));
    } else if (z_mod == 21){
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 31) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 1) & 0x6);
      zero = __float2half2_rn(-(scale_f * ((z_tmp) + 1)));
    } else {
      zero = __float2half2_rn(-(scale_f * (((as_unsigned(zeros[g * zero_width + z_w]) >> z_bit) & 0x7) + 1)));
    }

    std::memset(&res2, 0, sizeof(half2));
    tmp1 = as_unsigned(mat[i]);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
    res2 = __hfma2(__hfma2(deq2[tmp][off], scale, zero), blockvec[k + 5], res2);
    tmp2 >>= 4;
    k += 6;
    res2 = __hfma2(__hfma2(deq2[(tmp2 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp2 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp2 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp2 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
    res2 = __hfma2(__hfma2(deq2[tmp][off], scale, zero), blockvec[k + 4], res2);
    tmp1 >>= 2;
    k += 5;
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero), blockvec[k + 3], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero), blockvec[k + 4], res2);
    i += width;
    k += 5;
    res += __low2float(res2) + __high2float(res2);
  }

  atomicAdd(&mul[b * width + w], res);
}

void vecquant4matmul_faster_cuda_old(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelFaster_old<<<blocks, threads>>>(
    (half2*) vec.data_ptr(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    scales.data_ptr<float>(),
    zeros.data_ptr<int>(),
    batch, vec_height, height, width, zero_width, groupsize
  );
}

__global__ void VecQuant4MatMulKernelFaster_old(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  	 int* __restrict__ zeros,
	int batch,
	int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[256][8];
  int val = threadIdx.x / 8;
  int off = threadIdx.x % 8;
  for (; val < 256; val += BLOCKWIDTH / 8) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0xF), __int2half_rn(val >> 4)
    );
  }

  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;

  int z_w = w / 8;
  int z_mod = (w % 8) * 4;

  float res = 0;
  half2 res2;

  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
    int g = (g_h + (k * 2)) / groupsize;
	float scale_f = scales[g * width + w];

    half2 scale = __float2half2_rn(scale_f);
    half2 zero = __float2half2_rn(-(scale_f * (((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1)));

    //std::memset(&res2, 0, sizeof(half2));

    //res2 = __float2half2_rn((float)0.);

    std::memset(&res2, 0, sizeof(half2));
    tmp = as_unsigned(mat[i]);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  0) & 0xff][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  8) & 0xff][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 16) & 0xff][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 24) & 0xff][off], scale, zero), blockvec[k + 3], res2);
	i += width;
    k += 4;

    res += __low2float(res2) + __high2float(res2);

  }

  atomicAdd(&mul[b * width + w], res);
}