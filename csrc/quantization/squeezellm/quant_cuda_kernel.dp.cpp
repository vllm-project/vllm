#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <torch/all.h>
#include <torch/python.h>

// half-tensor
#include <ATen/Aten.cuh>
#inlude <ipex.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDATensorMethods.cuh>


#define BLOCKWIDTH 128
#define BLOCKHEIGHT4 16

namespace at{

namespace cuda{

dpct::queue_ptr getCurrentCUDAStream(){

  auto device_type = c10::DeviceType::XPU;
  c10::impl::VirtualGuardImpl impl(device_type);
  c10::Stream c10_stream = impl.getStream(c10::Device(device_type));
  auto& queue = xpu::get_queue_from_stream(c10_stream);
  return &queue;

}

}

}


namespace vllm {
namespace squeezellm {

inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

// 4-bit matvec kernel (LUT-based)
void NUQ4MatMulKernel(const sycl::half2 *__restrict__ vec,
                      const int *__restrict__ mat,
                      sycl::half2 *__restrict__ mul,
                      const sycl::half *__restrict__ lookup_table, int height,
                      int width, int batch, int vec_height,
                      const sycl::nd_item<3> &item_ct1) {

  const int blockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT4 * item_ct1.get_group(2);
  int col = BLOCKWIDTH * item_ct1.get_group(1) + item_ct1.get_local_id(2);
  
  extern __shared__ sycl::half2 blockvec[blockwidth2];
  extern __shared__ sycl::local_accessor<sycl::half, 2> deq2[16][BLOCKWIDTH];
  
  int off = item_ct1.get_local_id(2);
  int column_offset = col * 16;
  for (int val = 0; val < 16; val += 1) {
    int lut_index = column_offset + val;
    deq2[val][off] = lookup_table[lut_index];
  }

  sycl::half res;
  sycl::half2 res2;
  sycl::half2 tmp2;

  int i;
  int k;

  unsigned int tmp1;
  unsigned int lut_index1, lut_index2;

  for (int b = 0; b < batch; ++b){
    i = width * row + col;
    res =
        sycl::vec<int, 1>{0}.convert<sycl::half, sycl::rounding_mode::rtn>()[0];
    k = 0;

    /*
    DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (item_ct1.get_local_id(2) < blockwidth2)
      blockvec[item_ct1.get_local_id(2)] =
          vec[b * vec_height / 2 + (row / BLOCKHEIGHT4) * blockwidth2 +
              item_ct1.get_local_id(2)];
    /*
    DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    while (k < blockwidth2) {
      tmp1 = as_unsigned(mat[i]);

      res2 = {};
      tmp2 = {};

      lut_index1 = tmp1 & 0xF;
      lut_index2 = (tmp1 >> 4) & 0xF;
      tmp2.x() = deq2[lut_index1][off];
      tmp2.y() = deq2[lut_index2][off];
      res2 = sycl::fma(tmp2, blockvec[k + 0], res2);

      lut_index1 = (tmp1 >> 8) & 0xF;
      lut_index2 = (tmp1 >> 12) & 0xF;
      tmp2.x() = deq2[lut_index1][off];
      tmp2.y() = deq2[lut_index2][off];
      res2 = sycl::fma(tmp2, blockvec[k + 1], res2);

      lut_index1 = (tmp1 >> 16) & 0xF;
      lut_index2 = (tmp1 >> 20) & 0xF;
      tmp2.x() = deq2[lut_index1][off];
      tmp2.y() = deq2[lut_index2][off];
      res2 = sycl::fma(tmp2, blockvec[k + 2], res2);

      lut_index1 = (tmp1 >> 24) & 0xF;
      lut_index2 = (tmp1 >> 28) & 0xF;
      tmp2.x() = deq2[lut_index1][off];
      tmp2.y() = deq2[lut_index2][off];
      res2 = sycl::fma(tmp2, blockvec[k + 3], res2);

      /*
      DPCT1007:3: Migration of __hadd is not supported.
      */
      res = __hadd(__hadd(res2.x(), res2.y()), res);

      i += width;
      k += 4;
    }

    // col%2 -> only set one of the two values
    sycl::half2 res3 = {};
    if (col % 2 == 0) {
      res3.x() = res;
    } else {
      res3.y() = res;
    }

    /*
    DPCT1007:2: Migration of half version of atomicAdd is not supported.
    */
    atomicAdd(&mul[b * width / 2 + col / 2], res3);
  }
}

} // namespace squeezellm
} // namespace vllm

// 4-bit matvec kernel (LUT-based)
void squeezellm_gemm(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table
) {
  int height = mat.size(0);
  int width = mat.size(1);

  int batch = vec.size(0);
  int vec_height = vec.size(1);

  sycl::range<3> blocks(1, (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
                        (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4);
  sycl::range<3> threads(1, 1, BLOCKWIDTH);

  
  const dpct::queue_ptr stream = at::cuda::getCurrentCUDAStream();
  stream.submit([&](sycl::handler &cgh){
  
  cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
   [=](sycl::nd_item<3> item_ct1){
   vllm::squeezellm::NUQ4MatMulKernel(
        (sycl::half2*) vec.data<sycl::half2>(),
    mat.data_ptr<int>(),
    (sycl::half2*) mul.data<sycl::half2>(),
    (sycl::half*) lookup_table.data<sycl::half>(),
    height, width, batch, vec_height,item_ct1);
   });
    }
  );
  
  /*
  vllm::squeezellm::NUQ4MatMulKernel<<<blocks, threads>>>(
    (half2*) vec.data<at::Half>(),
    mat.data_ptr<int>(),
    (half2*) mul.data<at::Half>(),
    (__half*) lookup_table.data<at::Half>(),
    height, width, batch, vec_height
  );
  */
}

#undef BLOCKWIDTH
#undef BLOCKHEIGHT4
