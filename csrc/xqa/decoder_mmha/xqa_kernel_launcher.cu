/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <nvrtc.h>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "xqa_kernel_launcher.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include "attention/attention_dtypes.h"
#include "cuda_compat.h"
#include "dispatch_utils.h"
// #include "xqa/kernels/decoderMaskedMultiheadAttention/mha.h"
// #include "xqa/kernels/decoderMaskedMultiheadAttention/mhaUtils.cuh"
#include "decoder_xqa_impl_precompiled.h"
#include "decoder_xqa_runner.h"

// #include "cubin/xqa_kernel_cubin.h"

// // There are 4 ways to pass ctaRowMax backward from gemm1 warps to gemm0
// warps:
// //  1. Protect with xFwdBarriers+xBwdBarriers. This way, ctaRowMax is
// available to gemm0 warps together with x tiles and warpRowMax/warpRowSum. But
// ctaRowMax is required before warp tile online softmax, while the other
// buffers is needed only after online softmax. So xBwdBarriers wait will need
// to be moved before online softmax.
// //  2. Similar to approach 1, but we add an additional register copy of
// ctaRowMax in gemm0 warps. It's loaded from smem ctaRowMax after warp tile
// online softmax, so the current warp tile can't use it. But we can pass it to
// next iteration so softmax of next tile can use it. The update will be delayed
// by 1 more iteration and we need one or two more registers. Alternatively, put
// the extra copy in shared memory, so we have double buffer for ctaRowMax.
// //  3. Protected with dedicated backward barriers (xFwdBarriers +
// ctaRowmaxBwdBarriers). Then we don't have drawbacks of 1 or 2, but we need
// extra smem barriers and extra arrive/wait instructions.
// //  4. No protection, just use volatile read/write. This approach gives most
// timely update and has lowest cost, but the result is non-deterministic up to
// an small numeric error.
// // #define CTA_ROW_MAX_BACKWARD_METHOD 4
// // 1 is 8% slower than 4. 2/3 are 10% slower than 4.
// #define CTA_ROW_MAX_BACKWARD_METHOD 1

// static_assert(inputElemSize >= cacheElemSize);

// constexpr uint32_t cacheElemsPerGrain = exactDiv(grainBytes, cacheElemSize);
// constexpr uint32_t inputElemsPerGrain = exactDiv(grainBytes, inputElemSize);
// constexpr bool enableMicroFastPath = false;

// // x: horizontal stacking for cta horizontal tile size
// // y: vertical stacking for cta vertical tile size
// // z: must be 2 for warp specialization.
// constexpr uint3 ctaShapeInWarps = {4, 1, 2};

// static_assert(ctaShapeInWarps.z == 2); // for warp specialization
// constexpr uint32_t nbWarpsPerCta = ctaShapeInWarps.x * ctaShapeInWarps.y *
// ctaShapeInWarps.z; constexpr uint32_t ctaSize = warp_size * nbWarpsPerCta;

// constexpr uint32_t nbValidRows = headGrpSize * beamWidth;
// constexpr uint2 warpTile = {64, roundUp(nbValidRows, 16U)};
// static_assert(nbValidRows <= warpTile.y);

// constexpr uint32_t gemm1WarpsPerGrp = exactDiv(headElems, warpTile.x);
// constexpr uint32_t gemm1NbWarpGrps = exactDiv(ctaShapeInWarps.x,
// gemm1WarpsPerGrp); // warp groups split along seqLen dim.

// constexpr uint2 ctaTile = {
//     warpTile.x * ctaShapeInWarps.x, // if .x is greater than headSize, then
//     gemm1 uses split-K warpTile.y * ctaShapeInWarps.y
// };

// constexpr uint32_t cvtExpansion = exactDiv(inputElemSize, cacheElemSize);



// std::unique_ptr<DecoderXQAImplPrecompiled> decoder;
template <typename T, Data_type CACHE_T>
void xqa_paged_attention_launcher(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_value_cache,
    int64_t num_heads_dummy, int64_t num_kv_heads, int64_t rotary_embedding_dim,
    double scale, torch::Tensor& block_tables, torch::Tensor& seq_lens,
    int64_t block_size, int64_t max_seq_len, const std::string kv_cache_dtype,
    double k_scale, double v_scale) {
  /************** ************************/

  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);

  // int device;
  // checkCuda(cudaGetDevice(&device));
  // cudaDeviceProp prop;
  // checkCuda(cudaGetDeviceProperties(&prop, device));
  // const uint32_t nbSubSeqPerSeq = 1;  // no multi-block


  auto device = query.device();
  const auto stream = at::cuda::getCurrentCUDAStream(device.index());

  float const kScale = k_scale;
  float const vScale = v_scale;

  uint32_t seqLen = max_seq_len;
  uint32_t tokensPerPage = block_size;  // 16
#if USE_PAGED_KV_CACHE
  size_t maxSeqLen = roundUp(seqLen, tokensPerPage);  // max_num_blocks_per_seq
#else
  size_t maxSeqLen = seqLen;
#endif
  uint32_t nbKHeads = num_kv_heads;
  uint32_t nbVHeads = nbKHeads;
  // uint32_t nbQHeads = nbKHeads * HEAD_GRP_SIZE;
  // printf("**********cacheElemSize************ = %u \n", cacheElemSize);

  // uint32_t const totalNbCacheHeads =
  //     (nbKHeads + nbVHeads) * maxSeqLen * num_seqs;
  // size_t const totalNbCacheElems =
  //     validElemsPerHead * size_t(totalNbCacheHeads);
  // size_t const qElems = validElemsPerHead * nbQHeads * num_seqs;
  // size_t const outElems = validElemsPerHead * nbQHeads * num_seqs;
  // size_t const cacheBytes = sizeof(CACHE_T) * totalNbCacheElems;
  // size_t const inputBytes = sizeof(T) * qElems;
  // size_t const outputBytes = sizeof(T) * outElems;
  // size_t const seqLenListBytes = sizeof(uint32_t) * num_seqs;
  // size_t const ctxLenListBytes = sizeof(uint32_t) * num_seqs;
// #if USE_PAGED_KV_CACHE
//   uint32_t const nbPagesPerSeq = divUp<uint32_t>(maxSeqLen, tokensPerPage);
//   size_t const totalNbPages = nbPagesPerSeq * 2 * num_seqs;
//   size_t const pageListBytes = sizeof(int) * totalNbPages;
// #else
//   size_t const pageListBytes = 0U;
// #endif  
  auto batchSize = num_seqs;

  size_t const nbSeq = nbKHeads * num_seqs;  // what is this?



  int const beamwidth = num_seqs / batchSize;  // always 1

  auto qHeads = reinterpret_cast<T*>(query.data_ptr());
  auto output = reinterpret_cast<T*>(out.data_ptr());

  auto cacheHeads = reinterpret_cast<void*>(key_value_cache.data_ptr());
  auto pageListPtr= block_tables.data_ptr<KVCachePageIndex>();
  auto seqLenPtr = reinterpret_cast<SeqLenDataType const*>(seq_lens.data_ptr<int>());
  uint32_t const maxNbPagesPerSeq = exactDiv(maxSeqLen, tokensPerPage);

  printf("maxNbPagesPerSeq= %u tokensPerPage= %u, seqLen= %p\n",
         maxNbPagesPerSeq, tokensPerPage, seqLen);
  KVCacheListParams kvcacheList(cacheHeads, pageListPtr, seqLenPtr, maxNbPagesPerSeq);

  if (true) {
    printf("cacheHeads= %p q= %p output= %p\n", cacheHeads, qHeads,
           output);
    printf("maxSeqLen= %u, \n", maxSeqLen);
    // std::cout << "nbPagesPerSeq:" << nbPagesPerSeq << std::endl;
    printf("generating input data\n");
  }


  // auto scratch = reinterpret_cast<uintptr_t>(scratchBuf.get());
/*
  CUmodule cuModule{0};
  cuErrCheck(cuModuleLoadData(
      &cuModule,
      // xqa_kernel_dt_fp16_d_128_beam_1_kvt_fp16_pagedKV_16_nqpkv_8_m_8_sm_90_cubin)); //fp16
      xqa_kernel_dt_fp16_d_128_beam_1_kvt_e4m3_pagedKV_32_nqpkv_8_m_8_sm_90_cubin)); //e4m3


  TORCH_CHECK(cuModule != nullptr, "cuModule wasn't loaded properly");
  CUfunction xqaFunc = nullptr;
  cuErrCheck(cuModuleGetFunction(&xqaFunc, cuModule, "kernel_mha"));
  TORCH_CHECK(xqaFunc != nullptr, "kernel_mha wasn't loaded properly");
*/


  // Create the DecoderXQAImplPrecompiled object
  // static std::unique_ptr<DecoderXQAImplPrecompiled> decoder = DecoderXQAImplPrecompiled::create();
  // if (nullptr == decoder) {
  //   decoder = DecoderXQAImplPrecompiled::create();
  // }
  // static std::unique_ptr<DecoderXQARunner> runner = std::make_unique<DecoderXQARunner>();


  auto io_type = TypeToDataType<T>::value;
  bool use_multi_block = true;
  XQAParams xqa_params;

  xqa_params.data_type = io_type;
  xqa_params.kv_cache_data_type = CACHE_T;

  xqa_params.output = output;
  xqa_params.qHeads = qHeads;
  xqa_params.batch_size = batchSize;
  
  xqa_params.generation_input_length = maxSeqLen;
  xqa_params.layer_idx = 0;
  xqa_params.num_q_heads = num_heads;
  xqa_params.num_kv_heads = nbKHeads;
  xqa_params.beam_width = beamwidth;
  xqa_params.head_size = head_size;
  xqa_params.tokens_per_block = tokensPerPage;
  xqa_params.max_blocks_per_sequence = maxNbPagesPerSeq;
  xqa_params.multi_block_mode = use_multi_block;
  xqa_params.timestep = 1024;

  std::cout << "In xqaKernel.cu, iotype=" << io_type<<std::endl;
  DecoderXQARunner runner(io_type, num_heads, nbKHeads, head_size, use_multi_block);
  // int max_context_length = 4096;
  // int max_num_tokens = num_seqs * max_context_length;


  size_t const nbSemaphores = nbKHeads * batchSize;//roundUp<size_t>(nbSeq, 2) + 2 + nbSeq + 2;
  // auto const semaphores = ManagedMemBuf<uint32_t>(nbSemaphores);
  size_t workspace_size = runner.getWorkspaceSize(nbSemaphores); //max_num_tokens is only useful when medusa
  std::cout << "In xqaKernel.cu, workspace_size=" << workspace_size<<std::endl;  
  // size_t const scratchSize = workspace_size;//(256u << 20);
  // auto const scratchBuf = ManagedMemBuf<std::byte>(scratchSize);
  auto const kvCacheScale = ManagedMemBuf<float>(1);
  kvCacheScale[0] = kScale;  // only useful when fp8 cache

  // auto prefetchToDevice = [&](int dev) {
  //   semaphores.prefetch(dev, stream);
  //   // scratchBuf.prefetch(dev, stream);
  //   kvCacheScale.prefetch(dev, stream);
  // };
  // prefetchToDevice(device);
 
  torch::Tensor ws_tr = torch::empty(workspace_size, torch::TensorOptions().dtype(torch::kU8).device(device));


  // checkCuda(cudaMemsetAsync(semaphores.get(), 0, 1 * nbSemaphores, stream));
  // checkCuda(cudaStreamSynchronize(stream)) ;

  xqa_params.kv_scale_quant_orig = kvCacheScale.get();
  auto wk_ptr = ws_tr.mutable_data_ptr();
  xqa_params.semaphores = (uint32_t*)(wk_ptr);//semaphores.get();
  xqa_params.workspaces = (void*)(xqa_params.semaphores + nbSemaphores);//scratchBuf.get();


  // decoder.get()->runWithKVBlockArray(xqa_params, kvcacheList, stream) ;

  runner.dispatch(xqa_params, kvcacheList, stream);
  return;

  // static uint32_t const hostSmemSize = [&]() {
  //   uint32_t size;
  //   // Populate mSharedMemBytes.
  //   CUdeviceptr shmem_dev_ptr = 0;
  //   cuErrCheck(
  //       cuModuleGetGlobal(&shmem_dev_ptr, nullptr, cuModule, "smemSize"));
  //   cuErrCheck(cuMemcpyDtoH(&size, shmem_dev_ptr, sizeof(unsigned int)));
  //   cuErrCheck(cuFuncSetAttribute(
  //       xqaFunc, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, size));
  //   return size;
  // }();
  // // printf("hostSmemSize= %u\n", hostSmemSize);
  // bool isGmmaKernel = false;
  // auto kvCacheScalePtr = kvCacheScale.get();
  // auto semaphoresPtr = semaphores.get();
  // void* args[] = {&nbKHeads,         (void*)&output,        (void*)&qHeads,
  //                 (void*)&kvcacheList, &batchSize,     &kvCacheScalePtr,
  //                 &semaphoresPtr,    (void*)&scratch};
  // // https://github.com/NVIDIA/TensorRT-LLM/blob/8681b3a4c0ccc1028bb48d83aacbb690af8f55e7/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplPrecompiled.cpp#L301C54-L301C111
  // cuErrCheck(cuLaunchKernel(xqaFunc, nbSubSeqPerSeq, nbKHeads, batchSize, 128,
  //                           1, isGmmaKernel ? 3 : 2, hostSmemSize, stream, args,
  //                           /*extra=*/0));
}

// NOTE(shuw): XQA only support block_size,
// 16, 32, 64, 128
#define CALL_XQA_LAUNCHER(T, CACHE_T_ENUM )         \
  xqa_paged_attention_launcher<T, CACHE_T_ENUM>(  \
      out, query, key_value_cache, num_heads, num_kv_heads, \
      rotary_embedding_dim, scale, block_tables, seq_lens, block_size, \
      max_seq_len, kv_cache_dtype, k_scale, v_scale);

// The following macro is used to dispatch the conversion function based on
// the data type of the key and value cache. The FN is a macro that calls a
// function with template<typename scalar_t, typename cache_t>
#define DISPATCH_BY_KV_CACHE_ELEM_ENUM(SRC_DTYPE, KV_DTYPE, FN) \
if (KV_DTYPE == "auto") { \
  if (SRC_DTYPE == at::ScalarType::Half) { \
    FN(half, Data_type::DATA_TYPE_FP16); \
  } else if (SRC_DTYPE == at::ScalarType::BFloat16) { \
    FN(__nv_bfloat16, Data_type::DATA_TYPE_FP16); \
  } else { \
    TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
  } \
} else { \
  if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") { \
    if (SRC_DTYPE == at::ScalarType::Half) { \
      std::cout << "half is op and kv dtype is e4m3\n"; \
      FN(half, Data_type::DATA_TYPE_E4M3); \
    } else if (SRC_DTYPE == at::ScalarType::BFloat16) { \
    std::cout << "bf16 is op and kv dtype is e4m3\n"; \
      FN(__nv_bfloat16, Data_type::DATA_TYPE_E4M3); \
    } else { \
      TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
    } \
  } else { \
    TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE); \
  } \
}


void xqa_paged_attention(torch::Tensor& out, torch::Tensor& query,
                         torch::Tensor& key_value_cache, int64_t num_heads,
                         int64_t num_kv_heads, int64_t rotary_embedding_dim,
                         double scale, torch::Tensor& block_tables,
                         torch::Tensor& seq_lens, int64_t block_size,
                         int64_t max_seq_len, const std::string kv_cache_dtype,
                         double k_scale, double v_scale) {
  DISPATCH_BY_KV_CACHE_ELEM_ENUM(query.dtype(), kv_cache_dtype, CALL_XQA_LAUNCHER);
}


// XQA kernel: kv cache types: f16/bf16, int8, fp8_e4m3
// XQA kernel support list
    // # for llama v2 70b
    // [
    //     CompileMacroOption('DTYPE', 'dt', ['__half', '__nv_bfloat16']),
    //     CompileMacroOption('HEAD_ELEMS', 'd', [128, 256]),
    //     CompileMacroOption('BEAM_WIDTH', 'beam', [1]),
    //     CompileMacroOption('CACHE_ELEM_ENUM', 'kvt', [0, 1, 2]),
    //     CompileMacroOption('TOKENS_PER_PAGE', 'pagedKV', [0, 16, 32, 64, 128]), # 0 denotes contiguous kv cache.
    //     CompileMacroOption('HEAD_GRP_SIZE', 'nqpkv', [8]),
    //     CompileMacroOption('M_TILESIZE', 'm', [8]),
    // ],
