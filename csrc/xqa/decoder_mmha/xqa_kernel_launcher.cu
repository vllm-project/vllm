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
#include "decoder_xqa_impl_precompiled.h"
#include "decoder_xqa_runner.h"

template <typename T, Data_type CACHE_T>
void xqa_paged_attention_launcher(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_value_cache,
    int64_t num_heads_dummy, int64_t num_kv_heads, int64_t rotary_embedding_dim,
    double scale, torch::Tensor& block_tables, torch::Tensor& seq_lens,
    int64_t block_size, int64_t max_seq_len, const std::string kv_cache_dtype,
    double k_scale, double v_scale) {

  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);

  auto device = query.device();
  const auto stream = at::cuda::getCurrentCUDAStream(device.index());

  float const kScale = k_scale;
  float const vScale = v_scale;

  uint32_t seqLen = max_seq_len;
  uint32_t tokensPerPage = block_size;

  size_t maxSeqLen = roundUp(seqLen, tokensPerPage);  // max_num_blocks_per_seq

  uint32_t nbKHeads = num_kv_heads;
  uint32_t nbVHeads = nbKHeads;
  auto batchSize = num_seqs;

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

  DecoderXQARunner runner(io_type, num_heads, nbKHeads, head_size, use_multi_block);

  size_t const nbSemaphores = nbKHeads * batchSize;
  auto const semaphores = ManagedMemBuf<uint32_t>(nbSemaphores);
  size_t workspace_size = runner.getWorkspaceSize(0); //max_num_tokens is only useful when medusa
  auto const kvCacheScale = ManagedMemBuf<float>(1);
  kvCacheScale[0] = kScale;  // only useful when fp8 cache

  auto prefetchToDevice = [&](int dev) {
    semaphores.prefetch(dev, stream);
    kvCacheScale.prefetch(dev, stream);
  };
  prefetchToDevice(device.index());
  
  checkCuda(cudaMemsetAsync(semaphores.get(), 0, 1 * nbSemaphores, stream));
  checkCuda(cudaStreamSynchronize(stream));
  xqa_params.kv_scale_quant_orig = kvCacheScale.get();
  xqa_params.semaphores = semaphores.get();
  torch::Tensor ws_tr = torch::empty(workspace_size, torch::TensorOptions().dtype(torch::kU8).device(device));
  auto wk_ptr = ws_tr.mutable_data_ptr();
  xqa_params.workspaces = (void*)(wk_ptr);

  runner.dispatch(xqa_params, kvcacheList, stream);
  return;
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
      FN(half, Data_type::DATA_TYPE_E4M3); \
    } else if (SRC_DTYPE == at::ScalarType::BFloat16) { \
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
