// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Adapted from SGLang (Apache-2.0). The kernel is SGLang's rewrite of
// DeepSelect's bf16 cluster top-k, vendored in vllm/third_party/deep_select
// from:
// https://github.com/sgl-project/sglang/tree/656331295619a4d0267b40b7da0b5cd5b549fcbc/python/sglang/kernels/jit/csrc/deep_select/vendor
// Its configuration follows SGLang's launcher:
// https://github.com/sgl-project/sglang/blob/656331295619a4d0267b40b7da0b5cd5b549fcbc/python/sglang/kernels/ops/attention/deep_select.py
//
// The op `cluster_topk.py` builds; it writes request-local indices.

#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include "cuda_kernels/v3_cluster/topk_select.cuh"

namespace {

// The launcher's cluster tuning: int32 indices only, unsorted, the 1024 top-k
// bucket, 512 threads, no page transform.
template <uint32_t kCluster>
using Config = TopkSelectConfig<nv_bfloat16, int32_t, false, false, false, 1024,
                                512, 1, 8192, 4096, 5, 512, kCluster>;

void topk(const at::Tensor& logits, const at::Tensor& lengths, at::Tensor& out,
          int64_t cluster, int64_t smem_per_sm) {
  TopkSelectArgs args{};
  args.batch_size = logits.size(0);
  args.vocab_size = logits.size(1);
  args.topk = out.size(1);
  args.input = logits.data_ptr();
  args.output_index = out.data_ptr();
  args.end_ptr = lengths.data_ptr<int>();
  args.stride_input_batch = logits.stride(0);
  args.stride_output_index_batch = out.stride(0);
  args.idx_oob_fill_value = -1;
  args.shared_memory_size_per_sm = smem_per_sm;
  args.stream = c10::cuda::getCurrentCUDAStream(logits.get_device()).stream();
  if (cluster == 16) {
    topk_select_bf16_cluster::run_topk_select_kernel<Config<16>>(args);
  } else {
    TORCH_CHECK(cluster == 8, "cluster size ", cluster);
    topk_select_bf16_cluster::run_topk_select_kernel<Config<8>>(args);
  }
}

}  // namespace

TORCH_LIBRARY(sd_cluster_topk, m) {
  m.def(
      "topk(Tensor logits, Tensor lengths, Tensor(a!) out, int cluster, "
      "int smem_per_sm) -> ()",
      &topk);
}
