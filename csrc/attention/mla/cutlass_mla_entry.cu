/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <torch/all.h>

#if defined ENABLE_CUTLASS_MLA && ENABLE_CUTLASS_MLA
void cutlass_mla_decode_sm100a(torch::Tensor const& out,
                               torch::Tensor const& q_nope,
                               torch::Tensor const& q_pe,
                               torch::Tensor const& kv_c_and_k_pe_cache,
                               torch::Tensor const& seq_lens,
                               torch::Tensor const& page_table, double scale);
#endif

void cutlass_mla_decode(torch::Tensor const& out, torch::Tensor const& q_nope,
                        torch::Tensor const& q_pe,
                        torch::Tensor const& kv_c_and_k_pe_cache,
                        torch::Tensor const& seq_lens,
                        torch::Tensor const& page_table, double scale) {
#if defined ENABLE_CUTLASS_MLA && ENABLE_CUTLASS_MLA
  return cutlass_mla_decode_sm100a(out, q_nope, q_pe, kv_c_and_k_pe_cache,
                                   seq_lens, page_table, scale);
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No compiled cutlass MLA");
}
