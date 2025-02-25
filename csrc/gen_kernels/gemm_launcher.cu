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
#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <nvrtc.h>

#include "cuda_compat.h"

#include "dispatch_utils.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include "gemm_runner.h"

template <Data_type OUT_DTYPE>
void gen_w8a8_block_fp8_matmul_launcher(torch::Tensor& out, torch::Tensor const& a,
                                        torch::Tensor const& b, torch::Tensor const& As,
                                        torch::Tensor const& Bs, int64_t block_n,
                                        int64_t block_k) {
  int m = a.size(0);
  int n = b.size(0);
  int k = a.size(1);
  auto ptrA = reinterpret_cast<void*>(a.data_ptr());
  auto ptrB = reinterpret_cast<void*>(b.data_ptr());
  auto ptrAs = reinterpret_cast<float*>(As.data_ptr());
  auto ptrBs = reinterpret_cast<float*>(Bs.data_ptr());
  auto output = reinterpret_cast<void*>(out.data_ptr());
  auto mQDataType = DATA_TYPE_E4M3;
  auto mOutDataType = OUT_DTYPE;
  auto mTllmGenGemmRunner = TllmGenGemmRunner(mQDataType, mOutDataType);
  TllmGenGemmRunnerParams tllmRunnerParams;
  tllmRunnerParams.m = m;
  tllmRunnerParams.k = k;
  tllmRunnerParams.n = n;
  tllmRunnerParams.dtypeC = mOutDataType;
  tllmRunnerParams.dtypeElt = mQDataType;
  tllmRunnerParams.dtypeAcc = DATA_TYPE_FP32;
  tllmRunnerParams.oPtr = output;
  tllmRunnerParams.aPtr = ptrA;
  tllmRunnerParams.bPtr = ptrB;
  tllmRunnerParams.AsPtr = ptrAs;
  tllmRunnerParams.BsPtr = ptrBs;

  auto device = a.device();
  const auto stream = at::cuda::getCurrentCUDAStream(device.index());
  tllmRunnerParams.stream = stream;
  mTllmGenGemmRunner.run(tllmRunnerParams);
}

#define CALL_GEN_W8A8_BLOCK_FP8_MATMUL(OUT_DTYPE)                           \
  gen_w8a8_block_fp8_matmul_launcher<OUT_DTYPE>(out, a, b, As, Bs, block_n, \
                                                block_k);

#define DISPATCH_BY_OUTPUT_DTYPE(OUT_DTYPE, FN)            \
  if (OUT_DTYPE == at::ScalarType::Half) {                 \
    FN(Data_type::DATA_TYPE_FP16);                         \
  } else if (OUT_DTYPE == at::ScalarType::BFloat16) {      \
    FN(Data_type::DATA_TYPE_BF16);                         \
  } else if (OUT_DTYPE == at::ScalarType::Float) {         \
    FN(Data_type::DATA_TYPE_FP32);                         \
  } else {                                                 \
    TORCH_CHECK(false, "Unsupported data type of output"); \
  }

void gen_w8a8_block_fp8_matmul(torch::Tensor& out,
		               torch::Tensor const& a,
                               torch::Tensor const& b,
			       torch::Tensor const& As,
                               torch::Tensor const& Bs,
			       int64_t block_n,
                               int64_t block_k) {
  DISPATCH_BY_OUTPUT_DTYPE(out.dtype(), CALL_GEN_W8A8_BLOCK_FP8_MATMUL);
}
