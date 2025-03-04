/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "gemm_runner.h"

TllmGenGemmRunner::TllmGenGemmRunner(Data_type dtypeQ, Data_type dtypeOut)
    : mSM(getSMVersion()), mDtypeQ(dtypeQ), mDtypeOut(dtypeOut) {
  TORCH_CHECK(mSM == kSM_100, "Unsupported architecture");
  TORCH_CHECK(mDtypeQ == DATA_TYPE_E4M3, "Unsupported Input data type");
  // E4M3 is supported but not used in deekSeek, thus exclude.ff
  TORCH_CHECK((mDtypeOut == DATA_TYPE_FP32 || mDtypeOut == DATA_TYPE_FP16 ||
               mDtypeOut == DATA_TYPE_BF16),
              "Unsupported Output data type");
  mKernel = getTllmGemmKernels(mDtypeQ, mDtypeOut, mSM);
}

void TllmGenGemmRunner::run(TllmGenGemmRunnerParams const& runnerParams) {
  mKernel->run(runnerParams);
}

bool TllmGenGemmRunner::isSupported(
    TllmGenGemmRunnerParams const& runnerParams) const {
  return mKernel->checkIfKernelExist(runnerParams).first;
}

std::pair<bool, std::string> TllmGenGemmRunner::isSupportedWithInfo(
    TllmGenGemmRunnerParams const& runnerParams) const {
  return mKernel->checkIfKernelExist(runnerParams);
}
