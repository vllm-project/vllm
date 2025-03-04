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

#pragma once

#include <cuda_runtime.h>

#include "gemm_kernels.h"
#include "gemm_runner_params.h"

class TllmGenGemmRunner {
 public:
  // Constructor.
  explicit TllmGenGemmRunner(Data_type dtypeQ, Data_type dtypeOut);

  TllmGenGemmRunner() = default;

  // Check if gemm is supported.
  bool isSupported(TllmGenGemmRunnerParams const& runnerParams) const;

  std::pair<bool, std::string> isSupportedWithInfo(
      TllmGenGemmRunnerParams const& runnerParams) const;

  void run(TllmGenGemmRunnerParams const&);

 private:
  // The input/output datatype.
  Data_type mDtypeQ;
  Data_type mDtypeOut;
  // The SM version.
  int mSM;
  // The class that stores all the kernels.
  TllmGenGemmKernel const* mKernel;
};

