/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "cuda_runtime_api.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "cuda_utils.h"

#include "gemm_options.h"
#include "cubin/kernel_metainfo.h"
#include "gemm_runner_params.h"
#include "kernel_params.h"

inline int divUp(int a, int b) { return (a + b - 1) / b; }

template <typename T>
std::string toString(T e) {
  return std::to_string(e);
}

template <>
inline std::string toString(Dtype e) {
  return dtypeToString(e);
}
inline Dtype toTgDtype(Data_type dtype) {
  switch (dtype) {
    case DATA_TYPE_FP16:
      return Dtype::Fp16;
    case DATA_TYPE_BF16:
      return Dtype::Bfloat16;
    case DATA_TYPE_E4M3:
      return Dtype::E4m3;
    case DATA_TYPE_FP32:
      return Dtype::Fp32;
    default:
      return Dtype::Void;
  }
}

class TllmGenGemmKernel {
 public:
  using KernelMeta = TllmGenGemmKernelMetaInfo;
  using RunnerParams = TllmGenGemmRunnerParams;

  // Ctor.
  TllmGenGemmKernel(KernelMeta const* pMetaStart, unsigned int nMetaCount,
                    Data_type dtypeQ, Data_type dtypeOut, unsigned int smArch)
      : mDtypeQ(dtypeQ),
        mDtypeOut(dtypeOut),
        mKernelMeta(pMetaStart),
        mKernelMetaCount(nMetaCount),
        mSM(smArch) {}

  void loadKernels() {
    if (!mFunctions.empty()) {
      return;
    }
    for (unsigned int i = 0; i < mKernelMetaCount; ++i) {
      auto const& kernelMeta = mKernelMeta[i];
      if (kernelMeta.mSM == mSM && kernelMeta.mdtypeElt == mDtypeQ &&
          kernelMeta.mdtypeC == mDtypeOut) {
        CUmodule hmod{0};
        auto findModuleIter = mModules.find(kernelMeta.mCubin);
        if (findModuleIter != mModules.end()) {
          hmod = findModuleIter->second;
        } else {
          cuErrCheck(cuModuleLoadData(&hmod, kernelMeta.mCubin));
          mModules.insert(std::make_pair(kernelMeta.mCubin, hmod));
        }

        // Build a hash map, which maps from kernel meta info to kernel index
        KernelInfo funcInfo{};
        funcInfo.mMetaInfoIndex = i;
        funcInfo.mKernelName = kernelMeta.mFuncName;
        cuErrCheck(cuModuleGetFunction(&(funcInfo.mDeviceFunction), hmod,
                                       kernelMeta.mFuncName));
        assert(funcInfo.mDeviceFunction != nullptr);
        funcInfo.mSharedMemBytes = kernelMeta.mSharedMemBytes;
        if (kernelMeta.mSharedMemBytes > 48 * 1024) {
          cuErrCheck(cuFuncSetAttribute(
              funcInfo.mDeviceFunction,
              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
              kernelMeta.mSharedMemBytes));
        }
        mFunctions.insert(std::make_pair(hashID(kernelMeta), funcInfo));
      }
    }
  }

  inline uint64_t hashID(int dtypeC, int dtypeElt, int dtypeAcc) const {
    // Format of the hash key:
    // shuw: not correct, improve later
    return (static_cast<uint64_t>(dtypeC) << 0) |
           (static_cast<uint64_t>(dtypeElt) << 4) |
           (static_cast<uint64_t>(dtypeAcc) << 8);
  }

  uint64_t hashID(KernelMeta const& kernelMeta) const {
    return hashID(kernelMeta.mdtypeC, kernelMeta.mdtypeElt,
                  kernelMeta.mdtypeAcc);
  }

  std::pair<bool, std::string> checkIfKernelExist(
      RunnerParams const& params) const {
    auto [hashId, info] = hashFromRunnerParams(params);
    return std::make_pair(mFunctions.find(hashId) != mFunctions.end(), info);
  }

  void run(RunnerParams const& params) const {
    gemm::GemmOptions options;
    options.mM = params.m;
    options.mK = params.k;
    options.mN = params.n;
    options.mDtypeElt = toTgDtype(params.dtypeElt);
    options.mDtypeAcc = toTgDtype(params.dtypeAcc);
    options.mDtypeC = toTgDtype(params.dtypeC);
    // shuw(TODO):hard-coded
    options.mTileM = 128;
    options.mTileN = 128;
    options.mTileK = 128;
    options.mMmaM = 128;
    options.mMmaN = 64;
    options.mMmaK = 32;
    options.mEpilogueTileM = 128;
    options.mEpilogueTileN = 64;
    options.mClusterX = 1;
    options.mClusterY = 1;

    auto [hash, info] = hashFromRunnerParams(params);
    auto const findIter = mFunctions.find(hash);
    TORCH_CHECK(findIter != mFunctions.end(), "Kernel not found.");

    const CUfunction func = findIter->second.mDeviceFunction;
    assert(func != nullptr);

    int rank = 0;
    int worldSize = 1;

    // The input matrix A. The shape is m x k. Layout is row-major (contiguous
    // in the k dimension).
    void const* dPtrA = params.aPtr;
    // The input matrix B. The shape is n x k. Layout is row-major (contiguous
    // in the k dimension).
    void const* dPtrB = params.bPtr;
    // The temporary helper matrix matrix C for split-k slices. The shape is
    // split_k x m x n. Layout is row-major (contiguous in the n dimension).
    // The output matrix C. The shape is m x n. Layout is row-major (contiguous
    // in the n dimension).
    // The output matrix C reference on device.
    // The dequant factors for A when using DeepSeek FP8 recipe.
    float const* dDqSfsA = params.AsPtr;
    // The dequant factors for B when using DeepSeek FP8 recipe.
    float const* dDqSfsB = params.BsPtr;

    // shuw(TODO): do we shuffer A
    // // Allocate memory to swizzle the A matrix on the host (when we transpose
    // the output).
    // The number of bytes used by the dequantized vals (on the host) to
    // implement DeepSeek FP8.
    

    cudaStream_t cudaStream{params.stream};
    auto kernelParams = gemm::KernelParams::setKernelParams(
        options, dPtrA, dDqSfsA, dPtrB, dDqSfsB, params.oPtr,
        nullptr,                    // shuw(TODO):memHandleC.getPtr()?
        nullptr, nullptr, nullptr,  // memHandleSplitK, memHandleSplitK,
        nullptr, nullptr, nullptr, nullptr, nullptr, rank, worldSize);
    // The size of the grid.
    std::vector<int32_t> grid{divUp(options.mM, options.mTileM),
                              divUp(options.mN, options.mTileN),
                              options.mNumSlicesForSplitK};

    // Prepare kernel parameters list for cuLaunchKernelEx.
    void* kernelParamsList[] = {&kernelParams};
    CUlaunchConfig launch_config;
    launch_config.blockDimX = 320;  // shuw(TODO)
    launch_config.blockDimY = 1;
    launch_config.blockDimZ = 1;
    launch_config.gridDimX = divUp(options.mM, options.mTileM);
    launch_config.gridDimY = divUp(options.mN, options.mTileN);
    launch_config.gridDimZ = options.mNumSlicesForSplitK;

    launch_config.hStream = reinterpret_cast<CUstream>((void*)cudaStream);
    launch_config.sharedMemBytes = findIter->second.mSharedMemBytes;

    CUlaunchAttribute launch_attribute[2];
    launch_attribute[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    launch_attribute[0].value.clusterDim.x = 1;
    launch_attribute[0].value.clusterDim.y = 1;
    launch_attribute[0].value.clusterDim.z = 1;
    launch_attribute[1].id =
        CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
    launch_attribute[1].value.clusterSchedulingPolicyPreference =
        CU_CLUSTER_SCHEDULING_POLICY_DEFAULT;

    launch_config.attrs = launch_attribute;
    launch_config.numAttrs = 2;

    cuErrCheck(
        cuLaunchKernelEx(&launch_config, func, kernelParamsList, nullptr));
  }

 private:
  std::pair<uint64_t, std::string> hashFromRunnerParams(
      RunnerParams const& params) const {
    std::string info = "dtypeC=" + toString(toTgDtype(params.dtypeC)) +
                       ", dtypeElt=" + toString(toTgDtype(params.dtypeElt)) +
                       ", dtypeAcc=" + toString(toTgDtype(params.dtypeAcc));
    return std::make_pair(
        hashID(params.dtypeC, params.dtypeElt, params.dtypeAcc), info);
  }

  Data_type mDtypeQ;
  Data_type mDtypeOut;
  KernelMeta const* mKernelMeta;
  unsigned int mKernelMetaCount;
  unsigned int mSM;
  std::unordered_map<unsigned long long const*, CUmodule> mModules;

  struct KernelInfo {
    unsigned int mMetaInfoIndex;
    unsigned int mSharedMemBytes;
    const char* mKernelName;
    CUfunction mDeviceFunction;
  };

  std::unordered_map<uint64_t, KernelInfo> mFunctions;
};

class TllmGemmKernelFactory {
 public:
  using KernelType = TllmGenGemmKernel;

  KernelType const* getKernels(
      const typename KernelType::KernelMeta* pKernelList,
      unsigned int nbKernels, Data_type dtypeQ, Data_type dtypeOut,
      unsigned int sm) {
    static std::mutex s_mutex;
    std::lock_guard<std::mutex> lg(s_mutex);

    auto const id = hashID(dtypeQ, dtypeOut, sm);
    auto const findIter = mKernels.find(id);
    if (findIter == mKernels.end()) {
      KernelType* newKernel =
          new KernelType{pKernelList, nbKernels, dtypeQ, dtypeOut, sm};
      newKernel->loadKernels();
      mKernels.insert(
          std::make_pair(id, std::unique_ptr<KernelType>(newKernel)));
      return newKernel;
    }
    return findIter->second.get();
  }

  static TllmGemmKernelFactory& Get() {
    int deviceId;
    cudaGetDevice(&deviceId);
    static std::unique_ptr<TllmGemmKernelFactory> sFactory[32] = {nullptr};
    if (sFactory[deviceId] == nullptr) {
      TORCH_CHECK(deviceId < 32, "Invalid deviceId %d", deviceId);
      sFactory[deviceId] =
          std::make_unique<TllmGemmKernelFactory>(TllmGemmKernelFactory());
    }
    return *(sFactory[deviceId]);
  }

 private:
  TllmGemmKernelFactory() = default;

  inline uint64_t hashID(Data_type dtypeQ, Data_type dtypeOut,
                         unsigned int sm) const {
    return static_cast<uint64_t>(sm) | static_cast<uint64_t>(dtypeQ) << 16 |
           static_cast<uint64_t>(dtypeOut) << 20;
  }

  std::unordered_map<uint64_t, const std::unique_ptr<KernelType>> mKernels;
};

inline TllmGenGemmKernel const* getTllmGemmKernels(Data_type dtypeQ,
                                                   Data_type dtypeOut,
                                                   unsigned int sm) {
#ifndef EXCLUDE_SM_100
  return TllmGemmKernelFactory::Get().getKernels(
      sTllmGenGemmKernelMetaInfos,
      sizeof(sTllmGenGemmKernelMetaInfos) /
          sizeof(sTllmGenGemmKernelMetaInfos[0]),
      dtypeQ, dtypeOut, sm);
#else
  return nullptr;
#endif  // EXCLUDE_SM_100
}
