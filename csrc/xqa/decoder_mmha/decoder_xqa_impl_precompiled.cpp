#include "env_utils.h"
#include "decoder_xqa_impl_precompiled.h"
#include <cuda.h>
#include <functional>
#include <memory>
#include <mutex>
#include "cubin/xqa_kernel_cubin.h"
#include "decoder_xqa_runner.h"
uint32_t getElemBytes(CUtensorMapDataType_enum dataType) {
  switch (dataType) {
    case CU_TENSOR_MAP_DATA_TYPE_UINT8:
      return 1;
    case CU_TENSOR_MAP_DATA_TYPE_UINT16:
      return 2;
    case CU_TENSOR_MAP_DATA_TYPE_UINT32:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_INT32:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_UINT64:
      return 8;
    case CU_TENSOR_MAP_DATA_TYPE_INT64:
      return 8;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
      return 2;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
      return 8;
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
      return 2;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ:
      return 4;
  }
  throw std::runtime_error("unsupported data type");
}

CUtensorMap makeTensorMapForPagedKVCache(void const* addr,
                                         CUtensorMapDataType_enum dataType,
                                         uint32_t headElems, uint32_t nbKHeads,
                                         uint32_t tokensPerPage,
                                         uint32_t nbTokensPerTile = 64) {
  CUtensorMap tensorMap{};
  uint32_t elemBytes = getElemBytes(dataType);
  uint64_t const globalDims[] = {headElems, tokensPerPage, nbKHeads, 1U << 31};
  uint32_t const headBytes = elemBytes * headElems;
  uint64_t const globalStrides[] = {headBytes, headBytes * tokensPerPage,
                                    headBytes * tokensPerPage * nbKHeads};
  TORCH_CHECK(headElems <= 256);
  uint32_t const paddedHeadElems =
      headElems <= 64 ? 64 : (headElems <= 128 ? 128 : 256);
  uint32_t const partElems =
      std::min(elemBytes * paddedHeadElems, 128U) / elemBytes;
  uint32_t const boxDims[] = {partElems,
                              std::min(tokensPerPage, nbTokensPerTile), 1, 1};
  uint32_t const elemStrides[] = {1, 1, 1, 1};

  auto const swizzle = [&] {
    switch (partElems) {
      case 128:
        return CU_TENSOR_MAP_SWIZZLE_128B;
      case 64:
        return CU_TENSOR_MAP_SWIZZLE_64B;
      default:
        throw std::runtime_error("unsupported cache head size");
        // default: TLLM_THROW("unsupported cache head size");
    }
  }();

  cuErrCheck(cuTensorMapEncodeTiled(
      &tensorMap, dataType, 4, const_cast<void*>(addr), globalDims,
      globalStrides, boxDims, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tensorMap;
}

CUtensorMap makeTensorMapForKVCache(XQAParams const& xqaParams,
                                    KVCacheListParams const& kv_cache_buffer) {
  return makeTensorMapForPagedKVCache(
      kv_cache_buffer.pool, CU_TENSOR_MAP_DATA_TYPE_UINT8, xqaParams.head_size,
      xqaParams.num_kv_heads, xqaParams.tokens_per_block);
}

class XQAKernelList {
 public:
  using TKernelMeta = XQAKernelMetaInfo;

  XQAKernelList(Data_type type, unsigned int sm)
      : mDataType(type),
        mKernelMetaCount(sizeof(sXqaKernelMetaInfo) /
                         sizeof(sXqaKernelMetaInfo[0])),
        mKernelMeta(&sXqaKernelMetaInfo[0]),
        mSM(sm) {
    mForceXQA = forceXQAKernels();
  }

  void loadXQAKernels() {
    std::cout << "entering load XQA Kernels\n";
    if (!mFunctions.empty()) {
      return;
    }
    std::cout << "here mKernelMetaCount=" << mKernelMetaCount << std::endl;
    for (unsigned int i = 0; i < mKernelMetaCount; ++i) {
      auto const& kernelMeta = mKernelMeta[i];
      // std::cout << "00000000000000\n";
      // std::cout << kernelMeta.mSM << "; " << kernelMeta.mDataType <<
      // std::endl; std::cout << mSM << "; " << mDataType << std::endl;
      if (kernelMeta.mSM != mSM || kernelMeta.mDataType != mDataType) continue;

      // Cubins for kernels that would take the JIT path are removed from
      // kernelMeta.
      if (kernelMeta.mCubin == nullptr) continue;
      // std::cout << "11111111111111\n";
      CUmodule hmod{0};
      auto findModuleIter = mModules.find(kernelMeta.mCubin);
      if (findModuleIter != mModules.end()) {
        hmod = findModuleIter->second;
      } else {
        cuErrCheck(cuModuleLoadData(&hmod, kernelMeta.mCubin));
        mModules.insert(std::make_pair(kernelMeta.mCubin, hmod));
      }

      XQAKernelFuncInfo funcInfo{};
      funcInfo.mMetaInfoIndex = i;
      cuErrCheck(cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod,
                                     kernelMeta.mFuncName));
      // std::cout << "reading mDeviceFunction:" <<funcInfo.mDeviceFunction
      // <<std::endl;
      funcInfo.mSharedMemBytes =
          getGlobalVar<uint32_t>(hmod, "smemSize", true).value();
      funcInfo.mKernelType =
          getGlobalVar<XQAKernelType>(hmod, "kernelType", false)
              .value_or(XQAKernelType::kAMPERE_WARP_SPECIALIZED);
      /* Set 46KB threshold here because we have to take static/driver shared
       * memory into consideration. */
      if (funcInfo.mSharedMemBytes >= 46 * 1024) {
        cuErrCheck(
            cuFuncSetAttribute(funcInfo.mDeviceFunction,
                               CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               funcInfo.mSharedMemBytes));
      }

      XQAKernelRuntimeHashKey hash_key{
          kernelMeta.mKVDataType,   kernelMeta.mHeadDim,
          kernelMeta.mBeamWidth,    kernelMeta.mNumQHeadsOverKV,
          kernelMeta.mMTileSize,    kernelMeta.mTokensPerPage,
          kernelMeta.mPagedKVCache, kernelMeta.mMultiQueryTokens};

      mFunctions.insert(std::make_pair(hash_key, funcInfo));
    }
  }

  bool supportConfig(XQAParams const& xqaParams) const {
    unsigned int head_size = xqaParams.head_size;
    int num_q_heads = xqaParams.num_q_heads;
    int num_kv_heads = xqaParams.num_kv_heads;
    TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                "numQHeads should be multiple of numKVHeads.");
    unsigned int num_q_heads_over_kv = num_q_heads / num_kv_heads;
    unsigned int beam_width = xqaParams.beam_width;
    // MultiQueryToken kernels can support any num_q_heads_over_kv that is power
    // of 2.
    unsigned int kernel_num_q_heads_over_kv =
        xqaParams.multi_query_tokens ? 0 : num_q_heads_over_kv;
    unsigned int m_tilesize;
    if (xqaParams.multi_query_tokens) {
      // MultiQueryToken kernels can handle either 16/32 for M direction per
      // CTA.
      m_tilesize = xqaParams.generation_input_length <= 16 ? 16 : 32;
    } else {
      m_tilesize = num_q_heads_over_kv;
    }

    XQAKernelRuntimeHashKey hash_key = {
        xqaParams.kv_cache_data_type,
        head_size,
        beam_width,
        kernel_num_q_heads_over_kv,
        m_tilesize,
        xqaParams.paged_kv_cache
            ? static_cast<unsigned int>(xqaParams.tokens_per_block)
            : 0,
        xqaParams.paged_kv_cache,
        xqaParams.multi_query_tokens};
    auto const findIter = mFunctions.find(hash_key);
    return findIter != mFunctions.end();
  }

  bool mayHavePerfGain(XQAParams const& xqaParams,
                       int multiprocessor_count) const {
    return true;
  }

  template <typename T>
  void run(XQAParams const& xqaParams, KVCacheListParams const& kv_cache_buffer,
           int multiprocessor_count, cudaStream_t const& stream) const {
    unsigned int head_size = xqaParams.head_size;
    int num_q_heads = xqaParams.num_q_heads;
    int num_kv_heads = xqaParams.num_kv_heads;
    TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                "numQHeads should be multiple of numKVHeads.");
    unsigned int num_q_heads_over_kv = num_q_heads / num_kv_heads;
    unsigned int beam_width = xqaParams.beam_width;
    unsigned int batch_beam_size = xqaParams.batch_size * beam_width;

    XQALaunchParam launchParams;

    buildXQALaunchParams(launchParams, xqaParams, kv_cache_buffer);
    void* xqa_q_input_ptr = const_cast<void*>(xqaParams.qHeads);

    XQAKernelRuntimeHashKey hash_key =
        getRuntimeHashKeyFromXQAParams(xqaParams);

    auto const findIter = mFunctions.find(hash_key);
    // std::cout << "at running mDeviceFunction:"
    // <<findIter->second.mDeviceFunction <<std::endl;
    TORCH_CHECK(findIter != mFunctions.end(), "XQAKernelFunc not found.");

    auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
    const CUfunction func = findIter->second.mDeviceFunction;
    unsigned int const shared_mem_bytes = findIter->second.mSharedMemBytes;
    auto const kernelType = findIter->second.mKernelType;

    if (false && xqaParams.multi_query_tokens) {
      // pass
    } else {
      bool const isGmmaKernel =
          (kernelType == XQAKernelType::kHOPPER_WARP_SPECIALIZED);
      TORCH_CHECK(isGmmaKernel == (mSM == kSM_90 &&
                                   xqaParams.kv_cache_data_type ==
                                       XQADataType::DATA_TYPE_E4M3 &&
                                   xqaParams.beam_width == 1));
      constexpr uint32_t kMAX_NB_KERNEL_PARAMS = 11;
      uint32_t const maxNbKernelParams = (isGmmaKernel ? 11 : 10);
      uint32_t idxNextParam = 0;
      void* kernelParams[kMAX_NB_KERNEL_PARAMS];
      auto appendParam = [&](auto* p) mutable {
        TORCH_CHECK(idxNextParam < maxNbKernelParams);
        kernelParams[idxNextParam++] = p;
      };
      appendParam(&launchParams.num_k_heads);
      appendParam(&launchParams.output);
      appendParam(&xqa_q_input_ptr);
      appendParam(&launchParams.kvCacheParams);
      appendParam(&launchParams.batch_size);
      appendParam(&launchParams.kv_scale_quant_orig);
      CUtensorMap tensorMap{};
      if (isGmmaKernel) {
        tensorMap = makeTensorMapForKVCache(xqaParams, kv_cache_buffer);
        appendParam(&tensorMap);
      }
      appendParam(&launchParams.semaphores);
      appendParam(&launchParams.scratch);
      kernelParams[idxNextParam] =
          nullptr;  // one extra nullptr at end as guard.
      int multi_block = 1;
      if (xqaParams.multi_block_mode) {
        multi_block = computeMultiBlockCount(xqaParams, xqaParams.batch_size,
                                             multiprocessor_count);
      }
      auto blockz = isGmmaKernel ? 3 : 2;
      cuErrCheck(cuLaunchKernel(func, multi_block, xqaParams.num_kv_heads,
                                xqaParams.batch_size, 128, 1,
                                isGmmaKernel ? 3 : 2, shared_mem_bytes, stream,
                                kernelParams, nullptr));
    }
  }

 protected:
  Data_type mDataType;
  TKernelMeta const* mKernelMeta;
  unsigned int mKernelMetaCount;
  unsigned int mSM;
  std::unordered_map<unsigned long long const*, CUmodule> mModules;
  bool mForceXQA = false;

  struct XQAKernelFuncInfo {
    unsigned int mMetaInfoIndex;
    unsigned int mSharedMemBytes;
    CUfunction mDeviceFunction;
    XQAKernelType mKernelType;
  };

  std::unordered_map<XQAKernelRuntimeHashKey, XQAKernelFuncInfo,
                     XQAKernelRuntimeHasher>
      mFunctions;
};

class XQAKernelLoader {
 public:
  XQAKernelList const* getXQAKernels(Data_type type, unsigned int sm) {
    static std::mutex s_mutex;
    std::lock_guard<std::mutex> lg(s_mutex);
    XQAKernelLoadHashKey hash_key{type, sm};

    auto const findIter = mKernels.find(hash_key);
    if (findIter == mKernels.end()) {
      XQAKernelList* newKernel = new XQAKernelList{type, sm};
      newKernel->loadXQAKernels();
      mKernels.insert(
          std::make_pair(hash_key, std::unique_ptr<XQAKernelList>(newKernel)));
      return newKernel;
    } else {
      return findIter->second.get();
    }
  }

  static XQAKernelLoader& Get() {
    int device_id = getDevice();
    static std::unique_ptr<XQAKernelLoader> s_factory[32] = {nullptr};
    if (s_factory[device_id] == nullptr) {
      assert(device_id <= 32);
      s_factory[device_id] =
          std::make_unique<XQAKernelLoader>(XQAKernelLoader());
    }

    return *(s_factory[device_id]);
  }

 private:
  XQAKernelLoader() = default;

  std::unordered_map<XQAKernelLoadHashKey, const std::unique_ptr<XQAKernelList>,
                     XQAKernelLoadHasher>
      mKernels;
};

inline XQAKernelList const* getXQAKernels(Data_type type, unsigned int sm) {
  return XQAKernelLoader::Get().getXQAKernels(type, sm);
}

#define XQA_KERNEL_RUN(DATA_TYPE)                                  \
  xqa_kernel->template run<DATA_TYPE>(xqa_params, kv_cache_buffer, \
                                      multi_processor_count, stream);

void DecoderXQAImplPrecompiled::runDispatchBuffer(
    XQAParams const& xqa_params, KVCacheListParams const& kv_cache_buffer,
    cudaStream_t const& stream) {
  XQAKernelList const* xqa_kernel =
      getXQAKernels(/*mRunner->mDataType*/ mRunner->mDataType, getSMVersion());
  int multi_processor_count = mRunner->mMultiProcessorCount;
  if (mRunner->mDataType == DATA_TYPE_FP16) {
    XQA_KERNEL_RUN(__half);
  } else {
    XQA_KERNEL_RUN(__nv_bfloat16);
  }
}

void DecoderXQAImplPrecompiled::runWithKVBlockArray(
    XQAParams const& xqa_params, KVCacheListParams const& kv_block_array,
    cudaStream_t const& stream) {
  runDispatchBuffer(xqa_params, kv_block_array, stream);
}
#undef XQA_KERNEL_RUN