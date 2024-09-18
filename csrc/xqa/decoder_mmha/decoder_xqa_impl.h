#pragma once
#include "xqa_params.h"
#include "decoder_xqa_impl_common.h"

class DecoderXQARunner;
class DecoderXQAImpl {
 public:
  
// DecoderXQAImpl(DecoderXQARunner* runner) : mRunner(runner) {}

  void run(XQAParams const& xqa_params, KVCacheListParams const& kv_cache_buffer, cudaStream_t const& stream);
 enum class ImplType
    {
        kPrecompiled = 0,
    };
  static std::unique_ptr<DecoderXQAImpl> create(DecoderXQARunner* runner, ImplType implType) ;

 protected:
 DecoderXQAImpl(DecoderXQARunner* runner)
        : mRunner(runner)
    {
    }
  virtual  void runWithKVBlockArray(XQAParams const& xqa_params,
                           KVCacheListParams const& kv_block_array,
                           cudaStream_t const& stream) =0;
  DecoderXQARunner* mRunner;
};


// std::unique_ptr<DecoderXQAImpl> DecoderXQAImpl::create(DecoderXQARunner* runner)
// {

//     return std::unique_ptr<DecoderXQAImpl>(new DecoderXQAImplPrecompiled(runner));
  
// }
enum class XQAKernelType : int32_t {
  kAMPERE_WARP_SPECIALIZED = 0,
  kHOPPER_WARP_SPECIALIZED = 1
};