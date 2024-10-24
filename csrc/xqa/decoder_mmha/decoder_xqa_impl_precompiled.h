#pragma once

#include "decoder_xqa_impl.h"


class DecoderXQAImplPrecompiled  : public DecoderXQAImpl
{
 public:
  DecoderXQAImplPrecompiled(DecoderXQARunner* runner)
                                 : DecoderXQAImpl(runner)
  {}
  
  void runWithKVBlockArray(XQAParams const& xqa_params,
                           KVCacheListParams const& kv_block_array,
                           cudaStream_t const& stream)   override;
 private:
  void runDispatchBuffer(XQAParams const& xqa_params,
                         KVCacheListParams const& kv_cache_buffer,
                         cudaStream_t const& stream);

};
