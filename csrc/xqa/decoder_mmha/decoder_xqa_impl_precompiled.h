#pragma once
// #include "xqaImplCommon.h"
#include "decoder_xqa_impl.h"
// class DecoderXQARunner;

class DecoderXQAImplPrecompiled  : public DecoderXQAImpl
{
 public:
  DecoderXQAImplPrecompiled(DecoderXQARunner* runner)
                                 : DecoderXQAImpl(runner)
  {}
  // static std::unique_ptr<DecoderXQAImplPrecompiled> create();
  //   DecoderXQARunner* runner);
  // bool shouldUse(XQAParams const& xqaParams, bool forConfigurePlugin)
  // override; void prepare(XQAParams const& xqa_params) override;
  // size_t getWorkspaceSize(int max_num_tokens);
  void runWithKVBlockArray(XQAParams const& xqa_params,
                           KVCacheListParams const& kv_block_array,
                           cudaStream_t const& stream)   override;
 private:
  void runDispatchBuffer(XQAParams const& xqa_params,
                         KVCacheListParams const& kv_cache_buffer,
                         cudaStream_t const& stream);

};
