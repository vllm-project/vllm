
#include "decoder_xqa_impl.h"
#include "decoder_xqa_impl_precompiled.h"

void DecoderXQAImpl::run(XQAParams const& xqa_params,
                         KVCacheListParams const& kv_block_array,
                         cudaStream_t const& stream) {
  runWithKVBlockArray(xqa_params, kv_block_array, stream);
}

std::unique_ptr<DecoderXQAImpl> DecoderXQAImpl::create(DecoderXQARunner* runner,
                                                       ImplType implType) {
  std::cout << "Creating a decoderXQAImpl\n";
  switch (implType) {
    case ImplType::kPrecompiled:
      return std::unique_ptr<DecoderXQAImpl>(
          new DecoderXQAImplPrecompiled(runner));
  }
  throw std::invalid_argument("Unknown DecoderXQAImpl::ImplType");
}
