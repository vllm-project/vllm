#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillRaggedWrapper(nv_bfloat16, 4, 64, true, true, QKVLayout::kHND, PosEncodingMode::kRoPELlama)
