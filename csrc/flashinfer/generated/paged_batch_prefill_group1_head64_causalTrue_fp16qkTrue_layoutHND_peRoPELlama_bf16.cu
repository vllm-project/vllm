#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillPagedWrapper(nv_bfloat16, 1, 64, true, true, QKVLayout::kHND, PosEncodingMode::kRoPELlama)
