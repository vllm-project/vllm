#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillPagedWrapper(nv_half, 4, 256, true, true, QKVLayout::kHND, PosEncodingMode::kRoPELlama)
