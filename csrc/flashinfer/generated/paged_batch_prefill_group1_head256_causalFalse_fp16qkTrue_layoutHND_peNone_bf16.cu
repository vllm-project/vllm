#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillPagedWrapper(nv_bfloat16, 1, 256, false, true, QKVLayout::kHND, PosEncodingMode::kNone)
