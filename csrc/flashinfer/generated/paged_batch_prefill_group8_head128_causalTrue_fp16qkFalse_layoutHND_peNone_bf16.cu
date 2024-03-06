#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillPagedWrapper(nv_bfloat16, 8, 128, true, false, QKVLayout::kHND, PosEncodingMode::kNone)
