#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillPagedWrapper(nv_bfloat16, 4, 64, false, false, QKVLayout::kNHD, PosEncodingMode::kNone)
