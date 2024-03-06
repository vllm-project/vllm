#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillPagedWrapper(nv_half, 4, 256, false, true, QKVLayout::kHND, PosEncodingMode::kNone)
