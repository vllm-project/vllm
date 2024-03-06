#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillPagedWrapper(nv_half, 1, 256, false, false, QKVLayout::kHND, PosEncodingMode::kNone)
