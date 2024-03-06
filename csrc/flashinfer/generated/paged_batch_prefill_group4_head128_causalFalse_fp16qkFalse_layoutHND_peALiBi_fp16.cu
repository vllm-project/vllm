#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillPagedWrapper(nv_half, 4, 128, false, false, QKVLayout::kHND, PosEncodingMode::kALiBi)
