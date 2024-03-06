#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillPagedWrapper(nv_half, 8, 256, true, true, QKVLayout::kNHD, PosEncodingMode::kALiBi)
