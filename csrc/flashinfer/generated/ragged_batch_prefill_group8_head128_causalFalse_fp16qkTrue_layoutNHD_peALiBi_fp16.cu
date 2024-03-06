#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillRaggedWrapper(nv_half, 8, 128, false, true, QKVLayout::kNHD, PosEncodingMode::kALiBi)
