#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillRaggedWrapper(nv_half, 4, 128, false, false, QKVLayout::kNHD, PosEncodingMode::kALiBi)
