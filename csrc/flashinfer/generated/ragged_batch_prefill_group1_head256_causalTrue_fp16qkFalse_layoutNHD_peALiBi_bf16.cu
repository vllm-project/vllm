#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillRaggedWrapper(nv_bfloat16, 1, 256, true, false, QKVLayout::kNHD, PosEncodingMode::kALiBi)
