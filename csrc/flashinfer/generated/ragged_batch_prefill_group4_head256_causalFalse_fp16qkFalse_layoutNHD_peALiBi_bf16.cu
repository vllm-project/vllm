#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillRaggedWrapper(nv_bfloat16, 4, 256, false, false, QKVLayout::kNHD, PosEncodingMode::kALiBi)
