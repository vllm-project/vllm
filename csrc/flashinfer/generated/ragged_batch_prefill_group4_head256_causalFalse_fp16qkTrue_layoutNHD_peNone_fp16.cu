#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_BatchPrefillRaggedWrapper(nv_half, 4, 256, false, true, QKVLayout::kNHD, PosEncodingMode::kNone)
