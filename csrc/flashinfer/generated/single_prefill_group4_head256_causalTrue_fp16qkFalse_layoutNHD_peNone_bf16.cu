#include "../flashinfer_decl.h"

#include <flashinfer.cuh>

using namespace flashinfer;

INST_SinglePrefill(nv_bfloat16, 4, 256, true, false, QKVLayout::kNHD, PosEncodingMode::kNone)
