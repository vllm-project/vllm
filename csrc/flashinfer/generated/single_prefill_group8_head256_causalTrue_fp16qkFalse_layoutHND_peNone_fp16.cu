#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_SinglePrefill(nv_half, 8, 256, true, false, QKVLayout::kHND, PosEncodingMode::kNone)
