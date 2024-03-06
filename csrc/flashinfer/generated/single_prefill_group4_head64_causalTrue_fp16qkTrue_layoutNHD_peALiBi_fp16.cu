#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_SinglePrefill(nv_half, 4, 64, true, true, QKVLayout::kNHD, PosEncodingMode::kALiBi)
