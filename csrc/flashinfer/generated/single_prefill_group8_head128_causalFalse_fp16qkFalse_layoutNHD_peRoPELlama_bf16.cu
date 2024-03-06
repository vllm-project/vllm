#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_SinglePrefill(nv_bfloat16, 8, 128, false, false, QKVLayout::kNHD, PosEncodingMode::kRoPELlama)
