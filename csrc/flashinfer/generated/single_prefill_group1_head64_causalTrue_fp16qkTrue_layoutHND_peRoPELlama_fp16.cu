#include <flashinfer_decl.h>

#include <flashinfer.cuh>

using namespace flashinfer;

INST_SinglePrefill(nv_half, 1, 64, true, true, QKVLayout::kHND, PosEncodingMode::kRoPELlama)
