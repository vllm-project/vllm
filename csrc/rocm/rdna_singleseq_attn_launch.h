// Launch ABI of the RDNA3.5 attention kernels: each decode build exposes a
// LaunchFn, each prefill build a PrefillFn.
#pragma once

#include <hip/hip_runtime.h>

namespace rdna_singleseq {

// Raw pointers keep torch headers out of the kernels.
struct LaunchArgs {
  const void* q;
  const void* kv;
  const int* bt;
  float* acc;
  float* m;
  float* l;
  int* cnt;
  void* out;
  const int* seq_lens;
  int bt_width;
  float scale;
  hipStream_t stream;
  // Multimodal ranges in device memory, read at run time so CUDA graphs
  // follow them: mm_ranges[0] ranges, then (start, end) pairs, at most
  // mm_capacity. Null for no ranges.
  const int* mm_ranges;
  int mm_capacity;
};

using LaunchFn = void (*)(const LaunchArgs&);

// The launch function of each entry of kDecodeConfigs, null until its unit
// (cmake/rocm_rdna_singleseq_attn.cmake) registers it. Defined in
// rdna_singleseq_registry.cu.
LaunchFn* decode_launchers();

// One sequence of num_tokens query tokens. partial_o, partial_ml and counters
// are the split-KV scratch; max_seq_len only sizes the split. mm_ranges holds
// num_mm_ranges (start, end) multimodal ranges, empty ones with start == end.
struct PrefillArgs {
  const void* q;
  const void* kv;
  const int* bt;
  void* out;
  const int* seq_lens;
  float* partial_o;
  float* partial_ml;
  int* counters;
  int num_tokens;
  int max_seq_len;
  float scale;
  hipStream_t stream;
  const int* mm_ranges;
  int num_mm_ranges;
};

using PrefillFn = void (*)(const PrefillArgs&);

// Likewise for kPrefillConfigs.
PrefillFn* prefill_launchers();

}  // namespace rdna_singleseq
