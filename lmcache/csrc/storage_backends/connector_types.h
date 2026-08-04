// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// there are four "types" in this file:
// 1. Op: enum class for the three batched operations (GET, SET, EXISTS)
// 2. BatchState: shared communication state between threads executing a single
// batch operation
// 3. Request: the request data structure for a single operation (submitted to
// SQ)
// 4. Completion: the completion data structure for a single operation
// (collected from CQ)

namespace lmcache {
namespace connector {

// we only support batched operations
// benefits are fewer submissions and fewer completions
enum class Op : uint8_t {
  BATCH_TILE_GET,
  BATCH_TILE_SET,
  BATCH_TILE_EXISTS,
  BATCH_TILE_DELETE
};

/*
shared communication state between threads executing a single batch operation.
all threads need to complete before the completion is sent.

tiling refers to dividing work for batched operations between threads
beforehand.
*/
struct BatchState {
  std::atomic<uint32_t> remaining_tiles{0};
  std::atomic<bool> any_failed{false};

  std::mutex err_mu;
  std::string first_error;

  // Per-key success/failure results used by both EXISTS and GET.
  // For EXISTS: 1 = key found, 0 = not found.
  // For GET: 1 = read succeeded, 0 = read failed (e.g. file
  //   not found).  This enables per-key error tolerance on loads.
  // IMPORTANT: not vector<bool> due to concurrent write data race
  std::vector<uint8_t> per_key_results;

  Op batch_op;
};

/*
LIFETIME GUARANTEE:
we have a strict assumption that Python will NOT clean up any buffer memory
before all C++ operations finish. This is guaranteed by the Python-side design
where the caller holds references to all buffers until drain_completions()
returns the corresponding future_id. Therefore, we do NOT need to track
buf_owner references or acquire the GIL to prevent premature cleanup.
we can safely use raw pointers extracted under the GIL without additional
lifetime management on the C++ side.
*/

struct Request {
  uint64_t future_id = 0;
  Op op;

  // all operations use the batched structure (even single-item operations
  // are treated as batches of size 1)
  std::vector<std::string> keys;
  std::vector<void*> buf_ptrs;
  std::vector<size_t> buf_lens;

  // shared batch state between threads executing a single batch operation
  // so that they can coordinate when to send the completion
  std::shared_ptr<BatchState> batch;

  // for batch exists tiles, track which indices this tile is responsible for
  size_t start_idx = 0;

  // batch_chunk_num_bytes for get/set operations (passed per-operation, not
  // per-connection)
  size_t batch_chunk_num_bytes = 0;
};

struct Completion {
  uint64_t future_id = 0;

  bool ok = true;

  // for EXISTS operations, store boolean results as
  // bytes (0/1). Single EXISTS will have 1 element, batch EXISTS will have N
  // elements. No result in the completion for SET and GET.
  std::vector<uint8_t> result_bytes;

  std::string error;
};

}  // namespace connector
}  // namespace lmcache
