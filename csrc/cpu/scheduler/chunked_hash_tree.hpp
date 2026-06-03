#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Min-heap entry ordering requests by the number of KV-cache chunks they are
// missing. Ties are broken by request_id to ensure a deterministic order.
struct HeapEntry {
  int missing;
  uint32_t request_id;

  bool operator<(const HeapEntry& other) const {
    if (missing != other.missing) return missing > other.missing;
    return request_id > other.request_id;
  }
};

// ChunkedHashTree tracks a set of pending and active inference requests,
// each represented as a sequence of fixed-size token chunks. Every chunk is
// identified by a rolling XXH64 hash so that two requests that share a common
// token prefix share the same hashes at every level up to the divergence point.
//
// The tree answers three scheduling questions efficiently:
//   1. Which waiting request has the fewest KV-cache chunks still missing from
//      the current working set? (find_best_request)
//   2. What is the length of the prefix that ALL currently active requests
//      share? (shared_chain_length / shared_prefix_chain)
//   3. How many other waiting requests would benefit from the same prefix
//      extension that the next candidate request would trigger?
//
// All mutating operations keep the shared prefix chain and the per-request
// missing counts up to date incrementally, so lookups remain O(1) amortised.
class ChunkedHashTree {
 public:
  // chunk: number of tokens per chunk; determines hash granularity.
  explicit ChunkedHashTree(uint32_t chunk);

  // Hash the token sequence into chunks and register the request as waiting.
  // Returns the number of tokens inserted (0 if request_id already exists).
  uint32_t insert(uint32_t request_id, const std::vector<uint32_t>& tokens);

  // Return the waiting request that would extend the shared prefix chain the
  // most (fewest missing chunks). The tuple is:
  //   (request_id, chunks_before, chunks_after, requests_at_same_level)
  // where chunks_before is the current shared chain length, chunks_after is
  // the chain length after activating the candidate, and
  // requests_at_same_level is the number of OTHER waiting requests that
  // already share that extended prefix. Returns (0,0,0,0) if no request is
  // waiting. The result is cached until the next mutation.
  std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> find_best_request();

  // Move request_id from the waiting set into the active set and add its
  // chunks to the working set. Returns (chunks_added, total_chunks).
  // Chunks already present in the working set are reference-counted and not
  // double-counted in chunks_added.
  std::pair<uint32_t, uint32_t> activate_request(uint32_t request_id);

  // Remove request_id from the active set and evict chunks whose reference
  // count drops to zero. Returns (chunks_evicted, total_chunks).
  std::pair<uint32_t, uint32_t> finish_request(uint32_t request_id);

  // Unconditionally delete request_id from all internal structures. Safe to
  // call on both waiting and active requests; used for cancellation.
  void remove(uint32_t request_id);

 private:
  uint32_t chunk_size;

  // Per-request hash sequences computed by compute_hashes().
  // Index i holds the cumulative XXH64 digest of chunks 0..i.
  std::unordered_map<uint32_t, std::vector<uint64_t>> request_hashes;

  // Set of (level, hash) keys currently loaded in the KV cache.
  std::unordered_set<uint64_t> working_set;

  // Requests that have been activated and whose chunks are in working_set.
  std::unordered_set<uint32_t> active_requests;

  // Maps a (level, hash) key to every request (waiting or active) that has
  // that hash at that level. Used to update missing counts on cache changes.
  std::unordered_map<uint64_t, std::unordered_set<uint32_t>> hash_to_waiting;

  // Current number of chunks not yet in the working set for each waiting
  // request. Kept in sync with working_set so the heap stays accurate.
  std::unordered_map<uint32_t, int> missing_cache;

  // Reference counts for each (level, hash) key in the working set.
  // A key is evicted from working_set when its count reaches zero.
  std::unordered_map<uint64_t, uint32_t> hash_ref_counts;

  // Min-heap of (missing, request_id) pairs. May contain stale entries;
  // find_best_request() discards them on pop.
  std::priority_queue<HeapEntry> waiting_heap;

  // One-slot cache for find_best_request(). Invalidated by every mutation.
  bool cache_valid;
  std::optional<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>>
      cached_best_request;

  // Ordered sequence of hashes shared by every active request (the common
  // token prefix). Maintained incrementally by update_chain_on_activate()
  // and update_chain_on_finish().
  std::vector<uint64_t> shared_prefix_chain;
  uint32_t shared_chain_length;

  // Encode (level, hash) into a single 64-bit key: the top byte holds the
  // level and the lower 56 bits hold the truncated hash.
  uint64_t pack_key(uint32_t level, uint64_t hash);

  // Compute the rolling XXH64 hash for every chunk boundary in tokens.
  std::vector<uint64_t> compute_hashes(const std::vector<uint32_t>& tokens);

  // Mark the find_best_request() cache as dirty.
  void invalidate_cache();

  // Rebuild shared_prefix_chain from scratch by walking all active requests.
  // O(active * chain_length); only called when incremental updates are not
  // sufficient (e.g. after a request finishes and only one active remains).
  void recompute_shared_chain();

  // Incrementally shorten shared_prefix_chain to the longest prefix that the
  // newly activated request still matches. Called after inserting into
  // active_requests.
  void update_chain_on_activate(uint32_t request_id);

  // Attempt to extend shared_prefix_chain now that the finished request no
  // longer constrains it. Called before removing from active_requests.
  void update_chain_on_finish(uint32_t request_id);

  // Compute the shared chain length that would result if request_id were
  // activated, without actually modifying any state.
  uint32_t compute_chain_length_with_request(uint32_t request_id);

  // Count waiting (non-active) requests whose hash at level (prefix_length-1)
  // matches the current shared chain, i.e. requests that already share a
  // prefix of exactly prefix_length chunks.
  uint32_t count_requests_sharing_prefix(uint32_t prefix_length);
};