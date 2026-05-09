#include "chunked_hash_tree.hpp"

#include <algorithm>
#include <xxhash.h>

ChunkedHashTree::ChunkedHashTree(uint32_t chunk)
    : chunk_size(chunk), cache_valid(false), shared_chain_length(0) {
  request_hashes.reserve(1024);
  missing_cache.reserve(1024);
  hash_to_waiting.reserve(4096);
  working_set.reserve(4096);
  hash_ref_counts.reserve(4096);
  shared_prefix_chain.reserve(256);
}

uint64_t ChunkedHashTree::pack_key(uint32_t level, uint64_t hash) {
  // 16 bits for level (supports up to 65535 chunks = ~3.2M tokens at
  // chunk_size=50) 48 bits for truncated hash
  return (uint64_t(level & 0xFFFF) << 48) | (hash & 0x0000FFFFFFFFFFFFULL);
}

void ChunkedHashTree::invalidate_cache() {
  cache_valid = false;
  cached_best_request.reset();
}

// --------------------------------------------------------------------------
// Shared prefix chain maintenance
// --------------------------------------------------------------------------

void ChunkedHashTree::recompute_shared_chain() {
  shared_prefix_chain.clear();
  shared_chain_length = 0;

  if (active_requests.empty()) return;

  // Upper bound: the shortest active request limits the chain.
  uint32_t max_len = UINT32_MAX;
  for (auto req_id : active_requests) {
    auto it = request_hashes.find(req_id);
    if (it != request_hashes.end())
      max_len = std::min(max_len, (uint32_t)it->second.size());
  }

  if (max_len == 0 || max_len == UINT32_MAX) return;

  for (uint32_t level = 0; level < max_len; ++level) {
    uint64_t common_hash = 0;
    bool first = true;
    bool all_match = true;

    for (auto req_id : active_requests) {
      auto it = request_hashes.find(req_id);
      if (it == request_hashes.end() || level >= it->second.size()) {
        all_match = false;
        break;
      }
      uint64_t hash = it->second[level];
      if (first) {
        common_hash = hash;
        first = false;
      } else if (hash != common_hash) {
        all_match = false;
        break;
      }
    }

    if (!all_match) break;

    shared_prefix_chain.push_back(common_hash);
    shared_chain_length++;
  }
}

void ChunkedHashTree::update_chain_on_activate(uint32_t request_id) {
  auto it = request_hashes.find(request_id);
  if (it == request_hashes.end()) return;

  const auto& hashes = it->second;

  if (active_requests.size() == 1) {
    // First active request: the whole request becomes the shared chain.
    shared_prefix_chain = hashes;
    shared_chain_length = hashes.size();
    return;
  }

  // Because hashes are cumulative, matching is monotone: if level k matches,
  // all levels 0..k-1 match too. Binary-search for the new chain length.
  uint32_t lo = 0;
  uint32_t hi = std::min(shared_chain_length, (uint32_t)hashes.size());

  while (lo < hi) {
    // Use upper-mid to avoid an infinite loop when lo + 1 == hi.
    uint32_t mid = lo + (hi - lo + 1) / 2;
    if (hashes[mid - 1] == shared_prefix_chain[mid - 1])
      lo = mid;
    else
      hi = mid - 1;
  }

  shared_chain_length = lo;
  shared_prefix_chain.resize(shared_chain_length);
}

void ChunkedHashTree::update_chain_on_finish(uint32_t request_id) {
  if (active_requests.empty()) {
    shared_prefix_chain.clear();
    shared_chain_length = 0;
    return;
  }

  if (active_requests.size() == 1) {
    // Down to one request: recompute fully since the chain may now be longer.
    recompute_shared_chain();
    return;
  }

  // Removing a request can only extend the chain, never shorten it.
  // Find the new upper bound from the remaining active requests.
  uint32_t max_possible = UINT32_MAX;
  for (auto req_id : active_requests) {
    auto req_it = request_hashes.find(req_id);
    if (req_it != request_hashes.end())
      max_possible = std::min(max_possible, (uint32_t)req_it->second.size());
  }

  // Try to extend level by level until a mismatch is found.
  for (uint32_t level = shared_chain_length; level < max_possible; ++level) {
    uint64_t common_hash = 0;
    bool first = true;
    bool all_match = true;

    for (auto req_id : active_requests) {
      auto req_it = request_hashes.find(req_id);
      if (req_it == request_hashes.end() || level >= req_it->second.size()) {
        all_match = false;
        break;
      }
      uint64_t hash = req_it->second[level];
      if (first) {
        common_hash = hash;
        first = false;
      } else if (hash != common_hash) {
        all_match = false;
        break;
      }
    }

    if (!all_match) break;

    shared_prefix_chain.push_back(common_hash);
    shared_chain_length++;
  }
}

// --------------------------------------------------------------------------
// Scheduling helpers
// --------------------------------------------------------------------------

uint32_t ChunkedHashTree::compute_chain_length_with_request(
    uint32_t request_id) {
  auto it = request_hashes.find(request_id);
  if (it == request_hashes.end()) return shared_chain_length;

  const auto& hashes = it->second;

  if (active_requests.empty()) {
    // This would be the first active request, so its full length becomes
    // the shared chain.
    return hashes.size();
  }

  // Find how far the candidate's hashes agree with the current shared chain.
  uint32_t matching = 0;
  for (uint32_t level = 0; level < shared_chain_length && level < hashes.size();
       ++level) {
    if (hashes[level] == shared_prefix_chain[level])
      matching++;
    else
      break;
  }

  return matching;
}

uint32_t ChunkedHashTree::count_requests_sharing_prefix(
    uint32_t prefix_length) {
  if (prefix_length == 0 || prefix_length > shared_chain_length) return 0;

  // Because hashes are cumulative, matching level (prefix_length-1) implies
  // matching all earlier levels. We only need to check the last level.
  uint64_t key =
      pack_key(prefix_length - 1, shared_prefix_chain[prefix_length - 1]);

  auto it = hash_to_waiting.find(key);
  if (it == hash_to_waiting.end()) return 0;

  uint32_t count = 0;
  for (auto req_id : it->second) {
    if (!active_requests.count(req_id)) count++;
  }
  return count;
}

// --------------------------------------------------------------------------
// Public interface
// --------------------------------------------------------------------------

std::vector<uint64_t> ChunkedHashTree::compute_hashes(
    const std::vector<uint32_t>& tokens) {
  size_t num_chunks = (tokens.size() + chunk_size - 1) / chunk_size;
  std::vector<uint64_t> hashes;
  hashes.reserve(num_chunks);

  // Feed tokens through a single streaming state so each digest incorporates
  // all tokens from chunk 0 up to and including the current chunk boundary.
  XXH64_state_t* state = XXH64_createState();
  if (!state) return {};
  XXH64_reset(state, 0);

  for (size_t i = 0; i < tokens.size(); ++i) {
    XXH64_update(state, &tokens[i], sizeof(uint32_t));
    if ((i + 1) % chunk_size == 0 || i + 1 == tokens.size())
      hashes.push_back(XXH64_digest(state));
  }

  XXH64_freeState(state);
  return hashes;
}

uint32_t ChunkedHashTree::insert(uint32_t request_id,
                                 const std::vector<uint32_t>& tokens) {
  if (request_hashes.count(request_id)) return 0;

  auto hashes = compute_hashes(tokens);
  request_hashes[request_id] = hashes;

  int missing = 0;
  for (uint32_t level = 0; level < hashes.size(); ++level) {
    uint64_t key = pack_key(level, hashes[level]);

    if (!working_set.count(key)) missing++;

    auto& waiting_set = hash_to_waiting[key];
    if (waiting_set.empty()) waiting_set.reserve(16);
    waiting_set.insert(request_id);
  }

  missing_cache[request_id] = missing;
  waiting_heap.push({missing, request_id});

  invalidate_cache();
  return tokens.size();
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>
ChunkedHashTree::find_best_request() {
  if (cache_valid && cached_best_request.has_value())
    return cached_best_request.value();

  while (!waiting_heap.empty()) {
    auto top = waiting_heap.top();
    waiting_heap.pop();

    if (active_requests.count(top.request_id)) continue;

    auto it = missing_cache.find(top.request_id);
    if (it == missing_cache.end()) continue;  // request was removed
    if (it->second != top.missing) continue;  // stale heap entry

    uint32_t chunks_before = shared_chain_length;
    uint32_t chunks_after = compute_chain_length_with_request(top.request_id);
    uint32_t peers = count_requests_sharing_prefix(chunks_after);

    cached_best_request =
        std::make_tuple(top.request_id, chunks_before, chunks_after, peers);
    cache_valid = true;

    // Return the entry to the heap — we are only peeking, not activating.
    waiting_heap.push(top);
    return cached_best_request.value();
  }

  cached_best_request = std::make_tuple(0u, 0u, 0u, 0u);
  cache_valid = true;
  return cached_best_request.value();
}

std::pair<uint32_t, uint32_t> ChunkedHashTree::activate_request(
    uint32_t request_id) {
  if (active_requests.count(request_id)) return {0, 0};
  if (!request_hashes.count(request_id)) return {0, 0};

  active_requests.insert(request_id);
  auto& hashes = request_hashes[request_id];
  uint32_t added = 0;

  for (uint32_t level = 0; level < hashes.size(); ++level) {
    uint64_t key = pack_key(level, hashes[level]);

    if (hash_ref_counts[key]++ == 0) {
      // First reference: bring the chunk into the working set and
      // decrement the missing count for every waiting request that needs it.
      working_set.insert(key);

      auto waiting_it = hash_to_waiting.find(key);
      if (waiting_it != hash_to_waiting.end()) {
        for (auto w : waiting_it->second) {
          if (active_requests.count(w) || !missing_cache.count(w)) continue;
          auto& m = missing_cache[w];
          if (m > 0) {
            m--;
            waiting_heap.push({m, w});
          }
        }
      }
      added++;
    }
  }

  missing_cache.erase(request_id);
  update_chain_on_activate(request_id);
  invalidate_cache();

  return {added, (uint32_t)hashes.size()};
}

void ChunkedHashTree::remove(uint32_t request_id) {
  auto it = request_hashes.find(request_id);
  if (it != request_hashes.end()) {
    for (uint32_t level = 0; level < it->second.size(); ++level) {
      uint64_t key = pack_key(level, it->second[level]);
      auto map_it = hash_to_waiting.find(key);
      if (map_it != hash_to_waiting.end()) {
        map_it->second.erase(request_id);
        if (map_it->second.empty()) hash_to_waiting.erase(map_it);
      }
    }
  }

  request_hashes.erase(request_id);
  missing_cache.erase(request_id);
  invalidate_cache();
}

std::pair<uint32_t, uint32_t> ChunkedHashTree::finish_request(
    uint32_t request_id) {
  if (!active_requests.erase(request_id)) return {0, 0};

  auto it = request_hashes.find(request_id);
  if (it == request_hashes.end()) return {0, 0};

  auto& hashes = it->second;
  uint32_t evicted = 0;

  for (uint32_t level = 0; level < hashes.size(); ++level) {
    uint64_t key = pack_key(level, hashes[level]);

    if (--hash_ref_counts[key] == 0) {
      // Last reference dropped: evict the chunk and increment the missing
      // count for every waiting request that needs it.
      working_set.erase(key);
      hash_ref_counts.erase(key);

      auto waiting_it = hash_to_waiting.find(key);
      if (waiting_it != hash_to_waiting.end()) {
        for (auto w : waiting_it->second) {
          if (active_requests.count(w) || !missing_cache.count(w)) continue;
          missing_cache[w]++;
          waiting_heap.push({missing_cache[w], w});
        }
      }
      evicted++;
    }
  }

  // Chain may extend now that this request no longer constrains it.
  update_chain_on_finish(request_id);

  // Remove from all remaining bookkeeping.
  remove(request_id);
  invalidate_cache();

  return {evicted, (uint32_t)hashes.size()};
}