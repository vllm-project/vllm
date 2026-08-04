// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "bitmap.h"

namespace lmcache {

namespace storage_manager {

/**
 * @brief Fold per-(group, chunk, rank) presence into servable prefix lengths.
 *
 * For each object group, computes which prefix lengths it can serve under its
 * rule (full attention or a cross-chunk sliding window), and intersects across
 * groups. The result feeds :func:`Bitmap::highest_set_bit`; the model-wide hit
 * length is that index plus one (``-1`` -> hit length 0), then :func:`unfold`.
 *
 * The input ``found`` is group-major / chunk-major / rank-minor: bit
 * ``g * (num_chunks * num_ranks) + j * num_ranks + r`` is set iff chunk ``j``
 * of object group ``g`` is present for kv_rank ``r``. A chunk counts as present
 * for a group only when every kv_rank shard is present.
 *
 * @param found Presence bitmap of length
 *     ``group_windows.size() * num_chunks * num_ranks``.
 * @param num_chunks Number of LMCache chunks in the request.
 * @param num_ranks Number of kv_rank shards per chunk.
 * @param group_windows Per-object-group cross-chunk sliding-window size in
 *     chunks, in object-group order; ``<= 0`` means full attention.
 *
 * @return A bitmap of size ``num_chunks``; bit ``j`` set iff every group can
 *     serve a length-``j + 1`` prefix.
 */
Bitmap fold(const Bitmap& found, size_t num_chunks, size_t num_ranks,
            const std::vector<int64_t>& group_windows);

/**
 * @brief Expand a model-wide hit length into the per-group retain mask.
 *
 * Each group retains the chunks it needs to serve ``hit_length``: ``[0,
 * hit_length)`` for full attention, ``[hit_length - window, hit_length)`` for a
 * sliding window. The mask is over the same ranked layout as :func:`fold`'s
 * input (all kv_ranks of each retained ``(group, chunk)`` set).
 *
 * @param hit_length Model-wide prefix hit length in chunks (clamped to
 *     ``num_chunks``).
 * @param num_chunks Number of LMCache chunks in the request.
 * @param num_ranks Number of kv_rank shards per chunk.
 * @param group_windows Per-object-group cross-chunk sliding-window size in
 *     chunks, in object-group order; ``<= 0`` means full attention.
 *
 * @return Retain mask of length
 *     ``group_windows.size() * num_chunks * num_ranks``.
 */
Bitmap unfold(size_t hit_length, size_t num_chunks, size_t num_ranks,
              const std::vector<int64_t>& group_windows);

}  // namespace storage_manager

}  // namespace lmcache
