// SPDX-License-Identifier: Apache-2.0

#include "fold.h"

#include <algorithm>

namespace lmcache {

namespace storage_manager {

Bitmap fold(const Bitmap& found, size_t num_chunks, size_t num_ranks,
            const std::vector<int64_t>& group_windows) {
  const size_t num_groups = group_windows.size();
  const size_t group_stride = num_chunks * num_ranks;

  // ``servable[j]`` (bit ``j``, prefix length ``j + 1``) stays set only if
  // every group can serve a length-``j + 1`` prefix under its rule. ``run`` is
  // the count of consecutive present chunks ending at the current chunk, so a
  // length-L prefix needs the last ``min(window, L)`` chunks present, i.e.
  // ``run >= min(window, L)``.
  std::vector<char> servable(num_chunks, 1);
  for (size_t g = 0; g < num_groups; ++g) {
    const int64_t window = group_windows[g];
    const size_t eff_window =
        (window <= 0) ? num_chunks : static_cast<size_t>(window);
    const size_t gbase = g * group_stride;
    size_t run = 0;
    for (size_t prefix_len = 1; prefix_len <= num_chunks; ++prefix_len) {
      const size_t cbase = gbase + (prefix_len - 1) * num_ranks;
      bool chunk_present = true;
      for (size_t r = 0; r < num_ranks; ++r) {
        if (!found.test(cbase + r)) {
          chunk_present = false;
          break;
        }
      }
      run = chunk_present ? run + 1 : 0;
      if (servable[prefix_len - 1] && run < std::min(eff_window, prefix_len)) {
        servable[prefix_len - 1] = 0;
      }
    }
  }

  Bitmap servable_lengths(num_chunks);
  for (size_t j = 0; j < num_chunks; ++j) {
    if (servable[j]) servable_lengths.set(j);
  }
  return servable_lengths;
}

Bitmap unfold(size_t hit_length, size_t num_chunks, size_t num_ranks,
              const std::vector<int64_t>& group_windows) {
  if (hit_length > num_chunks) hit_length = num_chunks;
  const size_t num_groups = group_windows.size();
  const size_t group_stride = num_chunks * num_ranks;

  // The chunks each group needs to serve ``hit_length``, expanded over every
  // kv_rank. The retained cells of a group are a contiguous bit range
  // ``[gbase + lo * num_ranks, gbase + hit_length * num_ranks)``, so a single
  // ``set_range`` (whole-byte fill) covers each group.
  Bitmap retain_mask(num_groups * group_stride);
  if (hit_length == 0) return retain_mask;
  for (size_t g = 0; g < num_groups; ++g) {
    const int64_t window = group_windows[g];
    size_t lo = 0;
    if (window > 0 && hit_length > static_cast<size_t>(window)) {
      lo = hit_length - static_cast<size_t>(window);
    }
    const size_t gbase = g * group_stride;
    retain_mask.set_range(gbase + lo * num_ranks,
                          gbase + hit_length * num_ranks);
  }
  return retain_mask;
}

}  // namespace storage_manager

}  // namespace lmcache
