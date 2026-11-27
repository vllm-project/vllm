#pragma once

namespace DOSA_NAMESPACE::packed20 {
constexpr int PageIds = 128, LowBytes = 256, TrailerBytes = 16;
constexpr int PageBytes = LowBytes + TrailerBytes;
static_assert(PageBytes % 16 == 0);
__device__ __forceinline__ void store_low(int* pool, int64_t position, int id) {
  reinterpret_cast<uint16_t*>(pool)[position] = uint16_t(id);
}
__device__ __forceinline__ int load(const int* pool, int pages, int page_index,
                                    int slot) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(pool);
  const auto* ends =
      bytes + int64_t(pages) * LowBytes + page_index * TrailerBytes;
  const auto* low =
      reinterpret_cast<const uint16_t*>(pool) + page_index * PageIds;
  const uint32_t header = ends[15];
  uint32_t band = header & 15u;
  const uint32_t last = header >> 4;
  while (band < last && uint32_t(slot) >= ends[band]) ++band;
  return int(low[slot]) | int(band << 16);
}
// Preconditions inherited from compact_packed20: one CTA owns each stream's
// 4096-slot chunk; calls for that stream append ascending 64K bands, and the
// consumer runs after compaction. A new page is always first touched at slot0.
// A single thread owns each intersected page, including zero/init/publication.
__device__ __forceinline__ void fill_high(int* pool, int pages, int64_t begin,
                                          int count, uint32_t nibble,
                                          int thread) {
  const int64_t end = begin + count;
  auto* trailers = reinterpret_cast<uint8_t*>(pool) + int64_t(pages) * LowBytes;
  for (int64_t p = begin / PageIds + thread; p < (end + PageIds - 1) / PageIds;
       p += 256) {
    auto* trailer = trailers + p * TrailerBytes;
    const int lo = int(max(begin - p * PageIds, int64_t(0)));
    const int hi = int(min(end - p * PageIds, int64_t(PageIds)));
    if (lo == 0) {
      // The first band initializes the whole unshared trailer in one store.
      const uint32_t header = nibble * 17;
      const uint32_t entry =
          nibble == 15 ? 0u : uint32_t(hi) << (8 * (nibble % 4));
      const uint32_t lane = nibble / 4;
      *reinterpret_cast<uint4*>(trailer) = make_uint4(
          lane == 0 ? entry : 0u, lane == 1 ? entry : 0u,
          lane == 2 ? entry : 0u, (lane == 3 ? entry : 0u) | (header << 24));
    } else {
      const uint32_t first = trailer[15] & 15u;
      if (nibble < 15) trailer[nibble] = uint8_t(hi);
      trailer[15] = uint8_t(first | (nibble << 4));
    }
  }
}
}  // namespace DOSA_NAMESPACE::packed20
