// clang-format off

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace cpu_sgl::moe {

namespace detail {
inline int64_t div_up(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}
}  // namespace detail

// Cost-balanced [MB, NB] tiles grouped by expert runs and column strips.
struct TileGrid {
  std::vector<int64_t> cost;  // prefix sum of per-tile cost over mb
  std::vector<int64_t> runs;  // first mb of each expert run, then MB
  // Borrowed immutable row offsets; must outlive the grid and traversals.
  const int32_t* offsets;
  int64_t blocks = 0;
};

inline TileGrid make_grid(
    int64_t MB, const int32_t* expert_ids, const int32_t* offsets, int64_t tile_cost, int64_t fetch_cost) {
  TileGrid grid;
  grid.offsets = offsets;
  for (int64_t mb = 0; mb < MB; ++mb) {
    if (mb == 0 || expert_ids[mb] != expert_ids[mb - 1]) {
      grid.runs.push_back(mb);
    }
  }
  grid.runs.push_back(MB);
  grid.cost.resize(MB + 1, 0);
  for (size_t r = 0; r + 1 < grid.runs.size(); ++r) {
    const int64_t mb0 = grid.runs[r], mb1 = grid.runs[r + 1];
    const int64_t fetch = fetch_cost / (mb1 - mb0);
    for (int64_t mb = mb0; mb < mb1; ++mb) {
      const int64_t m_size = offsets[mb + 1] - offsets[mb];
      grid.cost[mb + 1] = grid.cost[mb] + (m_size ? m_size + tile_cost + fetch : 0);
      grid.blocks += m_size != 0;
    }
  }
  return grid;
}

// Maximum per-thread weight columns and activation rows loaded per K.
inline int64_t max_load(const TileGrid& grid, int64_t NB, int64_t W, int64_t n_per_block, int nth) {
  // Advance cost-slice boundaries without per-thread division.
  const int64_t total = grid.cost.back() * NB, step = total / nth, rem = total % nth;
  int64_t most = 0, load = 0, hi = step, carry = rem;
  for (size_t r = 0; r + 1 < grid.runs.size(); ++r) {
    const int64_t mb0 = grid.runs[r], mb1 = grid.runs[r + 1];
    const int64_t run_cost = grid.cost[mb1] - grid.cost[mb0];
    for (int64_t nb0 = 0; nb0 < NB && run_cost; nb0 += W) {
      const int64_t width = std::min(W, NB - nb0);
      const int64_t start = grid.cost[mb0] * NB + run_cost * nb0;
      const int64_t end = start + run_cost * width;
      // tiles of this strip that start before p
      auto tiles = [&](int64_t p) -> int64_t {
        const int64_t q = p - start;
        if (q <= 0) {
          return 0;
        }
        if (q >= run_cost * width) {
          return (mb1 - mb0) * width;
        }
        const int64_t row = std::lower_bound(grid.cost.begin() + mb0, grid.cost.begin() + mb1, q,
                                             [&](int64_t cost, int64_t v) { return (cost - grid.cost[mb0]) * width < v; }) -
                            grid.cost.begin() - 1;
        const int64_t before = (grid.cost[row] - grid.cost[mb0]) * width;
        return (row - mb0) * width + std::min(width, detail::div_up(q - before, grid.cost[row + 1] - grid.cost[row]));
      };
      for (int64_t from = start, t0 = 0; from < end;) {
        while (hi <= from) {
          most = std::max(most, load);
          load = 0;
          hi += step, carry += rem;
          if (carry >= nth) {
            carry -= nth, ++hi;
          }
        }
        const int64_t to = std::min(end, hi);
        if (from == start && to == end) {
          load += width * n_per_block + grid.offsets[mb1] - grid.offsets[mb0];
        } else {
          const int64_t t1 = tiles(to);
          if (t1 > t0) {
            load += std::min(width, t1 - t0) * n_per_block + grid.offsets[mb0 + (t1 - 1) / width + 1] -
                    grid.offsets[mb0 + t0 / width];
          }
          t0 = t1;
        }
        from = to;
      }
    }
  }
  return std::max(most, load);
}

// Select ownership width W and cache width S for roughly square worker blocks.
// Narrow W when avoiding strip boundaries reduces the maximum load by 25%.
inline std::pair<int64_t, int64_t> select_strips(const TileGrid& grid, int64_t NB, int64_t n_per_block, int64_t cap, int nth) {
  if (grid.blocks == 0) {
    return {NB, NB};
  }
  const double share = double(grid.blocks) * NB / nth;
  const double rows = double(grid.offsets[grid.runs.back()]) / grid.blocks;
  const double width = std::sqrt(share * rows / n_per_block);
  const double run = double(grid.blocks) / (grid.runs.size() - 1);
  auto strips_of = [&](int64_t W) -> std::pair<int64_t, int64_t> {
    const int64_t S = detail::div_up(W, detail::div_up(W, cap));
    return {share / W > run ? S : W, S};
  };
  const int64_t W = detail::div_up(NB, std::clamp<int64_t>(std::lround(NB / width), 1, NB));
  const auto strips = strips_of(W);
  const auto narrow = strips_of(detail::div_up(NB, detail::div_up(NB, std::clamp<int64_t>(std::lround(width), 1, NB))));
  if (narrow.first != strips.first && share >= NB && share / W <= run &&
      4 * max_load(grid, NB, narrow.first, n_per_block, nth) <=
          3 * max_load(grid, NB, strips.first, n_per_block, nth)) {
    return narrow;
  }
  return strips;
}

// Own tiles by start cost in [expert run, W-wide strip, mb, nb] order.
// Visit each owned nonempty tile once, in S-wide sub-strips with column rotation.
// NB, W, S and nth must be positive; 0 <= ith < nth and rotate >= 0.
template <typename func_t>
void for_each_tile(
    const TileGrid& grid, int64_t NB, int64_t W, int64_t S, int ith, int nth, const func_t& f, int64_t rotate = 0) {
  const int64_t total = grid.cost.back() * NB;
  const int64_t lo = total * ith / nth;
  const int64_t hi = total * (ith + 1) / nth;
  if (lo >= hi) {
    return;
  }
  const int64_t num_runs = grid.runs.size() - 1;
  int64_t r = std::upper_bound(grid.runs.begin(), grid.runs.begin() + num_runs, lo,
                               [&](int64_t value, int64_t mb) { return value < grid.cost[mb] * NB; }) -
              grid.runs.begin() - 1;
  for (; r < num_runs; ++r) {
    const int64_t mb0 = grid.runs[r], mb1 = grid.runs[r + 1];
    for (int64_t nb0 = 0; nb0 < NB; nb0 += W) {
      const int64_t width = std::min(W, NB - nb0);
      const int64_t start = grid.cost[mb0] * NB + (grid.cost[mb1] - grid.cost[mb0]) * nb0;
      auto row_start = [&](int64_t mb) { return start + (grid.cost[mb] - grid.cost[mb0]) * width; };
      if (start >= hi) {
        return;
      }
      if (row_start(mb1) <= lo) {
        continue;
      }
      int64_t first = mb0, last = mb1;
      while (first < last && row_start(first + 1) <= lo) {
        ++first;
      }
      while (last > first && row_start(last - 1) >= hi) {
        --last;
      }
      for (int64_t sb0 = nb0; sb0 < nb0 + width; sb0 += S) {
        const int64_t sb1 = std::min(sb0 + S, nb0 + width);
        for (int64_t mb = first; mb < last; ++mb) {
          const int64_t cost = grid.cost[mb + 1] - grid.cost[mb];
          if (cost == 0) {
            continue;
          }
          const int64_t begin = row_start(mb);
          const int64_t nb_begin = nb0 + (begin < lo ? detail::div_up(lo - begin, cost) : 0);
          const int64_t nb_end = std::min(sb1, nb0 + detail::div_up(hi - begin, cost));
          const int64_t nb_first = std::max(nb_begin, sb0), n = nb_end - nb_first;
          for (int64_t k = 0; k < n; ++k) {
            f(mb, nb_first + (k + rotate) % n);
          }
        }
      }
    }
  }
}

}  // namespace cpu_sgl::moe
