// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include "common.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <utility>

namespace sgl::dense_int8 {

inline constexpr int64_t kMinL2Bytes = 512 * 1024;
inline constexpr int64_t kStreamOutputThresholdBytes = 8LL * 1024 * 1024;

inline int64_t a_stride_for_k(int64_t k) { return (k % 128 == 0) ? k + 64 : k; }

inline int64_t partition_begin(int64_t size, int64_t parts, int64_t rank) {
  return rank * (size / parts) + std::min(rank, size % parts);
}

template <typename Plan>
inline const Plan& cached_select_plan(int64_t m, int64_t n, int64_t k,
                                      int64_t team, int64_t l2_bytes) {
  struct Entry {
    int64_t m = 0, n = 0, k = 0, team = 0, l2_bytes = 0;
    Plan plan;
  };
  static thread_local std::array<Entry, 16> entries;
  static thread_local size_t next = 0;
  for (const auto& entry : entries) {
    if (entry.team == team && entry.m == m && entry.n == n && entry.k == k &&
        entry.l2_bytes == l2_bytes) {
      return entry.plan;
    }
  }
  auto& entry = entries[next];
  next = (next + 1) % entries.size();
  entry.m = m;
  entry.n = n;
  entry.k = k;
  entry.team = team;
  entry.l2_bytes = l2_bytes;
  entry.plan = Plan::select(m, n, k, team, l2_bytes);
  return entry.plan;
}

struct DenseTileOwnership {
  int64_t workers = 0, group_m = 1;
  int64_t mb = 0, nb = 0;
  bool partial_m = false;

  template <typename func_t>
  void each(int64_t worker, const func_t& f) const {
    if (worker >= workers) {
      return;
    }
    const int64_t tasks = mb * nb;
    int64_t begin = partition_begin(tasks, workers, worker);
    const int64_t end = partition_begin(tasks, workers, worker + 1);
    int64_t pm0 = 0, pm1 = 0, pn0 = 0, pn1 = 0;
    bool pending = false;
    auto emit = [&](int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
      if (pending && pm0 == m0 && pm1 == m1 && pn1 == n0) {
        pn1 = n1;
      } else if (pending && pn0 == n0 && pn1 == n1 && pm1 == m0) {
        pm1 = m1;
      } else {
        if (pending) {
          f(pm0, pm1, pn0, pn1);
        }
        pm0 = m0;
        pm1 = m1;
        pn0 = n0;
        pn1 = n1;
        pending = true;
      }
    };
    auto visit = [&](int64_t body_mb, int64_t first, int64_t last) {
      while (first < last) {
        const int64_t m0 = first / (nb * group_m) * group_m;
        const int64_t height = std::min(group_m, body_mb - m0);
        const int64_t offset = first - m0 * nb;
        const int64_t row = offset % height;
        const int64_t count = std::min(last - first, height - row);
        emit(m0 + row, m0 + row + count, offset / height, offset / height + 1);
        first += count;
      }
    };
    const int64_t edge_rows = mb % group_m != 0 ? mb % group_m
                              : partial_m       ? 1
                                                : 0;
    if (edge_rows == 0) {
      visit(mb, begin, end);
    } else {
      // Distribute the short final row group across balanced rank ranges.
      const int64_t body_mb = mb - edge_rows;
      while (begin < end) {
        const int64_t n = begin / mb;
        const int64_t offset = begin % mb;
        if (offset < body_mb) {
          const int64_t count = std::min(end - begin, body_mb - offset);
          const int64_t first = begin - n * edge_rows;
          visit(body_mb, first, first + count);
          begin += count;
        } else {
          const int64_t count = std::min(end - begin, mb - offset);
          emit(offset, offset + count, n, n + 1);
          begin += count;
        }
      }
    }
    if (pending) {
      f(pm0, pm1, pn0, pn1);
    }
  }
};

struct DenseKernelConfig {
  int64_t m = 0, n = 0, k = 0;
  int64_t a_stride = 0, b_stride = 0, output_stride = 0;
  int64_t macro_m = 512, macro_n = 256, k_block = 512;
  bool stream_output = true;

  // Pad Ctmp rows by 16 int32s (64B) to avoid 4 KiB L1d set aliasing.
  int64_t c_stride() const { return macro_n + 16; }
  int64_t worker_scratch() const {
    return std::max<int64_t>(3072, macro_m * c_stride());
  }
};

// Coarse 2D macro-tile planner for large compute-bound shapes (M > 512,
// N > 2048). Uses balanced 2D macro-tile ownership with column-local core
// mapping across M row-bands for shared L3 weight reuse.
struct DenseCoarsePlan {
  struct Tile {
    int64_t m, n, rows, cols;
  };

  DenseKernelConfig kernel;
  DenseTileOwnership macro_ownership;
  int64_t productive_workers = 0;
  double imbalance = 0;
  bool col_locality = false;
  bool valid = false;

  static constexpr int64_t kDefaultKBlock = 512;
  static constexpr double kTargetImbalance = 1.15;
  static constexpr double kMaxImbalance = 1.35;

  static bool shape_eligible(int64_t m, int64_t n, int64_t k) {
    return m >= 64 && n >= 32 && n % 32 == 0 && k >= 128 && k % 64 == 0;
  }

  static bool l2_eligible(int64_t l2_bytes) { return l2_bytes >= kMinL2Bytes; }

  // Bijection on [0, team) grouping physical cores sharing the same N-column
  // slice across M row-bands onto contiguous IDs for shared L3 weight reuse.
  int64_t map_worker(int64_t worker) const {
    if (!col_locality || macro_ownership.mb <= 1 ||
        macro_ownership.workers <= 1) {
      return worker;
    }
    const int64_t team = macro_ownership.workers;
    const int64_t bands = std::min(macro_ownership.mb, team);
    const int64_t q = team / bands;
    const int64_t r = team % bands;
    const int64_t rank = worker / bands;
    const int64_t band = worker % bands;
    if (rank < q) {
      return band * q + std::min(band, r) + rank;
    }
    return band * q + band + q;
  }

  static DenseCoarsePlan make_geometry(int64_t m, int64_t n, int64_t k,
                                       int64_t team, int64_t macro_m,
                                       int64_t macro_n, int64_t k_block,
                                       bool row_band_local) {
    DenseCoarsePlan plan;
    plan.kernel.m = m;
    plan.kernel.n = n;
    plan.kernel.k = k;
    plan.kernel.a_stride = a_stride_for_k(k);
    plan.kernel.b_stride = k + int64_t(sizeof(int32_t));
    plan.kernel.output_stride = n;
    plan.kernel.macro_m = macro_m;
    plan.kernel.macro_n = macro_n;
    plan.kernel.k_block = k_block;
    plan.kernel.stream_output = (2 * m * n >= kStreamOutputThresholdBytes);
    plan.col_locality = row_band_local;
    plan.macro_ownership.mb = div_up(m, macro_m);
    plan.macro_ownership.nb = div_up(n, macro_n);
    plan.macro_ownership.workers = team;
    plan.macro_ownership.group_m =
        row_band_local
            ? 1
            : std::max<int64_t>(1, plan.macro_ownership.mb /
                                       div_up(team, plan.macro_ownership.nb));
    plan.macro_ownership.partial_m = m % macro_m != 0;
    double total = 0, largest = 0;
    for (int64_t worker = 0; worker < team; ++worker) {
      double live = 0;
      plan.each_macro_tile(worker, [&](int64_t tile) {
        const auto bounds = plan.tile_bounds(tile);
        live += double(bounds.rows) * double(bounds.cols);
      });
      plan.productive_workers += live > 0;
      total += live;
      largest = std::max(largest, live);
    }
    plan.imbalance = total > 0 ? largest * team / total : 0;
    plan.valid = (plan.productive_workers == team) &&
                 (team == 1 || plan.imbalance <= kMaxImbalance);
    return plan;
  }

  static DenseCoarsePlan select(int64_t m, int64_t n, int64_t k, int64_t team,
                                int64_t l2_bytes) {
    DenseCoarsePlan best;
    if (team <= 0 || !shape_eligible(m, n, k) || !l2_eligible(l2_bytes)) {
      return best;
    }
    if (div_up(m, int64_t(32)) * (n / 32) < team) {
      return best;
    }
    const int64_t a_stride = a_stride_for_k(k);
    const int64_t l2_budget = l2_bytes * 3 / 4;
    const int64_t l2_half = l2_bytes / 2;
    const int64_t max_mm = div_up(m, int64_t(32)) * 32;

    auto working_set = [&](int64_t mm, int64_t nn, int64_t kb) {
      const int64_t active_k = std::min(k, kb);
      return 4 * mm * (nn + 16) + mm * active_k + nn * active_k;
    };

    auto fits_l2 = [&](int64_t mm, int64_t nn, int64_t kb) {
      return working_set(mm, nn, kb) <= l2_budget;
    };

    auto choose_kb = [&](int64_t mm, int64_t nn) {
      return working_set(mm, nn, 1024) <= l2_half ? int64_t(1024)
                                                  : kDefaultKBlock;
    };

    auto evaluate = [&](int64_t cm, int64_t cn, int64_t kb,
                        bool row_band_local) -> bool {
      const int64_t mm = std::min(cm, max_mm);
      const int64_t nn = std::min(cn, n);
      if (!fits_l2(mm, nn, kb) ||
          (row_band_local && mm * a_stride > l2_budget)) {
        return false;
      }
      const auto candidate =
          make_geometry(m, n, k, team, mm, nn, kb, row_band_local);
      if (!candidate.valid) {
        return false;
      }
      if (candidate.imbalance <= kTargetImbalance) {
        best = candidate;
        return true;
      }
      if (!best.valid || candidate.imbalance < best.imbalance - 1e-6) {
        best = candidate;
      }
      return false;
    };

    const int64_t min_resident_rows = std::min<int64_t>(128, max_mm);
    if (min_resident_rows * a_stride <= l2_budget) {
      constexpr std::array<std::pair<int64_t, int64_t>, 4> kLowK = {{
          {256, 32},
          {128, 32},
          {64, 32},
          {32, 32},
      }};
      for (const auto& [cm, cn] : kLowK) {
        if (evaluate(cm, cn, 1024, true)) {
          return best;
        }
      }
    }

    const int64_t m512 = std::min<int64_t>(512, max_mm);
    const int64_t n256 = std::min<int64_t>(256, n);
    const int64_t kb512 = choose_kb(m512, n256);
    if (fits_l2(m512, n256, kb512)) {
      const auto c512 = make_geometry(m, n, k, team, m512, n256, kb512, false);
      if (c512.valid && c512.imbalance <= 1.20) {
        return c512;
      }
      if (c512.valid && !best.valid) {
        best = c512;
      }
    }

    constexpr std::array<std::pair<int64_t, int64_t>, 10> kHighK = {{
        {256, 256},
        {256, 128},
        {256, 64},
        {256, 32},
        {128, 128},
        {128, 64},
        {128, 32},
        {64, 64},
        {64, 32},
        {32, 32},
    }};
    for (const auto& [cm, cn] : kHighK) {
      if (cn < 128 && best.valid && best.kernel.macro_n >= 128 &&
          best.imbalance <= 1.25) {
        return best;
      }
      const int64_t mm = std::min(cm, max_mm);
      const int64_t nn = std::min(cn, n);
      if (evaluate(cm, cn, choose_kb(mm, nn), false)) {
        return best;
      }
    }
    return best;
  }

  static const DenseCoarsePlan& cached_select(int64_t m, int64_t n, int64_t k,
                                              int64_t team, int64_t l2_bytes) {
    return cached_select_plan<DenseCoarsePlan>(m, n, k, team, l2_bytes);
  }

  int64_t tiles() const {
    return div_up(kernel.m, kernel.macro_m) * div_up(kernel.n, kernel.macro_n);
  }
  Tile tile_bounds(int64_t tile) const {
    const int64_t nb = div_up(kernel.n, kernel.macro_n);
    const int64_t mc = tile / nb * kernel.macro_m;
    const int64_t nc = tile % nb * kernel.macro_n;
    return {mc, nc, std::min(kernel.macro_m, kernel.m - mc),
            std::min(kernel.macro_n, kernel.n - nc)};
  }
  template <typename func_t>
  void each_macro_tile(int64_t worker, const func_t& f) const {
    macro_ownership.each(map_worker(worker),
                         [&](int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
                           const int64_t span = n1 - n0;
                           if (col_locality && m1 - m0 == 1 && span > 1) {
                             const int64_t shift = (m0 % 2) * (span / 2);
                             for (int64_t step = 0; step < span; ++step) {
                               const int64_t n = n0 + (step + shift) % span;
                               f(m0 * macro_ownership.nb + n);
                             }
                           } else {
                             for (int64_t n = n0; n < n1; ++n) {
                               for (int64_t m = m0; m < m1; ++m) {
                                 f(m * macro_ownership.nb + n);
                               }
                             }
                           }
                         });
  }
};

// Roofline-driven 3D (gm x gn x gk) AMX BRGEMM planner for small-to-mid M
// (4 <= M <= 512), narrow N (N <= 2048), and narrow-N high-K projections
// (M <= 2048, N <= 4096, K > 4096). M <= 3 falls through to the upstream VNNI
// path.
struct DenseSmallGridPlan {
  struct Block {
    int64_t r0, r1, p0, p1, k0, k1, im, in, ik;
  };

  int64_t m = 0, n = 0, k = 0, team = 0, l2_bytes = 0;
  int64_t gm = 1, gn = 1, gk = 1;
  int64_t k_block = 512;
  int64_t n_chunk = 1;
  int64_t a_stride = 0, b_stride = 0;
  bool stream_output = false;
  double estimate_ns = 0;
  bool valid = false;

  static constexpr double kCoreFill = 10.0;
  static constexpr double kTeamFill = 550.0;
  static constexpr double kAmxOps = 2000.0;
  static constexpr double kQuantElems = 5.0;
  static constexpr double kL2Fill = 100.0;

  static double barrier_ns(int64_t workers) {
    return workers > 1 ? 470.0 * std::log2(double(workers)) : 0.0;
  }

  static double overlap(double a, double b) {
    return std::max(a, b) + 0.35 * std::min(a, b);
  }

  static bool shape_eligible(int64_t m, int64_t n, int64_t k) {
    return m >= 4 && n >= 32 && n % 32 == 0 && k >= 128 && k % 64 == 0;
  }

  static bool preferred(int64_t m, int64_t n, int64_t k, int64_t team,
                        int64_t l2_bytes) {
    if (!shape_eligible(m, n, k) || l2_bytes < kMinL2Bytes || team <= 0) {
      return false;
    }
    return m <= 512 || n <= 2048 || (m <= 2048 && n <= 4096 && k > 4096);
  }

  int64_t workers() const { return gm * gn * gk; }
  int64_t max_rows() const {
    const int64_t units = div_up(m, int64_t(16));
    return std::min(m, div_up(units, gm) * 16);
  }
  int64_t max_panels() const { return div_up(n / 32, gn); }
  int64_t max_k_slice() const { return div_up(k / 64, gk) * 64; }
  // Pad per-panel Ctmp stride by 16 int32s (64B) to avoid L2 set aliasing.
  int64_t panel_elems(int64_t rows) const { return rows * 32 + 16; }
  int64_t max_slot_elems() const {
    return std::max<int64_t>(1024, n_chunk * panel_elems(max_rows()));
  }

  // Locality order: ik innermost keeps Split-K reduction peers on contiguous
  // cores; im middle keeps M-bands sharing B columns adjacent.
  Block block(int64_t worker) const {
    const int64_t ik = worker % gk;
    const int64_t im = (worker / gk) % gm;
    const int64_t in = worker / (gk * gm);
    const int64_t m_units = div_up(m, int64_t(16));
    const int64_t panels = n / 32;
    const int64_t k_units = k / 64;
    return {
        partition_begin(m_units, gm, im) * 16,
        std::min(m, partition_begin(m_units, gm, im + 1) * 16),
        partition_begin(panels, gn, in),
        partition_begin(panels, gn, in + 1),
        partition_begin(k_units, gk, ik) * 64,
        partition_begin(k_units, gk, ik + 1) * 64,
        im,
        in,
        ik,
    };
  }

  bool finalize_and_cost() {
    a_stride = a_stride_for_k(k);
    b_stride = k + int64_t(sizeof(int32_t));
    stream_output = (2 * m * n >= kStreamOutputThresholdBytes);
    const int64_t rows = max_rows();
    const int64_t panels = max_panels();
    const int64_t cols = panels * 32;
    const int64_t k_slice = max_k_slice();
    if (rows <= 0 || panels <= 0 || k_slice <= 0) {
      return false;
    }

    const int64_t active_k = std::min(k_slice, k_block);
    const int64_t l2_budget = l2_bytes * 13 / 16;
    const int64_t a_ws = rows * a_stride_for_k(active_k);
    if (a_ws >= l2_budget) {
      return false;
    }
    if (gk > 1) {
      n_chunk = panels;
    } else {
      const int64_t per_panel =
          panel_elems(rows) * int64_t(sizeof(int32_t)) + 32 * active_k;
      const int64_t max_p =
          std::max<int64_t>(1, (l2_budget - a_ws) / per_panel);
      n_chunk = std::min<int64_t>({panels, max_p, int64_t(16)});
    }

    const int64_t chunk_cols = n_chunk * 32;
    const int64_t c_bytes =
        n_chunk * panel_elems(rows) * int64_t(sizeof(int32_t));
    const int64_t b_ws = chunk_cols * active_k;
    if (c_bytes + a_ws + b_ws > l2_budget) {
      return false;
    }

    const int64_t chunks = div_up(panels, n_chunk);
    const int64_t slabs = div_up(k_slice, k_block);
    const double dm = double(m), dn = double(n), dk = double(k);

    const double q_elems = double(div_up(m, team)) * dk;
    const double pass1_ns =
        overlap(3.0 * q_elems / kCoreFill, q_elems / kQuantElems);
    const double sync_ns =
        barrier_ns(team) + ((gk > 1) ? (1.5 * barrier_ns(team)) : 0.0);

    const bool a_fits_l2 = (rows * k_slice + c_bytes <= l2_bytes * 3 / 4);
    const double a_miss_bytes =
        double(rows) * double(k_slice) * (a_fits_l2 ? 1.0 : double(chunks));
    const double extra_team_bytes = 3.0 * dm * dk + double(gn) * dm * dk;

    const double b_bytes = double(cols) * double(k_slice);
    const double c_red_bytes =
        (gk > 1) ? (12.0 * double(rows) * double(cols)) : 0.0;
    const double out_bytes = 2.0 * double(rows) * double(cols) / double(gk);
    const double reload_ns =
        (slabs <= 1)
            ? 0.0
            : (8.0 * double(rows) * double(cols) * double(slabs - 1) / kL2Fill);
    const double eff_rows = double(div_up(rows, int64_t(16)) * 16);
    const double ops = 2.0 * eff_rows * double(cols) * double(k_slice);
    const double wrapper_ns = double(slabs) * double(panels) * 260.0;

    const double gemm_ns =
        overlap((a_miss_bytes + b_bytes + c_red_bytes + out_bytes) / kCoreFill,
                ops / kAmxOps + reload_ns + wrapper_ns);
    const double worker_ns = pass1_ns + gemm_ns;

    const double team_bytes = extra_team_bytes + double(gm) * dn * dk +
                              ((gk > 1) ? 4.0 * double(gk) * dm * dn : 0.0) +
                              2.0 * dm * dn;
    const double dram_bw_ns = (dn * dk + 2.0 * dm * dn) / kTeamFill;
    const double team_bw_ns = std::max(dram_bw_ns, team_bytes / kTeamFill);

    estimate_ns = std::max(worker_ns, team_bw_ns) + sync_ns;
    valid = true;
    return true;
  }

  static DenseSmallGridPlan select(int64_t m, int64_t n, int64_t k,
                                   int64_t team, int64_t l2_bytes) {
    DenseSmallGridPlan best;
    if (team <= 0 || !shape_eligible(m, n, k) || l2_bytes < kMinL2Bytes) {
      return best;
    }

    const int64_t m_units = div_up(m, int64_t(16));
    const int64_t panels = n / 32;
    const int64_t k_units = k / 64;
    const int64_t max_gm = std::min<int64_t>({team, m_units, int64_t(8)});
    const int64_t min_gm =
        (m > 512) ? std::min<int64_t>(max_gm, div_up(m, int64_t(512))) : 1;

    for (int64_t gm = min_gm; gm <= max_gm; ++gm) {
      const int64_t max_gn = std::min(team / gm, panels);
      for (int64_t gn = 1; gn <= max_gn; ++gn) {
        const int64_t max_gk = std::min<int64_t>(
            {team / (gm * gn), std::max<int64_t>(1, k / 256), k_units});
        for (int64_t gk = 1; gk <= max_gk; ++gk) {
          const int64_t k_slice = div_up(k_units, gk) * 64;
          auto try_cand = [&](int64_t kb) {
            DenseSmallGridPlan cand;
            cand.m = m;
            cand.n = n;
            cand.k = k;
            cand.team = team;
            cand.l2_bytes = l2_bytes;
            cand.gm = gm;
            cand.gn = gn;
            cand.gk = gk;
            cand.k_block = kb;
            if (!cand.finalize_and_cost()) {
              return;
            }
            const bool better = cand.estimate_ns < 0.995 * best.estimate_ns;
            const bool tied = cand.estimate_ns <= 1.005 * best.estimate_ns;
            if (!best.valid || better ||
                (tied &&
                 std::make_tuple(-cand.workers(), cand.gk, cand.gm) <
                     std::make_tuple(-best.workers(), best.gk, best.gm))) {
              best = cand;
            }
          };

          if (k_slice <= 1024) {
            try_cand(1024);
          }
          if (k_slice > 512) {
            try_cand(512);
          }
        }
      }
    }
    return best;
  }

  static const DenseSmallGridPlan& cached_select(int64_t m, int64_t n,
                                                 int64_t k, int64_t team,
                                                 int64_t l2_bytes) {
    return cached_select_plan<DenseSmallGridPlan>(m, n, k, team, l2_bytes);
  }
};

}  // namespace sgl::dense_int8
