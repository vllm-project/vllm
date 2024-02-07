/*
 * Adapted from https://github.com/InternLM/lmdeploy
 * Copyright (c) OpenMMLab. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#pragma once

#include <array>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace vllm {
namespace autoquant {

struct Metric {
    int  id;
    bool feasible;
    bool prefer;

    std::array<int, 3> cta_shape;
    std::array<int, 3> warp_shape;

    int   warps;
    int   stages;
    int   max_active_ctas;
    float smem;

    float cta_cnt_m;
    float cta_cnt_n;
    float cta_iter_k;
    float grid_size;

    int   active_ctas;
    float waves;
    float waves1;
    float occupancy;

    float tile_efficiency;
    float wave_efficiency;

    float grid_a0;
    float grid_b0;
    float grid_a1;
    float grid_b1;
    float grid_mm;

    float grid_sum;
    float grid_norm;

    float cta_sum;
    float cta_wave;

    int   best;
    float time;
    int   count;
};

inline void DumpMetrics(std::ostream& os, const std::vector<Metric>& metrics, const std::vector<int>& indices = {})
{
    auto dump_shape = [](const std::array<int, 3>& shape) {
        std::stringstream ss;
        ss << std::setw(4) << shape[0] << std::setw(4) << shape[1] << std::setw(4) << shape[2];
        return ss.str();
    };

    std::vector<std::tuple<std::string, int>> infos{
        {"id", 4},       {"valid", 6},      {"cta_mnk", 14},   {"warp_mnk", 14},   {"warps", 6},     {"stages", 8},
        {"smem", 8},     {"cta_cnt_m", 10}, {"cta_cnt_n", 10}, {"cta_iter_k", 11}, {"max_ctas", 9},  {"act_ctas", 10},
        {"waves", 12},   {"waves1", 12},    {"occupancy", 12}, {"%tile", 10},      {"%wave", 10},    {"grid_a0", 12},
        {"grid_b0", 12}, {"grid_a1", 12},   {"grid_b1", 12},   {"grid_mm", 12},    {"grid_sum", 12}, {"cta_cnt", 8},
        {"cta_sum", 8},  {"cta_wave", 9},   {"grid_norm", 12}, {"time", 12},       {"best", 7}};

    for (const auto& [name, width] : infos) {
        os << std::setw(width) << name;
    }
    os << "\n";

    for (size_t i = 0; i < metrics.size(); ++i) {
        auto& metric = indices.empty() ? metrics[i] : metrics[indices[i]];
        int   c      = 0;
        os << std::setw(std::get<1>(infos[c++])) << metric.id;
        os << std::setw(std::get<1>(infos[c++])) << metric.feasible;
        os << std::setw(std::get<1>(infos[c++])) << dump_shape(metric.cta_shape);
        os << std::setw(std::get<1>(infos[c++])) << dump_shape(metric.warp_shape);
        os << std::setw(std::get<1>(infos[c++])) << metric.warps;
        os << std::setw(std::get<1>(infos[c++])) << metric.stages;
        os << std::setw(std::get<1>(infos[c++])) << metric.smem;
        os << std::setw(std::get<1>(infos[c++])) << metric.cta_cnt_m;
        os << std::setw(std::get<1>(infos[c++])) << metric.cta_cnt_n;
        os << std::setw(std::get<1>(infos[c++])) << metric.cta_iter_k;
        os << std::setw(std::get<1>(infos[c++])) << metric.max_active_ctas;
        os << std::setw(std::get<1>(infos[c++])) << metric.active_ctas;
        os << std::setw(std::get<1>(infos[c++])) << metric.waves;
        os << std::setw(std::get<1>(infos[c++])) << metric.waves1;
        os << std::setw(std::get<1>(infos[c++])) << metric.occupancy;
        os << std::setw(std::get<1>(infos[c++])) << metric.tile_efficiency;
        os << std::setw(std::get<1>(infos[c++])) << metric.wave_efficiency;
        os << std::setw(std::get<1>(infos[c++])) << metric.grid_a0;
        os << std::setw(std::get<1>(infos[c++])) << metric.grid_b0;
        os << std::setw(std::get<1>(infos[c++])) << metric.grid_a1;
        os << std::setw(std::get<1>(infos[c++])) << metric.grid_b1;
        os << std::setw(std::get<1>(infos[c++])) << metric.grid_mm;
        os << std::setw(std::get<1>(infos[c++])) << metric.grid_sum;
        os << std::setw(std::get<1>(infos[c++])) << metric.grid_size;
        os << std::setw(std::get<1>(infos[c++])) << metric.cta_sum;
        os << std::setw(std::get<1>(infos[c++])) << metric.cta_wave;
        os << std::setw(std::get<1>(infos[c++])) << metric.grid_norm;
        os << std::setw(std::get<1>(infos[c++])) << metric.time * 1000 / metric.count;
        os << std::setw(std::get<1>(infos[c++])) << (metric.best ? "*" : "");
        os << "\n";
    }
}

}  // namespace autoquant
}  // namespace vllm
