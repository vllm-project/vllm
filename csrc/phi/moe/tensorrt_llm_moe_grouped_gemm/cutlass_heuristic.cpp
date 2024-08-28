/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "cutlass_heuristic.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // #ifndef _WIN32

#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"
#include "tensorrt_llm/common/assert.h"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif // #ifndef _WIN32

#include <cuda_runtime_api.h>
#include <set>
#include <vector>

using namespace tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

struct TileShape
{
    int m;
    int n;
};

TileShape get_cta_shape_for_config(CutlassTileConfig tile_config)
{
    switch (tile_config)
    {
    case CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64: return TileShape{16, 128};
    case CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64: return TileShape{16, 256};
    case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64: return TileShape{32, 128};
    case CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64: return TileShape{64, 64};
    case CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64:
    case CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64: return TileShape{64, 128};
    case CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64: return TileShape{128, 64};
    case CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8:
    case CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64:
    case CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64:
    case CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64: return TileShape{128, 128};
    case CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64: return TileShape{128, 256};
    case CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64: return TileShape{256, 128};
    default: TLLM_THROW("[get_grid_shape_for_config] Invalid config");
    }
}

bool is_valid_split_k_factor(const int64_t m, const int64_t n, const int64_t k, const TileShape tile_shape,
    int const split_k_factor, const size_t workspace_bytes, bool const is_weight_only)
{

    // All tile sizes have a k_tile of 64.
    static constexpr int k_tile = 64;

    // For weight-only quant, we need k and k_elements_per_split to be a multiple of cta_k
    if (is_weight_only)
    {
        if ((k % k_tile) != 0)
        {
            return false;
        }

        if ((k % split_k_factor) != 0)
        {
            return false;
        }

        int const k_elements_per_split = k / split_k_factor;
        if ((k_elements_per_split % k_tile) != 0)
        {
            return false;
        }
    }

    // Check that the workspace has sufficient space for this split-k factor
    int const ctas_in_m_dim = (m + tile_shape.m - 1) / tile_shape.m;
    int const ctas_in_n_dim = (n + tile_shape.n - 1) / tile_shape.n;
    int const required_ws_bytes = split_k_factor == 1 ? 0 : sizeof(int) * ctas_in_m_dim * ctas_in_n_dim;

    if (required_ws_bytes > workspace_bytes)
    {
        return false;
    }

    return true;
}

std::vector<CutlassTileConfig> get_candidate_tiles(
    int const sm, CutlassGemmConfig::CandidateConfigTypeParam const config_type_param)
{
    enum class CutlassGemmType : char
    {
        Default,
        WeightOnly,
        Simt,
        Int8
    };

    CutlassGemmType gemm_type = CutlassGemmType::Default;
    if (config_type_param & CutlassGemmConfig::SIMT_ONLY)
    {
        gemm_type = CutlassGemmType::Simt;
    }
    else if (config_type_param & CutlassGemmConfig::WEIGHT_ONLY)
    {
        gemm_type = CutlassGemmType::WeightOnly;
    }
    else if (config_type_param & CutlassGemmConfig::INT8_ONLY)
    {
        gemm_type = CutlassGemmType::Int8;
    }

    std::vector<CutlassTileConfig> base_configs{
        CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64, CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64};
    if (sm >= 75)
    {
        base_configs.push_back(CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64);
    }

    switch (gemm_type)
    {
    case CutlassGemmType::Simt: return {CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8};
    case CutlassGemmType::WeightOnly:
        if (sm >= 75)
        {
            return {CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64,
                CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64,
                CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
                CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
                CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64};
        }
        else
        {
            return {CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
                CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64};
        }
    case CutlassGemmType::Int8:
        return {CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64,
            CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
            CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64,
            CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64,
            CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64,
            CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64};
    default: return base_configs;
    }
}

std::vector<CutlassGemmConfig> get_candidate_configs(
    int sm, int const max_split_k, CutlassGemmConfig::CandidateConfigTypeParam const config_type_param)
{
    std::vector<CutlassTileConfig> tiles = get_candidate_tiles(sm, config_type_param);

    std::vector<CutlassGemmConfig> candidate_configs;
    bool const int8_configs_only = config_type_param & CutlassGemmConfig::INT8_ONLY;
    int const min_stages = int8_configs_only ? 3 : 2;
    int const max_stages = int8_configs_only ? 6 : (sm >= 80 ? 4 : 2);
    for (auto const& tile_config : tiles)
    {
        for (int stages = min_stages; stages <= max_stages; ++stages)
        {
            CutlassGemmConfig config(tile_config, SplitKStyle::NO_SPLIT_K, 1, stages);
            candidate_configs.push_back(config);
            if (sm >= 75)
            {
                for (int split_k_factor = 2; split_k_factor <= max_split_k; ++split_k_factor)
                {
                    auto config = CutlassGemmConfig{tile_config, SplitKStyle::SPLIT_K_SERIAL, split_k_factor, stages};
                    candidate_configs.push_back(config);
                }
            }
        }
    }

    return candidate_configs;
}

CutlassGemmConfig estimate_best_config_from_occupancies(std::vector<CutlassGemmConfig> const& candidate_configs,
    std::vector<int> const& occupancies, const int64_t m, const int64_t n, const int64_t k, const int64_t num_experts,
    int const split_k_limit, const size_t workspace_bytes, int const multi_processor_count, int const is_weight_only)
{

    if (occupancies.size() != candidate_configs.size())
    {
        TLLM_THROW(
            "[estimate_best_config_from_occupancies] occpancies and "
            "candidate configs vectors must have equal length.");
    }

    CutlassGemmConfig best_config;
    // Score will be [0, 1]. The objective is to minimize this score.
    // It represents the fraction of SM resources unused in the last wave.
    float config_score = 1.0f;
    int config_waves = INT_MAX;
    int current_m_tile = 0;

    int const max_split_k = n >= multi_processor_count * 256 ? 1 : split_k_limit;
    for (int ii = 0; ii < candidate_configs.size(); ++ii)
    {
        CutlassGemmConfig candidate_config = candidate_configs[ii];
        TileShape tile_shape = get_cta_shape_for_config(candidate_config.tile_config);
        int occupancy = occupancies[ii];

        if (occupancy == 0)
        {
            continue;
        }

        // Keep small tile sizes when possible.
        if (best_config.tile_config != CutlassTileConfig::ChooseWithHeuristic && m < current_m_tile
            && current_m_tile < tile_shape.m)
        {
            continue;
        }

        int const ctas_in_m_dim = (m + tile_shape.m - 1) / tile_shape.m;
        int const ctas_in_n_dim = (n + tile_shape.n - 1) / tile_shape.n;

        for (int split_k_factor = 1; split_k_factor <= max_split_k; ++split_k_factor)
        {
            if (is_valid_split_k_factor(m, n, k, tile_shape, split_k_factor, workspace_bytes, is_weight_only))
            {
                int const ctas_per_wave = occupancy * multi_processor_count;
                int const ctas_for_problem = ctas_in_m_dim * ctas_in_n_dim * split_k_factor;

                int const num_waves_total = (ctas_for_problem + ctas_per_wave - 1) / ctas_per_wave;
                float const num_waves_fractional = ctas_for_problem / float(ctas_per_wave);
                float const current_score = float(num_waves_total) - num_waves_fractional;

                float const score_slack = 0.1f;
                if (current_score < config_score
                    || ((config_waves > num_waves_total) && (current_score < config_score + score_slack)))
                {
                    config_score = current_score;
                    config_waves = num_waves_total;
                    SplitKStyle split_style
                        = split_k_factor > 1 ? SplitKStyle::SPLIT_K_SERIAL : SplitKStyle::NO_SPLIT_K;
                    best_config = CutlassGemmConfig(
                        candidate_config.tile_config, split_style, split_k_factor, candidate_config.stages);
                    current_m_tile = tile_shape.m;
                }
                else if (current_score == config_score
                    && (best_config.stages < candidate_config.stages || split_k_factor < best_config.split_k_factor
                        || current_m_tile < tile_shape.m))
                {
                    // Prefer deeper pipeline or smaller split-k
                    SplitKStyle split_style
                        = split_k_factor > 1 ? SplitKStyle::SPLIT_K_SERIAL : SplitKStyle::NO_SPLIT_K;
                    best_config = CutlassGemmConfig(
                        candidate_config.tile_config, split_style, split_k_factor, candidate_config.stages);
                    current_m_tile = tile_shape.m;
                    config_waves = num_waves_total;
                }
            }
        }
    }

    if (best_config.tile_config == CutlassTileConfig::ChooseWithHeuristic)
    {
        TLLM_THROW("Heurisitc failed to find a valid config.");
    }

    return best_config;
}

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
