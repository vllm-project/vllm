#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze MoE config files to extract patterns for better defaults."""

import json
import os
from collections import defaultdict

import numpy as np


def parse_all_configs():
    """Parse all config files and extract data."""
    configs_dir = os.path.dirname(os.path.abspath(__file__))

    configs_data = []

    for filename in os.listdir(configs_dir):
        if not filename.endswith('.json') or filename == 'README.json':
            continue

        # Parse filename
        name = filename[:-5]  # remove .json
        parts = name.split(',')

        config_info: dict[str, any] = {}
        for part in parts:
            if part.startswith('E='):
                config_info['E'] = int(part[2:])
            elif part.startswith('N='):
                config_info['N'] = int(part[2:])
            elif part.startswith('device_name='):
                config_info['device'] = part[12:]
            elif part.startswith('dtype='):
                config_info['dtype'] = part[6:]

        if 'E' not in config_info or 'N' not in config_info:
            continue

        config_info['filename'] = filename

        # Load the config file
        try:
            with open(os.path.join(configs_dir, filename)) as f:
                config_data = json.load(f)

            config_info['batch_configs'] = {}
            for batch_size, batch_config in config_data.items():
                config_info['batch_configs'][int(batch_size)] = batch_config

            configs_data.append(config_info)

        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

    return configs_data


def analyze_patterns(configs_data):
    """Analyze patterns in the config data."""

    print("=== MoE Config Analysis (All Vendors) ===\n")

    # Group by dtype
    by_dtype = defaultdict(list)
    for config in configs_data:
        dtype = config.get('dtype', 'unquantized')
        by_dtype[dtype].append(config)

    print("Configs by dtype:")
    for dtype, configs in by_dtype.items():
        print(f"  {dtype}: {len(configs)} configs")
    print()

    # Group by device families
    device_families = defaultdict(list)
    for config in configs_data:
        device = config.get('device', 'Unknown')
        if 'H100' in device:
            family = 'H100'
        elif 'A100' in device:
            family = 'A100'
        elif 'H200' in device:
            family = 'H200'
        elif 'H20' in device:
            family = 'H20'
        elif 'A800' in device:
            family = 'A800'
        elif 'MI300' in device:
            family = 'MI300'
        elif 'MI325' in device:
            family = 'MI325'
        elif 'AMD' in device:
            family = 'AMD_Other'
        else:
            family = 'Other'
        device_families[family].append(config)

    print("Configs by device family:")
    for family, configs in device_families.items():
        print(f"  {family}: {len(configs)} configs")
    print()

    # Analyze patterns for each dtype
    for dtype, configs in by_dtype.items():
        print(f"\n=== Analysis for {dtype} ===")
        analyze_dtype_patterns(configs, dtype)


def analyze_dtype_patterns(configs: list[dict], dtype: str):
    """Analyze patterns for a specific dtype."""

    # Collect all batch configs across all E/N combinations
    all_batch_configs = []
    E_values = set()
    N_values = set()

    for config in configs:
        E_values.add(config['E'])
        N_values.add(config['N'])

        for batch_size, batch_config in config['batch_configs'].items():
            entry = {
                'E': config['E'],
                'N': config['N'],
                'device': config['device'],
                'batch_size': batch_size,
                **batch_config
            }
            all_batch_configs.append(entry)

    print(f"Expert counts (E): {sorted(E_values)}")
    print(f"Hidden dims (N): {sorted(N_values)}")
    print(f"Total batch configs: {len(all_batch_configs)}")

    # Analyze block size patterns
    analyze_block_sizes(all_batch_configs)

    # Analyze warp and stage patterns
    analyze_warp_stage_patterns(all_batch_configs)

    # Analyze trends by E and N
    analyze_E_N_trends(all_batch_configs)


def analyze_block_sizes(configs):
    """Analyze block size patterns."""
    print("\n--- Block Size Analysis ---")
    if not configs:
        print("  No batch configs to analyze for block sizes.")
        return

    block_m_values = [c['BLOCK_SIZE_M'] for c in configs]
    block_n_values = [c['BLOCK_SIZE_N'] for c in configs]
    block_k_values = [c['BLOCK_SIZE_K'] for c in configs]

    print(
        f"BLOCK_SIZE_M: min={min(block_m_values)}, max={max(block_m_values)}, "
        f"most_common={max(set(block_m_values), key=block_m_values.count)}")
    print(
        f"BLOCK_SIZE_N: min={min(block_n_values)}, max={max(block_n_values)}, "
        f"most_common={max(set(block_n_values), key=block_n_values.count)}")
    print(
        f"BLOCK_SIZE_K: min={min(block_k_values)}, max={max(block_k_values)}, "
        f"most_common={max(set(block_k_values), key=block_k_values.count)}")

    # Look at combinations
    combinations: dict[tuple[int, int, int], int] = defaultdict(int)
    for c in configs:
        combo = (c['BLOCK_SIZE_M'], c['BLOCK_SIZE_N'], c['BLOCK_SIZE_K'])
        combinations[combo] += 1

    print("\nMost common (M,N,K) combinations:")
    for combo, count in sorted(combinations.items(),
                               key=lambda x: x[1],
                               reverse=True)[:10]:
        print(f"  {combo}: {count} times")


def analyze_warp_stage_patterns(configs):
    """Analyze warp and stage patterns."""
    print("\n--- Warp/Stage Analysis ---")

    warp_values = [c['num_warps'] for c in configs]
    stage_values = [c['num_stages'] for c in configs]
    group_m_values = [c['GROUP_SIZE_M'] for c in configs]

    print(f"num_warps: min={min(warp_values)}, max={max(warp_values)}, "
          f"most_common={max(set(warp_values), key=warp_values.count)}")
    print(f"num_stages: min={min(stage_values)}, max={max(stage_values)}, "
          f"most_common={max(set(stage_values), key=stage_values.count)}")
    print(
        f"GROUP_SIZE_M: min={min(group_m_values)}, max={max(group_m_values)}, "
        f"most_common={max(set(group_m_values), key=group_m_values.count)}")


def analyze_E_N_trends(configs):
    """Analyze trends based on E and N values."""
    print("\n--- E/N Trend Analysis ---")

    # Group by E ranges
    E_ranges = {
        'small_E': [c for c in configs if c['E'] <= 8],
        'medium_E': [c for c in configs if 8 < c['E'] <= 64],
        'large_E': [c for c in configs if c['E'] > 64]
    }

    # Group by N ranges
    N_ranges = {
        'small_N': [c for c in configs if c['N'] <= 1024],
        'medium_N': [c for c in configs if 1024 < c['N'] <= 4096],
        'large_N': [c for c in configs if c['N'] > 4096]
    }

    print("Trends by Expert Count (E):")
    for range_name, range_configs in E_ranges.items():
        if not range_configs:
            continue
        print(f"\n  {range_name} (n={len(range_configs)}):")
        analyze_range_stats(range_configs)

    print("\nTrends by Hidden Dim (N):")
    for range_name, range_configs in N_ranges.items():
        if not range_configs:
            continue
        print(f"\n  {range_name} (n={len(range_configs)}):")
        analyze_range_stats(range_configs)

    # Batch size trends
    print("\nTrends by Batch Size:")
    batch_ranges = {
        'tiny_batch': [c for c in configs if c['batch_size'] <= 8],
        'small_batch': [c for c in configs if 8 < c['batch_size'] <= 64],
        'medium_batch': [c for c in configs if 64 < c['batch_size'] <= 512],
        'large_batch': [c for c in configs if c['batch_size'] > 512]
    }

    for range_name, range_configs in batch_ranges.items():
        if not range_configs:
            continue
        print(f"\n  {range_name} (n={len(range_configs)}):")
        analyze_range_stats(range_configs)


def analyze_range_stats(configs):
    """Analyze stats for a range of configs."""
    if not configs:
        return

    # Collect values for analysis
    block_m_vals = [c['BLOCK_SIZE_M'] for c in configs]
    block_n_vals = [c['BLOCK_SIZE_N'] for c in configs]
    block_k_vals = [c['BLOCK_SIZE_K'] for c in configs]
    warps_vals = [c['num_warps'] for c in configs]
    stages_vals = [c['num_stages'] for c in configs]
    group_m_vals = [c['GROUP_SIZE_M'] for c in configs]

    def analyze_distribution(values, name):
        """Analyze distribution of values."""
        if not values:
            return

        # Basic stats
        avg = np.mean(values)
        std = np.std(values)
        median = np.median(values)

        # Mode (most common)
        from collections import Counter
        counter = Counter(values)
        mode = counter.most_common(1)[0][0]
        mode_count = counter.most_common(1)[0][1]
        mode_pct = (mode_count / len(values)) * 100

        # Top 3 most common values
        top3 = counter.most_common(3)

        print(f"    {name}:")
        print(f"      avg={avg:.1f} ± {std:.1f}, median={median:.1f}")
        print(f"      mode={mode} ({mode_pct:.1f}% of configs)")
        print(f"      top3: {[f'{val}({cnt})' for val, cnt in top3]}")

    analyze_distribution(block_m_vals, "BLOCK_SIZE_M")
    analyze_distribution(block_n_vals, "BLOCK_SIZE_N")
    analyze_distribution(block_k_vals, "BLOCK_SIZE_K")
    analyze_distribution(warps_vals, "num_warps")
    analyze_distribution(stages_vals, "num_stages")
    analyze_distribution(group_m_vals, "GROUP_SIZE_M")


if __name__ == "__main__":
    configs_data = parse_all_configs()
    print(f"Loaded {len(configs_data)} config files")
    analyze_patterns(configs_data)
