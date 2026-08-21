#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

source_root="${1:?source root is required}"
output_dir="${2:?output directory is required}"
reply_dir="$(find build -type d -path '*/.cmake/api/v1/reply' -print -quit)"
test -n "${reply_dir}"
build_dir="${reply_dir%/.cmake/api/v1/reply}"

ninja -C "${build_dir}" -t deps > /tmp/ninja-deps.txt
test -s /tmp/ninja-deps.txt
python3 tools/ci_test_selection/export_build_graph.py \
    "${build_dir}" --source-root "${source_root}" \
    --ninja-deps /tmp/ninja-deps.txt \
    --out /tmp/native-build-graph.jsonl \
    2> /tmp/native-build-graph-summary.json
python3 -c 'import json; d=json.load(open("/tmp/native-build-graph-summary.json")); assert d.get("header_dep_source") == "ninja_deps", d; assert d.get("dep_file_target_pairs", 0) > 0, d; assert d.get("include_edges", 0) > 0, d'
cat /tmp/native-build-graph-summary.json
python3 tools/ci_test_selection/extract_kernel_symbols.py \
    "$(find dist -maxdepth 1 -name '*.whl' -print -quit)" \
    --out "${output_dir}/kernel-map.jsonl"
python3 tools/ci_test_selection/extract_object_kernel_symbols.py \
    "${build_dir}" --source-root "${source_root}" \
    --ninja-deps /tmp/ninja-deps.txt \
    --build-graph /tmp/native-build-graph.jsonl \
    --out /tmp/object-kernel-map.jsonl \
    2> /tmp/object-kernel-map-summary.json
python3 -c 'import json; d=json.load(open("/tmp/object-kernel-map-summary.json")); assert d.get("kernel_objects", 0) > 0, d; assert d.get("translation_unit_kernel_edges", 0) > 0, d'
cat /tmp/object-kernel-map-summary.json
cat /tmp/native-build-graph.jsonl \
    "${output_dir}/kernel-map.jsonl" \
    /tmp/object-kernel-map.jsonl \
    > "${output_dir}/build-graph.jsonl"
test -s "${output_dir}/build-graph.jsonl"
python3 tools/ci_test_selection/validate_build_graph.py \
    "${output_dir}/build-graph.jsonl" \
    --required-file cmake/external_projects/flashmla.cmake \
    --required-target _flashmla_C
