// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// The CTA shape of the sparse-indexer top-k (`cooperative_topk` and
// `persistent_topk`: 1024 threads, one CTA per SM). Never launched: build.py
// asks CUDA's occupancy query how many clusters of such CTAs a locality
// domain's SMs hold at once.
extern "C" __global__ void __launch_bounds__(1024, 1) topk_cta_shape() {}
