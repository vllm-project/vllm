# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local sparse-attention indexer decode (`VLLM_INDEXER_LOCALIZE`).

On GPUs with two memory locality domains, the indexer's decode scan reads each
page with the SMs of the domain that holds it. `placement` alternates the KV
cache between the domains and gives each a green-context stream; every decode
step, `build` splits each request's pages by domain and builds both domains'
DeepGEMM schedules; `scan` scans and selects per domain and merges (or joins
the scores); `kernels` compiles the CUDA kernels at runtime with NVRTC.
"""
