# Vendored DeepGEMM headers (DSA indexer fused top-k)

A minimal, self-contained subset (10 headers) of DeepSeek's DeepGEMM
(<https://github.com/deepseek-ai/DeepGEMM>, MIT) needed by the fused DSA
indexer top-k kernel (`csrc/dsa_indexer/`). These are header-only SM100
primitives (tcgen05 UMMA, TMA copy, PTX wrappers). They depend only on
CUTLASS/CuTe (FlashInfer's 3rdparty/cutlass) and the C++ stdlib.

Upstream license: see `LICENSE.deepseek-deepgemm` (MIT, Copyright (c) 2025 DeepSeek).
