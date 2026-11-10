# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The routed-expert front of the specialized decode's TP 1 MoE layers: one
CuTe DSL kernel for the softmax top-10, its routing metadata and the NVFP4
quantization of the layer input (`front.py`), which `runtime/moe.py` runs
before FlashInfer's TRT-LLM expert GEMMs."""
