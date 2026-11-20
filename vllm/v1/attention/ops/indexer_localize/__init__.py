# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Die-local sparse-attention indexer decode (`VLLM_INDEXER_LOCALIZE`).

On GPUs with two memory locality domains, the indexer's decode scan reads
each page with the SMs of the domain that holds it:

- `placement`: the KV cache alternates between the domains in runs of whole
  blocks, and each domain gets a green-context stream on its SMs.
"""
