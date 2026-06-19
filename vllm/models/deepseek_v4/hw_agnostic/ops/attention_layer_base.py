# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AttentionLayerBase — re-exported from upstream.

The worker's ``GPUModelRunner.get_kv_cache_spec()`` walks the model's
static-forward-context with ``get_layers_from_vllm_config(cfg,
AttentionLayerBase)`` — using the **upstream** ``AttentionLayerBase``
class — to collect every layer that owns a KV cache. If the vendored
DSv4 attention layers (``DeepseekV4MultiHeadLatentAttention``,
``DeepseekV4SWACache``, ``DeepseekV4IndexerCache``,
``CompressorStateCache``) inherit from a *vendored* ``AttentionLayerBase``
copy, ``isinstance(layer, upstream.AttentionLayerBase)`` returns False
and the worker reports zero KV cache specs — which collapses
``HybridKVCacheCoordinator`` to "needs at least two attention groups".

Re-exporting upstream keeps both sides referencing the same ABC. Same
justification as the
``MoEActivation`` / ``FusedMoEMethodBase`` / ``MoERunnerInterface`` /
``SparseAttnIndexer`` carve-outs: pure ABC, must have identity-matching
cross-boundary.
"""

from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

__all__ = ["AttentionLayerBase"]
