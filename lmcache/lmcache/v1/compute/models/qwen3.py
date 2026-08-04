# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.compute.models.base import LMCBaseModel


class LMCQwen3Model(LMCBaseModel):
    def _process_qkv(self, q, k, v, layer):
        """Process QKV tensors for Qwen3 model with q_norm and k_norm layers."""
        # Qwen3 has q_norm and k_norm layers
        q_by_head = q.view(
            *q.shape[:-1],
            q.shape[-1] // layer.self_attn.head_dim,
            layer.self_attn.head_dim,
        )
        q_by_head = layer.self_attn.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(
            *k.shape[:-1],
            k.shape[-1] // layer.self_attn.head_dim,
            layer.self_attn.head_dim,
        )
        k_by_head = layer.self_attn.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        return q, k, v
