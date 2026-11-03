# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mono kernels of the decoder layers: vLLM tensors -> launches. A
``MonoLayer`` runs one whole decoder layer (GatedDeltaNet or full attention,
then the MoE with the shared expert, both TP all-reduces in the kernel) on a
planned step. The kernels read vLLM's loaded weights and caches in place;
``bind`` registers them with the kernels (again whenever the caches move)."""

import torch
from torch import nn


def _kv(attn) -> torch.Tensor:
    kv = attn.kv_cache
    return kv[0] if isinstance(kv, (list, tuple)) else kv


class MonoLayer:
    """One decoder layer. ``kid``: its kernel-side id (the scratch and mailbox
    tag slot); ``name``: its mailbox."""

    def __init__(self, layer: nn.Module, kid: int, name: str):
        self.layer, self.kid, self.name = layer, kid, name
        self.attn = layer.layer_type == "full_attention"
        # the key of this layer's metadata in the forward context
        self.prefix = (
            layer.self_attn.attn.layer_name if self.attn else layer.linear_attn.prefix
        )

    def caches(self) -> list[torch.Tensor]:
        if self.attn:
            return [_kv(self.layer.self_attn.attn)]
        state = self.layer.linear_attn.kv_cache
        return [state[0], state[1]]

    def weights(self) -> list[torch.Tensor]:
        """The kernel's tensor list: norms, router, shared gate, the mixer's
        output projection, routed and shared experts, then the mixer."""
        layer, mlp = self.layer, self.layer.mlp
        experts = getattr(mlp.experts, "routed_experts", mlp.experts)
        mixer = layer.self_attn if self.attn else layer.linear_attn
        out = mixer.o_proj if self.attn else mixer.out_proj
        t = [
            layer.input_layernorm.weight,
            layer.post_attention_layernorm.weight,
            mlp.gate.weight,
            mlp.shared_expert_gate.weight,
            out.weight,
            out.weight_scale,
            experts.w13_weight,
            experts.w13_weight_scale,
            experts.w2_weight,
            experts.w2_weight_scale,
            mlp.shared_expert.gate_up_proj.weight,
            mlp.shared_expert.gate_up_proj.weight_scale,
            mlp.shared_expert.down_proj.weight,
            mlp.shared_expert.down_proj.weight_scale,
        ]
        if self.attn:
            return t + [
                mixer.qkv_proj.weight,
                mixer.qkv_proj.weight_scale,
                mixer.q_norm.weight,
                mixer.k_norm.weight,
            ]
        return t + [
            mixer.in_proj_qkvz.weight,
            mixer.in_proj_qkvz.weight_scale,
            mixer.in_proj_ba.weight,
            mixer.in_proj_ba.weight_scale,
            mixer.conv1d.weight,
            mixer.A_log,
            mixer.dt_bias,
            mixer.norm.weight,
        ]

    def bind(self, mailbox: tuple[int, int]) -> None:
        """Registers the weights and the current caches (the kernel checks every
        shape, dtype and stride and raises on a mismatch). Registering a layer
        again only updates its pointers."""
        own, peer = mailbox
        torch.ops._xpu_C.mono_register_layer(
            self.kid, self.attn, own, peer, self.weights() + self.caches()
        )

    def _metadata(self, md, verify: bool):
        m = md[self.prefix]
        if self.attn:
            return m.block_table, m.seq_lens, m.slot_mapping
        if verify:
            return m.spec_state_indices_tensor, m.num_accepted_tokens, None
        return m.non_spec_state_indices_tensor, None, None

    def launch(
        self,
        q: int,
        md,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The layer's (hidden_states, residual) out, like the stock layer's
        forward: ``q`` 1 is a decode step, else a verify step of q tokens a
        request (the plan checked the metadata and the width)."""
        if residual is None:
            # layer 0: the stock layer takes the input as the residual
            residual = torch.zeros_like(hidden_states)
        out = torch.empty_like(hidden_states)
        res = torch.empty_like(hidden_states)
        m0, m1, m2 = self._metadata(md, q > 1)
        ops = torch.ops._xpu_C
        if q > 1:
            served = ops.mono_decode_layer_spec(
                self.kid, q, hidden_states, residual, out, res, positions, m0, m1, m2
            )
            # the plan only picks widths the verify kernels have
            assert served, f"no verify kernel for {hidden_states.shape[0]} rows x {q}"
        else:
            ops.mono_decode_layer(
                self.kid, hidden_states, residual, out, res, positions, m0, m1, m2
            )
        return out, res
