# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Five-layer, one-request DSpark decode for the gfx950 specialized backend."""

import torch

from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.models.kimi_k3.amd.specialized_decode.runtime.peer import PeerBuffer, tp_agree


class PersistentDraft:
    def __init__(self, model, alarm):
        self.model = model
        self.alarm = alarm
        self.buffers = []
        self.peers = []
        self.epoch = None

    @classmethod
    def create(cls, model, speculator, alarm):
        config = speculator.vllm_config
        if (
            config.additional_config.get("decode_backend") != "latency_specialized"
            or speculator.num_query_per_req != 7
            or config.lora_config is not None
            or config.weight_transfer_config is not None
            or config.model_config.enable_sleep_mode
        ):
            return None
        from vllm.platforms.rocm import on_gfx950

        tp = get_tp_group()
        supported = (
            on_gfx950()
            and tp.world_size == 8
            and torch.cuda.get_device_properties().multi_processor_count == 256
            and len(model.layers) == 5
        )
        if supported:
            for module in model.layers:
                a = module.self_attn
                if (
                    getattr(a, "W_UK_T", None) is None
                    or getattr(a, "W_UV", None) is None
                ):
                    supported = False
                    break
                weights = [
                    a.fused_qkv_a_proj.weight,
                    a.q_b_proj.weight,
                    a.W_UK_T,
                    a.W_UV,
                    a.o_proj.weight,
                    module.mlp.gate_up_proj.weight,
                    module.mlp.down_proj.weight,
                ]
                supported &= (
                    a.num_local_heads == 8
                    and a.dcp_world_size == 1
                    and a.kv_cache_dtype in ("fp8", "fp8_e4m3")
                    and a.rotary_emb is not None
                    and a.rotary_emb.cos_sin_cache.dtype == torch.float32
                    and a.rotary_emb.cos_sin_cache.shape[-1] == 64
                    and a.rotary_emb.cos_sin_cache.is_contiguous()
                    and not a.use_output_gate
                    and a.qk_nope_head_dim == 128
                    and a.rms_norm_eps == 1e-5
                    and module.input_layernorm.variance_epsilon == 1e-5
                    and module.post_attention_layernorm.variance_epsilon == 1e-5
                    and tuple(a.fused_qkv_a_proj.weight.shape) == (2112, 7168)
                    and tuple(a.q_b_proj.weight.shape) == (1536, 1536)
                    and tuple(a.W_UK_T.shape) == (8, 128, 512)
                    and tuple(a.W_UV.shape) == (8, 512, 128)
                    and tuple(a.o_proj.weight.shape) == (7168, 1024)
                    and tuple(module.mlp.gate_up_proj.weight.shape) == (3584, 7168)
                    and tuple(module.mlp.down_proj.weight.shape) == (7168, 1792)
                    and all(w.dtype == torch.bfloat16 and w.is_cuda for w in weights)
                    and all(
                        w.is_contiguous()
                        for w in (weights[0], weights[1], *weights[4:])
                    )
                )
        if not tp_agree(supported, tp.cpu_group):
            return None
        draft = cls(model, alarm)
        draft._prepare()
        return draft

    def _prepare(self):
        from . import layer

        assert not torch.cuda.is_current_stream_capturing()
        layer.precompile()
        tp = get_tp_group()
        device = next(self.model.parameters()).device
        self.epoch = torch.ones(1, device=device, dtype=torch.int32)
        for i, module in enumerate(self.model.layers):
            a = module.self_attn
            with torch.inference_mode(False):
                peer = PeerBuffer(
                    layer.peer_bytes(8), tp.cpu_group, tp.rank_in_group, 8, device
                )
            peer.bytes.zero_()
            # (layer.SENT_*) the sentinel regions armed (all ones), both
            # parities: [arin | arout] a parity, ``arbytes`` each
            ab = layer.peer_bytes(8) // 4
            for par in range(2):
                if layer.SENT_NEXT:
                    peer.bytes[2 * par * ab : 2 * par * ab + ab].fill_(255)
                if layer.SENT_AR:
                    peer.bytes[2 * par * ab + ab : 2 * par * ab + 2 * ab].fill_(255)
            self.peers.append(peer)
            b = dict(
                in_weight=module.input_layernorm.weight,
                post_weight=module.post_attention_layernorm.weight,
                wqkv=a.fused_qkv_a_proj.weight,
                wqb=a.q_b_proj.weight,
                wuk=torch.empty_like(
                    a.W_UK_T.transpose(1, 2), memory_format=torch.contiguous_format
                ),
                wuv=torch.empty_like(
                    a.W_UV.transpose(1, 2), memory_format=torch.contiguous_format
                ),
                wo=a.o_proj.weight,
                wgu=module.mlp.gate_up_proj.weight,
                wd=module.mlp.down_proj.weight,
                qnw=a.q_a_layernorm.weight,
                kvnw=a.kv_a_layernorm.weight,
                cos_sin=a.rotary_emb.cos_sin_cache,
                qsi=a._q_scale_inv,
                ksi=a._k_scale_inv,
                query=torch.zeros((8, 8, 576), device=device, dtype=torch.bfloat16),
                part_o=torch.empty(
                    (128, 8, 8, 512), device=device, dtype=torch.bfloat16
                ),
                part_lse=torch.empty((128, 8, 8), device=device, dtype=torch.float32),
                scale=torch.empty(1, device=device, dtype=torch.float32),
                vscale=torch.empty(1, device=device, dtype=torch.float32),
                scratch=torch.zeros(layer.NBYTES, device=device, dtype=torch.uint8),
                peers=peer.addresses,
                rank=tp.rank_in_group,
                epoch=self.epoch,
                layer=i,
                health=torch.zeros(257 * 8, device=device, dtype=torch.int32),
                alarm=self.alarm,
            )
            for name in ("x", "residual_mid", "residual_out", "out"):
                b[name] = torch.empty((7, 7168), device=device, dtype=torch.bfloat16)
            self.buffers.append(b)
        self.refresh_weights()
        # a layer's down pushes its rows into the next layer's AR inbox
        # (the last layer's own table: unused)
        for i, b in enumerate(self.buffers):
            b["npeers"] = self.peers[min(i + 1, len(self.peers) - 1)].addresses
        if layer.SENT_AR or layer.SENT_NEXT:
            # no rank pushes into a peer's region before every rank armed its own
            torch.cuda.current_stream().synchronize()
            tp_agree(True, tp.cpu_group)

    def refresh_weights(self):
        """Refill derived weights in place so captured launches keep their pointers."""
        for b, module in zip(self.buffers, self.model.layers):
            a = module.self_attn
            b["wuk"].copy_(a.W_UK_T.transpose(1, 2))
            b["wuv"].copy_(a.W_UV.transpose(1, 2))
            b["scale"].copy_(a.scale * a._q_scale * a._k_scale)
            b["vscale"].copy_(a._k_scale)

    def forward(self, positions, hidden_states):
        if (
            hidden_states.shape != (7, 7168)
            or hidden_states.dtype != torch.bfloat16
            or not hidden_states.is_contiguous()
            or positions.shape != (7,)
            or positions.dtype != torch.int64
            or not positions.is_contiguous()
            or not is_forward_context_available()
        ):
            return None
        from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAMetadata

        ctx = get_forward_context()
        if not isinstance(ctx.attn_metadata, dict) or not isinstance(
            ctx.slot_mapping, dict
        ):
            return None
        bindings = []
        for module in self.model.layers:
            a = module.self_attn
            md = ctx.attn_metadata.get(a.layer_name)
            slots = ctx.slot_mapping.get(a.layer_name)
            if (
                not isinstance(md, AiterMLAMetadata)
                or slots is None
                or slots.ndim != 1
                or slots.numel() < 7
                or slots.dtype != torch.int64
                or not slots.is_contiguous()
                or md.num_actual_tokens != 7
                or md.num_decode_tokens != 7
                or md.decode is None
                or md.decode.varlen_q_rows is not None
            ):
                return None
            d = md.decode
            if (
                d.paged_kv_indices is None
                or d.paged_kv_indptr is None
                or d.paged_kv_indices.dtype != torch.int32
                or d.paged_kv_indptr.dtype != torch.int32
                or d.paged_kv_indptr.numel() != 2
                or not d.paged_kv_indices.is_contiguous()
                or not d.paged_kv_indptr.is_contiguous()
                or a.kv_cache.ndim < 2
                or not a.kv_cache.is_contiguous()
                or a.kv_cache.shape[-1] != 576
                or a.kv_cache.element_size() != 1
            ):
                return None
            bindings.append(
                dict(
                    cache=a.kv_cache,
                    positions=positions[:7],
                    slots=slots[:7],
                    pages=d.paged_kv_indices,
                    indptr=d.paged_kv_indptr,
                )
            )
        from . import layer

        assert self.epoch is not None
        self.epoch.add_(1)
        residual = hidden_states
        steps = []
        for b, binding in zip(self.buffers, bindings):
            steps.append(b | binding | dict(hidden=hidden_states, residual=residual))
            hidden_states, residual = b["out"], b["residual_out"]
        # front 0, (back i, front i + 1) fused, the last back
        layer.run_step(steps)
        return hidden_states, residual
