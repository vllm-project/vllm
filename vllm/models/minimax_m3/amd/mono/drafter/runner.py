# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the DSpark drafter kernels: one rank's weights, scratch, peers and
the launches of a draft step."""

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.models.minimax_m3.amd.mono.config import MonoUnsupported
from vllm.models.minimax_m3.amd.mono.drafter import layout
from vllm.models.minimax_m3.amd.mono.drafter.layer import build_drafter
from vllm.models.minimax_m3.amd.mono.drafter.wide import build_wide
from vllm.models.minimax_m3.amd.mono.peer_buffer import PeerBuffer
from vllm.platforms import current_platform

# the symmetric buffer: the narrow kernel's all-reduce slots, then the wide kernel's
PEER_BYTES = layout.SYM_BYTES + layout.SYM_W_BYTES
WIDE_BASE = layout.SYM_BYTES
WAIT_S = 10.0

_GEMV = {
    "qkv_w": ((2304, 6144), layout.T_QKV_W),
    "qkv_s": ((2304, 192), layout.T_QKV_S),
    "o_w": ((6144, 2048), layout.T_O_W),
    "o_s": ((6144, 64), layout.T_O_S),
    "gu_w": ((6144, 6144), layout.T_GU_W),
    "gu_s": ((6144, 192), layout.T_GU_S),
    "d_w": ((6144, 3072), layout.T_D_W),
    "d_s": ((6144, 96), layout.T_D_S),
}
_NORMS = {
    "in_ln": layout.T_IN_LN,
    "post_ln": layout.T_POST_LN,
    "q_norm": layout.T_Q_NORM,
    "k_norm": layout.T_K_NORM,
}


def _bf16(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.bfloat16).contiguous()


def _layer_weights(layer) -> dict[str, torch.Tensor]:
    a, m = layer.self_attn, layer.mlp
    w = {
        "qkv_w": a.qkv_proj.weight,
        "qkv_s": a.qkv_proj.weight_scale,
        "o_w": a.o_proj.weight,
        "o_s": a.o_proj.weight_scale,
        "gu_w": m.gate_up_proj.weight,
        "gu_s": m.gate_up_proj.weight_scale,
        "d_w": m.down_proj.weight,
        "d_s": m.down_proj.weight_scale,
        "in_ln": _bf16(layer.input_layernorm.weight),
        "post_ln": _bf16(layer.post_attention_layernorm.weight),
        "q_norm": _bf16(a.q_norm.weight),
        "k_norm": _bf16(a.k_norm.weight),
    }
    for name, (shape, _) in _GEMV.items():
        t = w[name]
        if tuple(t.shape) != shape or t.element_size() != 1 or not t.is_contiguous():
            raise MonoUnsupported(f"{name} {tuple(t.shape)} {t.dtype}")
    return w


def _kv_cache(layer) -> torch.Tensor:
    kv = layer.self_attn.attn.kv_cache
    return kv[0] if isinstance(kv, (list, tuple)) else kv


class DrafterRunner:
    """The drafter kernels of one TP4 rank for one ``MiniMaxM3DSparkModel``.

    Built outside CUDA graph capture (the peer buffer's IPC exchange is collective);
    a re-bound KV cache (``stale``) needs a new runner.
    """

    def __init__(self, model) -> None:
        device_index = torch.accelerator.current_device_index()
        cus = current_platform.get_device_properties(device_index).multi_processor_count
        if cus != layout.BLOCKS:
            raise MonoUnsupported(
                f"{cus} CUs: the kernels need all {layout.BLOCKS} CTAs co-resident"
            )
        if get_tensor_model_parallel_world_size() != 4:
            raise MonoUnsupported("TP != 4")
        if len(model.layers) != layout.LAYERS:
            raise MonoUnsupported(f"{len(model.layers)} draft layers")
        cos_sin = model._rope_cos_sin_cache
        if cos_sin.dtype != torch.bfloat16 or cos_sin.shape[-1] != layout.ROT:
            raise MonoUnsupported("rope cache")
        self.cos_sin = cos_sin
        self.layers = [_layer_weights(layer) for layer in model.layers]
        self.final_norm = _bf16(model.norm.weight)
        self.layer_names = [layer.self_attn.attn.layer_name for layer in model.layers]
        self._layers = model.layers
        self.caches = [_kv_cache(layer) for layer in model.layers]
        for kv in self.caches:
            if kv.numel() == 0 or tuple(kv.shape[1:]) != (1, layout.BLOCK, 256):
                raise MonoUnsupported(f"draft KV cache {tuple(kv.shape)}")
            if kv.element_size() != 1 or not kv.is_contiguous():
                raise MonoUnsupported(f"draft KV cache {kv.dtype}")

        dev = torch.device("cuda", device_index)
        self.rank = get_tensor_model_parallel_rank()
        self.peers = PeerBuffer(PEER_BYTES, get_tp_group().cpu_group, self.rank, 4, dev)
        self.pe = [int(a) for a in self.peers.addresses.tolist()]
        self.packed = {
            name: torch.stack([w[name] for w in self.layers]).contiguous()
            for name in _GEMV
        }
        tab = torch.zeros(layout.LAYERS, 16, dtype=torch.int64)
        for i, (w, kv) in enumerate(zip(self.layers, self.caches)):
            for name, slot in _NORMS.items():
                tab[i, slot] = w[name].data_ptr()
            tab[i, layout.T_CACHE] = kv.data_ptr()
            tab[i, layout.T_NBLK] = kv.shape[0]
        self.tab = tab.to(dev)
        self.scratch = torch.zeros(layout.SCRATCH_BYTES, dtype=torch.uint8, device=dev)
        self.scratch_w = torch.zeros(
            layout.W_SCRATCH_BYTES, dtype=torch.uint8, device=dev
        )
        self.health = torch.zeros(4, dtype=torch.int32, device=dev)
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)
        wait = int(WAIT_S * 1e8)
        self.narrow = build_drafter(wait_ticks=wait)
        self.wide = build_wide(WIDE_BASE, wait_ticks=wait)
        self.no_slots = torch.full(
            (layout.QPR * layout.MAX_REQS,), -1, dtype=torch.int64, device=dev
        )

    def stale(self) -> bool:
        """Whether vLLM re-bound the draft KV caches since the build."""
        now = [_kv_cache(layer).data_ptr() for layer in self._layers]
        return now != [kv.data_ptr() for kv in self.caches]

    def close(self) -> None:
        self.peers.close()

    def _launch(self, fn, scratch, x, out, bt, pos, slots, seq_lens, n) -> None:
        pk = self.packed
        fn(
            x.data_ptr(),
            out.data_ptr(),
            self.tab.data_ptr(),
            pk["qkv_w"].data_ptr(),
            pk["qkv_s"].data_ptr(),
            pk["o_w"].data_ptr(),
            pk["o_s"].data_ptr(),
            pk["gu_w"].data_ptr(),
            pk["gu_s"].data_ptr(),
            pk["d_w"].data_ptr(),
            pk["d_s"].data_ptr(),
            self.final_norm.data_ptr(),
            self.cos_sin.data_ptr(),
            bt.data_ptr(),
            pos.data_ptr(),
            slots.data_ptr(),
            seq_lens.data_ptr(),
            scratch.data_ptr(),
            int(self.peers.local),
            *self.pe,
            self.step.data_ptr(),
            self.health.data_ptr(),
            self.rank,
            bt.stride(0),
            n,
            stream=torch.cuda.current_stream(),  # flydsl rejects torch.Stream
        )

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        """The 5 layers and the final norm of a draft step: (n, 6144) bf16."""
        md, _, _, slots = get_attention_context(self.layer_names[0])
        n = inputs_embeds.shape[0]
        if slots is None:
            slots = self.no_slots[:n]
        bt = md.block_table
        x = inputs_embeds.contiguous()
        out = torch.empty((n, layout.HIDDEN), dtype=torch.bfloat16, device=x.device)
        q = layout.QPR
        n_req = n // q
        # a request reads only its own rows and cache and every GEMV row is per
        # token, so a step splits into independent launches (a fresh tag each)
        for r0 in range(0, n_req, 8):
            k = min(8, n_req - r0)
            a, b = r0 * q, (r0 + k) * q
            if k >= 3:
                fn, scratch = self.wide, self.scratch_w
            else:
                fn, scratch = self.narrow, self.scratch
            self._launch(
                fn,
                scratch,
                x[a:b],
                out[a:b],
                bt[r0:],
                positions[a:],
                slots[a:],
                md.seq_lens[r0:],
                b - a,
            )
            self.step.add_(1)
        return out
