# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load one GLM-5.2 MoE decoder layer's TP-rank shard for the MonoKernel, straight
from the checkpoint safetensors (independent of vLLM's loaded, requantized and
preshuffled parameters).

Kernel contract (``kernel.weights.LayerWeights.t``, TP8, 8 local heads, inter 256):

  g_in/g_q/g_kv/g_post  BF16 RMSNorm gammas
  w_qkv_a  BF16 [2624, 6144]  cat(q_a_proj, kv_a_proj_with_mqa), replicated
  w_q_b    BF16 [2048, 2048]  q_b_proj rows of this rank's 8 heads
                              (per head nope192 | rope64)
  w_uk     BF16 [8*512, 192]  kv_b_proj per head [:192]^T (q_lat = W_UK^T q_nope)
  w_uv     BF16 [8*256, 512]  kv_b_proj per head [192:]
  w_o      BF16 [6144, 2048]  o_proj columns of this rank's heads
  w_r      BF16 [256, 6144], bias FP32 [256]
  w_ug U8 [257, 512, 3072] / s_ug U8 [257, 512, 192]
                              gate rows | up rows (this rank's 256 of 2048)
  w_dn U8 [257, 6144, 128] / s_dn U8 [257, 6144, 8]
                              down K-columns of this rank

Expert 256 is the shared expert; MXFP4 is e2m1 low-nibble-first with an E8M0 scale per
32 along K.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from safetensors import safe_open

N_ROUTED = 256
HEADS_TOTAL = 64
NOPE, ROPE, VDIM, KV_LORA = 192, 64, 256, 512
INTER_TOTAL = 2048


class _Ckpt:
    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self._files: dict[str, object] = {}

    def _f(self, name):
        fn = self.map[name]
        if fn not in self._files:
            self._files[fn] = safe_open(
                os.path.join(self.path, fn), framework="pt", device="cpu"
            )
        return self._files[fn]

    def get(self, name) -> torch.Tensor:
        return self._f(name).get_tensor(name)

    def slice(self, name):
        return self._f(name).get_slice(name)


def load_glm5_layer(
    ckpt_dir: str,
    layer: int,
    rank: int,
    npes: int = 8,
    device="cuda",
    threads: int = 16,
) -> dict[str, torch.Tensor]:
    assert npes == 8, "the GLM MonoKernel geometry is TP8 here"
    ck = _Ckpt(ckpt_dir)
    p = f"model.layers.{layer}."
    bf = torch.bfloat16
    heads = HEADS_TOTAL // npes
    inter = INTER_TOTAL // npes
    t: dict[str, torch.Tensor] = {}

    def dev(x, dtype=None):
        x = x if dtype is None else x.to(dtype)
        return x.contiguous().to(device, non_blocking=False)

    t["g_in"] = dev(ck.get(p + "input_layernorm.weight"), bf)
    t["g_post"] = dev(ck.get(p + "post_attention_layernorm.weight"), bf)
    t["g_q"] = dev(ck.get(p + "self_attn.q_a_layernorm.weight"), bf)
    t["g_kv"] = dev(ck.get(p + "self_attn.kv_a_layernorm.weight"), bf)
    q_a = ck.get(p + "self_attn.q_a_proj.weight")
    kv_a = ck.get(p + "self_attn.kv_a_proj_with_mqa.weight")
    assert q_a.shape == (2048, 6144) and kv_a.shape == (576, 6144), (
        q_a.shape,
        kv_a.shape,
    )
    t["w_qkv_a"] = dev(torch.cat([q_a, kv_a], 0), bf)
    qb_rows = heads * (NOPE + ROPE)
    t["w_q_b"] = dev(
        ck.slice(p + "self_attn.q_b_proj.weight")[
            rank * qb_rows : (rank + 1) * qb_rows
        ],
        bf,
    )
    kvb_rows = heads * (NOPE + VDIM)
    kv_b = ck.slice(p + "self_attn.kv_b_proj.weight")[
        rank * kvb_rows : (rank + 1) * kvb_rows
    ]
    by_head = kv_b.view(heads, NOPE + VDIM, KV_LORA)
    t["w_uk"] = dev(
        by_head[:, :NOPE].transpose(1, 2).reshape(heads * KV_LORA, NOPE), bf
    )
    t["w_uv"] = dev(by_head[:, NOPE:].reshape(heads * VDIM, KV_LORA), bf)
    o_cols = heads * VDIM
    t["w_o"] = dev(
        ck.slice(p + "self_attn.o_proj.weight")[:, rank * o_cols : (rank + 1) * o_cols],
        bf,
    )
    t["w_r"] = dev(ck.get(p + "mlp.gate.weight"), bf)
    t["bias"] = dev(ck.get(p + "mlp.gate.e_score_correction_bias"), torch.float32)

    # Experts: native MXFP4, gate rows then up rows of this rank; shared expert = index
    # 256.
    E = N_ROUTED + 1
    w_ug = torch.empty(E, 2 * inter, 6144 // 2, dtype=torch.uint8)
    s_ug = torch.empty(E, 2 * inter, 6144 // 32, dtype=torch.uint8)
    w_dn = torch.empty(E, 6144, inter // 2, dtype=torch.uint8)
    s_dn = torch.empty(E, 6144, inter // 32, dtype=torch.uint8)
    r0, r1 = rank * inter, (rank + 1) * inter

    def one(e):
        q = p + (f"mlp.experts.{e}." if e < N_ROUTED else "mlp.shared_experts.")
        for j, proj in enumerate(("gate_proj", "up_proj")):
            w = ck.slice(q + proj + ".weight")
            s = ck.slice(q + proj + ".weight_scale")
            w_ug[e, j * inter : (j + 1) * inter] = w[r0:r1]
            s_ug[e, j * inter : (j + 1) * inter] = s[r0:r1]
        w_dn[e] = ck.slice(q + "down_proj.weight")[
            :, rank * inter // 2 : (rank + 1) * inter // 2
        ]
        s_dn[e] = ck.slice(q + "down_proj.weight_scale")[
            :, rank * inter // 32 : (rank + 1) * inter // 32
        ]

    # Open every shard file once on this thread (safe_open handles are then read-only).
    for e in range(E):
        q = p + (f"mlp.experts.{e}." if e < N_ROUTED else "mlp.shared_experts.")
        for n in ("gate_proj", "up_proj", "down_proj"):
            ck._f(q + n + ".weight")
            ck._f(q + n + ".weight_scale")
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, range(E)))
    for name, x in (("w_ug", w_ug), ("s_ug", s_ug), ("w_dn", w_dn), ("s_dn", s_dn)):
        t[name] = x.to(device)
    for name in ("w_ug", "s_ug", "w_dn", "s_dn"):
        assert t[name].dtype == torch.uint8
    return t
