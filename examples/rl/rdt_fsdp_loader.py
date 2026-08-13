# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FSDP2 shard loader for the sharded-RDT examples.

`rlhf_sharded_rdt_fsdp_ep.py` needs a trainer whose weights are real (so a
weight sync transfers something meaningful) without ever materializing the whole
model on one GPU. Kept out of the example so that file stays about the weight
sync rather than about checkpoint loading, alongside `rdt_vllm_serve.py`.
"""

import os

import torch


def load_sharded_from_disk(model, model_name: str, config) -> None:
    """Stream each FSDP rank's local shard directly from the on-disk safetensors.

    The whole model is NEVER materialized on any single GPU. Call after
    ``fully_shard`` + ``model.to_empty('cuda')``.

    Three cases:
      * Normal params: FSDP2 shards them ``Shard(dim=0)``, so each rank reads only
        its rows ``disk[name][offset : offset + local_rows]``.
      * MoE experts: FUSED in the model (``experts.gate_up_proj`` [E, 2*I, H] and
        ``experts.down_proj`` [E, H, I]) but stored PER-EXPERT on disk. Each rank
        loads only its local experts' gate/up/down and fuses them
        (``gate_up = cat([gate, up], 0)``; down copied directly).
      * Buffers (rotary ``inv_freq``): recomputed from config after ``to_empty``.
    """
    import glob
    import json

    import regex as re
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    snap = snapshot_download(model_name)
    index = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        weight_map = {}
        for f in glob.glob(os.path.join(snap, "*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys():  # noqa: SIM118 (safe_open is not iterable)
                    weight_map[k] = os.path.basename(f)

    handles: dict = {}

    def handle(key: str):
        fn = weight_map[key]
        h = handles.get(fn)
        if h is None:
            h = safe_open(os.path.join(snap, fn), framework="pt", device="cuda:0")
            handles[fn] = h
        return h

    expert_re = re.compile(r"^(.*\.experts)\.(gate_up_proj|down_proj)$")

    with torch.no_grad():
        for name, param in model.named_parameters():
            local = param.to_local().detach()
            lshape, goff = compute_local_shape_and_global_offset(
                param.shape, param.device_mesh, param.placements
            )
            local.zero_()
            n0 = lshape[0]
            if n0 == 0:
                continue
            m = expert_re.match(name)
            if m:
                prefix, kind = m.group(1), m.group(2)
                e0 = goff[0]
                for i in range(n0):
                    e = e0 + i
                    if kind == "gate_up_proj":
                        gk = f"{prefix}.{e}.gate_proj.weight"
                        uk = f"{prefix}.{e}.up_proj.weight"
                        g = handle(gk).get_tensor(gk)
                        u = handle(uk).get_tensor(uk)
                        local[i].copy_(torch.cat([g, u], dim=0))
                    else:
                        dk = f"{prefix}.{e}.down_proj.weight"
                        local[i].copy_(handle(dk).get_tensor(dk))
            else:
                if name not in weight_map:
                    raise RuntimeError(
                        f"param {name!r} is not in the checkpoint and is not a "
                        f"fused expert param (tied weights not handled here)."
                    )
                sliced = handle(name).get_slice(name)
                local[:n0].copy_(sliced[goff[0] : goff[0] + n0])

    rot = model.model.rotary_emb
    fresh = type(rot)(config=config, device=torch.device("cuda"))
    rot.inv_freq = fresh.inv_freq.to("cuda")
    if hasattr(rot, "original_inv_freq"):
        rot.original_inv_freq = rot.inv_freq
    if hasattr(fresh, "attention_scaling"):
        rot.attention_scaling = fresh.attention_scaling
