# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FSDP2 checkpoint loading for ``rlhf_sharded_rdt_small_ep.py``.

The example needs a trainer whose weights are real, so the weight sync transfers
something meaningful, without ever materializing the whole model on one GPU.
``load_sharded_from_disk`` does that; ``CheckpointReader`` (snapshot resolution,
the weight map, cached ``safe_open`` handles) and ``local_shard`` (the FSDP2
``Shard(0)`` slice math) are the pieces it is built from. Kept out of the example
so that file stays about the weight sync rather than about checkpoint loading,
alongside ``rdt_vllm_serve.py``.
"""

import json
import os

import torch


def _weight_map(snap: str) -> dict[str, str]:
    """Parameter name -> shard filename, from the index if there is one."""
    import glob

    from safetensors import safe_open

    index = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            return json.load(f)["weight_map"]
    # Single-shard checkpoints ship no index; recover it by scanning.
    weight_map = {}
    for path in glob.glob(os.path.join(snap, "*.safetensors")):
        with safe_open(path, framework="pt") as sf:
            for k in sf.keys():  # noqa: SIM118 (safe_open is not iterable)
                weight_map[k] = os.path.basename(path)
    return weight_map


class CheckpointReader:
    """Cached safetensors reader over a downloaded HF checkpoint.

    The snapshot directory is resolved through ``snapshot_download``, which for
    an already-cached model is a pure cache lookup and otherwise fetches it.
    Nothing here assumes a path: set ``HF_HOME`` (or ``HF_HUB_CACHE``) if the
    cache is not in its default location. Pre-downloading is still worth it --
    every trainer rank builds one of these, so a cold cache becomes one
    concurrent download per GPU.
    """

    def __init__(self, model_name: str, device: str = "cuda:0"):
        from huggingface_hub import snapshot_download

        self.snap = snapshot_download(model_name)
        self.weight_map = _weight_map(self.snap)
        self._device = device
        self._handles: dict = {}

    def __contains__(self, key: str) -> bool:
        return key in self.weight_map

    def names(self) -> list[str]:
        return list(self.weight_map)

    def handle(self, key: str):
        """The open safetensors file holding ``key``, opened at most once."""
        from safetensors import safe_open

        fn = self.weight_map[key]
        h = self._handles.get(fn)
        if h is None:
            h = safe_open(
                os.path.join(self.snap, fn), framework="pt", device=self._device
            )
            self._handles[fn] = h
        return h

    def get_slice(self, key: str):
        """A lazy slice handle -- reads only the rows actually indexed."""
        return self.handle(key).get_slice(key)

    def get_tensor(self, key: str) -> torch.Tensor:
        return self.handle(key).get_tensor(key)


def local_shard(param) -> tuple[torch.Tensor, int, int]:
    """Zeroed local storage of an FSDP2 ``Shard(0)`` param, its rows and offset.

    Returns ``(local, rows, offset)`` so the caller can fill
    ``local[:rows]`` from ``disk[offset : offset + rows]``. ``rows == 0`` on
    ranks the param does not reach.
    """
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    local = param.to_local().detach()
    lshape, goff = compute_local_shape_and_global_offset(
        param.shape, param.device_mesh, param.placements
    )
    local.zero_()
    return local, lshape[0], goff[0]


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
    import regex as re

    reader = CheckpointReader(model_name)
    expert_re = re.compile(r"^(.*\.experts)\.(gate_up_proj|down_proj)$")

    with torch.no_grad():
        for name, param in model.named_parameters():
            local, n0, off = local_shard(param)
            if n0 == 0:
                continue
            m = expert_re.match(name)
            if m:
                prefix, kind = m.group(1), m.group(2)
                for i in range(n0):
                    e = off + i
                    if kind == "gate_up_proj":
                        gk = f"{prefix}.{e}.gate_proj.weight"
                        uk = f"{prefix}.{e}.up_proj.weight"
                        g = reader.get_tensor(gk)
                        u = reader.get_tensor(uk)
                        local[i].copy_(torch.cat([g, u], dim=0))
                    else:
                        dk = f"{prefix}.{e}.down_proj.weight"
                        local[i].copy_(reader.get_tensor(dk))
            else:
                if name not in reader:
                    raise RuntimeError(
                        f"param {name!r} is not in the checkpoint and is not a "
                        f"fused expert param (tied weights not handled here)."
                    )
                local[:n0].copy_(reader.get_slice(name)[off : off + n0])

    rot = model.model.rotary_emb
    fresh = type(rot)(config=config, device=torch.device("cuda"))
    rot.inv_freq = fresh.inv_freq.to("cuda")
    if hasattr(rot, "original_inv_freq"):
        rot.original_inv_freq = rot.inv_freq
    if hasattr(fresh, "attention_scaling"):
        rot.attention_scaling = fresh.attention_scaling
