# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loading and caching of OSCAR per-layer rotation matrices.

The reference exporter (``rotation/compute_kv_rotation.py``) saves a
checkpoint of the form::

    {
        "format_version": 1,
        "objective": "qqt_r_h_pbr",
        "source_grouping": "layer",
        "layers": {
            <layer_id>: {
                "layer_id": int,
                "rotation": Tensor[head_dim, head_dim] (fp32, orthogonal),
                "eigenvalues": Tensor[head_dim] (fp32),
            },
            ...
        },
    }

Layer ids may be stored as ``int`` or ``str``. A stacked ``[num_layers,
head_dim, head_dim]`` tensor (one matrix per layer) is also accepted.
"""

from __future__ import annotations

import re
from functools import lru_cache

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def layer_index_from_name(layer_name: str) -> int | None:
    """Best-effort extraction of the global decoder layer index.

    vLLM attention layer names look like
    ``model.layers.<idx>.self_attn.attn``. Returns ``None`` when no index
    can be parsed (e.g. non-standard naming) so the caller can fall back to
    an identity rotation.
    """
    m = _LAYER_IDX_RE.search(layer_name)
    if m is not None:
        return int(m.group(1))
    # Fallback: last integer token in the name.
    ints = re.findall(r"\d+", layer_name)
    return int(ints[-1]) if ints else None


@lru_cache(maxsize=8)
def _load_checkpoint(path: str) -> dict[int, torch.Tensor]:
    """Load and normalize a rotation checkpoint into ``{layer_id: matrix}``.

    Cached per path: the same file is shared across all attention layers and
    loaded once. Returns fp32 CPU tensors; callers move/cast as needed.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)

    out: dict[int, torch.Tensor] = {}
    if isinstance(obj, dict) and "layers" in obj:
        for k, entry in obj["layers"].items():
            lid = int(k)
            rot = entry["rotation"] if isinstance(entry, dict) else entry
            out[lid] = rot.float().contiguous()
    elif isinstance(obj, dict):
        # Flat ``{layer_id: matrix}`` mapping.
        for k, rot in obj.items():
            out[int(k)] = rot.float().contiguous()
    elif torch.is_tensor(obj) and obj.dim() == 3:
        # Stacked [num_layers, head_dim, head_dim].
        for lid in range(obj.shape[0]):
            out[lid] = obj[lid].float().contiguous()
    else:
        raise ValueError(
            f"Unrecognized OSCAR rotation checkpoint structure at {path!r}: {type(obj)}"
        )
    logger.info("Loaded OSCAR rotation checkpoint %s (%d layers)", path, len(out))
    return out


def get_layer_rotation(
    path: str,
    layer_name: str,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the ``[head_dim, head_dim]`` rotation for one attention layer.

    Falls back to the identity matrix when ``path`` is empty, the layer index
    cannot be parsed, or the checkpoint has no entry for this layer — in which
    case OSCAR degrades to (clipped) INT2 with no rotation.
    """
    if not path:
        return torch.eye(head_dim, device=device, dtype=dtype)

    table = _load_checkpoint(path)
    lid = layer_index_from_name(layer_name)
    rot = table.get(lid) if lid is not None else None
    if rot is None:
        logger.warning_once(
            "OSCAR: no rotation for layer %r (idx=%s) in %s; using identity.",
            layer_name,
            lid,
            path,
        )
        return torch.eye(head_dim, device=device, dtype=dtype)
    if rot.shape != (head_dim, head_dim):
        raise ValueError(
            f"OSCAR rotation for layer {lid} has shape {tuple(rot.shape)}, "
            f"expected ({head_dim}, {head_dim})."
        )
    return rot.to(device=device, dtype=dtype).contiguous()
