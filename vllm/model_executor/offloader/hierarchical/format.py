# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-disk ExpertStore layout and safetensors converter."""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

MANIFEST_NAME = "manifest.json"


@dataclass
class LayerExpertMeta:
    layer_id: int
    num_experts: int
    row_nbytes: int
    file_name: str
    tensor_specs: list[dict]
    """Per-row tensor layout: list of {name, nbytes, dtype, shape} in order."""


@dataclass
class ExpertStoreManifest:
    version: int
    model_id: str
    quant: str
    layers: list[LayerExpertMeta]
    checksum: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "model_id": self.model_id,
                "quant": self.quant,
                "checksum": self.checksum,
                "layers": [asdict(layer) for layer in self.layers],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> ExpertStoreManifest:
        data = json.loads(text)
        layers = [LayerExpertMeta(**layer) for layer in data["layers"]]
        return cls(
            version=data["version"],
            model_id=data["model_id"],
            quant=data.get("quant", "unknown"),
            layers=layers,
            checksum=data.get("checksum", ""),
        )


def expert_file_name(layer_id: int) -> str:
    return f"L{layer_id:03d}.experts"


def row_nbytes_from_tensors(tensors: list[torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def pack_expert_row_torch(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Pack expert row tensors into a flat uint8 CPU tensor."""
    flats = [
        t.detach().contiguous().cpu().view(torch.uint8).reshape(-1) for t in tensors
    ]
    return torch.cat(flats, dim=0)


def pack_expert_row(tensors: list[torch.Tensor]) -> bytes:
    """Pack one expert's tensors into a contiguous byte blob (CPU)."""
    return pack_expert_row_torch(tensors).numpy().tobytes()


def unpack_expert_row(
    blob: torch.Tensor, specs: list[dict]
) -> list[torch.Tensor]:
    """Unpack a flat uint8 row into typed tensors (CPU, contiguous)."""
    out: list[torch.Tensor] = []
    offset = 0
    for spec in specs:
        nbytes = int(spec["nbytes"])
        dtype = getattr(torch, spec["dtype"].replace("torch.", ""))
        shape = tuple(spec["shape"])
        chunk = blob[offset : offset + nbytes]
        offset += nbytes
        out.append(chunk.view(dtype).reshape(shape).clone())
    return out


def write_layer_experts(
    path: Path,
    expert_rows: list[list[torch.Tensor]],
) -> tuple[int, list[dict]]:
    """Write contiguous expert file. Returns (row_nbytes, tensor_specs)."""
    assert expert_rows, "no experts to write"
    first = expert_rows[0]
    specs = [
        {
            "name": f"t{i}",
            "nbytes": t.numel() * t.element_size(),
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
        }
        for i, t in enumerate(first)
    ]
    row_nbytes = sum(s["nbytes"] for s in specs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for row in expert_rows:
            packed = pack_expert_row_torch(row)
            assert packed.numel() == row_nbytes
            f.write(packed.numpy().tobytes())
    return row_nbytes, specs


def convert_layer_from_device_params(
    disk_path: str | Path,
    layer_id: int,
    weight_tensors: list[torch.Tensor],
    *,
    model_id: str = "unknown",
    quant: str = "unknown",
) -> LayerExpertMeta:
    """Convert packed expert weight tensors [E, ...] into an ExpertStore layer.

    ``weight_tensors`` is a list of packed params (e.g. [w13, w2, ...scales]),
    each with dim0 = num_experts.
    """
    disk_path = Path(disk_path)
    num_experts = weight_tensors[0].shape[0]
    rows: list[list[torch.Tensor]] = []
    for e in range(num_experts):
        rows.append([w[e].contiguous() for w in weight_tensors])
    file_name = expert_file_name(layer_id)
    row_nbytes, specs = write_layer_experts(disk_path / file_name, rows)
    # Name specs after source parameter order
    for i, w in enumerate(weight_tensors):
        specs[i]["name"] = f"param{i}"
        specs[i]["dtype"] = str(w.dtype).replace("torch.", "")
    meta = LayerExpertMeta(
        layer_id=layer_id,
        num_experts=num_experts,
        row_nbytes=row_nbytes,
        file_name=file_name,
        tensor_specs=specs,
    )
    _upsert_manifest(disk_path, meta, model_id=model_id, quant=quant)
    return meta


def _upsert_manifest(
    disk_path: Path,
    layer_meta: LayerExpertMeta,
    *,
    model_id: str,
    quant: str,
) -> None:
    manifest_path = disk_path / MANIFEST_NAME
    if manifest_path.exists():
        manifest = ExpertStoreManifest.from_json(manifest_path.read_text())
        layers = [layer for layer in manifest.layers if layer.layer_id != layer_meta.layer_id]
        layers.append(layer_meta)
        layers.sort(key=lambda x: x.layer_id)
        manifest.layers = layers
    else:
        manifest = ExpertStoreManifest(
            version=1,
            model_id=model_id,
            quant=quant,
            layers=[layer_meta],
        )
    manifest_path.write_text(manifest.to_json())


def load_manifest(disk_path: str | Path) -> ExpertStoreManifest | None:
    path = Path(disk_path) / MANIFEST_NAME
    if not path.exists():
        return None
    return ExpertStoreManifest.from_json(path.read_text())


def read_expert_row_bytes(
    file_path: Path, expert_id: int, row_nbytes: int
) -> bytes:
    """Synchronous pread of one expert row."""
    offset = expert_id * row_nbytes
    with open(file_path, "rb") as f:
        f.seek(offset)
        data = f.read(row_nbytes)
    if len(data) != row_nbytes:
        raise IOError(
            f"Short read for expert {expert_id} in {file_path}: "
            f"{len(data)}/{row_nbytes}"
        )
    return data


# Keep struct import used for potential header extensions.
_ = struct
