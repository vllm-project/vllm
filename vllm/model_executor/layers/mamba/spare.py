# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline SPARE basis loading and homogeneous Mamba2 projection."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load


def pack_basis(basis: torch.Tensor, groups: int) -> torch.Tensor:
    """Pack contiguous heads of each native group for a grouped GEMM."""
    heads, native_dim, rank = basis.shape
    return (
        basis.reshape(groups, heads // groups, native_dim, rank)
        .transpose(1, 2)
        .reshape(groups, native_dim, heads // groups * rank)
        .contiguous()
    )


def project_bc(
    b: torch.Tensor, c: torch.Tensor, packed_basis: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project each native B/C group to its heads, keeping input storage dtype."""

    def project(vectors: torch.Tensor) -> torch.Tensor:
        grouped = torch.bmm(vectors.transpose(0, 1).float(), packed_basis)
        grouped = grouped.to(vectors.dtype).transpose(0, 1)
        return grouped.unflatten(-1, (-1, rank)).flatten(1, 2)

    return project(b), project(c)


def load_bases(
    path: str,
    sha256: str,
    method: str,
    model: str,
    revision: str,
    layer_indices: tuple[int, ...],
    heads: int,
    native_dim: int,
    rank: int,
) -> dict[int, torch.Tensor]:
    """Read one engine's bases on CPU and verify their checkpoint binding."""
    artifact = Path(path)
    # Read and parse the same bytes so an edited file cannot race the hash check.
    data = artifact.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise ValueError(f"SPARE basis SHA-256 mismatch: {digest}")

    # SafeTensors' in-memory loader validates the header and tensor layout.
    # The header metadata holds the artifact identity alongside its tensors.
    header_size = int.from_bytes(data[:8], "little")
    with torch.device("cpu"):
        tensors = load(data)
    metadata = json.loads(data[8 : 8 + header_size]).get("__metadata__", {})
    expected = {
        "format": "spare-mamba2-v1",
        "method": method,
        "model": model,
        "revision": revision,
        "native_dim": str(native_dim),
        "rank": str(rank),
        "heads": str(heads),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"SPARE basis metadata mismatch for {key!r}")
    keys = {f"layer.{layer}" for layer in layer_indices}
    if set(tensors) != keys:
        raise ValueError("SPARE basis tensors must match the Mamba layer indices")

    bases: dict[int, torch.Tensor] = {}
    with torch.device("cpu"):
        identity = torch.eye(rank, dtype=torch.float64)
        for layer in layer_indices:
            basis = tensors[f"layer.{layer}"]
            if basis.dtype != torch.float32 or basis.shape != (heads, native_dim, rank):
                raise ValueError(f"Invalid SPARE basis dtype or shape at layer {layer}")
            check = basis.double()
            defect = (check.transpose(-2, -1) @ check - identity).abs().amax()
            if not torch.isfinite(defect) or defect.item() > 1e-5:
                raise ValueError(f"Nonorthonormal SPARE basis at layer {layer}")
            if method == "identity" and not torch.equal(
                basis, identity.float().expand_as(basis)
            ):
                raise ValueError(f"Expected an identity SPARE basis at layer {layer}")
            bases[layer] = basis
    return bases
